"""CPU-only protocol and fake-artifact checks; no dataset or real model training."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from scripts import run_pretrain_component_strength as sweep


class ComponentStrengthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('global_pi_ctx5_noce_noneg', 'projection_only_v1_baseline', 'pretrain_component_strength', 'current_tuning_reference'):
            shutil.copytree(sweep.ROOT/'configs'/name, self.root/'configs'/name)
        (self.root/'scripts').mkdir()
        shutil.copy(sweep.ROOT/'scripts/run_projection_only_v1_baseline.py', self.root/'scripts')
        for obj, values in ((sweep, {'ROOT':self.root}), (sweep.baseline, {
            'ROOT':self.root, 'CONFIG':self.root/'configs/projection_only_v1_baseline'})):
            p = patch.multiple(obj, **values)
            p.start()
            self.addCleanup(p.stop)

    def fake_trainer(self, runner):
        class Process:
            def __init__(self, command, **kwargs):
                assert command[1] == 'train_cm_pretrain.py'
                assert '--resume' not in command
                cfg=runner.validate('pretrain'); d,table=runner.paths('pretrain')
                (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                checkpoint=d/'best_component_anchor_pretrain.pth'
                prefixes=('iqa.backbone.fusion.semantic_proj.', 'iqa.quality_adapter.',
                          'iqa.shared_image_projector.', 'iqa.shared_text_projector.',
                          'iqa.local_patch_proj.', 'iqa.quality_local_encoder.')
                torch.save(dict(cfg=cfg,run_id=cfg['run_id'],epoch=4,checkpoint_scope='representation',
                    selection_metric='val_component_anchor_SROCC',
                    model={p+'weight':torch.zeros(1) for p in prefixes}), checkpoint)
                with table.open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=['event','metrics_json']); writer.writeheader()
                    writer.writerow(dict(event='final',metrics_json=json.dumps(dict(
                        selection_metric='val_component_anchor_SROCC',best_component_anchor_epoch=4,
                        best_component_anchor_checkpoint=str(checkpoint)))))
                self.stdout=iter(['fake pretraining completed\n'])
            def wait(self): return 0
        return Process

    def test_only_intended_weights_change_and_independent_paths(self):
        original=sweep.baseline.validate('pretrain'); paths=[]
        for variant, (loss, multiplier) in sweep.VARIANTS.items():
            runner=sweep.make_runner(variant); cfg=runner.validate('pretrain')
            expected=copy.deepcopy(original)
            for key in ('run_id','output_dir','results_csv'): expected[key]=cfg[key]
            for component in ('artifact','aesthetics'):
                key=f'lambda_{component}_{loss}'
                expected['train'][key]*=multiplier
            self.assertEqual(cfg,expected)
            self.assertFalse(any(k in cfg['train'] for k in ('resume','pretrained_checkpoint','refit_plan')))
            self.assertEqual(cfg['dataset']['split_strategy'],'official')
            paths.append(runner.paths('pretrain'))
        self.assertEqual(len({str(p) for p in paths}),4)
        self.assertEqual(sweep.make_runner('anchor050').validate('pretrain')['train']['lambda_artifact_anchor'],.005)
        with self.assertRaises(ValueError): sweep.make_runner('invalid')

    def test_dry_run_cpu_refusal_and_preservation(self):
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v)
            with patch.object(r.subprocess,'Popen',side_effect=AssertionError('no trainer')):
                with contextlib.redirect_stdout(io.StringIO()): r.run('pretrain',True)
            with patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'CUDA'): r.run('pretrain')
            d,table=r.paths('pretrain'); self.assertFalse(d.exists())
            table.parent.mkdir(exist_ok=True); table.write_text('preserve')
            with self.assertRaises(FileExistsError): r.run('pretrain',True)
            self.assertEqual(table.read_text(),'preserve')

    def test_config_and_downstream_rejected(self):
        r=sweep.make_runner('anchor050')
        with self.assertRaises(ValueError): r.validate('unknown')
        p=r.CONFIG/'pretrain.yaml'; cfg=yaml.safe_load(p.read_text())
        cfg['train']['pretrained_checkpoint']='old.pth'; p.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError): r.run('pretrain',True)

    def test_fake_completion_pack_and_cross_variant_rejection(self):
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v)
            with patch('torch.cuda.is_available',return_value=True), patch.object(
                r.subprocess,'Popen',self.fake_trainer(r)), patch(
                'utils.run_metadata.collect_run_metadata',return_value={'fake':True}):
                with contextlib.redirect_stdout(io.StringIO()): r.run('pretrain')
            source=r.completed('pretrain')
            self.assertEqual(source['selected_epoch'],4)
            self.assertEqual(len(source['sha256']),64)
            with contextlib.redirect_stdout(io.StringIO()): r.pack('pretrain')
            archive=self.root/f'{r.VERSION}_pretrain_results.tar.gz'
            with tarfile.open(archive) as tar:
                names=tar.getnames()
                for suffix in ('console.log','config_snapshot.yaml','completion_manifest.json','run_metadata.json','source_fingerprint.json'):
                    self.assertTrue(any(n.endswith(suffix) for n in names))
                self.assertTrue(any(n.endswith('.csv') for n in names))
                self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
            with self.assertRaises(FileExistsError): r.pack('pretrain')
            with self.assertRaises(FileExistsError): r.run('pretrain')
            p=Path(source['checkpoint']); checkpoint=torch.load(p,weights_only=False)
            checkpoint['run_id']='other_variant'; torch.save(checkpoint,p)
            with self.assertRaises(ValueError): r.completed('pretrain')

    def test_downstream_configs_preserve_reference_and_use_own_source(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            for stage in sweep.STAGES[1:]:
                cfg=r.validate(stage)
                expected=yaml.safe_load((self.root/f'configs/current_tuning_reference/{stage}.yaml').read_text())
                for key in ('run_id','output_dir','results_csv'): expected[key]=cfg[key]
                expected['train']['pretrained_checkpoint']=sweep.expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
                self.assertEqual(cfg,expected)
                self.assertEqual(cfg['train']['layerwise_lr']['shared_projection_mult'],1.0)
                self.assertEqual(cfg['split_seed'],3407 if stage=='agiqa1k' else 42)
                self.assertEqual(cfg['train']['lambda_srocc_quality'],{'agiqa1k':.03,'agiqa3k':.02,'aigciqa2023':.04}[stage])

    def test_pipeline_order_failure_and_existing_preflight(self):
        r=sweep.make_runner('anchor050')
        with patch.object(sweep,'run_stage') as launch:
            sweep.run_pipeline(r,'all')
            self.assertEqual([c.args[1] for c in launch.call_args_list],list(sweep.STAGES))
        with patch.object(sweep,'run_stage',side_effect=RuntimeError('failed')) as launch:
            with self.assertRaises(RuntimeError): sweep.run_pipeline(r,'all')
            self.assertEqual(launch.call_count,1)
        with patch.object(sweep,'run_stage') as launch:
            with self.assertRaises(FileNotFoundError): sweep.run_pipeline(r,'downstream')
            launch.assert_not_called()
        d,_=r.paths('aigciqa2023');d.mkdir(parents=True)
        with patch.object(sweep,'run_stage') as launch:
            with self.assertRaises(FileExistsError): sweep.run_pipeline(r,'all')
            launch.assert_not_called()

    def test_fake_downstream_load_identity_budget_and_split(self):
        r=sweep.make_runner('anchor050')
        with patch('torch.cuda.is_available',return_value=True), patch.object(
            r.subprocess,'Popen',self.fake_trainer(r)), patch(
            'utils.run_metadata.collect_run_metadata',return_value={'fake':True}):
            with contextlib.redirect_stdout(io.StringIO()): r.run('pretrain')
        source=r.completed('pretrain')
        class Downstream:
            def __init__(inner, command, **kwargs):
                self.assertEqual(command[1],'train_cm_iqa.py')
                cfg=r.validate('agiqa3k');d,table=r.paths('agiqa3k')
                (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                r.write_new(d/'run_metadata.json',dict(checkpoint_verified_after_load=True,checkpoint={'sha256':source['sha256']}))
                r.write_new(d/'data_split_manifest.json',dict(train_seed=42,split_seed=42,indices={}))
                with table.open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader()
                    for epoch in range(1,61):writer.writerow(dict(event='epoch',epoch=epoch,metrics_json='{}'))
                    writer.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
                inner.stdout=iter(['fake downstream completed\n'])
            def wait(inner):return 0
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Downstream),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
            r.run('agiqa3k')
        self.assertEqual(r.completed('agiqa3k'),source)
        d,_=r.paths('agiqa3k');evidence=d/'data_split_manifest.json'
        evidence.write_text(json.dumps(dict(train_seed=42,split_seed=123,indices={})))
        with self.assertRaisesRegex(ValueError,'seed'):r.completed('agiqa3k')

    def test_incomplete_refuses_pack(self):
        with self.assertRaises(FileNotFoundError): sweep.make_runner('anchor050').pack('pretrain')


if __name__=='__main__': unittest.main()
