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
        for name in ('global_pi_ctx5_noce_noneg', 'projection_only_v1_baseline', 'pretrain_component_strength'):
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
        with self.assertRaises(ValueError): r.validate('agiqa3k')
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

    def test_incomplete_refuses_pack(self):
        with self.assertRaises(FileNotFoundError): sweep.make_runner('anchor050').pack('pretrain')


if __name__=='__main__': unittest.main()
