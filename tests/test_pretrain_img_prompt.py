"""Image-Prompt weight sweep protocol tests; fake artifacts only, no real training."""
import contextlib
import copy
import io
import csv
import json
import tarfile
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import yaml
from scripts import run_pretrain_img_prompt as sweep
import test_pretrain_component_strength as component_tests


class ImagePromptSweepTests(unittest.TestCase):
    def setUp(self):
        self.fixture=component_tests.ComponentStrengthTests('test_incomplete_refuses_pack')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        shutil.copytree(sweep.ROOT/sweep.CONFIG,self.root/sweep.CONFIG)
        p=patch.object(sweep,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def test_only_img_prompt_changes_with_baseline_anchors(self):
        for variant,weight in sweep.VARIANTS.items():
            r=sweep.make_runner(variant)
            cfg=r.validate('pretrain');expected=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain'))
            for k in ('run_id','output_dir','results_csv'):expected[k]=cfg[k]
            expected['train']['lambda_img_prompt']=weight
            self.assertEqual(cfg,expected)
            self.assertEqual(cfg['train']['lambda_img_siam'],.15)
            self.assertEqual(cfg['train']['lambda_artifact_anchor'],.01)
            self.assertEqual(cfg['train']['lambda_artifact_ordinal'],.015)
            for ds in sweep.STAGES[1:]:
                cfg=r.validate(ds);expected=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
                for k in ('run_id','output_dir','results_csv'):expected[k]=cfg[k]
                expected['train']['pretrained_checkpoint']=r.validate('pretrain')['output_dir']+'/best_component_anchor_pretrain.pth'
                self.assertEqual(cfg,expected)
            self.assertNotEqual(r.paths('pretrain'),sweep.pipeline.make_runner('anchor050').paths('pretrain'))
        self.assertNotEqual(sweep.make_runner('prompt020').paths('pretrain'),sweep.make_runner('prompt030').paths('pretrain'))
        with self.assertRaises(ValueError):sweep.make_runner('unknown')

    def test_dry_run_and_cpu_no_outputs(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):
                sweep.pipeline.run_pipeline(r,'all',True)
            with patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('pretrain')
            self.assertFalse(r.paths('pretrain')[0].exists())
            with self.assertRaises(FileNotFoundError):r.run('agiqa3k',True)

    def test_completed_source_tamper_and_pack(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                r.run('pretrain');source=r.completed('pretrain');r.pack('pretrain')
            self.assertEqual(source['version'],f'projection_only_v1_ctx5_img_{variant}')
            self.assertTrue((self.root/f'{r.VERSION}_pretrain_results.tar.gz').exists())
            with self.assertRaises(FileExistsError):r.run('pretrain')
            d,_=r.paths('pretrain');(d/'best_component_anchor_pretrain.pth').write_bytes(b'wrong-source')
            with self.assertRaises(ValueError):r.run('agiqa3k',True)

    def test_changed_protocol_rejected(self):
        r=sweep.make_runner('prompt020');p=r.CONFIG/'pretrain.yaml'
        cfg=yaml.safe_load(p.read_text());cfg['train']['lambda_img_siam']=.10
        p.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):r.validate('pretrain')


    def test_six_downstreams_verify_sources_splits_budget_and_pack(self):
        from utils.agiqa1k_subset1078 import expected_resplit_manifest
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                r.run('pretrain')
            source=r.completed('pretrain')
            outer=self
            class Downstream:
                def __init__(inner,command,**kwargs):
                    outer.assertEqual(command[1],'train_cm_iqa.py')
                    stage=Path(command[3]).stem
                    cfg=r.validate(stage);d,table=r.paths(stage)
                    (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                    r.write_new(d/'run_metadata.json',dict(seed=42,split_seed=cfg['split_seed'],checkpoint_verified_after_load=True,checkpoint={'sha256':source['sha256']}))
                    if cfg['train'].get('record_split_manifest',False):
                        indices=expected_resplit_manifest(3407)['split_indices'] if stage=='agiqa1k' else {}
                        r.write_new(d/'data_split_manifest.json',dict(train_seed=42,split_seed=cfg['split_seed'],indices=indices))
                    with table.open('w') as f:
                        writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader()
                        for epoch in range(1,61):writer.writerow(dict(event='epoch',epoch=epoch,metrics_json='{}'))
                        writer.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
                    inner.stdout=iter(['fake downstream completed\n'])
                def wait(inner):return 0
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Downstream),patch.object(r.subprocess,'run'),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                for stage in sweep.STAGES[1:]:r.run(stage)
                r.pack('downstream')
                for stage in sweep.STAGES[1:]:
                    self.assertEqual(r.completed(stage),source)
                    self.assertEqual(r.read_json(r.paths(stage)[0]/'source_fingerprint.json'),source)
                    with self.assertRaises(FileExistsError):r.run(stage)
            with tarfile.open(self.root/f'{r.VERSION}_downstream_results.tar.gz') as tar:
                names=tar.getnames()
                self.assertEqual(sum(n.endswith('.csv') for n in names),4)
                self.assertEqual(sum(n.endswith('completion_manifest.json') for n in names),4)
                self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
            self.assertFalse((r.paths('agiqa3k')[0]/'data_split_manifest.json').exists())
            # A subsequent file change must invalidate both completion and packaging.
            metadata=r.paths('agiqa1k')[0]/'run_metadata.json'
            data=r.read_json(metadata);data['checkpoint']['sha256']='0'*64
            metadata.write_text(json.dumps(data))
            with self.assertRaises(ValueError):r.completed('agiqa1k')

    def test_pipeline_order_failure_stops_and_outputs_are_preserved(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch.object(sweep.pipeline,'run_stage') as launch:
                sweep.pipeline.run_pipeline(r,'all')
                self.assertEqual([c.args[1] for c in launch.call_args_list],list(sweep.STAGES))
            with patch.object(sweep.pipeline,'run_stage',side_effect=RuntimeError('pretrain failed')) as launch:
                with self.assertRaises(RuntimeError):sweep.pipeline.run_pipeline(r,'all')
                self.assertEqual(launch.call_count,1)
            d,table=r.paths('aigciqa2023');table.parent.mkdir(exist_ok=True);table.write_text('keep')
            with patch.object(sweep.pipeline,'run_stage') as launch:
                with self.assertRaises(FileExistsError):sweep.pipeline.run_pipeline(r,'all')
                launch.assert_not_called()
            self.assertEqual(table.read_text(),'keep')


if __name__=='__main__':unittest.main()
