"""I-I weight sweep protocol tests; fake artifacts only, no real training."""
import contextlib
import copy
import io
import shutil
import unittest
from unittest.mock import patch

import yaml
from scripts import run_pretrain_img_siam as sweep
import test_pretrain_component_strength as component_tests


class ImageSiamSweepTests(unittest.TestCase):
    def setUp(self):
        self.fixture=component_tests.ComponentStrengthTests('test_incomplete_refuses_pack')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        shutil.copytree(sweep.ROOT/sweep.CONFIG,self.root/sweep.CONFIG)
        p=patch.object(sweep,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def test_only_img_siam_changes_with_baseline_anchors(self):
        for variant,weight in sweep.VARIANTS.items():
            r=sweep.make_runner(variant)
            cfg=r.validate('pretrain');expected=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain'))
            for k in ('run_id','output_dir','results_csv'):expected[k]=cfg[k]
            expected['train']['lambda_img_siam']=weight
            self.assertEqual(cfg,expected)
            self.assertEqual(cfg['train']['lambda_artifact_anchor'],.01)
            self.assertEqual(cfg['train']['lambda_artifact_ordinal'],.015)
            for ds in sweep.STAGES[1:]:
                cfg=r.validate(ds);expected=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
                for k in ('run_id','output_dir','results_csv'):expected[k]=cfg[k]
                expected['train']['pretrained_checkpoint']=r.validate('pretrain')['output_dir']+'/best_component_anchor_pretrain.pth'
                self.assertEqual(cfg,expected)
            self.assertNotEqual(r.paths('pretrain'),sweep.pipeline.make_runner('anchor050').paths('pretrain'))
        self.assertNotEqual(sweep.make_runner('siam010').paths('pretrain'),sweep.make_runner('siam020').paths('pretrain'))
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
        r=sweep.make_runner('siam010');p=r.CONFIG/'pretrain.yaml'
        cfg=yaml.safe_load(p.read_text());cfg['train']['lambda_artifact_anchor']=.005
        p.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):r.validate('pretrain')


if __name__=='__main__':unittest.main()
