"""Stage-isolated freeze protocols and reused-source identity; no real training."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
from unittest.mock import patch

import torch
import yaml
from scripts import run_clip_freeze_stages as sweep
import test_pretrain_component_strength as component_tests
from test_frozen_clip_baseline import small_model
from test_projection_only import config as tiny_config
from train_cm_pretrain import _extract_representation_state
from models.two_stage_models import load_pretrained_backbone


class CLIPFreezeStageTests(unittest.TestCase):
    def setUp(self):
        state=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,state)
        self.fixture=component_tests.ComponentStrengthTests('test_incomplete_refuses_pack')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.root=self.fixture.root
        for name in ('clip_freeze_stages','pretrain_frozen_clip'):
            shutil.copytree(sweep.ROOT/'configs'/name,self.root/'configs'/name)
        (self.root/'docs').mkdir()
        shutil.copy2(sweep.ROOT/'docs/current_tuning_reference.json',self.root/'docs/current_tuning_reference.json')
        for module in (sweep,sweep.frozen):
            p=patch.object(module,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def source(self,variant):
        r=sweep.source_runner(variant)
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        s=r.completed('pretrain')
        if variant=='downstream_only':
            (self.root/'docs/current_tuning_reference.json').write_text(json.dumps({'source':s}))
        return r,s

    def test_only_expected_freeze_flags_change_and_source_is_shared(self):
        paths=[]
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v)
            for stage in sweep.STAGES:
                cfg=r.validate(stage)
                expected=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain')) if stage=='pretrain' else yaml.safe_load((self.root/f'configs/current_tuning_reference/{stage}.yaml').read_text())
                for key in ('run_id','output_dir','results_csv'):expected[key]=cfg[key]
                expected['model']['global_encoder']['freeze']=(v!='downstream_only') if stage=='pretrain' else v!='pretrain_only'
                if stage!='pretrain':expected['train']['pretrained_checkpoint']=sweep.expected_config(v)['output_dir']+'/best_component_anchor_pretrain.pth'
                self.assertEqual(cfg,expected)
            paths.append(r.paths('agiqa3k')[0])
        self.assertEqual(len(set(paths)),3)
        self.assertEqual(sweep.make_runner('both').paths('pretrain'),sweep.make_runner('pretrain_only').paths('pretrain'))
        self.assertEqual(sweep.make_runner('both').paths('agiqa3k'),sweep.frozen.make_runner().paths('agiqa3k'))
        with self.assertRaises(ValueError):sweep.make_runner('unknown')
        r=sweep.make_runner('pretrain_only');p=r.CONFIG/'agiqa1k.yaml';cfg=r.validate('agiqa1k');cfg['model']['global_encoder']['freeze']=True;p.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):r.validate('agiqa1k')

    def test_shared_frozen_pretrain_runs_once_and_all_reuses_verified(self):
        source,s=self.source('both')
        for v in ('both','pretrain_only'):
            r=sweep.make_runner(v)
            self.assertEqual(r.completed('pretrain'),s)
            with patch.object(sweep.pipeline,'run_stage') as launch,contextlib.redirect_stdout(io.StringIO()):sweep.run_pipeline(v,'all')
            self.assertEqual([c.args[1] for c in launch.call_args_list],list(sweep.STAGES[1:]))
        with self.assertRaises(FileExistsError):sweep.make_runner('pretrain_only').run('pretrain')
        d,_=source.paths('pretrain');(d/'best_component_anchor_pretrain.pth').write_bytes(b'changed')
        with patch.object(sweep.pipeline,'run_stage') as launch:
            with self.assertRaises(ValueError):sweep.run_pipeline('pretrain_only','all')
            launch.assert_not_called()

    def test_downstream_only_requires_pinned_baseline_and_never_retrains(self):
        r=sweep.make_runner('downstream_only')
        with patch.object(r.subprocess,'Popen',side_effect=AssertionError('never retrain')):
            with self.assertRaises(FileNotFoundError):r.run('pretrain')
        _,s=self.source('downstream_only')
        self.assertEqual(r.completed('pretrain'),s)
        with patch.object(sweep.pipeline,'run_stage') as launch,contextlib.redirect_stdout(io.StringIO()):sweep.run_pipeline('downstream_only','all')
        self.assertEqual([c.args[1] for c in launch.call_args_list],list(sweep.STAGES[1:]))
        ref=self.root/'docs/current_tuning_reference.json';value=json.loads(ref.read_text());value['source']['sha256']='0'*64;ref.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Pinned'):r.completed('pretrain')
        self.assertFalse(r.paths('agiqa1k')[0].exists())

    def test_dry_plans_cpu_refusal_and_partial_output_preservation(self):
        for v in sweep.VARIANTS:
            with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):sweep.run_pipeline(v,'all',True)
        r=sweep.make_runner('pretrain_only')
        with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('pretrain')
        d,table=r.paths('pretrain');table.parent.mkdir(exist_ok=True);table.write_text('partial original')
        with patch.object(sweep.pipeline,'run_stage') as launch:
            with self.assertRaises(FileNotFoundError):sweep.run_pipeline('both','all')
            launch.assert_not_called()
        self.assertEqual(table.read_text(),'partial original')

    def test_three_conditions_nine_fake_downstreams_pack_and_identity(self):
        from utils.agiqa1k_subset1078 import expected_resplit_manifest
        self.source('both');self.source('downstream_only')
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v);source=r.completed('pretrain')
            class Downstream:
                def __init__(inner,command,**kwargs):
                    assert command[1]=='train_cm_iqa.py'
                    stage=Path(command[3]).stem;cfg=r.validate(stage);d,table=r.paths(stage)
                    (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                    r.write_new(d/'run_metadata.json',dict(seed=42,split_seed=cfg['split_seed'],checkpoint_verified_after_load=True,checkpoint={'sha256':source['sha256']}))
                    if cfg['train'].get('record_split_manifest',False):r.write_new(d/'data_split_manifest.json',dict(train_seed=42,split_seed=cfg['split_seed'],indices=expected_resplit_manifest(3407)['split_indices'] if stage=='agiqa1k' else {}))
                    with table.open('w') as f:
                        w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                        for epoch in range(1,61):w.writerow(dict(event='epoch',epoch=epoch,metrics_json='{}'))
                        w.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
                    inner.stdout=iter(['fake downstream complete\n'])
                def wait(inner):return 0
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Downstream),patch.object(r.subprocess,'run'),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                for stage in sweep.STAGES[1:]:r.run(stage);self.assertEqual(r.completed(stage),source)
                r.pack('downstream')
                with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('must reuse')):sweep.run_pipeline(v,'all')
            with tarfile.open(self.root/f'{r.VERSION}_downstream_results.tar.gz') as t:
                names=t.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),4);self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
            d,_=r.paths('agiqa3k');meta=d/'run_metadata.json';value=r.read_json(meta);value['checkpoint']['sha256']='wrong';meta.write_text(json.dumps(value))
            with self.assertRaises(ValueError):r.completed('agiqa3k')

    def test_transfer_respects_target_freeze_in_both_directions(self):
        torch.manual_seed(42)
        for prefreeze,downfreeze in [(True,False),(False,True),(True,True)]:
            cfg=tiny_config(True);cfg['model']['global_encoder']['freeze']=prefreeze
            source=small_model(cfg,joint=True).eval()
            target_cfg=copy.deepcopy(cfg);target_cfg['model']['global_encoder']['freeze']=downfreeze
            target=small_model(target_cfg)
            with patch('torch.load',return_value={'model':_extract_representation_state(source.state_dict())}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
            target.train()
            self.assertEqual(target.backbone.global_encoder.training,not downfreeze)
            self.assertTrue(all(p.requires_grad!=downfreeze for p in target.backbone.global_encoder.parameters()))
            for name,x in source.iqa.backbone.global_encoder.state_dict().items():torch.testing.assert_close(x,target.backbone.global_encoder.state_dict()[name],rtol=0,atol=0)
            target.forward_single(torch.randn(4,3,16,16),['cat','dog','bird','fish'])['s_quality'].sum().backward()
            for p in target.backbone.global_encoder.parameters():
                if downfreeze:self.assertIsNone(p.grad)
                else:self.assertIsNotNone(p.grad)
            self.assertGreater(target.backbone.fusion.semantic_proj.weight.grad.abs().sum().item(),0.)


if __name__=='__main__':unittest.main()
