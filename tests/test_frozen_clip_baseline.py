"""Freeze-only protocol, real small-model gradients/optimizers, and fake pipeline checks."""
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
from torch import nn
import yaml
from scripts import run_frozen_clip as sweep
import test_pretrain_component_strength as component_tests
from test_projection_only import config as tiny_config
from test_frozen_visual_encoder_mode import GlobalBN, LocalBN
from models.two_stage_models import CrossModalIQAFullModel, CrossModalIQAJointPretrain, load_pretrained_backbone
from train_cm_iqa import build_optimizer as downstream_optimizer
from train_cm_pretrain import build_optimizer as pretrain_optimizer, _extract_representation_state


class ContextText(nn.Module):
    output_dim=6
    def __init__(self,**kw):
        super().__init__()
        self.frozen_text=nn.Linear(6,6).requires_grad_(False)
        self.context_vectors=nn.Parameter(torch.randn(5,6))
    def forward(self,prompts,device):
        signals=torch.stack([torch.sin(torch.arange(6,device=device,dtype=torch.float32)+(sum(map(ord,p))%17)) for p in prompts])
        return torch.tanh(self.frozen_text(signals+self.context_vectors.mean(0)))


def small_model(cfg,joint=False):
    with patch('models.two_stage_models.GlobalVisualEncoder',GlobalBN),patch('models.two_stage_models.ResNet50LocalEncoder',LocalBN),patch('models.two_stage_models.FrozenCLIPTextEncoder',ContextText):
        return CrossModalIQAJointPretrain(cfg) if joint else CrossModalIQAFullModel(cfg)


class FrozenCLIPModelTests(unittest.TestCase):
    def setUp(self):
        rng=torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state,rng)
        torch.manual_seed(42)

    def test_frozen_weights_buffers_and_optimizer_step_preserve_other_gradient_paths(self):
        for stage in sweep.STAGES:
            with self.subTest(stage=stage):
                real=sweep.expected_config(stage=stage)
                cfg=tiny_config(True)
                cfg['model']['global_encoder']['freeze']=True
                for key in ('lr','weight_decay','layerwise_lr'):
                    if key in real['train']:cfg['train'][key]=copy.deepcopy(real['train'][key])
                joint=stage=='pretrain';model=small_model(cfg,joint);iqa=model.iqa if joint else model
                model.train()
                self.assertFalse(iqa.backbone.global_encoder.training)
                self.assertTrue(iqa.backbone.local_encoder.training)
                self.assertTrue(iqa.quality_adapter.training)
                before={k:v.clone() for k,v in iqa.backbone.global_encoder.state_dict().items()}
                projection_before=iqa.backbone.fusion.semantic_proj.weight.detach().clone()
                local_before=iqa.backbone.local_encoder.proj.weight.detach().clone()
                build=pretrain_optimizer if joint else downstream_optimizer
                opt=build(cfg,model)
                frozen={id(p) for p in iqa.backbone.global_encoder.parameters()}
                self.assertTrue(all(not p.requires_grad for p in iqa.backbone.global_encoder.parameters()))
                self.assertFalse(any(id(p) in frozen for g in opt.param_groups for p in g['params']))
                for target,lr in [(iqa.backbone.fusion.semantic_proj.weight,1e-5),(iqa.shared_image_projector.net[0].weight,1e-4),(iqa.shared_text_projector.net[0].weight,1e-4)]:
                    group=next(g for g in opt.param_groups if any(p is target for p in g['params']))
                    self.assertAlmostEqual(group['lr'],lr)
                out=iqa.forward_single(torch.randn(4,3,16,16),['cat','dog','bird','fish'])
                loss=out['s_quality'].square().sum()+out['s_align'].square().sum()
                loss.backward()
                for parameter in [iqa.backbone.fusion.semantic_proj.weight,iqa.backbone.local_encoder.proj.weight,iqa.local_patch_proj.weight,iqa.quality_adapter.net[0].weight,iqa.shared_image_projector.net[0].weight,iqa.shared_text_projector.net[0].weight,iqa.backbone.text_encoder.context_vectors]:
                    self.assertTrue(parameter.requires_grad)
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.abs().sum().item(),0.)
                opt.step()
                self.assertFalse(torch.equal(iqa.backbone.fusion.semantic_proj.weight,projection_before))
                self.assertFalse(torch.equal(iqa.backbone.local_encoder.proj.weight,local_before))
                for name,value in iqa.backbone.global_encoder.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
                self.assertTrue(all(p.grad is None for p in iqa.backbone.global_encoder.parameters()))
                self.assertTrue(all(not p.requires_grad for p in iqa.backbone.text_encoder.frozen_text.parameters()))
                model.eval();model.train()
                self.assertFalse(iqa.backbone.global_encoder.training)
                self.assertTrue(iqa.backbone.local_encoder.training)

    def test_other_optimizer_groups_and_lr_unchanged(self):
        for stage in sweep.STAGES:
            real=sweep.expected_config(stage=stage);cfg=tiny_config(True)
            for key in ('lr','weight_decay','layerwise_lr'):
                if key in real['train']:cfg['train'][key]=copy.deepcopy(real['train'][key])
            joint=stage=='pretrain';model=small_model(cfg,joint)
            build=pretrain_optimizer if joint else downstream_optimizer
            def describe(opt):
                names={id(p):n for n,p in model.named_parameters()}
                return {g['name']:(g['lr'],g['weight_decay'],[names[id(p)] for p in g['params']]) for g in opt.param_groups}
            before=describe(build(cfg,model))
            for p in (model.iqa if joint else model).backbone.global_encoder.parameters():p.requires_grad_(False)
            after=describe(build(cfg,model));before.pop('global_encoder')
            self.assertEqual(before,after)

    def test_transfer_frozen_global_is_complete_and_eval_output_parity(self):
        cfg=tiny_config(True);source=small_model(cfg,joint=True).eval()
        frozen=copy.deepcopy(cfg);frozen['model']['global_encoder']['freeze']=True
        target=small_model(frozen).eval()
        state=_extract_representation_state(source.state_dict())
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        for name,value in target.backbone.global_encoder.state_dict().items():torch.testing.assert_close(value,source.iqa.backbone.global_encoder.state_dict()[name],rtol=0,atol=0)
        target.load_state_dict(source.iqa.state_dict())
        x=torch.randn(4,3,16,16);a=source.iqa.forward_single(x,['cat']*4);b=target.forward_single(x,['cat']*4)
        for key in ['s_quality','s_align']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in target.backbone.global_encoder.parameters()))


class FrozenCLIPPipelineTests(unittest.TestCase):
    def setUp(self):
        rng=torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state,rng)
        self.fixture=component_tests.ComponentStrengthTests('test_incomplete_refuses_pack')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.root=self.fixture.root
        shutil.copytree(sweep.ROOT/sweep.CONFIG,self.root/sweep.CONFIG)
        p=patch.object(sweep,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def test_only_freeze_flag_and_independent_source_change(self):
        r=sweep.make_runner()
        for stage in sweep.STAGES:
            cfg=r.validate(stage)
            expected=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain')) if stage=='pretrain' else yaml.safe_load((self.root/f'configs/current_tuning_reference/{stage}.yaml').read_text())
            for key in ('run_id','output_dir','results_csv'):expected[key]=cfg[key]
            expected['model']['global_encoder']['freeze']=True
            if stage!='pretrain':expected['train']['pretrained_checkpoint']=r.validate('pretrain')['output_dir']+'/best_component_anchor_pretrain.pth'
            self.assertEqual(cfg,expected)
            self.assertFalse(cfg['model']['local_encoder']['freeze'])
        cfg=r.validate('pretrain')
        self.assertEqual(cfg['train']['lambda_img_siam'],.15)
        self.assertEqual(cfg['train']['lambda_img_prompt'],.25)
        self.assertFalse(any(k in cfg['train'] for k in ('resume','pretrained_checkpoint','refit_plan')))
        with self.assertRaises(ValueError):sweep.expected_config('other')
        path=r.CONFIG/'agiqa3k.yaml';cfg=r.validate('agiqa3k');cfg['model']['global_encoder']['freeze']=False
        path.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):r.validate('agiqa3k')

    def test_dry_run_cpu_refusal_and_pipeline_failure(self):
        r=sweep.make_runner()
        with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):sweep.pipeline.run_pipeline(r,'all',True)
        with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('pretrain')
        self.assertFalse(r.paths('pretrain')[0].exists())
        with patch.object(sweep.pipeline,'run_stage',side_effect=RuntimeError('failed')) as launch:
            with self.assertRaises(RuntimeError):sweep.pipeline.run_pipeline(r,'all')
            self.assertEqual(launch.call_count,1)

    def test_fake_source_three_downstreams_pack_and_no_overwrite(self):
        from utils.agiqa1k_subset1078 import expected_resplit_manifest
        r=sweep.make_runner()
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        source=r.completed('pretrain');self.assertEqual(source['version'],sweep.VERSION)
        class Downstream:
            def __init__(inner,command,**kwargs):
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
        with tarfile.open(self.root/f'{sweep.VERSION}_downstream_results.tar.gz') as t:
            names=t.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),4);self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
        for stage in sweep.STAGES:
            with self.assertRaises(FileExistsError):r.run(stage)
        d,_=r.paths('pretrain');(d/'best_component_anchor_pretrain.pth').write_bytes(b'other source')
        with self.assertRaises(ValueError):r.run('agiqa3k',True)


if __name__=='__main__':unittest.main()
