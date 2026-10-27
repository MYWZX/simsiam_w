"""Freeze PI/PT BN statistics only; preserve affine/linear training and source."""
import contextlib
import copy
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import train_one_epoch, build_optimizer, evaluate
from test_dual_quality_residual import residual_config
from test_dual_quality_representation import batch
from test_projection_ablations import make
from test_quality_route_ablations import Bank
from utils.train_utils import load_yaml

ROOT=Path(__file__).resolve().parents[1]
def config(frozen=True):
    c=residual_config();c['model']['shared_simsiam_projection']=True
    if frozen:c['model']['shared_projection_freeze_bn_stats']=True
    return c

def buffers(m):
    return {n:b.clone() for n,b in m.named_buffers() if n.startswith(('shared_image_projector.','shared_text_projector.'))}

class SharedProjectionBNFrozen(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_only_bn_is_eval_and_train_calls_reapply_policy(self):
        m=make(config(),joint=False)
        for mode in [True,False,True,True]:
            m.train(mode)
            for name in ['shared_image_projector','shared_text_projector']:
                projector=getattr(m,name)
                self.assertEqual(projector.training,mode)
                for layer in projector.modules():
                    if isinstance(layer,nn.modules.batchnorm._BatchNorm):self.assertFalse(layer.training)
                    if isinstance(layer,nn.Dropout):self.assertEqual(layer.training,mode)
                self.assertTrue(all(p.requires_grad for p in projector.parameters()))
            self.assertEqual(m.quality_adapter.net[1].training,mode)
        source=make(config(False));target=make(config(),joint=False)
        with torch.no_grad():
            for name in ['shared_image_projector','shared_text_projector']:
                for layer in getattr(source.iqa,name).modules():
                    if isinstance(layer,nn.modules.batchnorm._BatchNorm):
                        layer.running_mean.fill_(.25);layer.running_var.fill_(1.3);layer.num_batches_tracked.fill_(11)
        state=_extract_representation_state(source.state_dict())
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        for key,value in source.iqa.state_dict().items():
            if key.startswith(('shared_image_projector.','shared_text_projector.')):torch.testing.assert_close(value,target.state_dict()[key])
        with self.assertRaisesRegex(ValueError,'requires shared_simsiam_projection'):
            bad=config();bad['model']['shared_simsiam_projection']=False;make(bad,joint=False)

    def test_real_downstream_steps_keep_buffers_but_update_bn_affine_and_linear(self):
        c=config();c['train']['grad_clip']=0.;m=make(c,joint=False);m.quality_text_bank=Bank()
        b=batch();down={'image':b['image_clean'],'prompt':b['prompt'],'score':b['quality_score'],
            'quality_score':b['quality_score'],'align_score':b['align_score']}
        opt=build_optimizer(c,m);saved=buffers(m);qa_counter=m.quality_adapter.net[1].num_batches_tracked.clone()
        params=[m.shared_image_projector.net[0].weight,m.shared_text_projector.net[0].weight,
                m.shared_image_projector.net[1].weight,m.shared_text_projector.net[1].weight]
        before=[p.detach().clone() for p in params]
        with contextlib.redirect_stdout(io.StringIO()):
            for epoch in [1,2]:
                train_one_epoch(m,[down],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,epoch)
        for name,value in saved.items():torch.testing.assert_close(value,dict(m.named_buffers())[name],rtol=0,atol=0)
        for old,p in zip(before,params):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            self.assertFalse(torch.equal(old,p))
        self.assertGreater(m.quality_adapter.net[1].num_batches_tracked.item(),qa_counter.item())
        metrics=evaluate(m,[down],torch.device('cpu'),c)
        self.assertEqual(metrics['val_shared_projection_bn_stats_frozen'],1.)

    def test_baseline_stats_still_update_and_eval_outputs_are_identical_before_training(self):
        baseline=make(config(False),joint=False);torch.manual_seed(42);frozen=make(config(),joint=False)
        baseline.eval();frozen.eval();b=batch()
        a=baseline.forward_single(b['image_clean'],b['prompt']);z=frozen.forward_single(b['image_clean'],b['prompt'])
        for key in ['s_quality','s_align']:torch.testing.assert_close(a[key],z[key],rtol=0,atol=0)
        saved=buffers(baseline);baseline.train();baseline.forward_single(b['image_clean'],b['prompt'])
        self.assertTrue(any(not torch.equal(v,dict(baseline.named_buffers())[k]) for k,v in saved.items()))

    def test_three_effective_configs_change_only_bn_policy_and_outputs(self):
        import train_cm_iqa as module
        class Captured(Exception):pass
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            old=load_yaml(str(ROOT/f'configs/dual_residual_ablations/{ds}_shared_pi_pt.yaml'))
            new=load_yaml(str(ROOT/f'configs/shared_projection_bn_frozen/{ds}.yaml'))
            expected=copy.deepcopy(old);expected['model']['shared_projection_freeze_bn_stats']=True
            for key in ['run_id','output_dir','results_csv']:expected[key]=expected[key].replace('_shared_pi_pt_seed42','_shared_pi_pt_bnfrozen_seed42')
            self.assertEqual(new,expected)
            command=subprocess.check_output(['bash',str(ROOT/'scripts/run_shared_projection_bn_frozen.sh'),ds,'--dry-run'],text=True)
            self.assertIn('layer3_shared_pi_pt_seed42/best_component_anchor_pretrain.pth',command)
            self.assertNotIn('--reinit-local-projection',command)
            parts=shlex.split(command);argv=parts[parts.index('train_cm_iqa.py'):];seen=[]
            def snapshot(c,*a,**kw):seen.append(copy.deepcopy(c));raise Captured()
            with tempfile.TemporaryDirectory() as d:
                argv[argv.index('--output-dir')+1]=d;argv[argv.index('--results-csv')+1]=d+'/r.csv'
                with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Captured):module.main()
            c=seen[0]
            self.assertTrue(c['model']['shared_projection_freeze_bn_stats'])
            self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3')
            self.assertEqual(c['model']['quality_adapter']['mode'],'residual_feature')
            self.assertEqual(c['train']['lambda_quality_text_ce'],.005)
            self.assertEqual(c['train']['layerwise_lr']['global_encoder_mult'],.02)
        p=subprocess.run(['bash',str(ROOT/'scripts/run_shared_projection_bn_frozen.sh'),'pretrain','--dry-run'],capture_output=True)
        self.assertNotEqual(p.returncode,0)

if __name__=='__main__':unittest.main()
