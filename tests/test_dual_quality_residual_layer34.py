"""Matched layer3+4 control for dual-quality residual scoring."""
import contextlib
import copy
import io
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from models.two_stage_models import CrossModalIQAJointPretrain, CrossModalIQAFullModel, load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state, _compute_quality_text_loss
from utils.train_utils import load_yaml
from utils.clip_experiments import local_stage_diagnostics
from utils.quality_representation import dual_quality_diagnostics
from test_dual_quality_residual import residual_config
from test_dual_quality_representation import gradient
from test_quality_route_ablations import Global, Text, Bank

ROOT=Path(__file__).resolve().parents[1]
def make_real(c,joint=True):
    with patch('models.two_stage_models.GlobalVisualEncoder',Global),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text):
        return (CrossModalIQAJointPretrain(c) if joint else CrossModalIQAFullModel(c)).eval()

class DualQualityResidualLayer34Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_presets_only_change_local_stage_and_output_paths(self):
        for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
            original=load_yaml(str(ROOT/f'configs/dual_quality_residual/{stage}.yaml'))
            actual=load_yaml(str(ROOT/f'configs/dual_quality_residual_layer34/{stage}.yaml'))
            expected=copy.deepcopy(original);expected['model']['local_encoder']['feature_stage']='layer3_4'
            for key in ['run_id','output_dir','results_csv']:expected[key]=expected[key].replace('layer3','layer34')
            if stage!='pretrain':expected['train']['pretrained_checkpoint']=expected['train']['pretrained_checkpoint'].replace('layer3','layer34')
            self.assertEqual(actual,expected)

    def test_real_resnet_three_views_supervise_both_scales_and_transfer_all_local_weights(self):
        c=residual_config();c['model']['local_encoder']['feature_stage']='layer3_4'
        source=make_real(c);source.iqa.quality_text_bank=Bank()
        images=[torch.randn(2,3,64,64) for _ in range(3)]
        seen=[];hook=source.iqa.backbone.local_encoder.register_forward_hook(lambda m,i,o:seen.append(tuple(o.shape)))
        out=source.forward_views(images[0],images[1],['cat','dog'],['cat','dog'],image_clean=images[2]);hook.remove()
        self.assertEqual(seen,[(2,4,4,4)]*3)
        self.assertEqual(out['align_patch_features_clean'].shape,(2,16,8))
        loss=_compute_quality_text_loss(out,torch.tensor([.3,.8]),torch.ones(2),source.iqa.quality_text_bank.centers)['quality_text_ce']
        local=source.iqa.backbone.local_encoder
        for p in [local.proj3.weight,local.proj4.weight,local.stage_mix_logit,local.backbone.conv1.weight]:
            g=gradient(loss,p);self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0.)
        with torch.no_grad():
            source.iqa.quality_adapter_gate_logit.fill_(math.log(.31/.69));local.stage_mix_logit.fill_(.2)
        state=_extract_representation_state(source.state_dict());target=make_real(c,joint=False)
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        for key,value in local.state_dict().items():torch.testing.assert_close(value,target.backbone.local_encoder.state_dict()[key],rtol=0,atol=0)
        self.assertAlmostEqual(dual_quality_diagnostics(target)['quality_residual_alpha'],.31,places=6)
        self.assertAlmostEqual(local_stage_diagnostics(target)['local_layer3_weight'],torch.sigmoid(torch.tensor(.2)).item())
        for key in ['iqa.backbone.local_encoder.proj4.weight','iqa.backbone.local_encoder.stage_mix_logit']:
            bad=dict(state);bad.pop(key)
            with patch('torch.load',return_value={'model':bad}),self.assertRaisesRegex(RuntimeError,'Incomplete multiscale'):
                load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        single=copy.deepcopy(c);single['model']['local_encoder']['feature_stage']='layer3';old_target=make_real(single,joint=False)
        with patch('torch.load',return_value={'model':state}),self.assertRaisesRegex(RuntimeError,'Local feature stage checkpoint mismatch'):
            load_pretrained_backbone(old_target,'mock.pth',load_scope='representation')

    def test_four_effective_main_configs_are_multiscale_and_keep_residual_alpha(self):
        import train_cm_pretrain as pre
        import train_cm_iqa as down
        class Captured(Exception):pass
        for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
            command=subprocess.check_output(['bash',str(ROOT/'scripts/run_dual_quality_residual_layer34.sh'),stage,'--dry-run'],text=True)
            self.assertIn('--local-feature-stage layer3_4',command);self.assertNotIn('--reinit-local-projection',command)
            parts=shlex.split(command);idx=next(i for i,v in enumerate(parts) if v in ['train_cm_pretrain.py','train_cm_iqa.py'])
            argv=parts[idx:];module=pre if stage=='pretrain' else down;seen=[]
            def snapshot(c,*args,**kwargs):seen.append(copy.deepcopy(c));raise Captured()
            with tempfile.TemporaryDirectory() as d:
                argv[argv.index('--output-dir')+1]=d;argv[argv.index('--results-csv')+1]=d+'/results.csv'
                with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Captured):module.main()
            c=seen[0]
            self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3_4')
            self.assertEqual(c['model']['quality_global_feature'],'g')
            self.assertEqual(c['model']['quality_adapter']['mode'],'residual_feature')
            self.assertEqual(c['model']['quality_adapter']['residual_init'],.1)
            self.assertEqual(c['train']['lambda_quality_text_ce'],.005)
            self.assertFalse(c['train'].get('reinit_local_projection',False))
            if stage!='pretrain':self.assertIn('residual_layer34_seed42',c['train']['pretrained_checkpoint'])

if __name__=='__main__':unittest.main()
