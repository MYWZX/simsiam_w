"""One-change dual-quality control: global residual scoring and alpha transfer."""
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
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import (_extract_representation_state, _compute_quality_text_loss,
    train_one_epoch, evaluate, build_optimizer)
from utils.quality_representation import dual_quality_diagnostics
from utils.train_utils import load_yaml
from test_dual_quality_representation import config, batch, views, gradient
from test_projection_ablations import make
from test_quality_route_ablations import Bank

ROOT=Path(__file__).resolve().parents[1]
def residual_config():
    c=config();c['model']['quality_adapter'].update(mode='residual_feature',residual_init=.1)
    return c

class DualQualityResidualTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_only_global_score_changes_and_auxiliary_features_do_not_use_alpha(self):
        base=make(config());torch.manual_seed(42);m=make(residual_config())
        for key,value in base.state_dict().items():
            if not key.endswith('dual_quality_signature'):torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
        b=batch();before=views(base,b);m.iqa.quality_text_bank=Bank();out=views(m,b)
        for key in ['quality_clean_z','quality_global_clean_z','quality_local_clean_z']:
            torch.testing.assert_close(before[key],out[key])
        torch.testing.assert_close(before['align_pred_clean'],out['align_pred_clean'])
        image=b['image_clean'];single=m.iqa.forward_single(image,b['prompt'])
        g=m.iqa.backbone.fusion.semantic_proj(single['f_sem'])
        qg=single['z_quality_global'];alpha=torch.sigmoid(m.iqa.quality_adapter_gate_logit)
        expected=g+alpha*g.float().norm(dim=-1,keepdim=True).detach()*torch.nn.functional.normalize(qg.float(),dim=-1)
        torch.testing.assert_close(single['s_quality_global'],torch.sigmoid(m.iqa.global_quality_head(expected).squeeze(-1)))
        self.assertFalse(torch.allclose(before['quality_pred_clean'],out['quality_pred_clean']))
        ce=_compute_quality_text_loss(out,b['quality_score'],torch.ones(3),m.iqa.quality_text_bank.centers)['quality_text_ce']
        self.assertIsNone(gradient(ce,m.iqa.quality_adapter_gate_logit))
        for p in [m.iqa.quality_adapter.net[0].weight,m.iqa.backbone.local_encoder.proj.weight]:
            self.assertGreater(gradient(ce,p).abs().sum().item(),0.)
        self.assertGreater(gradient(out['quality_pred_clean'].sum(),m.iqa.quality_adapter_gate_logit).abs().sum().item(),0.)
        self.assertIsNone(gradient(single['s_quality_global'].sum(),m.iqa.backbone.local_encoder.proj.weight))
        # The magnitude estimate is detached, while the g bypass remains live.
        raw=torch.randn(3,8,requires_grad=True);adapted=torch.randn(3,8,requires_grad=True)
        result,_=m.iqa._quality_residual_feature(raw,adapted)
        torch.testing.assert_close(gradient(result.sum(),raw),torch.ones_like(raw))

    def test_alpha_is_transferred_and_missing_or_wrong_mode_is_rejected(self):
        c=residual_config();source=make(c);target=make(c,joint=False)
        with torch.no_grad():source.iqa.quality_adapter_gate_logit.fill_(math.log(.27/.73))
        state=_extract_representation_state(source.state_dict())
        self.assertIn('iqa.quality_adapter_gate_logit',state)
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        self.assertAlmostEqual(dual_quality_diagnostics(target)['quality_residual_alpha'],.27,places=6)
        bad=dict(state);bad.pop('iqa.quality_adapter_gate_logit')
        with patch('torch.load',return_value={'model':bad}),self.assertRaisesRegex(RuntimeError,'Incomplete dual-quality'):
            load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        replace=make(config());old=_extract_representation_state(replace.state_dict())
        with patch('torch.load',return_value={'model':old}),self.assertRaises(RuntimeError):
            load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        with patch('torch.load',return_value={'model':state}),self.assertRaisesRegex(RuntimeError,'scoring mode mismatch'):
            load_pretrained_backbone(replace.iqa,'mock.pth',load_scope='representation')

    def test_real_pcgrad_step_updates_alpha_and_full_state_roundtrip(self):
        c=residual_config()
        # Identical fake text vectors make BN gradients enormous in this tiny
        # fixture. Disable clipping here to observe alpha's optimizer update;
        # shipped presets retain the baseline clipping value of 1.0.
        c['train']['grad_clip']=0.
        m=make(c);m.iqa.quality_text_bank=Bank();b=batch();opt=build_optimizer(c,m)
        before=m.iqa.quality_adapter_gate_logit.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(m,[b],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,1)
            metrics=evaluate(m,[b],torch.device('cpu'),c)
        self.assertFalse(torch.equal(before,m.iqa.quality_adapter_gate_logit))
        self.assertTrue(math.isfinite(metrics['quality_residual_alpha']))
        clone=make(c);clone.iqa.quality_text_bank=Bank();clone.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(views(m,b)['quality_pred_clean'],views(clone,b)['quality_pred_clean'])
        self.assertEqual(sum(p is m.iqa.quality_adapter_gate_logit for group in opt.param_groups for p in group['params']),1)

    def test_presets_and_four_effective_main_configs_change_only_residual_scoring(self):
        import train_cm_pretrain as pre
        import train_cm_iqa as down
        class Captured(Exception):pass
        for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
            base=load_yaml(str(ROOT/f'configs/dual_quality/{stage}.yaml'))
            new=load_yaml(str(ROOT/f'configs/dual_quality_residual/{stage}.yaml'))
            expected=copy.deepcopy(base);expected['model']['quality_adapter'].update(mode='residual_feature',residual_init=.1)
            for key in ['run_id','output_dir','results_csv']:expected[key]=expected[key].replace('dualquality','dualquality_residual')
            if stage!='pretrain':expected['train']['pretrained_checkpoint']=expected['train']['pretrained_checkpoint'].replace('dualquality','dualquality_residual')
            self.assertEqual(new,expected)
            command=subprocess.check_output(['bash',str(ROOT/'scripts/run_dual_quality_residual.sh'),stage,'--dry-run'],text=True)
            self.assertNotIn('--reinit-local-projection',command)
            self.assertIn('dualquality_residual',command)
            parts=shlex.split(command);idx=next(i for i,v in enumerate(parts) if v in ['train_cm_pretrain.py','train_cm_iqa.py'])
            argv=parts[idx:];module=pre if stage=='pretrain' else down;seen=[]
            def snapshot(c,*args,**kwargs):seen.append(copy.deepcopy(c));raise Captured()
            with tempfile.TemporaryDirectory() as d:
                argv[argv.index('--output-dir')+1]=d;argv[argv.index('--results-csv')+1]=d+'/results.csv'
                with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Captured):module.main()
            c=seen[0]
            self.assertEqual(c['model']['quality_adapter']['mode'],'residual_feature')
            self.assertEqual(c['model']['quality_adapter']['residual_init'],.1)
            self.assertEqual(c['model']['quality_global_feature'],'g')
            self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3')
            self.assertEqual(c['train']['lambda_quality_text_ce'],.005)
            self.assertTrue(c['model']['dual_quality_representation']['enabled'])

if __name__=='__main__':unittest.main()
