"""Downstream P0 bypasses PI; global image/text projections are retained."""
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
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state
from test_shared_projection_bn_frozen import config as frozen_config
from test_dual_quality_representation import batch, gradient
from test_projection_ablations import make
from utils.quality_representation import dual_quality_diagnostics
from utils.train_utils import load_yaml

ROOT=Path(__file__).resolve().parents[1]
def config():
    c=frozen_config();c['model']['shared_projection_project_patches']=False
    return c

class SharedProjectionGlobalOnly(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_exact_routing_and_local_score_does_not_depend_on_pi(self):
        m=make(config(),joint=False);b=batch();seen={};pi_calls=[];pt_calls=[]
        hooks=[getattr(m,n).register_forward_pre_hook(lambda mod,args,n=n:seen.update({n:args[0]})) for n in ['align_global_adapter','align_patch_adapter','align_text_adapter']]
        hooks += [m.shared_image_projector.register_forward_pre_hook(lambda mod,args:pi_calls.append(args[0])),m.shared_text_projector.register_forward_pre_hook(lambda mod,args:pt_calls.append(args[0]))]
        out=m.forward_single(b['image_clean'],b['prompt'])
        for h in hooks:h.remove()
        g=m.backbone.fusion.semantic_proj(out['f_sem']);p0=m._make_patch_tokens(out['f_loc'])
        self.assertEqual(len(pi_calls),1);torch.testing.assert_close(pi_calls[0],g)
        self.assertEqual(len(pt_calls),1);torch.testing.assert_close(pt_calls[0],out['text_fused'])
        torch.testing.assert_close(seen['align_global_adapter'],m.project_shared_image(g))
        torch.testing.assert_close(seen['align_patch_adapter'],p0)
        torch.testing.assert_close(seen['align_text_adapter'],torch.nn.functional.normalize(m.shared_text_projector(out['text_fused']),dim=-1))
        pi=m.shared_image_projector.net[0].weight;pt=m.shared_text_projector.net[0].weight
        self.assertIsNone(gradient(out['s_align_patch'].sum(),pi))
        self.assertGreater(gradient(out['s_align_global'].sum(),pi).abs().sum().item(),0.)
        self.assertGreater(gradient(out['s_align_patch'].sum(),pt).abs().sum().item(),0.)
        self.assertGreater(gradient(out['s_align_patch'].sum(),m.backbone.local_encoder.proj.weight).abs().sum().item(),0.)
        self.assertIsNone(gradient(out['s_quality'].sum(),pi))
        self.assertEqual(dual_quality_diagnostics(m)['align_patch_uses_shared_pi'],0.)

    def test_quality_and_global_alignment_are_unchanged_and_source_loads_completely(self):
        baseline=make(frozen_config(),joint=False);torch.manual_seed(42);m=make(config(),joint=False)
        for key,v in baseline.state_dict().items():torch.testing.assert_close(v,m.state_dict()[key],rtol=0,atol=0)
        b=batch();a=baseline.forward_single(b['image_clean'],b['prompt']);z=m.forward_single(b['image_clean'],b['prompt'])
        for key in ['s_quality','s_quality_global','s_quality_patch','s_align_global']:torch.testing.assert_close(a[key],z[key],rtol=0,atol=0)
        self.assertFalse(torch.allclose(a['s_align_patch'],z['s_align_patch']))
        c=frozen_config(False);source=make(c);state=_extract_representation_state(source.state_dict())
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(m,'mock.pth',load_scope='representation')
        for name in ['shared_image_projector','shared_text_projector','align_patch_adapter','quality_adapter']:
            for key,v in getattr(source.iqa,name).state_dict().items():torch.testing.assert_close(v,getattr(m,name).state_dict()[key])
        m.train()
        self.assertTrue(all(not bn.training for projector in [m.shared_image_projector,m.shared_text_projector] for bn in projector.modules() if isinstance(bn,torch.nn.modules.batchnorm._BatchNorm)))
        c=config();c['model']['shared_simsiam_projection']=False
        with self.assertRaisesRegex(ValueError,'requires shared_simsiam_projection'):make(c,joint=False)

    def test_three_effective_configs_only_change_patch_route_and_outputs(self):
        import train_cm_iqa as module
        class Captured(Exception):pass
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            old=load_yaml(str(ROOT/f'configs/shared_projection_bn_frozen/{ds}.yaml'))
            new=load_yaml(str(ROOT/f'configs/shared_projection_global_only/{ds}.yaml'))
            expected=copy.deepcopy(old);expected['model']['shared_projection_project_patches']=False
            for key in ['run_id','output_dir','results_csv']:expected[key]=expected[key].replace('_bnfrozen_seed42','_bnfrozen_piglobal_seed42')
            self.assertEqual(new,expected)
            command=subprocess.check_output(['bash',str(ROOT/'scripts/run_shared_projection_global_only.sh'),ds,'--dry-run'],text=True)
            self.assertIn('shared_pi_pt_seed42/best_component_anchor_pretrain.pth',command)
            self.assertNotIn('--reinit-local-projection',command)
            parts=shlex.split(command);argv=parts[parts.index('train_cm_iqa.py'):];seen=[]
            def snapshot(c,*args,**kw):seen.append(copy.deepcopy(c));raise Captured()
            with tempfile.TemporaryDirectory() as d:
                argv[argv.index('--output-dir')+1]=d;argv[argv.index('--results-csv')+1]=d+'/r.csv'
                with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Captured):module.main()
            c=seen[0]
            self.assertFalse(c['model']['shared_projection_project_patches'])
            self.assertTrue(c['model']['shared_projection_freeze_bn_stats'])
            self.assertEqual(c['model']['quality_adapter']['mode'],'residual_feature')
            self.assertEqual(c['train']['lambda_quality_text_ce'],.005)
            self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3')

if __name__=='__main__':unittest.main()
