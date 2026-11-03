"""Projection-only version: transfer, routes, eval parity and LR ownership."""
import tempfile
import unittest
from pathlib import Path
import torch
import test_image_image_fused_feature as fused
from test_projection_ablations import make
from test_dual_quality_representation import batch, views
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import build_optimizer
from models.two_stage_models import load_pretrained_backbone
from utils.downstream_controls import audit_fusion_gradients


def config(new=False):
    c=fused.FusedIITests().config();c['model']['image_text_feature']='g'
    c['train']['lambda_final']=0
    if new:c['model']['fusion']['mode']='projection_only_v1'
    return c

class ProjectionOnlyTests(unittest.TestCase):
    def test_transfer_and_eval_parity(self):
        old=make(config());new=make(config(True));state=_extract_representation_state(old.state_dict())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.pth';torch.save({'model':state},p)
            load_pretrained_backbone(new.iqa,p,load_scope='representation')
            for k,v in new.iqa.state_dict().items():
                source='iqa.'+k
                if source in state:torch.testing.assert_close(v,state[source],rtol=0,atol=0)
            # Scoring heads deliberately are not representation transfer; match them
            # explicitly to isolate the computational route rather than initialization.
            compatible={k:v for k,v in old.state_dict().items() if k in new.state_dict()}
            new.load_state_dict(compatible,strict=True)
            b=batch();a=views(old,b);z=views(new,b)
            for k in a:
                if isinstance(a[k],torch.Tensor) and k in z and z[k] is not None:
                    torch.testing.assert_close(a[k],z[k],rtol=0,atol=0)
            self.assertIsNone(new.iqa.backbone.encode_image(b['image1'])['z'])
            self.assertFalse(any(isinstance(m,torch.nn.MultiheadAttention) for m in new.iqa.backbone.fusion.modules()))
            for fragment in ('fusion.semantic_proj.weight', 'quality_adapter.net.0.weight',
                             'shared_image_projector.net.0.weight', 'shared_text_projector.net.0.weight',
                             'local_patch_proj.weight', 'quality_local_encoder.net.0.weight'):
                key = next((k for k in state if fragment in k), None)
                if key is None:
                    continue
                incomplete = dict(state); del incomplete[key]
                torch.save({'model':incomplete},p)
                with self.assertRaises(RuntimeError):load_pretrained_backbone(new.iqa,p,load_scope='representation')

    def test_optimizer_and_audit(self):
        c=config(True);c['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'fusion_mult':.1,'shared_projection_mult':.1})
        m=make(c).iqa;o=build_optimizer(c,m)
        group=next(g for g in o.param_groups if any(p is m.backbone.fusion.semantic_proj.weight for p in g['params']))
        self.assertEqual(group['name'],'fusion');self.assertEqual(group['lr'],1e-5)
        out=m.forward_single(torch.randn(3,3,16,16),['cat']*3)
        audit=audit_fusion_gradients(m,{'quality':out['s_quality'].sum()})
        self.assertEqual(audit['gradient_audit_cross_attention_present'],0)
        self.assertGreater(audit['gradient_global_projection_quality_norm'],0)
        self.assertGreater(audit['gradient_local_projection_quality_norm'],0)

    def test_invalid_routes(self):
        for path,value in [('quality_global_feature','z'),('align_global_feature','z'),('image_image_feature','z'),('image_text_feature','z')]:
            c=config(True);c['model'][path]=value
            with self.assertRaises(ValueError):make(c)
        c=config(True);c['train']['lambda_fusion_gate']=.1
        with self.assertRaises(ValueError):make(c)

    def test_independent_configs_and_pretrain_lr(self):
        from utils.train_utils import load_yaml
        from train_cm_pretrain import build_optimizer as pretrain_optimizer
        for name in ('pretrain', 'agiqa1k', 'agiqa3k', 'aigciqa2023'):
            c=load_yaml(f'configs/global_pi_ctx5_projection_only_v1/{name}.yaml')
            model=make(c,joint=name=='pretrain')
            self.assertTrue((model.iqa if name=='pretrain' else model).backbone.projection_only)
        c=config(True);c['train'].update(lr=2e-5,layerwise_lr={'enabled':True,'fusion_mult':.5})
        model=make(c);optimizer=pretrain_optimizer(c,model)
        group=next(g for g in optimizer.param_groups if any(p is model.iqa.backbone.fusion.semantic_proj.weight for p in g['params']))
        self.assertEqual(group['name'],'fusion');self.assertEqual(group['lr'],1e-5)
