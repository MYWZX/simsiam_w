"""Fused z I-I is opt-in, reaches fusion/local trunk, and leaves Clean PQ alone."""
import unittest
import torch
from test_dual_quality_residual import residual_config
from test_projection_ablations import make
from test_dual_quality_representation import batch, views
from train_cm_pretrain import _quality_view_simsiam_loss

class FusedIITests(unittest.TestCase):
    def config(self):
        c=residual_config();c['model']['shared_simsiam_projection']=True
        return c
    def test_explicit_z_is_the_input_to_shared_pi_and_has_local_gradients(self):
        c=self.config();c['model']['image_image_feature']='z';m=make(c)
        image=torch.randn(3,3,16,16)
        feature=m.iqa.backbone.encode_image(image)
        self.assertIs(m._quality_view_feature(feature),feature['z'])
        out=views(m,batch());loss=_quality_view_simsiam_loss(out,torch.ones(3))
        params=[m.iqa.shared_image_projector.net[0].weight,
                m.iqa.backbone.local_encoder.proj.weight]
        gradients=torch.autograd.grad(loss,params,retain_graph=True)
        for g in gradients:self.assertGreater(g.abs().sum().item(),0.)
        pq=m.iqa.quality_adapter.net[0].weight
        self.assertIsNone(torch.autograd.grad(loss,pq,retain_graph=True,allow_unused=True)[0])
        self.assertEqual(m.iqa.quality_global_feature,'g')
    def test_legacy_default_stays_g(self):
        m=make(self.config());f=m.iqa.backbone.encode_image(torch.randn(3,3,16,16))
        torch.testing.assert_close(m._quality_view_feature(f),m.iqa.backbone.fusion.semantic_proj(f['f_sem']))
    def test_invalid_option_rejected(self):
        c=self.config();c['model']['image_image_feature']='unknown'
        with self.assertRaises(ValueError):make(c)

if __name__=='__main__':unittest.main()
