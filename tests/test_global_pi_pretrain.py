import unittest
import torch
from test_image_image_fused_feature import FusedIITests
from test_projection_ablations import make
from test_dual_quality_representation import batch, views

class GlobalPITraining(unittest.TestCase):
    def test_all_pi_calls_are_global_and_local_quality_still_has_gradients(self):
        c=FusedIITests().config()
        c['model'].update(image_image_feature='g',image_text_feature='g',
                          align_global_feature='g',shared_projection_project_patches=False)
        m=make(c); features=[];inputs=[]
        h=m.iqa.backbone.register_forward_hook(lambda *args:None)
        # encode_image is a method: capture actual tensors through the encoders.
        original=m.iqa.backbone.encode_image
        def encode(image):
            f=original(image);features.append(f);return f
        m.iqa.backbone.encode_image=encode
        hook=m.iqa.shared_image_projector.register_forward_pre_hook(lambda mod,args:inputs.append(args[0]))
        out=views(m,batch());hook.remove();h.remove()
        gs=[m.iqa.backbone.fusion.semantic_proj(f['f_sem']) for f in features]
        self.assertTrue(inputs)
        for value in inputs:
            self.assertTrue(any(value.shape==g.shape and torch.allclose(value,g) for g in gs))
        local=m.iqa.backbone.local_encoder.proj.weight
        semantic=torch.autograd.grad(out['clean_p'].sum(),local,retain_graph=True,allow_unused=True)[0]
        self.assertTrue(semantic is None or semantic.abs().sum()==0)
        quality=torch.autograd.grad(out['quality_clean_z'][:,0].sum(),local,retain_graph=True)[0]
        self.assertGreater(quality.abs().sum().item(),0.)

    def test_invalid_semantic_feature_is_rejected(self):
        c=FusedIITests().config();c['model']['image_text_feature']='invalid'
        with self.assertRaises(ValueError):make(c)
