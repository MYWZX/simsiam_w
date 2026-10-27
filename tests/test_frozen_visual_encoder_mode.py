"""Explicit encoder freeze must persist through parent train() calls."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from models.two_stage_models import ImageTextBackbone
from test_quality_route_ablations import config, Text

class GlobalBN(nn.Module):
    def __init__(self,**kw):
        super().__init__();self.proj=nn.Linear(3,8);self.bn=nn.BatchNorm1d(8);self.dropout=nn.Dropout(.4)
    def forward(self,x):return self.dropout(self.bn(self.proj(x.mean((2,3)))))
class LocalBN(nn.Module):
    def __init__(self,**kw):
        super().__init__();self.proj=nn.Conv2d(3,4,1);self.bn=nn.BatchNorm2d(4);self.dropout=nn.Dropout(.4)
    def forward(self,x):return self.dropout(self.bn(self.proj(torch.nn.functional.adaptive_avg_pool2d(x,(2,2)))))

class FrozenVisualEncoderMode(unittest.TestCase):
    def test_explicit_freeze_keeps_stats_and_dropout_fixed_without_freezing_fusion(self):
        for frozen in [False,True]:
            c=config();c['model']['global_encoder']['freeze']=frozen;c['model']['local_encoder']['freeze']=frozen
            with patch('models.two_stage_models.GlobalVisualEncoder',GlobalBN),patch('models.two_stage_models.ResNet50LocalEncoder',LocalBN),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text):
                m=ImageTextBackbone(c)
            for mode in [True,False,True]:
                m.train(mode)
                for encoder in [m.global_encoder,m.local_encoder]:
                    self.assertEqual(encoder.training,False if frozen else mode)
                    self.assertTrue(all(p.requires_grad!=frozen for p in encoder.parameters()))
                self.assertEqual(m.fusion.training,mode)
            before=m.local_encoder.bn.num_batches_tracked.clone();x=torch.randn(3,3,16,16)
            a=m.encode_image(x);b=m.encode_image(x)
            if frozen:
                self.assertEqual(m.local_encoder.bn.num_batches_tracked.item(),before.item())
                torch.testing.assert_close(a['f_sem'],b['f_sem'],rtol=0,atol=0)
                torch.testing.assert_close(a['f_loc'],b['f_loc'],rtol=0,atol=0)
            else:self.assertGreater(m.local_encoder.bn.num_batches_tracked.item(),before.item())
            a['z'].square().sum().backward()
            self.assertIsNotNone(m.fusion.semantic_proj.weight.grad)
            if frozen:self.assertIsNone(m.global_encoder.proj.weight.grad)

    def test_dual_transfer_does_not_silently_skip_alignment_adapters(self):
        from models.two_stage_models import load_pretrained_backbone
        from train_cm_pretrain import _extract_representation_state
        from test_dual_quality_residual import residual_config
        from test_projection_ablations import make
        c=residual_config();c['model']['align_adapter']={'enabled':True,'adapt_text':True,'hidden_dim':16,'dropout':0.}
        source=make(c);target=make(c,joint=False);state=_extract_representation_state(source.state_dict())
        for key in ['iqa.align_global_adapter.net.1.weight','iqa.align_patch_adapter.net.1.bias','iqa.text_to_fused.bias']:
            bad=dict(state);bad.pop(key)
            with patch('torch.load',return_value={'model':bad}),self.assertRaisesRegex(RuntimeError,'Incomplete dual-quality'):
                load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        c['model']['align_adapter']['hidden_dim']=32;target=make(c,joint=False)
        with patch('torch.load',return_value={'model':state}),self.assertRaisesRegex(RuntimeError,'Incomplete dual-quality'):
            load_pretrained_backbone(target,'mock.pth',load_scope='representation')

if __name__=='__main__':unittest.main()
