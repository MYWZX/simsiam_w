"""Layer3 preset isolation, real three-view ResNet path, matched transfer."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn
from models.two_stage_models import CrossModalIQAJointPretrain, CrossModalIQAFullModel, load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state, _map_to_patch_weights
from utils.train_utils import load_yaml
from utils.clip_experiments import configure_local_feature_stage
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

class TinyGlobal(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(); self.proj=nn.Linear(3,8)
    def forward(self,x):return self.proj(x.mean((2,3)))

class TinyText(nn.Module):
    output_dim=6
    def __init__(self,**kwargs):super().__init__()
    def forward(self,texts,device):return torch.ones(len(texts),6,device=device)

class Layer3PretrainingTests(unittest.TestCase):
    def preset(self):return load_yaml(str(ROOT/'configs/richhf18k_cm_pretrain_decoupled_quality_align_layer3.yaml'))

    def test_preset_changes_only_stage_and_output_identifiers(self):
        old=load_yaml(str(ROOT/'configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml'))
        new=self.preset()
        self.assertEqual(new['model']['local_encoder']['feature_stage'],'layer3')
        self.assertNotEqual(old['output_dir'],new['output_dir'])
        self.assertNotEqual(old['results_csv'],new['results_csv'])
        expected=copy.deepcopy(old)
        expected['model']['local_encoder']['feature_stage']='layer3'
        expected['output_dir']=new['output_dir'];expected['results_csv']=new['results_csv']
        self.assertEqual(new,expected)
        configure_local_feature_stage(new,SimpleNamespace(local_feature_stage=None,reinit_local_projection=False))
        self.assertEqual(new,expected)
        self.assertTrue(new['train']['pcgrad']['enabled'])
        self.assertEqual(new['train']['pretrain_selection_metric'],'val_component_anchor_SROCC')

    def test_three_view_real_resnet_gradients_and_matched_representation_transfer(self):
        cfg=self.preset();m=cfg['model']
        m['global_encoder'].update(pretrained=False,output_dim=8)
        m['local_encoder']['pretrained']=False
        m['fusion'].update(embed_dim=8,num_heads=2,dropout=0.)
        m['heads'].update(mlp_hidden_dim=16,dropout=0.)
        m['simsiam'].update(proj_dim=8,proj_hidden_dim=16,pred_hidden_dim=4)
        with patch('models.two_stage_models.GlobalVisualEncoder',TinyGlobal), patch('models.two_stage_models.FrozenCLIPTextEncoder',TinyText):
            source=CrossModalIQAJointPretrain(cfg).eval()
        seen=[]
        handle=source.iqa.backbone.local_encoder.register_forward_hook(lambda mod,ins,out:seen.append(tuple(out.shape)))
        images=[torch.randn(2,3,224,224) for _ in range(3)]
        out=source._forward_three_views(*images,['cat','dog'],['cat','dog'])
        handle.remove()
        self.assertEqual(seen,[(2,256,14,14)]*3)
        self.assertEqual(out['align_patch_features_clean'].shape,(2,196,8))
        loss=out['quality_img1_p'].square().mean()+out['quality_img2_p'].square().mean()+out['quality_clean_z'][:,0].mean()+out['quality_pred'].sum()+out['align_pred'].sum()
        loss.backward()
        local=source.iqa.backbone.local_encoder
        self.assertEqual(local.proj.in_channels,1024)
        self.assertFalse(any('layer4' in k for k in local.state_dict()))
        for param in [local.proj.weight,local.backbone.conv1.weight,source.quality_image_projector.net[0].weight]:
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(),0.)
        # RichHF region maps adapt to the actual number of local tokens.
        region=_map_to_patch_weights(torch.rand(2,1,32,32),196)
        self.assertEqual(region.shape,(2,196))
        del out,loss
        # Exercise the exact representation extraction used for component checkpoints.
        with torch.no_grad():local.proj.weight.fill_(.123)
        state=_extract_representation_state(source.state_dict())
        self.assertIn('iqa.backbone.local_encoder.proj.weight',state)
        self.assertIn('quality_image_projector.net.0.weight',state)
        target_cfg=copy.deepcopy(cfg)
        target_cfg['model']['quality_adapter'].update(enabled=True,hidden_dim=16,output_dim=8,dropout=0.,last_bn=True,mode='residual_feature',residual_init=.1)
        target_cfg['model']['quality_auxiliary_heads']['enabled']=False
        with patch('models.two_stage_models.GlobalVisualEncoder',TinyGlobal), patch('models.two_stage_models.FrozenCLIPTextEncoder',TinyText):
            target=CrossModalIQAFullModel(target_cfg).eval()
        before=target.backbone.local_encoder.proj.weight.detach().clone()
        with patch('models.two_stage_models.torch.load',return_value={'model':state,'cfg':cfg,'checkpoint_scope':'representation'}):
            load_pretrained_backbone(target,'mock_layer3_component.pth',load_scope='representation')
        self.assertFalse(torch.equal(before,target.backbone.local_encoder.proj.weight))
        torch.testing.assert_close(target.backbone.local_encoder.proj.weight,local.proj.weight)
        torch.testing.assert_close(target.quality_adapter.net[0].weight,source.quality_image_projector.net[0].weight)
        self.assertEqual(target.backbone.local_encoder.feature_stage,'layer3')

if __name__=='__main__':unittest.main()
