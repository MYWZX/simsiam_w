"""Actual ResNet multiscale math, gradients, checkpoint contracts and consumers."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from models.backbones import ResNet50LocalEncoder
from models.two_stage_models import CrossModalIQAJointPretrain, CrossModalIQAFullModel, load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state, _map_to_patch_weights
from train_cm_iqa import build_optimizer, evaluate
from utils.clip_experiments import configure_local_feature_stage, local_stage_diagnostics
from utils.train_utils import load_yaml
from test_layer3_pretraining import TinyGlobal, TinyText

ROOT=Path(__file__).resolve().parents[1]

def local_model(stage):
    m=nn.Module();m.backbone=nn.Module();m.backbone.local_encoder=ResNet50LocalEncoder(pretrained=False,feature_stage=stage);return m

class MultiscaleLocalTests(unittest.TestCase):
    def test_actual_shapes_formula_initial_weights_and_both_gradients(self):
        e=ResNet50LocalEncoder(pretrained=False,feature_stage='layer3_4').eval()
        x=torch.randn(2,3,224,224)
        with torch.no_grad():
            f3,f4=e.backbone(x)
            self.assertEqual(f3.shape,(2,1024,14,14));self.assertEqual(f4.shape,(2,2048,7,7))
            expected=.5*e.proj3(f3)+.5*F.interpolate(e.proj4(f4),size=(14,14),mode='bilinear',align_corners=False)
        feature_grads=[]
        def retain_features(mod, inputs, outputs):
            for feature in outputs:feature.retain_grad()
            feature_grads.extend(outputs)
        hook=e.backbone.register_forward_hook(retain_features)
        y=e(x);hook.remove();self.assertEqual(y.shape,(2,256,14,14));torch.testing.assert_close(y,expected)
        (y*torch.randn_like(y)).mean().backward()
        for param in [e.proj3.weight,e.proj4.weight,e.stage_mix_logit,e.backbone.conv1.weight]:
            self.assertTrue(torch.isfinite(param.grad).all());self.assertGreater(param.grad.abs().sum().item(),0.)
        for feature in feature_grads:
            self.assertTrue(torch.isfinite(feature.grad).all());self.assertGreater(feature.grad.abs().sum().item(),0.)
        self.assertEqual((e.proj3.in_channels,e.proj4.in_channels),(1024,2048))
        # Spatial sizes derive from actual features, not hard-coded 14/7.
        with torch.no_grad():self.assertEqual(e(torch.randn(1,3,256,256)).shape,(1,256,16,16))

    def test_matching_multiscale_restores_trunk_both_projections_and_gate(self):
        src,dst=local_model('layer3_4'),local_model('layer3_4')
        with torch.no_grad():src.backbone.local_encoder.stage_mix_logit.fill_(1.2)
        state={'module.iqa.'+k:v for k,v in src.state_dict().items()}
        with patch('models.two_stage_models.torch.load',return_value={'model':state}):
            load_pretrained_backbone(dst,'mock.pth',load_scope='representation')
        for key,value in src.state_dict().items():torch.testing.assert_close(value,dst.state_dict()[key])
        diag=local_stage_diagnostics(dst)
        self.assertAlmostEqual(diag['local_layer3_weight'],torch.sigmoid(torch.tensor(1.2)).item())
        self.assertAlmostEqual(sum(diag.values()),1.)

    def test_stage_mismatch_rejected_both_directions_before_loading(self):
        for a,b in [('layer3','layer3_4'),('layer4','layer3_4'),('layer3_4','layer3'),('layer3_4','layer4')]:
            src,dst=local_model(a),local_model(b)
            before=dst.backbone.local_encoder.backbone.conv1.weight.detach().clone()
            with patch('models.two_stage_models.torch.load',return_value={'model':src.state_dict()}):
                with self.assertRaisesRegex(RuntimeError,'Local feature stage checkpoint mismatch'):
                    load_pretrained_backbone(dst,'mock.pth',load_scope='representation')
            torch.testing.assert_close(before,dst.backbone.local_encoder.backbone.conv1.weight)

    def test_incomplete_or_wrong_shape_multiscale_checkpoint_rejected(self):
        src,dst=local_model('layer3_4'),local_model('layer3_4');state=src.state_dict()
        for key in ['backbone.local_encoder.proj3.bias','backbone.local_encoder.proj4.weight','backbone.local_encoder.stage_mix_logit','backbone.local_encoder.backbone.layer4.0.conv1.weight']:
            partial=dict(state);partial.pop(key)
            with patch('models.two_stage_models.torch.load',return_value={'model':partial}):
                with self.assertRaisesRegex(RuntimeError,'Incomplete multiscale local checkpoint'):
                    load_pretrained_backbone(dst,'mock.pth',load_scope='representation')
        partial=dict(state);partial['backbone.local_encoder.proj4.weight']=torch.randn(1)
        with patch('models.two_stage_models.torch.load',return_value={'model':partial}):
            with self.assertRaisesRegex(RuntimeError,'Multiscale local checkpoint shape mismatch'):
                load_pretrained_backbone(dst,'mock.pth',load_scope='representation')

    def test_config_isolated_and_reinit_not_allowed(self):
        old=load_yaml(str(ROOT/'configs/richhf18k_cm_pretrain_decoupled_quality_align_layer3.yaml'))
        new=load_yaml(str(ROOT/'configs/richhf18k_cm_pretrain_decoupled_quality_align_layer34.yaml'))
        expected=copy.deepcopy(old);expected['model']['local_encoder']['feature_stage']='layer3_4';expected['output_dir']=new['output_dir'];expected['results_csv']=new['results_csv'];self.assertEqual(new,expected)
        cfg=copy.deepcopy(old);configure_local_feature_stage(cfg,SimpleNamespace(local_feature_stage='layer3_4',reinit_local_projection=False));self.assertEqual(cfg['model']['local_encoder']['feature_stage'],'layer3_4')
        cfg['train']['pretrained_checkpoint']='mock.pth'
        with self.assertRaisesRegex(ValueError,'requires layer3'):
            configure_local_feature_stage(cfg,SimpleNamespace(reinit_local_projection=True))
        self.assertEqual(local_stage_diagnostics(nn.Linear(1,1)),{})

    def test_three_views_and_all_downstream_consumers_use_multiscale(self):
        cfg=load_yaml(str(ROOT/'configs/richhf18k_cm_pretrain_decoupled_quality_align_layer34.yaml'));m=cfg['model']
        m['global_encoder'].update(pretrained=False,output_dim=8);m['local_encoder']['pretrained']=False
        m['fusion'].update(embed_dim=8,num_heads=2,dropout=0.);m['heads'].update(mlp_hidden_dim=16,dropout=0.)
        m['simsiam'].update(proj_dim=8,proj_hidden_dim=16,pred_hidden_dim=4)
        with patch('models.two_stage_models.GlobalVisualEncoder',TinyGlobal),patch('models.two_stage_models.FrozenCLIPTextEncoder',TinyText):src=CrossModalIQAJointPretrain(cfg).eval()
        seen=[];hook=src.iqa.backbone.local_encoder.register_forward_hook(lambda mod,ins,out:seen.append(tuple(out.shape)))
        imgs=[torch.randn(2,3,224,224) for _ in range(3)]
        out=src._forward_three_views(*imgs,['cat','dog'],['cat','dog']);hook.remove()
        self.assertEqual(seen,[(2,256,14,14)]*3);self.assertEqual(out['align_patch_features_clean'].shape,(2,196,8))
        loss=out['quality_pred'].sum()+out['align_pred'].sum()+out['quality_clean_z'][:,0].mean();loss.backward()
        for p in [src.iqa.backbone.local_encoder.proj3.weight,src.iqa.backbone.local_encoder.proj4.weight,src.iqa.backbone.local_encoder.stage_mix_logit]:self.assertGreater(p.grad.abs().sum().item(),0.)
        self.assertEqual(_map_to_patch_weights(torch.rand(2,1,32,32),196).shape,(2,196))
        state=_extract_representation_state(src.state_dict());self.assertIn('iqa.backbone.local_encoder.stage_mix_logit',state)
        down=copy.deepcopy(cfg);down['model']['quality_adapter'].update(enabled=True,hidden_dim=16,output_dim=8,last_bn=True,dropout=0.,mode='residual_feature',residual_init=.1);down['model']['quality_auxiliary_heads']['enabled']=False
        with patch('models.two_stage_models.GlobalVisualEncoder',TinyGlobal),patch('models.two_stage_models.FrozenCLIPTextEncoder',TinyText):dst=CrossModalIQAFullModel(down).eval()
        with patch('models.two_stage_models.torch.load',return_value={'model':state}):load_pretrained_backbone(dst,'mock_component.pth',load_scope='representation')
        torch.testing.assert_close(dst.backbone.local_encoder.stage_mix_logit,src.iqa.backbone.local_encoder.stage_mix_logit)
        optim=build_optimizer(down,dst);group=next(g for g in optim.param_groups if g['name']=='local_encoder')
        self.assertTrue(any(p is dst.backbone.local_encoder.stage_mix_logit for p in group['params']))
        batch={'image':imgs[0],'prompt':['cat','dog'],'score':torch.tensor([.2,.8]),'quality_score':torch.tensor([.2,.8]),'align_score':torch.tensor([.3,.7])}
        metrics=evaluate(dst,[batch],torch.device('cpu'),down)
        self.assertAlmostEqual(metrics['val_local_layer3_weight']+metrics['val_local_layer4_weight'],1.)

if __name__=='__main__':unittest.main()
