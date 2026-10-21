"""Stage extraction and explicit local projection transfer; no downloads."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.backbones import ResNet50LocalEncoder
from models.two_stage_models import CrossModalIQAFullModel, load_pretrained_backbone
from utils.clip_experiments import configure_local_feature_stage


def local_model(stage):
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.local_encoder = ResNet50LocalEncoder(pretrained=False, feature_stage=stage)
    return model


class LocalFeatureStageTests(unittest.TestCase):
    def test_real_resnet_shapes_and_default_keys(self):
        for stage, channels, spatial in [('layer3', 1024, 14), ('layer4', 2048, 7)]:
            encoder = ResNet50LocalEncoder(pretrained=False, feature_stage=stage).eval()
            with torch.no_grad():
                result = encoder(torch.randn(1, 3, 224, 224))
            self.assertEqual(result.shape, (1, 256, spatial, spatial))
            self.assertEqual(encoder.proj.in_channels, channels)
            if stage == 'layer3':
                self.assertFalse(any('layer4' in key for key in encoder.backbone.state_dict()))
        default = ResNet50LocalEncoder(pretrained=False)
        explicit = ResNet50LocalEncoder(pretrained=False, feature_stage='layer4')
        self.assertEqual({k: v.shape for k,v in default.state_dict().items()},
                         {k: v.shape for k,v in explicit.state_dict().items()})
        with self.assertRaises(ValueError):
            ResNet50LocalEncoder(pretrained=False, feature_stage='layer2')

    def test_stage_mismatch_fails_without_permission(self):
        source, target = local_model('layer4'), local_model('layer3')
        before = target.backbone.local_encoder.proj.weight.detach().clone()
        with patch('torch.load', return_value={'model': source.state_dict()}):
            with self.assertRaisesRegex(RuntimeError, 'Local projection checkpoint mismatch'):
                load_pretrained_backbone(target, 'mock.pth', load_scope='representation')
        torch.testing.assert_close(target.backbone.local_encoder.proj.weight, before)

    def test_explicit_transfer_restores_trunk_and_preserves_both_new_projection_tensors(self):
        source, target = local_model('layer4'), local_model('layer3')
        state = {'iqa.'+key: value for key,value in source.state_dict().items()}
        state['iqa.backbone.local_encoder.backbone.conv1.weight'].fill_(.123)
        weight = target.backbone.local_encoder.proj.weight.detach().clone()
        bias = target.backbone.local_encoder.proj.bias.detach().clone()
        with patch('torch.load', return_value={'model': state}):
            load_pretrained_backbone(target, 'mock.pth', load_scope='representation', reinit_local_projection=True)
        torch.testing.assert_close(target.backbone.local_encoder.proj.weight, weight)
        torch.testing.assert_close(target.backbone.local_encoder.proj.bias, bias)
        torch.testing.assert_close(target.backbone.local_encoder.backbone.conv1.weight,
                                  source.backbone.local_encoder.backbone.conv1.weight)

    def test_matching_stage_restores_projection_without_reset(self):
        source, target = local_model('layer3'), local_model('layer3')
        with patch('torch.load', return_value={'model': source.state_dict()}):
            load_pretrained_backbone(target, 'mock.pth', load_scope='representation')
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[key])
        with patch('torch.load', return_value={'model': source.state_dict()}):
            with self.assertRaisesRegex(ValueError, 'only supported'):
                load_pretrained_backbone(target, 'mock.pth', reinit_local_projection=True)

    def test_missing_projection_or_trunk_is_not_silently_skipped(self):
        source, target = local_model('layer4'), local_model('layer3')
        state = dict(source.state_dict())
        state.pop('backbone.local_encoder.proj.bias')
        with patch('torch.load', return_value={'model': state}):
            with self.assertRaisesRegex(RuntimeError, 'lacks local projection'):
                load_pretrained_backbone(target, 'mock.pth', reinit_local_projection=True)
        state = dict(source.state_dict())
        state.pop('backbone.local_encoder.backbone.conv1.weight')
        with patch('torch.load', return_value={'model': state}):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete local trunk'):
                load_pretrained_backbone(target, 'mock.pth', reinit_local_projection=True)

    def test_cli_config_is_independent_and_checks_requirements(self):
        cfg = {'model': {'local_encoder': {}}, 'train': {'pretrained_checkpoint': 'mock.pth'}}
        original = copy.deepcopy(cfg)
        configure_local_feature_stage(cfg, SimpleNamespace())
        self.assertEqual(cfg, original)
        configure_local_feature_stage(cfg, SimpleNamespace(local_feature_stage='layer3', reinit_local_projection=True))
        self.assertEqual(cfg['model']['local_encoder']['feature_stage'], 'layer3')
        self.assertTrue(cfg['train']['reinit_local_projection'])
        for invalid in [{}, {'model': {'local_encoder': {'feature_stage': 'layer4'}}, 'train': {'pretrained_checkpoint': 'mock.pth'}}]:
            with self.assertRaises(ValueError):
                configure_local_feature_stage(invalid, SimpleNamespace(reinit_local_projection=True))

    def test_full_model_uses_real_layer3_for_fusion_and_both_outputs_with_gradients(self):
        class TinyGlobal(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.proj = nn.Linear(3, 8)
            def forward(self, x):
                return self.proj(x.mean((2,3)))
        class TinyText(nn.Module):
            output_dim = 6
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, texts, device):
                return torch.ones(len(texts), 6, device=device)
        cfg = {'dataset': {'input_size': 224}, 'model': {
            'global_encoder': {'output_dim': 8, 'pretrained': False},
            'local_encoder': {'feature_stage': 'layer3', 'output_dim': 256, 'pretrained': False},
            'fusion': {'embed_dim': 8, 'num_heads': 2, 'dropout': 0.},
            'heads': {'mlp_hidden_dim': 8, 'dropout': 0.},
            'quality_branch_use_text_gate': False}}
        with patch('models.two_stage_models.GlobalVisualEncoder', TinyGlobal), patch('models.two_stage_models.FrozenCLIPTextEncoder', TinyText):
            model = CrossModalIQAFullModel(cfg).eval()
        out = model.forward_single(torch.randn(2,3,224,224), prompts=['cat','dog'])
        self.assertEqual(out['f_loc'].shape, (2,256,14,14))
        self.assertEqual(out['quality_patch_weights'].shape, (2,196))
        self.assertEqual(out['align_patch_weights'].shape, (2,196))
        self.assertEqual(out['z'].shape, (2,8))
        loss = out['s_quality'].sum() + out['s_align'].sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.backbone.local_encoder.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.local_encoder.backbone.conv1.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
