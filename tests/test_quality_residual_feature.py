"""Real residual Quality forward/backward with tiny encoders; no downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from models.two_stage_models import CrossModalIQAFullModel, load_pretrained_backbone


class TinyBackbone(nn.Module):
    fused_dim, local_dim, text_dim = 8, 4, 6

    def __init__(self, cfg):
        super().__init__()
        self.global_encoder = nn.Linear(3, 8)
        self.local_encoder = nn.Conv2d(3, 4, 1)
        self.fusion = nn.Identity()

    def encode_image(self, image, local_crops=None):
        return dict(z=self.global_encoder(image.mean((2, 3))),
                    f_loc=self.local_encoder(F.adaptive_avg_pool2d(image, (2, 2))))

    def encode_text(self, prompts, device):
        return torch.ones(len(prompts), 6, device=device)


def build(mode='residual_feature'):
    cfg = dict(model=dict(heads=dict(mlp_hidden_dim=16, dropout=0.),
                          fusion=dict(num_heads=2), quality_global_only=True,
                          quality_branch_use_text_gate=False,
                          quality_adapter=dict(enabled=True, output_dim=8, hidden_dim=16,
                                               dropout=0., mode=mode, residual_init=.1)))
    with patch('models.two_stage_models.ImageTextBackbone', TinyBackbone):
        return CrossModalIQAFullModel(cfg)


class ResidualFeatureTests(unittest.TestCase):
    def test_residual_scale_and_gradients_to_both_features_and_gate(self):
        model = build()
        shared = torch.randn(4, 8, requires_grad=True)
        adapted = torch.randn(4, 8, requires_grad=True)
        mixed, alpha = model._quality_residual_feature(shared, adapted)
        ratio = (mixed-shared).norm(dim=-1) / shared.norm(dim=-1)
        torch.testing.assert_close(ratio, torch.full((4,), .1))
        (mixed * torch.randn_like(mixed)).sum().backward()
        for grad in [shared.grad, adapted.grad, model.quality_adapter_gate_logit.grad]:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)

    def test_full_quality_route_backward_and_unchanged_align_forward(self):
        torch.manual_seed(42)
        residual = build().eval()
        torch.manual_seed(42)
        baseline = build('replace').eval()
        image = torch.randn(3, 3, 8, 8)
        out = residual.forward_single(image, prompts=['test']*3)
        old = baseline.forward_single(image, prompts=['test']*3)
        torch.testing.assert_close(out['s_align'], old['s_align'])
        torch.testing.assert_close(out['s_quality'], out['s_quality_global'])
        features = residual.backbone.encode_image(image)['z']
        mixed, alpha = residual._quality_residual_feature(features, out['z_quality_global'])
        expected = torch.sigmoid(residual.global_quality_head(mixed).squeeze(-1))
        torch.testing.assert_close(out['s_quality'], expected)
        out['s_quality'].sum().backward()
        for param in [residual.quality_adapter.net[0].weight,
                      residual.backbone.global_encoder.weight,
                      residual.quality_adapter_gate_logit]:
            self.assertGreater(param.grad.abs().sum().item(), 0.)

    def test_representation_checkpoint_restores_adapter_preserves_new_gate(self):
        original = build('replace')
        resumed = build()
        state = {'iqa.'+k: v for k, v in original.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'pretrain.pth')
            torch.save({'model': state}, path)
            load_pretrained_backbone(resumed, path, load_scope='representation')
        torch.testing.assert_close(resumed.quality_adapter.net[0].weight,
                                   original.quality_adapter.net[0].weight)
        self.assertAlmostEqual(torch.sigmoid(resumed.quality_adapter_gate_logit).item(), .1, places=6)

    def test_text_bank_keeps_adapter_only_input(self):
        class CaptureBank(nn.Module):
            def forward(self, features):
                self.seen = features
                return dict(quality_text_logits=features[:, :5],
                            quality_text_score=features[:, 0].sigmoid())
        model = build().eval()
        model.quality_text_bank = CaptureBank()
        out = model.forward_single(torch.randn(3, 3, 8, 8))
        torch.testing.assert_close(model.quality_text_bank.seen, out['z_quality_global'])
        torch.testing.assert_close(out['z_quality_global'].norm(dim=-1), torch.ones(3))

    def test_zero_features_are_finite(self):
        model = build()
        mixed, alpha = model._quality_residual_feature(torch.zeros(3, 8), torch.zeros(3, 8))
        self.assertTrue(torch.isfinite(mixed).all())


if __name__ == '__main__':
    unittest.main()
