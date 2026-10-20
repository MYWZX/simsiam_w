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


def build(mode='residual_feature', fixed=None, freeze_bn_stats=False, adapter_dropout=0.):
    cfg = dict(model=dict(heads=dict(mlp_hidden_dim=16, dropout=0.),
                          fusion=dict(num_heads=2), quality_global_only=True,
                          quality_branch_use_text_gate=False,
                          quality_adapter=dict(enabled=True, output_dim=8, hidden_dim=16,
                                               dropout=adapter_dropout, mode=mode, residual_init=.1, residual_fixed=fixed,
                                               freeze_bn_stats=freeze_bn_stats)))
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

    def test_fixed_zero_is_identity_and_auxiliary_gradient_survives(self):
        class CaptureBank(nn.Module):
            def forward(self, features):
                return dict(quality_text_logits=features[:, :5],
                            quality_text_score=features[:, 0].sigmoid())
        model = build(fixed=0.).eval()
        model.quality_text_bank = CaptureBank()
        self.assertFalse(model.quality_adapter_gate_logit.requires_grad)
        shared = torch.randn(3, 8, requires_grad=True)
        mixed, alpha = model._quality_residual_feature(shared, torch.randn(3, 8))
        self.assertIs(mixed, shared)
        self.assertEqual(alpha.item(), 0.)
        out = model.forward_single(torch.randn(3, 3, 8, 8))
        torch.testing.assert_close(out['s_quality_global'], out['s_quality_global_shared'])
        self.assertEqual(out['quality_adapter_gate'].item(), 0.)
        out['s_quality'].sum().backward(retain_graph=True)
        self.assertIsNone(model.quality_adapter.net[0].weight.grad)
        out['quality_text_logits'].square().sum().backward()
        self.assertGreater(model.quality_adapter.net[0].weight.grad.abs().sum().item(), 0.)

    def test_invalid_fixed_residual_is_rejected(self):
        for value in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                build(fixed=value)
        with self.assertRaises(ValueError):
            build(mode='replace', fixed=0.)

    def test_frozen_adapter_bn_statistics_and_trainable_parameters(self):
        model = build(freeze_bn_stats=True, adapter_dropout=.2)
        # Stand-in for normalization outside the Quality Adapter.
        model.unrelated_bn = nn.BatchNorm1d(8)
        bns = [m for m in model.quality_adapter.modules() if isinstance(m, nn.BatchNorm1d)]
        self.assertGreater(len(bns), 0)
        for bn in bns:
            bn.running_mean.fill_(.2)
            bn.running_var.fill_(1.3)
            bn.num_batches_tracked.fill_(12)
        originals = [(bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()) for bn in bns]
        for _ in range(2):
            model.eval()
            model.train()
            self.assertTrue(model.unrelated_bn.training)
            self.assertTrue(model.quality_adapter.training)
            self.assertTrue(all(not bn.training for bn in bns))
            self.assertTrue(all(m.training for m in model.quality_adapter.modules() if isinstance(m, nn.Dropout)))
        model.zero_grad()
        out = model.forward_single(torch.randn(4, 3, 8, 8), prompts=['test']*4)
        out['s_quality'].sum().backward()
        self.assertGreater(model.quality_adapter.net[0].weight.grad.abs().sum().item(), 0.)
        for bn, original in zip(bns, originals):
            for current, before in zip((bn.running_mean, bn.running_var, bn.num_batches_tracked), original):
                torch.testing.assert_close(current, before, rtol=0, atol=0)
            if bn.affine:
                self.assertTrue(bn.weight.requires_grad and bn.bias.requires_grad)
                self.assertGreater(bn.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(bn.bias.grad.abs().sum().item(), 0.)

    def test_default_adapter_bn_keeps_updating(self):
        model = build().train()
        bn = next(m for m in model.quality_adapter.modules() if isinstance(m, nn.BatchNorm1d))
        before = bn.num_batches_tracked.clone()
        model.forward_single(torch.randn(4, 3, 8, 8))
        self.assertTrue(bn.training)
        self.assertEqual(bn.num_batches_tracked.item(), before.item()+1)

    def test_zero_features_are_finite(self):
        model = build()
        mixed, alpha = model._quality_residual_feature(torch.zeros(3, 8), torch.zeros(3, 8))
        self.assertTrue(torch.isfinite(mixed).all())


if __name__ == '__main__':
    unittest.main()
