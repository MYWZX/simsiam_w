"""Zero-init equivalence, trainability, and masked branch diagnostics."""
import unittest
from unittest.mock import patch
import torch
from models.two_stage_models import CrossModalIQAFullModel
from test_quality_residual_feature import TinyBackbone
from train_cm_iqa import evaluate, build_optimizer


def build(enabled=False, global_only=False):
    cfg = {'model': {'heads': {'mlp_hidden_dim': 16, 'dropout': 0.},
                     'fusion': {'num_heads': 2}, 'quality_branch_use_text_gate': False,
                     'quality_global_only': global_only,
                     'quality_sample_gate': {'enabled': enabled}},
           'train': {'two_output_only': True, 'lr': 1e-4, 'layerwise_lr': {'enabled': True}}}
    with patch('models.two_stage_models.ImageTextBackbone', TinyBackbone):
        return CrossModalIQAFullModel(cfg).eval(), cfg


class QualitySampleGateTests(unittest.TestCase):
    def test_initial_equivalence_preserves_existing_weights_and_align(self):
        torch.manual_seed(42)
        base, _ = build()
        torch.manual_seed(42)
        gated, cfg = build(True)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, gated.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(4, 3, 8, 8)
        a, b = base.forward_single(x), gated.forward_single(x)
        torch.testing.assert_close(a['s_quality'], b['s_quality'], rtol=0, atol=0)
        torch.testing.assert_close(a['s_align'], b['s_align'], rtol=0, atol=0)
        b['s_quality'].sum().backward()
        self.assertGreater(gated.quality_sample_gate[-1].weight.grad.abs().sum().item(), 0)
        optimizer = build_optimizer(cfg, gated)
        group = next(g for g in optimizer.param_groups if g['name'] == 'quality')
        self.assertTrue(any(p is gated.quality_sample_gate[-1].weight for p in group['params']))
        optimizer.step()
        out = gated.forward_single(x)
        self.assertGreater(out['quality_mix'].std().item(), 0)
        self.assertTrue(((out['quality_mix'] > 0) & (out['quality_mix'] < 1)).all())

    def test_global_only_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            build(True, True)

    def test_masked_evaluation_diagnostics(self):
        model, cfg = build(True)
        batch = {'image': torch.randn(4, 3, 8, 8), 'prompt': ['x']*4,
                 'score': torch.tensor([0., .2, .7, 1.]),
                 'quality_score': torch.tensor([0., .2, .7, 1.]),
                 'align_score': torch.tensor([0., .2, .7, 1.]),
                 'quality_mask': torch.tensor([1., 0., 1., 1.])}
        metrics = evaluate(model, [batch], torch.device('cpu'), cfg)
        self.assertEqual(metrics['val_quality_global_N'], 3)
        self.assertEqual(metrics['val_quality_patch_N'], 3)
        self.assertAlmostEqual(metrics['val_quality_mix_mean'], .5)
        self.assertEqual(metrics['val_quality_mix_std'], 0.)
        self.assertIn('val_quality_branch_error_PLCC', metrics)

if __name__ == '__main__':
    unittest.main()
