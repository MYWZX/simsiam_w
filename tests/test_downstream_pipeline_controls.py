import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import savemat
import torch
from torch import nn
from torch.nn import functional as F

from datasets import build_dataloaders
from train_cm_iqa import compute_losses, evaluate, get_selection_value
from utils.downstream_controls import (configure_downstream_controls, distinct_prompt_mask,
                                      audit_fusion_gradients, validate_local_crops)


class ToyScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.3))
        self.calls = 0
    def forward_single(self, image, prompts=None, local_crops=None):
        self.calls += 1
        q = torch.sigmoid(self.weight * image.mean((1, 2, 3)))
        a = q * .8
        return dict(s_quality=q, s_align=a, s_final=(q+a)/2,
                    s_align_global=a, s_align_patch=a, s_quality_global_adapter=q,
                    s_quality_global_shared=q, quality_adapter_gate=q.new_tensor(0.),
                    fusion_w_img=q.new_tensor(.5), align_gate=q.new_full(q.shape, .5),
                    align_gate_input=q[:, None].expand(-1, 4), align_gate_logit=q*0)


def batch(prompts=None):
    return dict(image=torch.randn(4, 3, 8, 8), prompt=prompts or ['same']*4,
                quality_score=torch.tensor([.1, .3, .6, .9]),
                align_score=torch.tensor([.2, .5, .7, .8]), score=torch.full((4,), .5))


class PipelineControlsTests(unittest.TestCase):
    def test_controls_are_independent(self):
        cfg = dict(dataset={}, train=dict(lambda_neg=.05))
        configure_downstream_controls(cfg, SimpleNamespace(quality_safe_input=True))
        self.assertEqual(cfg['dataset']['quality_safe_flip_prob'], 0.)
        self.assertEqual(cfg['train']['lambda_neg'], .05)
        configure_downstream_controls(cfg, SimpleNamespace(disable_legacy_negative=True))
        self.assertEqual(cfg['train']['lambda_neg'], 0.)

    def test_negative_mask_and_all_invalid_skip_forward(self):
        valid = distinct_prompt_mask(['A', ' a ', '', 'B'], [' a ', 'B', 'C', ''], 'cpu')
        torch.testing.assert_close(valid, torch.tensor([0., 1., 0., 0.]))
        model = ToyScores()
        _, losses = compute_losses(model, batch(), 'cpu', dict(train=dict(two_output_only=True, lambda_neg=.05), dataset={}))
        self.assertEqual(model.calls, 1)
        self.assertEqual(losses['neg'].item(), 0.)
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_valid_negative_still_runs_and_disabled_negative_does_not(self):
        b = batch(['A', 'B', 'C', 'D'])
        for coefficient, calls in [(.05, 2), (0., 1)]:
            model = ToyScores()
            compute_losses(model, b, 'cpu', dict(train=dict(two_output_only=True, lambda_neg=coefficient), dataset={}))
            self.assertEqual(model.calls, calls)

    def test_evaluation_reports_real_quality_and_align_losses(self):
        model = ToyScores().eval()
        b = batch()
        cfg = dict(train=dict(two_output_only=True, lambda_quality=1., lambda_align=1.))
        out = model.forward_single(b['image'])
        m = evaluate(model, [b], 'cpu', cfg)
        expected_q = F.mse_loss(out['s_quality'], b['quality_score']).item()
        expected_a = F.mse_loss(out['s_align'], b['align_score']).item()
        self.assertAlmostEqual(m['val_quality_loss'], expected_q, places=6)
        self.assertAlmostEqual(m['val_align_loss'], expected_a, places=6)
        self.assertAlmostEqual(m['val_reg_loss'], expected_q+expected_a, places=6)

    def test_selection_typo_and_nan_raise(self):
        with self.assertRaises(ValueError):
            get_selection_value({'val_reg_loss': 0.}, {'train': {'selection_metric': 'typo'}})
        with self.assertRaises(ValueError):
            get_selection_value({'val_quality_SROCC': float('nan')}, {'train': {'selection_metric': 'val_quality_SROCC'}})
        value, maximize, _ = get_selection_value({'val_quality_SROCC': .8}, {'train': {'selection_metric': 'val_quality_SROCC', 'selection_mode': 'max'}})
        self.assertEqual(value, .8)
        self.assertTrue(maximize)

    def test_gradient_probe_preserves_optimizer_gradients(self):
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.fusion = nn.Linear(3, 2)
        data = torch.randn(4, 3)
        loss = model.backbone.fusion(data).square().mean()
        result = audit_fusion_gradients(model, {'quality_mse': loss})
        self.assertGreater(result['gradient_fusion_quality_mse_norm'], 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_enabled_crops_cannot_be_silently_ignored(self):
        with self.assertRaises(ValueError):
            validate_local_crops({}, {'dataset': {'local_multicrop': {'enabled': True, 'input_size': 336}}})

    def test_all_three_loaders_honor_resize_only_and_3k_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'images').mkdir()
            rows = []
            for i in range(10):
                pixels = np.zeros((24, 40, 3), dtype=np.uint8)
                pixels[:, :20, 0] = 230
                pixels[4:12, :, 1] = 120
                Image.fromarray(pixels).save(root/'images'/f'{i}.png')
                rows.append(dict(name=f'{i}.png', prompt='left red object', answer='red',
                                 mos_quality=1+i*.3, mos_align=2+i*.1, MOS=1+i*.3))
            pd.DataFrame(rows).to_csv(root/'data.csv', index=False)
            pd.DataFrame({'prompt': ['left red object']*10}).to_csv(root/'prompts.csv', index=False)
            savemat(root/'mosz1.mat', {'MOSz': np.linspace(30, 70, 10)[:, None]})
            savemat(root/'mosz3.mat', {'MOSz': np.linspace(35, 65, 10)[:, None]})
            for kind in ['agiqa1k_iqa', 'agiqa3k_simsiam', 'aigciqa2023_iqa']:
                cfg = dict(seed=42, dataset=dict(type=kind, root_dir=tmp, image_dir='images',
                           csv_file='data.csv', prompt_file='prompts.csv', input_size=16, mos_max=100 if kind=='aigciqa2023_iqa' else 5,
                           mosz1_key='MOSz', mosz3_key='MOSz', use_perturb=False),
                           loader=dict(batch_size=2, num_workers=0, pin_memory=False), train={})
                configure_downstream_controls(cfg, SimpleNamespace(quality_safe_input=True))
                if kind == 'agiqa3k_simsiam':
                    cfg['dataset']['local_multicrop'] = dict(enabled=True, input_size=336, train_num_crops=5, eval_num_crops=5)
                train, val, test = build_dataloaders(cfg)
                dataset = train.dataset.dataset
                def no_legacy_transform(*args):
                    raise AssertionError('resize-only must not call legacy training augmentation')
                with patch.object(dataset, 'train_transform', side_effect=no_legacy_transform):
                    first, second = dataset[0], dataset[0]
                torch.testing.assert_close(first['image'], second['image'], rtol=0, atol=0)
                with Image.open(root/'images'/'0.png') as image:
                    expected = dataset.eval_transform(image)
                torch.testing.assert_close(first['image'], expected, rtol=0, atol=0)
                self.assertEqual(first['prompt'], 'left red object')
                if kind == 'agiqa3k_simsiam':
                    self.assertEqual(tuple(first['local_crops'].shape), (5, 3, 336, 336))
                    validate_local_crops(next(iter(train)), cfg)
                train_idx, val_idx, test_idx = map(lambda loader: set(loader.dataset.indices), [train, val, test])
                self.assertFalse(train_idx & val_idx or train_idx & test_idx or val_idx & test_idx)


if __name__ == '__main__':
    unittest.main()
