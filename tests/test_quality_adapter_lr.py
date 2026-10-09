"""Quality Adapter LR isolation checks; no datasets or pretrained weights."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

import train_cm_iqa
from train_cm_iqa import (
    build_optimizer, configure_downstream_pcgrad, configure_quality_adapter_lr, parse_args,
)
from utils.train_utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]


class TinyIQA(nn.Module):
    """Exercise the real optimizer's name-based grouping without big encoders."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.global_encoder = nn.Linear(2, 2)
        self.backbone.local_encoder = nn.Linear(2, 2)
        self.backbone.fusion = nn.Linear(2, 2)
        self.quality_adapter = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.global_quality_head = nn.Linear(2, 1)
        self.local_patch_proj = nn.Linear(2, 2)
        self.align_global_head = nn.Linear(2, 1)
        self.align_global_adapter = nn.Linear(2, 2)
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.other_weight = nn.Parameter(torch.tensor(0.0))


class QualityAdapterLRTests(unittest.TestCase):
    def args(self, *flags):
        with patch('sys.argv', ['train_cm_iqa.py', '--config', 'unused.yaml', *flags]):
            return parse_args()

    def base(self):
        return {
            'train': {'lr': 1e-4, 'layerwise_lr': {'enabled': True, 'quality_mult': 1.0}},
            'model': {'quality_adapter': {'enabled': True}},
        }

    def test_omitting_flag_preserves_configuration(self):
        cfg = self.base()
        expected = copy.deepcopy(cfg)
        configure_quality_adapter_lr(cfg, self.args())
        self.assertEqual(cfg, expected)

    def test_only_one_config_leaf_changes(self):
        cfg = self.base()
        cfg['train']['pcgrad'] = {'enabled': True, 'mode': 'global_shared', 'projection_strength': 0.5}
        cfg['model']['quality_adapter'].update({'freeze': False, 'detach_input': False})
        expected = copy.deepcopy(cfg)
        expected['train']['layerwise_lr']['quality_adapter_mult'] = 0.10
        configure_quality_adapter_lr(cfg, self.args('--quality-adapter-lr-mult', '0.10'))
        self.assertEqual(cfg, expected)

    def test_explicit_yaml_multiplier_can_be_overridden(self):
        cfg = self.base()
        cfg['train']['layerwise_lr']['quality_adapter_mult'] = 0.75
        configure_quality_adapter_lr(cfg, self.args('--quality-adapter-lr-mult', '0.10'))
        self.assertEqual(cfg['train']['layerwise_lr']['quality_adapter_mult'], 0.10)

    def test_invalid_values_rejected_without_mutation(self):
        for value in ['0', '-0.1', 'nan', 'inf']:
            cfg = self.base()
            expected = copy.deepcopy(cfg)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                configure_quality_adapter_lr(cfg, self.args('--quality-adapter-lr-mult', value))
            self.assertEqual(cfg, expected)

    def test_disabled_layerwise_lr_rejected(self):
        cfg = self.base()
        cfg['train']['layerwise_lr']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'layerwise_lr.enabled=true'):
            configure_quality_adapter_lr(cfg, self.args('--quality-adapter-lr-mult', '0.10'))

    def test_disabled_or_frozen_adapter_rejected(self):
        for qcfg in [{'enabled': False}, {'enabled': True, 'freeze': True}]:
            cfg = self.base()
            cfg['model']['quality_adapter'] = qcfg
            with self.subTest(qcfg=qcfg), self.assertRaisesRegex(ValueError, 'enabled, unfrozen'):
                configure_quality_adapter_lr(cfg, self.args('--quality-adapter-lr-mult', '0.10'))

    def test_all_three_configs_change_only_adapter_optimizer_group(self):
        for dataset in ['agiqa3k', 'aigciqa2023', 'agiqa1k']:
            with self.subTest(dataset=dataset):
                source = load_yaml(str(ROOT / 'configs' / f'{dataset}_cm_iqa_decoupled_quality_align.yaml'))
                low_cfg = copy.deepcopy(source)
                args = self.args('--pcgrad-mode', 'global_shared', '--pcgrad-projection-strength', '0.5', '--quality-adapter-lr-mult', '0.10')
                configure_downstream_pcgrad(low_cfg, args)
                before_cfg = copy.deepcopy(low_cfg)
                configure_quality_adapter_lr(low_cfg, args)
                expected = copy.deepcopy(before_cfg)
                expected['train']['layerwise_lr']['quality_adapter_mult'] = 0.10
                self.assertEqual(low_cfg, expected)
                model = TinyIQA()
                state = {k: v.clone() for k, v in model.state_dict().items()}
                before = {g['name']: g for g in build_optimizer(before_cfg, model).param_groups}
                after = {g['name']: g for g in build_optimizer(low_cfg, model).param_groups}
                self.assertEqual(before.keys(), after.keys())
                changed = [name for name in before if before[name]['lr'] != after[name]['lr']]
                self.assertEqual(changed, ['quality_adapter'])
                self.assertAlmostEqual(before['quality_adapter']['lr'], 1e-4)
                self.assertAlmostEqual(after['quality_adapter']['lr'], 1e-5)
                for name in before:
                    self.assertEqual([id(p) for p in before[name]['params']], [id(p) for p in after[name]['params']])
                    self.assertEqual(before[name]['weight_decay'], after[name]['weight_decay'])
                self.assertAlmostEqual(after['quality']['lr'], 1e-4)
                self.assertAlmostEqual(after['align']['lr'], 1e-4)
                self.assertAlmostEqual(after['fusion']['lr'], 1e-5)
                self.assertAlmostEqual(after['global_encoder']['lr'], 2e-6)
                self.assertAlmostEqual(after['local_encoder']['lr'], 2e-6)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, state[key]))
                self.assertTrue(model.quality_adapter[1].training)
                self.assertTrue(all(p.requires_grad for p in model.quality_adapter.parameters()))

    def test_main_records_override_before_loading_data(self):
        cfg = load_yaml(str(ROOT / 'configs' / 'agiqa3k_cm_iqa_decoupled_quality_align.yaml'))
        cfg['train']['pretrained_checkpoint'] = ''
        class StopBeforeData(RuntimeError):
            pass
        flags = ['--config', 'unused.yaml', '--quality-adapter-lr-mult', '0.10', '--pcgrad-mode', 'global_shared', '--pcgrad-projection-strength', '0.5']
        with (
            patch('sys.argv', ['train_cm_iqa.py', *flags]),
            patch('train_cm_iqa.load_yaml', return_value=cfg),
            patch('train_cm_iqa.set_seed'),
            patch('train_cm_iqa.ensure_dir'),
            patch('train_cm_iqa.dump_config_snapshot', return_value='unused-snapshot.yaml') as snapshot,
            patch('train_cm_iqa.CSVExperimentLogger') as logger,
            patch('train_cm_iqa.torch.cuda.is_available', return_value=False),
            patch('train_cm_iqa.build_dataloaders', side_effect=StopBeforeData),
            patch('train_cm_iqa.build_iqa_model') as build_model,
        ):
            with self.assertRaises(StopBeforeData):
                train_cm_iqa.main()
        effective = snapshot.call_args.args[0]
        self.assertEqual(effective['train']['layerwise_lr']['quality_adapter_mult'], 0.10)
        self.assertEqual(effective['train']['pcgrad']['projection_strength'], 0.5)
        self.assertEqual(logger.call_args.kwargs['cfg'], effective)
        build_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
