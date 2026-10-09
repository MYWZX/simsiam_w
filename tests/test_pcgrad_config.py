"""CLI/config regression tests; no data or model construction."""
import copy
import unittest
from unittest.mock import patch

from train_cm_iqa import configure_downstream_pcgrad, parse_args


class PCGradConfigTests(unittest.TestCase):
    def configure(self, cfg=None, flags=()):
        cfg = copy.deepcopy(cfg or {'train': {'two_output_only': True}})
        with patch('sys.argv', ['train_cm_iqa.py', '--config', 'unused.yaml', *flags]):
            args = parse_args()
        configure_downstream_pcgrad(cfg, args)
        return cfg

    def test_normal_config_unchanged(self):
        cfg = {'train': {'two_output_only': True}}
        self.assertEqual(self.configure(cfg), cfg)

    def test_cli_flag_defaults_to_full_global_projection(self):
        cfg = self.configure(flags=['--enable-pcgrad'])
        self.assertEqual(cfg['train']['pcgrad'], {
            'enabled': True, 'mode': 'global_shared', 'projection_strength': 1.0,
        })

    def test_strength_flag_enables_half_global_projection(self):
        cfg = self.configure(flags=['--pcgrad-projection-strength', '0.5'])
        self.assertEqual(cfg['train']['pcgrad'], {
            'enabled': True, 'mode': 'global_shared', 'projection_strength': 0.5,
        })

    def test_cli_overrides_yaml_strength_and_preserves_other_fields(self):
        cfg = {'train': {
            'two_output_only': True, 'lambda_quality': 1.0,
            'pcgrad': {'enabled': True, 'mode': 'global_shared', 'projection_strength': 0.75},
        }, 'seed': 42, 'split_seed': 42, 'model': {'quality_adapter': {'enabled': True}}}
        expected = copy.deepcopy(cfg)
        expected['train']['pcgrad']['projection_strength'] = 0.5
        self.assertEqual(self.configure(cfg, ['--pcgrad-projection-strength', '0.5']), expected)

    def test_old_global_snapshot_defaults_to_one(self):
        cfg = self.configure({'train': {
            'two_output_only': True, 'pcgrad': {'enabled': True, 'mode': 'global_shared'},
        }})
        self.assertEqual(cfg['train']['pcgrad']['projection_strength'], 1.0)

    def test_old_snapshot_without_mode_remains_legacy(self):
        cfg = self.configure({'train': {'pcgrad': {'enabled': True}}})
        self.assertEqual(cfg['train']['pcgrad'], {'enabled': True, 'mode': 'per_parameter'})

    def test_yaml_strength_preserved(self):
        cfg = {'train': {'two_output_only': True, 'pcgrad': {
            'enabled': True, 'mode': 'global_shared', 'projection_strength': 0.5,
        }}}
        self.assertEqual(self.configure(cfg), cfg)

    def test_zero_strength_not_treated_as_missing(self):
        cfg = self.configure(flags=['--pcgrad-projection-strength', '0'])
        self.assertEqual(cfg['train']['pcgrad']['projection_strength'], 0.0)

    def test_invalid_strength_rejected_before_construction(self):
        for strength in ['-0.01', '1.01', 'nan', 'inf']:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                self.configure(flags=['--pcgrad-projection-strength', strength])

    def test_legacy_mode_rejects_strength(self):
        with self.assertRaisesRegex(ValueError, 'only by global_shared'):
            self.configure(flags=['--pcgrad-mode', 'per_parameter', '--pcgrad-projection-strength', '0.5'])

    def test_global_mode_still_rejects_final_supervision(self):
        for train in [
            {'two_output_only': False},
            {'two_output_only': True, 'final_auxiliary_enabled': True},
            {'two_output_only': True, 'lambda_fusion_gate': 0.1},
        ]:
            with self.subTest(train=train), self.assertRaisesRegex(ValueError, 'two-output'):
                self.configure({'train': train}, ['--pcgrad-projection-strength', '0.5'])


if __name__ == '__main__':
    unittest.main()
