"""Verify CLI/config and real optimizer group isolation across all datasets."""
import copy
from pathlib import Path
from unittest.mock import patch
import unittest
from train_cm_iqa import parse_args, configure_visual_backbone_lr, build_optimizer
from utils.train_utils import load_yaml
from test_quality_adapter_lr import TinyIQA

ROOT = Path(__file__).resolve().parents[1]

class VisualBackboneLRTests(unittest.TestCase):
    def args(self, *flags):
        with patch('sys.argv', ['train_cm_iqa.py', '--config', 'unused.yaml', *flags]):
            return parse_args()

    def test_default_and_validation_leave_config_unchanged(self):
        cfg = {'train': {'lr': 1e-4, 'layerwise_lr': {'enabled': True}}}
        before = copy.deepcopy(cfg)
        configure_visual_backbone_lr(cfg, self.args())
        self.assertEqual(cfg, before)
        for value in ['0', '-1', 'nan', 'inf']:
            with self.assertRaises(ValueError):
                configure_visual_backbone_lr(cfg, self.args('--visual-backbone-lr-mult', value))
            self.assertEqual(cfg, before)
        cfg['train']['layerwise_lr']['enabled'] = False
        before = copy.deepcopy(cfg)
        with self.assertRaisesRegex(ValueError, 'layerwise_lr.enabled=true'):
            configure_visual_backbone_lr(cfg, self.args('--visual-backbone-lr-mult', '.01'))
        self.assertEqual(cfg, before)

    def test_only_visual_groups_change_for_all_datasets(self):
        for ds in ['agiqa1k', 'agiqa3k', 'aigciqa2023']:
            with self.subTest(dataset=ds):
                cfg = load_yaml(str(ROOT / 'configs' / f'{ds}_cm_iqa_decoupled_quality_align.yaml'))
                before_cfg = copy.deepcopy(cfg)
                model = TinyIQA()
                before = {g['name']: g for g in build_optimizer(cfg, model).param_groups}
                configure_visual_backbone_lr(cfg, self.args('--visual-backbone-lr-mult', '.01'))
                expected = copy.deepcopy(before_cfg)
                expected['train']['layerwise_lr'].update(global_encoder_mult=.01, local_encoder_mult=.01)
                self.assertEqual(cfg, expected)
                after = {g['name']: g for g in build_optimizer(cfg, model).param_groups}
                self.assertEqual(before.keys(), after.keys())
                changed = [n for n in before if before[n]['lr'] != after[n]['lr']]
                self.assertEqual(changed, ['global_encoder', 'local_encoder'])
                for n in before:
                    self.assertEqual([id(p) for p in before[n]['params']], [id(p) for p in after[n]['params']])
                    self.assertEqual(before[n]['weight_decay'], after[n]['weight_decay'])
                for n in changed:
                    self.assertAlmostEqual(before[n]['lr'], 2e-6)
                    self.assertAlmostEqual(after[n]['lr'], 1e-6)
                self.assertAlmostEqual(after['quality_adapter']['lr'], 1e-4)

if __name__ == '__main__':
    unittest.main()
