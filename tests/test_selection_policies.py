import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack, redirect_stdout

import torch
from torch import nn

import train_cm_iqa as training
from utils.selection_policies import (SelectionComparison, cpu_state_dict,
                                     configure_selection_comparison)
from utils.run_metadata import checkpoint_fingerprint, collect_run_metadata


def metrics(q, a):
    return {'val_quality_SROCC': q, 'val_align_SROCC': a, 'val_align_N': 4}


class SelectionPolicyTests(unittest.TestCase):
    def test_cpu_snapshot_has_independent_parameters_buffers_and_metadata(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        state = cpu_state_dict(model)
        baseline = copy.deepcopy(state)
        for key, value in model.state_dict().items():
            self.assertNotEqual(value.data_ptr(), state[key].data_ptr())
            self.assertEqual(state[key].device.type, 'cpu')
            value.add_(1)
        for key in state:
            torch.testing.assert_close(state[key], baseline[key])
        self.assertEqual(state._metadata, model.state_dict()._metadata)
        self.assertIsNot(state._metadata, model.state_dict()._metadata)
        model.load_state_dict(state)

    def test_different_policy_epochs_ties_and_test_is_not_selection(self):
        model = nn.Linear(1, 1, bias=False)
        tracker = SelectionComparison()
        with torch.no_grad():
            model.weight.fill_(1)
        tracker.update(model, metrics(.8, .8), 1, training.get_selection_value)
        self.assertIs(tracker.best['quality']['state'], tracker.best['joint']['state'])
        with torch.no_grad():
            model.weight.fill_(2)
        tracker.update(model, metrics(.9, .2), 2, training.get_selection_value)
        tracker.update(model, metrics(.9, .2), 3, training.get_selection_value)
        self.assertEqual(tracker.best['quality']['epoch'], 2)
        self.assertEqual(tracker.best['joint']['epoch'], 1)
        seen = []
        def evaluate():
            seen.append(model.weight.item())
            return {}, {'test_quality_SROCC': -model.weight.item()}
        result = tracker.evaluate_selected(model, evaluate)
        self.assertEqual(seen, [2., 1.])
        self.assertEqual(result['quality']['best_epoch'], 2)
        self.assertEqual(result['joint']['best_epoch'], 1)

    def test_shared_epoch_evaluated_once_and_invalid_metrics_fail_before_update(self):
        tracker = SelectionComparison()
        model = nn.Linear(1, 1)
        for bad in [metrics(.8, float('nan')), {**metrics(.8, .8), 'val_align_N': 0}]:
            with self.assertRaises(ValueError):
                tracker.update(model, bad, 1, training.get_selection_value)
            self.assertFalse(tracker.best)
        tracker.update(model, metrics(.8, .8), 1, training.get_selection_value)
        with patch.object(model, 'load_state_dict', wraps=model.load_state_dict) as load:
            with patch('builtins.print'):
                tracker.evaluate_selected(model, lambda: ({'val': 1}, {'test': 1}))
            self.assertEqual(load.call_count, 1)

    def test_config_override_and_1k_rejection(self):
        cfg = {'train': {'epochs': 2, 'early_stop_patience': 1}}
        args = SimpleNamespace(compare_selection_policies=True, selection_metric=None)
        self.assertTrue(configure_selection_comparison(cfg, args))
        self.assertEqual(cfg['train']['epochs'], 60)
        self.assertFalse(cfg['train']['early_stopping_enabled'])
        with self.assertRaises(ValueError):
            configure_selection_comparison({'dataset': {'type': 'agiqa1k_iqa'}}, args)
        with self.assertRaises(ValueError):
            configure_selection_comparison({}, SimpleNamespace(compare_selection_policies=True, selection_metric='val_quality_SROCC'))

    def test_checkpoint_digest_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.pth'
            path.write_bytes(b'checkpoint')
            fingerprint = checkpoint_fingerprint(path)
            self.assertEqual(fingerprint['sha256'], hashlib.sha256(b'checkpoint').hexdigest())
            path.write_bytes(b'other')
            self.assertNotEqual(fingerprint, checkpoint_fingerprint(path))
        meta = collect_run_metadata(None, {'seed': 42, 'split_seed': 42})
        json.dumps(meta)
        self.assertEqual(meta['pytorch'], str(torch.__version__))
        self.assertEqual(meta['deterministic_algorithms'], torch.are_deterministic_algorithms_enabled())

    def run_main(self, compare):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            cfg = dict(seed=42, dataset={'type': 'agiqa3k_simsiam'}, model={},
                       train=dict(epochs=4, early_stop_patience=1, amp=False,
                                  selection_metric='val_quality_SROCC', selection_mode='max'))
            argv = ['train_cm_iqa.py', '--config', 'unused.yaml', '--output-dir', tmp,
                    '--results-csv', str(Path(tmp)/'results.csv')]
            if compare:
                argv.append('--compare-selection-policies')
            stack.enter_context(patch('sys.argv', argv))
            stack.enter_context(patch.object(training, 'load_yaml', return_value=cfg))
            stack.enter_context(patch.object(training, 'build_dataloaders', return_value=('train', 'val', 'test')))
            model = nn.Linear(1, 1, bias=False)
            stack.enter_context(patch.object(training, 'build_iqa_model', return_value=model))
            stack.enter_context(patch.object(training, 'build_optimizer', side_effect=lambda cfg, m: torch.optim.SGD(m.parameters(), lr=.1)))
            stack.enter_context(patch.object(training, 'apply_progressive_finetune_stage', return_value=('all', {'all': 1})))
            stack.enter_context(patch.object(training, 'format_metrics', return_value='metrics'))
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            def train(m, loader, optimizer, *args, **kwargs):
                optimizer.zero_grad()
                m(torch.ones(1, 1)).sum().backward()
                optimizer.step()
                return {'loss': 1.}
            train_mock = stack.enter_context(patch.object(training, 'train_one_epoch', side_effect=train))
            calls = []
            def evaluate(m, loader, device, cfg, split):
                epoch = train_mock.call_count
                calls.append((split, epoch))
                if split == 'test':
                    self.assertEqual(epoch, 60 if compare else 2)
                    return {'test_quality_SROCC': .5}
                # Quality picks epoch2; joint picks epoch1. Long plateau must
                # not trigger the configured patience=1 in comparison mode.
                return metrics(.8, .8) if epoch == 1 else metrics(.9, .1)
            stack.enter_context(patch.object(training, 'evaluate', side_effect=evaluate))
            disk_save = stack.enter_context(patch('torch.save', side_effect=AssertionError('No disk weights')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            training.main()
            with (Path(tmp)/'results.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            parsed = [(r['event'], json.loads(r['metrics_json'])) for r in rows]
            disk_save.assert_not_called()
            self.assertEqual(len(list(Path(tmp).glob('*.pth'))), 0)
            self.assertEqual(train_mock.call_count, 60 if compare else 2)
            if compare:
                self.assertEqual(calls[:60], [('val', i) for i in range(1, 61)])
                self.assertEqual(len(calls), 64)
                self.assertNotIn('early_stop', [e for e, _ in parsed])
                self.assertEqual([e for e, _ in parsed][-3:], ['quality_selected', 'joint_selected', 'final'])
                final = parsed[-1][1]
                self.assertEqual(final['selection_policies']['quality']['best_epoch'], 2)
                self.assertEqual(final['selection_policies']['joint']['best_epoch'], 1)
                self.assertEqual(final['last_epoch'], 60)
                self.assertTrue(final['comparison_complete'])
                self.assertTrue((Path(tmp)/'run_metadata.json').is_file())
                from yaml import safe_load
                self.assertEqual(safe_load((Path(tmp)/'config_snapshot.yaml').read_text())['train']['epochs'], 60)
            else:
                self.assertIn('early_stop', [e for e, _ in parsed])
                self.assertIn('best_test', [e for e, _ in parsed])
                self.assertIn('best_test', parsed[-1][1])
                self.assertNotIn('selection_policies', parsed[-1][1])

    def test_main_comparison_single_trajectory_and_csv(self):
        self.run_main(True)

    def test_main_ordinary_early_stop_and_csv_unchanged(self):
        # Use a plateau starting at epoch1 for this ordinary-mode check.
        with patch.object(training, 'get_selection_value', side_effect=lambda m, cfg: (0.5, True, 'val_quality_SROCC')):
            self.run_main(False)


if __name__ == '__main__':
    unittest.main()
