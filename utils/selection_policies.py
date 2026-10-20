"""Validation-only policy comparison and immutable CPU model snapshots."""
from collections import OrderedDict
import copy

import torch


POLICIES = {'quality': 'val_quality_SROCC', 'joint': 'val_component_hmean_SROCC'}


def configure_selection_comparison(cfg, args):
    enabled = bool(getattr(args, 'compare_selection_policies', False) or
                   cfg.get('train', {}).get('compare_selection_policies', False))
    if not enabled:
        return False
    if getattr(args, 'selection_metric', None) is not None:
        raise ValueError('--compare-selection-policies cannot use --selection-metric.')
    if cfg.get('dataset', {}).get('type') in {'agiqa1k', 'agiqa1k_iqa'}:
        raise ValueError('Selection comparison requires Align labels; do not rerun 1K.')
    train = cfg.setdefault('train', {})
    train.update(compare_selection_policies=True, epochs=60, early_stopping_enabled=False,
                 selection_policies=dict(POLICIES))
    return True


def cpu_state_dict(model):
    """Copy tensors even for CPU models; preserve state_dict version metadata."""
    original = model.state_dict()
    state = OrderedDict((key, value.detach().to(device='cpu', copy=True)
                         if isinstance(value, torch.Tensor) else copy.deepcopy(value))
                        for key, value in original.items())
    if hasattr(original, '_metadata'):
        state._metadata = copy.deepcopy(original._metadata)
    return state


class SelectionComparison:
    def __init__(self):
        self.best = {}

    def update(self, model, metrics, epoch, selector):
        if float(metrics.get('val_align_N', 0)) < 2:
            raise ValueError('Selection comparison requires at least two validation Align labels.')
        # Validate both values before mutating either policy's state.
        values = {}
        for policy, key in POLICIES.items():
            value, maximize, _ = selector(metrics, {'train': {
                'selection_metric': key, 'selection_mode': 'max'}})
            assert maximize
            values[policy] = value
        improved = {name: name not in self.best or value > self.best[name]['value']
                    for name, value in values.items()}
        if any(improved.values()):
            snapshot = cpu_state_dict(model)
            for name, changed in improved.items():
                if changed:
                    self.best[name] = dict(value=values[name], epoch=epoch, state=snapshot)
        return {name: dict(selection_metric=POLICIES[name], value=self.best[name]['value'],
                           epoch=self.best[name]['epoch'], is_best=improved[name])
                for name in POLICIES}

    def evaluate_selected(self, model, evaluate_fn):
        """Test only after training; identical selected epochs are evaluated once."""
        if set(self.best) != set(POLICIES):
            raise RuntimeError('Both validation selection policies must have a best state.')
        results, cached = {}, {}
        for name, record in self.best.items():
            epoch = record['epoch']
            if epoch not in cached:
                model.load_state_dict(record['state'])
                cached[epoch] = evaluate_fn()
            val, test = cached[epoch]
            results[name] = dict(selection_policy=name, selection_metric=POLICIES[name],
                                 best_epoch=epoch, best_selection_value=record['value'],
                                 best_val=val, best_test=test or {},
                                 best_checkpoint='', last_checkpoint='')
        return results
