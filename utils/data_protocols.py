"""Opt-in split-local RNG and training-only 1K label scaling."""
import random
import numpy as np
import torch


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader_rng_kwargs(cfg, split):
    if not cfg.get('loader', {}).get('isolate_rng', False):
        return {}
    offset = {'train': 0, 'val': 1, 'test': 2}[split]
    return {'generator': torch.Generator().manual_seed(int(cfg.get('seed',42))+offset),
            'worker_init_fn': seed_worker}


def fit_1k_train_scaler(train_dataset, eval_dataset, indices, cfg, fit_source='train_indices'):
    if cfg.get('dataset',{}).get('mos_normalization','legacy_full_data') != 'train_minmax':
        return
    scores = train_dataset.df.iloc[list(indices)]['MOS'].to_numpy(dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Training MOS must be nonempty and finite')
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        raise ValueError('Training MOS needs at least two different values for train_minmax')
    for ds in (train_dataset, eval_dataset):
        ds.min_mos, ds.max_mos = lo, hi
    cfg['dataset']['fitted_mos_normalization'] = dict(mode='train_minmax', min=lo, max=hi,
        fit_count=len(scores), fit_source=fit_source, clip_to_unit_interval=True)
