"""Opt-in evidence for actual downstream memberships; no pixel reads or RNG use."""
import json
from pathlib import Path
import torch
from utils.refit_protocol import split_manifest, digest
from utils.agiqa1k_subset1078 import RESPLIT_REVISION, expected_manifest


def record_split(cfg, loaders, output_dir):
    if not cfg.get('train', {}).get('record_split_manifest', False):
        return None
    actual = split_manifest(*loaders)
    counts = actual['split_sizes']; total = sum(counts.values())
    seed = int(cfg.get('split_seed', cfg.get('seed', 42)))
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    n, v = counts['train'], counts['val']
    expected = dict(train=order[:n],val=order[n:n+v],test=order[n+v:])
    if actual['indices'] != expected:
        raise ValueError('Actual loaders do not use the requested random split')
    if cfg['dataset'].get('data_revision') == RESPLIT_REVISION:
        reference = expected_manifest()['split_indices']
    else:
        baseline = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
        reference = dict(train=baseline[:n],val=baseline[n:n+v],test=baseline[n+v:])
    if seed != 42 and actual['indices'] == reference:
        raise ValueError('New split unexpectedly identical to seed42')
    actual.update(train_seed=int(cfg['seed']), split_seed=seed,
                  reference_seed42_indices_sha256=digest(reference),
                  split_indices_sha256=digest(actual['indices']),
                  test_overlap_with_seed42=len(set(actual['indices']['test']) & set(reference['test'])))
    path = Path(output_dir)/'data_split_manifest.json'
    with path.open('x') as f:json.dump(actual,f,indent=2)
    return actual
