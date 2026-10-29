"""Pinned independent 1078-image revision, preserving unaffected split members."""
import hashlib
import json
from pathlib import Path
import torch

SOURCE_CSV_SHA256 = '40c19836a31a3d07dd3ad7f61824962fc259c23fd3ccb813eb2c0aa99fba8bbd'
SOURCE_XLSX_SHA256 = 'f958adab2712d6ea6c2b0777f1ca892f3d35d3956a893e78cf6090309e6a8ea3'
SUBSET_CSV_SHA256 = '60e7d83a36bf5daa545d05ef979618d72821fcb3bd9e5e87874fe2563b22fde3'
REVISION = 'agiqa1k_subset1078_preserve_members_v1'
RELATIVE_DIR = 'revisions/subset1078_v1'
CSV_NAME = 'AIGC-1K_answer_unambiguous1078.csv'


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_manifest():
    # Only the confirmed duplicate's two records are excluded. Keep every
    # remaining sample in its original seed42 split, rather than reshuffling.
    keep = list(range(1078))
    mapping = {old: new for new, old in enumerate(keep)}
    order = torch.randperm(1080, generator=torch.Generator().manual_seed(42)).tolist()
    original = {'train': order[:756], 'val': order[756:864], 'test': order[864:]}
    splits = {name: [mapping[i] for i in indices if i in mapping] for name,indices in original.items()}
    return {'revision': REVISION, 'source_csv_sha256': SOURCE_CSV_SHA256,
            'source_xlsx_sha256': SOURCE_XLSX_SHA256, 'subset_csv_sha256': SUBSET_CSV_SHA256,
            'source_rows': 1080, 'subset_rows': 1078, 'split_seed': 42,
            'removed_original_row_indices': [1078,1079],
            'removed_name': 'dreamStudio_8_9_1_1.jpg',
            'kept_original_row_indices': keep, 'split_indices': splits,
            'split_sizes': {k:len(v) for k,v in splits.items()},
            'policy': 'preserve original split membership except removal of the conflicting image'}


def read_revision_split(cfg, total):
    dataset = cfg['dataset']
    if dataset.get('data_revision') != REVISION or dataset.get('type') != 'agiqa1k_iqa':
        raise ValueError('Unsupported explicit subset split revision')
    if total != 1078 or int(cfg.get('split_seed', cfg.get('seed',42))) != 42:
        raise ValueError('Subset1078 requires 1078 records and split_seed=42')
    if [float(dataset.get(k,v)) for k,v in [('train_ratio',.7),('val_ratio',.1),('test_ratio',.2)]] != [.7,.1,.2]:
        raise ValueError('Subset1078 preserves the original 70/10/20 split before exclusion')
    root = Path(dataset['root_dir'])
    manifest = json.loads((root/dataset['split_manifest_file']).read_text())
    expected = expected_manifest()
    if manifest != expected:
        raise ValueError('Subset1078 split manifest differs from the pinned original membership')
    if file_sha256(root/dataset['csv_file']) != SUBSET_CSV_SHA256:
        raise ValueError('Subset1078 CSV changed; do not continue with a different annotation revision')
    dataset['actual_split_sizes'] = expected['split_sizes']
    return tuple(expected['split_indices'][name] for name in ('train','val','test'))
