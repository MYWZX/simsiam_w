#!/usr/bin/env python3
"""Read-only split/identity audit from an effective config snapshot; no models."""
import argparse
import hashlib
import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import build_dataloaders
from utils.train_utils import load_yaml


def identity(dataset, index):
    if hasattr(dataset, 'df'):
        row = dataset.df.iloc[index]
        name = str(row['name'])
        if hasattr(dataset, '_resolve_image_path'):
            path = dataset._resolve_image_path(name)
            prompt = dataset._make_prompt(row)
        else:
            path = dataset._path(name)
            prompt = str(row['prompt'])
    elif hasattr(dataset, 'images'):
        path = dataset.images[index]
        prompt = dataset._prompt(index)
    else:
        raise TypeError('Audit supports the active 1K/3K/2023 datasets only')
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path), ' '.join(str(prompt).strip().lower().split())



def raw_scores(dataset, index):
    if hasattr(dataset, 'df'):
        row = dataset.df.iloc[index]
        if 'MOS' in row:
            return {'quality': float(row['MOS'])}
        return {'quality': float(row['mos_quality']), 'align': float(row['mos_align'])}
    return {'quality': float(dataset.quality[index]), 'align': float(dataset.align[index])}


def audit(cfg):
    loaders = build_dataloaders(cfg)
    records, image_sets, prompt_sets, index_sets = {}, {}, {}, {}
    for name, loader in zip(('train', 'val', 'test'), loaders):
        subset = loader.dataset
        indices = [int(i) for i in subset.indices]
        rows = [identity(subset.dataset, i) for i in indices]
        paths = [row[0] for row in rows]
        image_sets[name] = set(paths)
        prompt_sets[name] = {row[1] for row in rows if row[1]}
        index_sets[name] = set(indices)
        labels = [raw_scores(subset.dataset, i) for i in indices]
        stats = {}
        for component in (labels[0] if labels else {}):
            values = np.asarray([row[component] for row in labels], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite raw labels in {name}/{component}')
            stats[component] = dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()))
        records[name] = dict(
            n=len(indices), raw_label_stats=stats, duplicate_image_references=len(paths)-len(set(paths)),
            blank_prompts=sum(not row[1] for row in rows),
            indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
            image_paths_sha256=hashlib.sha256(json.dumps(sorted(paths)).encode()).hexdigest(),
            examples=[dict(index=i, image=p, prompt=text) for i,(p,text) in list(zip(indices,rows))[:5]],
        )
    overlap = {}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap[a+'_'+b] = dict(index_overlap=len(index_sets[a]&index_sets[b]),
                                image_overlap=len(image_sets[a]&image_sets[b]),
                                prompt_overlap=len(prompt_sets[a]&prompt_sets[b]))
        if overlap[a+'_'+b]['index_overlap'] or overlap[a+'_'+b]['image_overlap']:
            raise ValueError(f'Data leakage by repeated sample/image between {a} and {b}: {overlap[a+"_"+b]}')
    base = loaders[0].dataset.dataset
    normalization = {key: float(getattr(base,key)) for key in ('mos_max','min_mos','max_mos') if hasattr(base,key)}
    return dict(dataset_type=cfg['dataset']['type'], seed=cfg.get('split_seed',cfg.get('seed',42)), label_normalization=normalization,
                splits=records, overlap=overlap,
                note='Prompt overlap is diagnostic, not automatic leakage under image-wise splits. Paths are checked, not image contents or annotation provenance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Use the effective config_snapshot.yaml from a completed run')
    parser.add_argument('--output-json',required=True)
    args=parser.parse_args()
    result=audit(load_yaml(args.config))
    output=Path(args.output_json);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='splits'},ensure_ascii=False,indent=2))
    print('Split sizes:',{k:v['n'] for k,v in result['splits'].items()})
    print('Saved:',output)

if __name__=='__main__':main()
