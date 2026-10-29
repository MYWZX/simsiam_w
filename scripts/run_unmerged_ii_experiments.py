"""Unmerged downstream: reuse verified completed g baseline, train new z I-I."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import yaml
from utils.refit_protocol import canonical_cfg
from utils.run_metadata import checkpoint_fingerprint
from utils.agiqa1k_subset1078 import REVISION, read_revision_split

ROOT=Path(__file__).resolve().parents[1]
G_CONFIGS={
    'agiqa1k':'configs/agiqa1k_subset1078/both.yaml',
    'agiqa3k':'configs/protocol_v2/agiqa3k_both_noce_noneg.yaml',
    'aigciqa2023':'configs/protocol_v2/aigciqa2023_both_noce_noneg.yaml',
}
Z_PRETRAIN='configs/zii_noce_noneg/pretrain_dev.yaml'


def comparable_cfg(cfg):
    result=canonical_cfg(cfg)
    # These are populated by the loaders, not scientific hyperparameters.
    result.get('dataset',{}).pop('actual_split_sizes',None)
    result.get('dataset',{}).pop('pretrain_development_manifest',None)
    return result


def verify_completed(cfg, entry):
    output=Path(cfg['output_dir']);csv_path=Path(cfg['results_csv'])
    snapshot=output/'config_snapshot.yaml'
    if not csv_path.is_file() or not snapshot.is_file():
        raise RuntimeError(f'Incomplete output exists: {output}. Preserve it; no automatic overwrite.')
    actual=yaml.safe_load(snapshot.read_text())
    if (actual.get('train',{}).get('refit_plan') or
            actual.get('dataset',{}).get('refit_split_manifest')):
        raise ValueError('Merged refit snapshot cannot be reused as an unmerged result')
    if comparable_cfg(actual)!=comparable_cfg(cfg):
        raise ValueError(f'Completed run config differs: {snapshot}')
    csv.field_size_limit(20_000_000)
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    if sum(r['event']=='run_start' for r in rows)!=1 or sum(r['event']=='final' for r in rows)!=1:
        raise RuntimeError(f'Run is incomplete or appended: {csv_path}')
    if any(r['event'] in {'refit_start','refit_epoch'} for r in rows):
        raise ValueError('Refit output cannot be reused as an unmerged development result')
    epochs=[int(r['epoch']) for r in rows if r['event']=='epoch']
    if not epochs or epochs!=list(range(1,len(epochs)+1)):
        raise ValueError('Completed epoch sequence is invalid')
    final=json.loads(next(r['metrics_json'] for r in rows if r['event']=='final'))
    if entry=='train_cm_iqa.py':
        if len(epochs)!=int(cfg['train']['epochs']):
            raise ValueError('Expected the full configured downstream development trajectory')
        metadata=json.loads((output/'run_metadata.json').read_text())
        if not metadata.get('checkpoint_verified_after_load'):
            raise ValueError('Existing source was not verified after load')
        if metadata.get('checkpoint')!=checkpoint_fingerprint(cfg['train']['pretrained_checkpoint']):
            raise ValueError('Pretraining source differs from completed downstream result')
        if 'best_test' not in final or 'best_val' not in final:
            raise ValueError('Expected validation-selected unmerged result')
    else:
        checkpoint=output/'best_component_anchor_pretrain.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Completed z pretraining checkpoint missing: {checkpoint}')
        if cfg['model'].get('image_image_feature')!='z':
            raise ValueError('Expected fused-z I-I source')
    return final


def run_stage(config_path, entry, dry_run=False):
    cfg=yaml.safe_load(Path(config_path).read_text())
    cmd=[sys.executable,entry,'--config',config_path]
    if dry_run:
        print('RUN OR REUSE: '+shlex.join(cmd));return
    if cfg['dataset'].get('data_revision')==REVISION:
        read_revision_split(copy.deepcopy(cfg),1078)
    if Path(cfg['output_dir']).exists() or Path(cfg['results_csv']).exists():
        verify_completed(cfg,entry)
        print(f'REUSE completed unmerged result: {cfg["results_csv"]}',flush=True)
        return
    source=cfg['train'].get('pretrained_checkpoint')
    if entry=='train_cm_iqa.py' and (not source or not Path(source).is_file()):
        raise FileNotFoundError(f'Pretraining source missing: {source}')
    print('RUN: '+shlex.join(cmd),flush=True)
    subprocess.run(cmd,check=True,cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1'})
    verify_completed(cfg,entry)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',default='all',choices=['g','z','z-downstream','pretrain-z','all'])
    parser.add_argument('dataset',nargs='?',default='all',choices=['all','agiqa1k','agiqa3k','aigciqa2023'])
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    datasets=['agiqa1k','agiqa3k','aigciqa2023'] if args.dataset=='all' else [args.dataset]
    if args.mode=='pretrain-z' and args.dataset!='all':
        parser.error('Pretraining is shared across datasets; do not select a downstream dataset')
    if 'agiqa1k' in datasets and args.mode!='pretrain-z':
        prep=[sys.executable,'scripts/prepare_agiqa1k_subset1078.py']
        if args.dry_run:print('PREPARE/VERIFY: '+shlex.join(prep))
        else:subprocess.run(prep,check=True,cwd=ROOT)
    if args.mode in {'g','all'}:
        for dataset in datasets:run_stage(G_CONFIGS[dataset],'train_cm_iqa.py',args.dry_run)
    if args.mode in {'z','pretrain-z','all'}:
        run_stage(Z_PRETRAIN,'train_cm_pretrain.py',args.dry_run)
    if args.mode in {'z','z-downstream','all'}:
        for dataset in datasets:
            run_stage(f'configs/zii_noce_noneg/{dataset}_unmerged.yaml','train_cm_iqa.py',args.dry_run)
    print('Unmerged-only workflow complete: no train+val refit or budget-plan stage.',flush=True)

if __name__=='__main__':main()
