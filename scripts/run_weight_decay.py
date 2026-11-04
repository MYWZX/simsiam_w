"""Downstream weight-decay control; fixed accepted source, no pretraining launch."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_projection_only_v1_baseline as base

VARIANTS = {'wd005': .005, 'wd010': .01}
DATASETS = ('agiqa1k', 'agiqa3k', 'aigciqa2023')
VERSION = 'projection_only_v1_weight_decay'


def config(variant, ds):
    actual = yaml.safe_load((ROOT/f'configs/weight_decay_sweep/{variant}/{ds}.yaml').read_text())
    expected = yaml.safe_load((ROOT/f'configs/projection_only_v1_baseline/{ds}.yaml').read_text())
    for k in ('output_dir', 'results_csv', 'run_id'):
        expected[k] = expected[k].replace('projection_only_v1_baseline_ctx5', f'projection_only_v1_{variant}_ctx5')
    expected['train']['weight_decay'] = VARIANTS[variant]
    if actual != expected:
        raise ValueError('Only weight_decay and output identifiers may differ from baseline')
    return actual


def source():
    observed = base.completed('pretrain')
    pinned = base.read_json(ROOT/'docs/current_baseline.json')['source']
    if any(observed[k] != pinned[k] for k in ('version', 'sha256', 'config_sha256', 'selected_epoch', 'completion_sha256')):
        raise ValueError('Source is not the accepted best baseline')
    return observed


def paths(variant, ds):
    c = config(variant, ds)
    return ROOT/c['output_dir'], ROOT/c['results_csv']


def verify(variant, ds, fingerprint):
    c = config(variant, ds);d, table = paths(variant, ds)
    launch = base.read_json(d/'launch_manifest.json')
    done = base.read_json(d/'completion_manifest.json')
    if (launch['version'] != VERSION or launch['variant'] != variant or launch['dataset'] != ds
            or launch['config_sha256'] != base.digest(ROOT/f'configs/weight_decay_sweep/{variant}/{ds}.yaml')
            or launch['source'] != fingerprint):
        raise ValueError('Run provenance mismatch')
    for name, digest in done['files_sha256'].items():
        if base.digest(d/name) != digest:raise ValueError(f'Artifact changed: {name}')
    if base.digest(table) != done['csv_sha256']:raise ValueError('CSV changed')
    base.require_subset(c, yaml.safe_load((d/'config_snapshot.yaml').read_text()))
    if base.read_json(d/'source_fingerprint.json') != fingerprint:raise ValueError('Source changed')
    metadata = base.read_json(d/'run_metadata.json')
    if not metadata.get('checkpoint_verified_after_load') or metadata['checkpoint']['sha256'] != fingerprint['sha256']:
        raise ValueError('Source load was not verified')
    if metadata.get('qa_alpha_initialization') is not None:
        raise ValueError('QA alpha must inherit the accepted source unchanged')
    csv.field_size_limit(50_000_000)
    with table.open(encoding='utf-8-sig', newline='') as f:rows=list(csv.DictReader(f))
    finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
    if (len(finals)!=1 or finals[0]['selection_metric']!='val_quality_SROCC'
            or finals[0]['last_epoch']!=60 or any(r['event']=='resume' for r in rows)):
        raise ValueError('Expected completed original-split 60-epoch quality-SROCC run')
    events=[json.loads(r['metrics_json']) for r in rows if r['event']=='post_load_qa_alpha']
    if events:raise ValueError('Unexpected QA alpha override')


def run(variant, ds, dry_run=False):
    cfg=config(variant, ds);fingerprint=source();d,table=paths(variant, ds)
    if d.exists() or table.exists():raise FileExistsError(f'Preserve existing output: {d}')
    command=[sys.executable, 'train_cm_iqa.py', '--config', str(ROOT/f'configs/weight_decay_sweep/{variant}/{ds}.yaml')]
    if dry_run:
        print(json.dumps(dict(command=command, source=fingerprint), indent=2));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; cloud CPU training refused')
    if ds=='agiqa1k':subprocess.run([sys.executable,'scripts/prepare_agiqa1k_subset1078.py'],cwd=ROOT,check=True)
    d.mkdir(parents=True,exist_ok=False)
    base.write_new(d/'source_fingerprint.json',fingerprint)
    base.write_new(d/'launch_manifest.json',dict(version=VERSION,variant=variant,dataset=ds,command=command,
        config_sha256=base.digest(ROOT/f'configs/weight_decay_sweep/{variant}/{ds}.yaml'),source=fingerprint))
    with (d/'console.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        if process.wait()!=0:raise RuntimeError('Run failed; partial output preserved')
    files=('config_snapshot.yaml','run_metadata.json','launch_manifest.json','source_fingerprint.json','console.log')
    base.write_new(d/'completion_manifest.json',dict(files_sha256={p:base.digest(d/p) for p in files},csv_sha256=base.digest(table)))
    verify(variant,ds,fingerprint)


def pack():
    fingerprint=source();files=set((ROOT/'configs/weight_decay_sweep').glob('*/*.yaml'))
    files.update(ROOT/'docs'/name for name in ('current_baseline.json','current_baseline.csv'))
    for variant in VARIANTS:
        for ds in DATASETS:
            verify(variant,ds,fingerprint);d,table=paths(variant,ds);files.add(table)
            for name in ('config_snapshot.yaml','run_metadata.json','launch_manifest.json','completion_manifest.json','source_fingerprint.json','console.log'):
                files.add(d/name)
    # Include the accepted control's original results and provenance, never rerun it.
    for stage in base.STAGES:
        base.completed(stage);d,table=base.paths(stage);files.add(table)
        for name in ('config_snapshot.yaml','run_metadata.json','launch_manifest.json','completion_manifest.json','source_fingerprint.json','console.log'):
            if (d/name).is_file():files.add(d/name)
    archive=ROOT/f'{VERSION}_results.tar.gz'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        data=json.dumps(fingerprint,indent=2).encode();info=tarfile.TarInfo('source_fingerprint.json');info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    print(archive)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=tuple(VARIANTS)+('pack','check-configs'))
    p.add_argument('dataset',nargs='?',choices=DATASETS)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT)
    if a.variant in VARIANTS:
        if a.dataset is None:p.error('Specify one dataset; no implicit all-dataset launch')
        run(a.variant,a.dataset,a.dry_run)
    else:
        if a.dataset or a.dry_run:p.error('pack/check-configs take no dataset or --dry-run')
        if a.variant=='pack':pack()
        else:
            for v in VARIANTS:
                for ds in DATASETS:config(v,ds)
            print('All six configs match the baseline except weight_decay and output identifiers')

if __name__=='__main__':main()
