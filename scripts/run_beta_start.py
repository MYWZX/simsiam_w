"""Learnable quality score beta starts with explicitly selected 1K split."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_projection_only_v1_baseline as base
from scripts import run_baseline_splitcheck as splits
VARIANTS={'beta025':.25,'beta075':.75,'control':.5}
DATASETS=('agiqa1k','agiqa3k','aigciqa2023')
VERSION='projection_only_v1_beta_start'
FILES=('launch_config.yaml','config_snapshot.yaml','console.log','launch_manifest.json',
       'completion_manifest.json','run_metadata.json','source_fingerprint.json','data_split_manifest.json')


def config(variant,ds,split=None):
    if ds=='agiqa1k':
        if split not in (3047,3407):raise ValueError('Explicitly select 1K --split-seed 3047 or 3407')
    elif split is not None:raise ValueError('3K/2023 split is fixed at42; omit --split-seed')
    seed=split if ds=='agiqa1k' else 42
    if variant=='control' and (ds!='agiqa1k' or seed!=3047):
        raise ValueError('Only new 1K split3047 needs a new control; reuse completed controls elsewhere')
    c=yaml.safe_load((ROOT/f'configs/projection_only_v1_baseline/{ds}.yaml').read_text())
    for k in ('output_dir','results_csv','run_id'):
        c[k]=c[k].replace('_seed42',f'_train42_split{seed}_{variant}')
    c['split_seed']=seed
    beta=VARIANTS[variant]
    c['model']['nss_text_gate']['quality_mix_init']=math.log(beta/(1-beta))
    c['train'].update(record_split_manifest=True,record_quality_mix_initialization=True)
    if ds=='agiqa1k':
        c['dataset'].update(data_revision='agiqa1k_subset1078_resplit_v1',
            split_manifest_file=f'revisions/subset1078_resplit_v1/split_seed{seed}.json')
    return c


def paths(v,ds,split=None):
    c=config(v,ds,split);return ROOT/c['output_dir'],ROOT/c['results_csv']


def source():return splits.source()


def verify(v,ds,split,fingerprint):
    c=config(v,ds,split);d,table=paths(v,ds,split)
    launch=base.read_json(d/'launch_manifest.json');done=base.read_json(d/'completion_manifest.json')
    if launch != dict(version=VERSION,variant=v,dataset=ds,split_seed=c['split_seed'],source=fingerprint,
                      config_sha256=base.digest(d/'launch_config.yaml')):raise ValueError('Launch provenance mismatch')
    for name,sha in done['files_sha256'].items():
        if base.digest(d/name)!=sha:raise ValueError(f'Artifact changed: {name}')
    if base.digest(table)!=done['csv_sha256']:raise ValueError('CSV changed')
    if yaml.safe_load((d/'launch_config.yaml').read_text())!=c:raise ValueError('Unexpected configuration changes')
    base.require_subset(c,yaml.safe_load((d/'config_snapshot.yaml').read_text()))
    if base.read_json(d/'source_fingerprint.json')!=fingerprint:raise ValueError('Wrong source')
    m=base.read_json(d/'run_metadata.json')
    if (m['seed']!=42 or m['split_seed']!=c['split_seed'] or not m.get('checkpoint_verified_after_load')
            or m['checkpoint']['sha256']!=fingerprint['sha256'] or m.get('qa_alpha_initialization')):raise ValueError('Runtime source/seed/alpha mismatch')
    beta=m['quality_mix_initialization']
    if not beta['trainable'] or not math.isclose(beta['applied_beta'],VARIANTS[v],abs_tol=1e-7,rel_tol=0):raise ValueError('Wrong beta initialization')
    evidence=base.read_json(d/'data_split_manifest.json')
    if evidence['train_seed']!=42 or evidence['split_seed']!=c['split_seed']:raise ValueError('Wrong actual split')
    from utils.refit_protocol import digest
    import torch
    sizes=evidence['split_sizes'];n,va=sizes['train'],sizes['val']
    order=torch.randperm(sum(sizes.values()),generator=torch.Generator().manual_seed(c['split_seed'])).tolist()
    if evidence['indices']!=dict(train=order[:n],val=order[n:n+va],test=order[n+va:]) or evidence['split_indices_sha256']!=digest(evidence['indices']):raise ValueError('Wrong actual memberships')
    if m['data_split']!={k:v for k,v in evidence.items() if k!='indices'}:raise ValueError('Split metadata mismatch')
    csv.field_size_limit(50_000_000)
    with table.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
    if len(finals)!=1 or finals[0]['last_epoch']!=60 or finals[0]['selection_metric']!='val_quality_SROCC' or any(r['event']=='resume' for r in rows):raise ValueError('Expected complete 60-epoch validation-quality selection')
    records=[json.loads(r['metrics_json']) for r in rows if r['event']=='quality_mix_initialization']
    if records!=[beta]:raise ValueError('Missing beta event')


def control(ds,split,fingerprint):
    if ds!='agiqa1k':
        base.completed(ds);return base.paths(ds)
    if split==3407:
        splits.verify('split3407',ds,fingerprint);return splits.paths('split3407',ds)
    verify('control',ds,split,fingerprint);return paths('control',ds,split)


def run(v,ds,split=None,dry_run=False):
    c=config(v,ds,split);fingerprint=source();d,table=paths(v,ds,split)
    if v!='control':control(ds,split,fingerprint)
    if d.exists() or table.exists():raise FileExistsError(f'Preserve existing outputs: {d}')
    command=[sys.executable,'train_cm_iqa.py','--config',str(d/'launch_config.yaml')]
    if dry_run:print(json.dumps(dict(command=command,config=c,source=fingerprint),indent=2));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; cloud CPU training refused')
    if ds=='agiqa1k':
        subprocess.run([sys.executable,'scripts/prepare_agiqa1k_subset1078.py'],cwd=ROOT,check=True)
        from utils.agiqa1k_subset1078 import prepare_resplit_manifest
        prepare_resplit_manifest(ROOT/c['dataset']['root_dir'],split)
    d.mkdir(parents=True,exist_ok=False)
    with (d/'launch_config.yaml').open('x') as f:yaml.safe_dump(c,f,sort_keys=False)
    base.write_new(d/'source_fingerprint.json',fingerprint)
    base.write_new(d/'launch_manifest.json',dict(version=VERSION,variant=v,dataset=ds,split_seed=c['split_seed'],source=fingerprint,config_sha256=base.digest(d/'launch_config.yaml')))
    with (d/'console.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        if process.wait()!=0:raise RuntimeError('Trainer failed; partial outputs preserved')
    base.write_new(d/'completion_manifest.json',dict(files_sha256={n:base.digest(d/n) for n in FILES if n!='completion_manifest.json'},csv_sha256=base.digest(table)))
    verify(v,ds,split,fingerprint)


def pack(split):
    if split not in (3047,3407):raise ValueError('Explicitly select 1K split for the archive')
    fingerprint=source();files={ROOT/'docs/current_baseline.json',ROOT/'docs/beta_start.md'}
    for ds in DATASETS:
        selected=split if ds=='agiqa1k' else None
        cd,ct=control(ds,selected,fingerprint);files.add(ct)
        files.update(cd/n for n in FILES if (cd/n).is_file())
        for v in ('beta025','beta075'):
            verify(v,ds,selected,fingerprint);d,table=paths(v,ds,selected);files.add(table);files.update(d/n for n in FILES)
            if (cd/'data_split_manifest.json').exists():
                a=base.read_json(cd/'data_split_manifest.json');b=base.read_json(d/'data_split_manifest.json')
                if any(a[k]!=b[k] for k in ('dataset_ids_sha256','indices')):raise ValueError('Candidate/control memberships differ')
    archive=ROOT/f'{VERSION}_1ksplit{split}_results.tar.gz'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        data=json.dumps(fingerprint,indent=2).encode();info=tarfile.TarInfo('source_fingerprint.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    print(archive)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('variant',choices=tuple(VARIANTS)+('pack','check-configs'))
    p.add_argument('dataset',nargs='?',choices=DATASETS);p.add_argument('--split-seed',type=int,choices=(3047,3407));p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT)
    if a.variant=='check-configs':
        if a.dataset or a.split_seed or a.dry_run:p.error('check-configs takes no other options')
        for s in (3047,3407):
            for v in ('beta025','beta075'):config(v,'agiqa1k',s)
        for ds in DATASETS[1:]:
            for v in ('beta025','beta075'):config(v,ds)
        config('control','agiqa1k',3047);print('Configs valid; 1K requires an explicit split choice')
    elif a.variant=='pack':
        if a.dataset or a.dry_run or a.split_seed is None:p.error('pack requires only --split-seed')
        pack(a.split_seed)
    else:
        if a.dataset is None:p.error('Specify one dataset')
        run(a.variant,a.dataset,a.split_seed,a.dry_run)
if __name__=='__main__':main()
