"""Four downstream-only alignment controls from the pinned original v2 baseline."""
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
from scripts import run_quality_srocc as previous_srocc
VARIANTS={'alignlr050':('lr',.5),'alignlr200':('lr',2.),'asrocc050':('srocc',.02),'asrocc200':('srocc',.08)}
QUEUES={'lr':('alignlr050','alignlr200'),'srocc':('asrocc050','asrocc200')}
DATASETS=('agiqa3k','aigciqa2023')
VERSION='projection_only_v1_v2_align_tuning'
PROTOCOL_DOC='docs/align_tuning.md'
FILES=('launch_config.yaml','config_snapshot.yaml','console.log','launch_manifest.json',
       'completion_manifest.json','run_metadata.json','source_fingerprint.json','data_split_manifest.json')


def config(variant,ds,split=None):
    if variant not in VARIANTS or ds not in DATASETS:raise ValueError('Unknown alignment variant/dataset; 1K is excluded')
    if split is not None and split != 42:
        raise ValueError('Fixed split_seed=42')
    seed=42
    c=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())
    name=f'{ds}_projection_only_v1_v2_{variant}_train42_split{seed}'
    c.update(output_dir=f'outputs/{name}',results_csv=f'outputs/{name}.csv',run_id=name)
    if c['seed']!=42 or c['split_seed']!=seed or c['train']['lambda_srocc_quality']!={'agiqa1k':.03,'agiqa3k':.02,'aigciqa2023':.04}[ds]:
        raise ValueError('Selected v2 config seed/split/Soft-SROCC policy mismatch')
    pinned=reference()['source']['checkpoint']
    expected_checkpoint='outputs/'+pinned.split('/outputs/',1)[1]
    if (c['train']['pretrained_checkpoint']!=expected_checkpoint or
            c['train']['pretrained_load_scope']!='representation' or
            c['train']['epochs']!=60 or c['train']['selection_metric']!='val_quality_SROCC' or
            c['train']['selection_mode']!='max' or c['model']['fusion']['mode']!='projection_only_v1'):
        raise ValueError('Wrong baseline source/mode/budget/selection configuration')
    c['split_seed']=seed
    if c['model']['nss_text_gate'].get('descriptor','legacy')!='legacy' or c['model']['nss_text_gate']['nss_dim']!=6 or c['model']['nss_text_gate'].get('pathway','both')!='both':
        raise ValueError('This study requires the original six-dimensional NSS baseline')
    if c['train']['srocc_tau']!=.02 or c['train']['lambda_srocc_align']!=.04 or c['train']['layerwise_lr']['align_mult']!=1.0:
        raise ValueError('Expected original alignment baseline settings')
    kind,value=VARIANTS[variant]
    if kind=='lr':c['train']['layerwise_lr']['align_mult']=value
    else:c['train']['lambda_srocc_align']=value
    c['train'].update(record_split_manifest=True,record_optimizer_initialization=True)
    return c


def paths(v,ds,split=None):
    c=config(v,ds,split);return ROOT/c['output_dir'],ROOT/c['results_csv']


def reference():
    data=base.read_json(ROOT/'docs/current_tuning_reference.json')
    if data['version']!='projection_only_v1_selected_tuning_reference_v2':
        raise ValueError('This study requires the selected v2 tuning reference')
    return data


def source():
    observed=splits.source();pinned=reference()['source']
    if any(observed[k]!=pinned[k] for k in ('version','sha256','config_sha256','selected_epoch','completion_sha256')):
        raise ValueError('Pretraining source differs from v2')
    return observed


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
    groups=m['optimizer_initialization']
    expected=expected_group_lrs(c)
    if (len({g['name'] for g in groups})!=len(groups)
            or not {'quality_adapter','quality','align','fusion','global_encoder','local_encoder','text_context','shared_projection'}.issubset({g['name'] for g in groups})
            or any(g['name'] not in expected
                   or not math.isclose(g['lr'],expected[g['name']],rel_tol=1e-12)
                   or g['weight_decay']!=.002 for g in groups)):
        raise ValueError('Wrong optimizer group initialization')
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
    records=[json.loads(r['metrics_json']) for r in rows if r['event']=='optimizer_initialization']
    if records!=[{'groups':groups}]:raise ValueError('Missing optimizer initialization event')
    epochs=[r for r in rows if r['event']=='epoch']
    if [int(r['epoch']) for r in epochs]!=list(range(1,61)):
        raise ValueError('Missing or repeated epoch records')
    selected=max(epochs,key=lambda row:json.loads(row['metrics_json'])['val_val_quality_SROCC'])
    best_test=[r for r in rows if r['event']=='best_test']
    best_val=[r for r in rows if r['event']=='best_val']
    if (len(best_test)!=1 or len(best_val)!=1 or
            int(selected['epoch'])!=int(best_test[0]['epoch']) or
            int(selected['epoch'])!=int(best_val[0]['epoch']) or
            int(selected['epoch'])!=finals[0]['best_epoch'] or
            json.loads(best_test[0]['metrics_json'])!=finals[0]['best_test']):
        raise ValueError('Best-test/validation selection mismatch')
    for row in epochs:
        metrics=json.loads(row['metrics_json'])
        # All non-listed terms have zero coefficients in this fixed protocol.
        t=c['train'];prefix='train_train_'
        expected_loss=sum(coef*metrics[prefix+key] for key,coef in (
            ('quality',t['lambda_quality']),('align',t['lambda_align']),
            ('align_global',t['lambda_align_aux']),('align_patch',t['lambda_align_aux']),
            ('rank_quality',t['lambda_rank_quality']),('rank_align',t['lambda_rank_align']),
            ('srocc_quality',t['lambda_srocc_quality']),('srocc_align',t['lambda_srocc_align']),
            ('pert',t['lambda_pert'])))
        if not math.isclose(metrics[prefix+'loss'],expected_loss,rel_tol=1e-3,abs_tol=1e-6):
            raise ValueError('Effective total alignment/quality objective mismatch')
        for raw,weighted,coefficient in (
            ('train_train_srocc_quality','train_train_weighted_quality_srocc',c['train']['lambda_srocc_quality']),
            ('train_train_rank_quality','train_train_weighted_quality_rank',c['train']['lambda_rank_quality']),
            ('train_train_quality','train_train_weighted_quality_mse',c['train']['lambda_quality'])):
            if not math.isclose(metrics[weighted],coefficient*metrics[raw],rel_tol=1e-3,abs_tol=1e-7):
                raise ValueError(f'Effective loss weight mismatch at epoch {row["epoch"]}: {weighted}')


def control(ds,split,fingerprint):
    if ds=='aigciqa2023':
        previous_srocc.verify('qsrocc200',ds,None,fingerprint)
        d,table=previous_srocc.paths('qsrocc200',ds)
    else:
        base.completed(ds);d,table=base.paths(ds)
    record=next(r for r in reference()['results'] if r['dataset']==ds)
    if table.stem!=record['run_id'] or base.digest(table)!=record['csv_sha256']:
        raise ValueError('Control CSV is not the user-selected v2 result')
    base.require_subset(yaml.safe_load((ROOT/record['config']).read_text()),yaml.safe_load((d/'config_snapshot.yaml').read_text()))
    return d,table


def run(v,ds,split=None,dry_run=False):
    c=config(v,ds,split);fingerprint=source();d,table=paths(v,ds,split)
    control(ds,split,fingerprint)
    if d.exists() or table.exists():raise FileExistsError(f'Preserve existing outputs: {d}')
    command=[sys.executable,'train_cm_iqa.py','--config',str(d/'launch_config.yaml')]
    if dry_run:print(json.dumps(dict(command=command,config=c,source=fingerprint),indent=2));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; cloud CPU training refused')
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


def pack():
    fingerprint=source()
    files={ROOT/'docs/current_tuning_reference.json',ROOT/'docs/current_tuning_reference.md',
           ROOT/PROTOCOL_DOC}
    files.update(ROOT/f'configs/current_tuning_reference/{ds}.yaml' for ds in DATASETS)
    for ds in DATASETS:
        cd,ct=control(ds,None,fingerprint);files.add(ct)
        files.update(cd/n for n in FILES if (cd/n).is_file())
        for v in VARIANTS:
            verify(v,ds,None,fingerprint);d,table=paths(v,ds);files.add(table)
            files.update(d/n for n in FILES)
            if (cd/'data_split_manifest.json').exists():
                a=base.read_json(cd/'data_split_manifest.json');b=base.read_json(d/'data_split_manifest.json')
                if any(a[k]!=b[k] for k in ('dataset_ids_sha256','indices')):
                    raise ValueError('Candidate/control memberships differ')
    archive=ROOT/f'{VERSION}_results.tar.gz'
    with archive.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        data=json.dumps(fingerprint,indent=2).encode();info=tarfile.TarInfo('source_fingerprint.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    print(archive)


def run_all(queue='all',dry_run=False,continue_completed=False,dataset=None):
    variants=tuple(VARIANTS) if queue=='all' else QUEUES[queue]
    datasets=DATASETS if dataset is None else (dataset,)
    if any(ds not in DATASETS for ds in datasets):raise ValueError('Only 3K and 2023')
    if dry_run:
        print(json.dumps({'configs':[config(v,ds) for ds in datasets for v in variants],
            'source':reference()['source'],'note':'Config plan only; actual source/control hashes checked before launch'},indent=2));return
    fingerprint=source();pending=[]
    # Validate ALL requested controls/output paths before any trainer starts.
    for ds in datasets:
        control(ds,None,fingerprint)
        for v in variants:
            d,table=paths(v,ds)
            if d.exists() or table.exists():
                if not continue_completed:raise FileExistsError(f'Preserve existing outputs: {d}')
                verify(v,ds,None,fingerprint);print('Verified completed; skip:',v,ds)
            else:pending.append((v,ds))
    for v,ds in pending:run(v,ds)


def expected_group_lrs(c):
    l=c['train']['layerwise_lr'];rate=c['train']['lr']
    return {name:rate*mult for name,mult in dict(
        global_encoder=l['global_encoder_mult'],local_encoder=l['local_encoder_mult'],
        fusion=l['fusion_mult'],text_context=l.get('text_context_mult',.1),
        fusion_gate=l.get('fusion_gate_mult',l['fusion_mult']),
        quality_adapter=l.get('quality_adapter_mult',l['quality_mult']),
        quality=l['quality_mult'],align=l['align_mult'],other=l['other_mult'],
        shared_projection=l['shared_projection_mult']).items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=tuple(VARIANTS)+tuple(QUEUES)+('all','pack','check-configs'))
    p.add_argument('dataset',nargs='?',choices=DATASETS)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--continue',dest='continue_completed',action='store_true',help='Verify and skip completed runs; never resume or overwrite partial runs')
    a=p.parse_args();os.chdir(ROOT)
    if a.variant in ('all',*QUEUES):run_all(a.variant,a.dry_run,a.continue_completed,a.dataset)
    elif a.variant=='check-configs':
        if a.dataset or a.dry_run or a.continue_completed:p.error('check-configs takes no other arguments')
        for ds in DATASETS:
            for v in VARIANTS:config(v,ds)
        print(f'{len(VARIANTS)*len(DATASETS)} downstream configs: train42/split42; original source, six-channel NSS and Quality selection preserved')
    elif a.variant=='pack':
        if a.dataset or a.dry_run or a.continue_completed:p.error('pack takes no other arguments')
        pack()
    else:
        if a.dataset is None:p.error('Specify agiqa3k or aigciqa2023')
        if a.dry_run:
            print(json.dumps({'config':config(a.variant,a.dataset),'source':reference()['source'],'note':'Config plan only'},indent=2));return
        if a.continue_completed and any(x.exists() for x in paths(a.variant,a.dataset)):
            fingerprint=source();control(a.dataset,None,fingerprint);verify(a.variant,a.dataset,None,fingerprint)
        else:run(a.variant,a.dataset)
if __name__=='__main__':main()
