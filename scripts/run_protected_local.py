"""Protected local adaptation: mature foundation, matched control, and optional RichHF warmup."""
import argparse
import copy
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_projection_only_v1_baseline as base
from scripts import run_align_tuning as original
from utils.protected_local import FULL_FORMAT,LOCAL_FORMAT,ALLOWED
VERSION='projection_only_v1_v2_protected_local_v1'
DATASETS=('agiqa3k','aigciqa2023')
VARIANTS=('old','scratch','warm')


def source():return original.source()


def reference_config(ds):
    if ds not in DATASETS:raise ValueError('Only labelled alignment datasets 3K/2023')
    return yaml.safe_load((ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())


def foundation_config(ds):
    c=reference_config(ds)
    expected_source='outputs/'+original.reference()['source']['checkpoint'].split('/outputs/',1)[1]
    if (c['train']['pretrained_checkpoint']!=expected_source or c['train']['pretrained_load_scope']!='representation'
            or c['seed']!=42 or c['split_seed']!=42 or c['train']['epochs']!=60
            or c['train']['selection_metric']!='val_quality_SROCC'
            or c['model']['fusion']['mode']!='projection_only_v1'
            or c['model'].get('align_region_word') or c['model'].get('align_branch_control')):
        raise ValueError('Restore the original v2 baseline before foundation training')
    name=f'{ds}_{VERSION}_foundation_train42_split42'
    c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
    c['train'].update(save_selected_downstream=FULL_FORMAT,record_split_manifest=True,record_optimizer_initialization=True)
    return c


def foundation_runner():
    spec=importlib.util.spec_from_file_location('_protected_foundation',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.ROOT=ROOT
    r.VERSION=VERSION+'_foundation';r.VARIANTS={'foundation':('foundation',1)};r.QUEUES={}
    def config_request(v,ds,split=None):
        if v!='foundation' or split not in (None,42):raise ValueError('Invalid foundation request')
        return foundation_config(ds)
    r.config=config_request
    r.FILES=(*r.FILES,'selected_checkpoint_manifest.json')
    old=r.verify
    def verify(v,ds,split,upstream):
        old(v,ds,split,upstream)
        d,_=r.paths(v,ds);m=base.read_json(d/'selected_checkpoint_manifest.json')
        if (m['format']!=FULL_FORMAT or m['run_id']!=foundation_config(ds)['run_id']
                or m['last_epoch']!=60 or m['upstream_sha256']!=upstream['sha256']
                or m['checkpoint']!='best_selected_downstream.pth'
                or base.digest(d/m['checkpoint'])!=m['sha256']):raise ValueError('Foundation checkpoint changed/wrong source')
        rows=read_rows(r.paths(v,ds)[1]);final=next(x[1] for x in rows if x[0]=='final')
        if m['selected_epoch']!=final['best_epoch']:raise ValueError('Foundation selected checkpoint epoch differs')
    r.verify=verify
    return r


def foundation_path(ds):return ROOT/foundation_config(ds)['output_dir']/'best_selected_downstream.pth'


def verify_foundation(ds):
    r=foundation_runner();upstream=source();r.verify('foundation',ds,None,upstream)
    return dict(format=FULL_FORMAT,sha256=base.digest(foundation_path(ds)),
        run_id=foundation_config(ds)['run_id'],pretraining_sha256=upstream['sha256'])


def config(variant,ds):
    if variant not in (*VARIANTS,'warmup'):raise ValueError('Unknown local variant')
    if (variant=='warmup') != (ds=='richhf'):raise ValueError('Warmup is RichHF-only; downstream needs 3K/2023')
    c=reference_config('agiqa3k' if variant=='warmup' else ds)
    t=c['train'];m=c['model']
    if variant=='warmup':
        original_pre=yaml.safe_load((ROOT/'configs/projection_only_v1_baseline/pretrain.yaml').read_text())
        c['dataset']=copy.deepcopy(original_pre['dataset']);c['loader']=copy.deepcopy(original_pre['loader'])
        t['human_supervision_view']='three_view'
    else:
        t['pretrained_checkpoint']=str(foundation_path(ds).relative_to(ROOT))
        t['pretrained_load_scope']=FULL_FORMAT
    if variant!='old':m['align_region_word']={'version':'clip_region_word_v1'}
    name=f'{ds}_{VERSION}_{variant}_train42_split42'
    c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
    t.update(epochs=20,early_stop_patience=20,early_stopping_enabled=False,
        use_joint_iqa_pretraining=False,two_output_only=True,
        selection_metric='val_align_SROCC',selection_mode='max',
        lambda_quality=0.,lambda_rank_quality=0.,lambda_srocc_quality=0.,lambda_pert=0.,
        record_split_manifest=variant!='warmup',record_optimizer_initialization=True,
        audit_loss_gradients=False,record_run_metadata=True,pcgrad={'enabled':False},
        protected_local_variant=variant,protected_local_dataset=ds)
    if variant=='warmup':t.update(lambda_align_aux=0.)
    if (c['seed']!=42 or c['split_seed']!=42 or m['nss_text_gate']['nss_dim']!=6
            or m['fusion']['mode']!='projection_only_v1' or t['layerwise_lr']['shared_projection_mult']!=1.):
        raise ValueError('Wrong baseline local protocol')
    return c


def paths(v,ds):
    c=config(v,ds);return ROOT/c['output_dir'],ROOT/c['results_csv']


def read_rows(path):
    csv.field_size_limit(50_000_000)
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [(r['event'],json.loads(r['metrics_json']),int(r['epoch'])) for r in csv.DictReader(f)]


def dependencies(v,ds):
    if v=='warmup':return source()
    result=verify_foundation(ds)
    if v=='warm':
        verify('warmup','richhf')
        d,_=paths('warmup','richhf');result['local_pretraining_sha256']=base.digest(d/'best_local.pth')
    return result


def local_files(v):
    return ('launch_config.yaml','config_snapshot.yaml','launch_manifest.json','console.log','initial_metadata.json',
            'run_metadata.json','local_checkpoint_manifest.json',
            'official_split_manifest.json' if v=='warmup' else 'data_split_manifest.json')


def verify(v,ds):
    c=config(v,ds);d,table=paths(v,ds);dep=dependencies(v,ds)
    done=base.read_json(d/'completion_manifest.json');launch=base.read_json(d/'launch_manifest.json')
    if launch!=dict(version=VERSION,variant=v,dataset=ds,dependencies=dep,config_sha256=base.digest(d/'launch_config.yaml')):
        raise ValueError('Local launch/source identity mismatch')
    if yaml.safe_load((d/'launch_config.yaml').read_text())!=c or yaml.safe_load((d/'config_snapshot.yaml').read_text())!=c:
        raise ValueError('Local configuration differs')
    if set(done['files_sha256'])!=set(local_files(v)):raise ValueError('Incomplete local completion hashes')
    for name,sha in done['files_sha256'].items():
        if base.digest(d/name)!=sha:raise ValueError('Changed local artifact: '+name)
    if done['csv_sha256']!=base.digest(table):raise ValueError('Changed local CSV')
    ck=base.read_json(d/'local_checkpoint_manifest.json')
    if (ck['format']!=LOCAL_FORMAT or ck['variant']!=v or ck['last_epoch']!=20 or ck['upstream']!=dep
            or ck['checkpoint']!='best_local.pth' or base.digest(d/'best_local.pth')!=ck['sha256']):
        raise ValueError('Changed/foreign local checkpoint')
    rows=read_rows(table);epochs=[r for r in rows if r[0]=='epoch'];finals=[r for r in rows if r[0]=='final']
    if [e[2] for e in epochs]!=list(range(1,21)) or len(finals)!=1 or any(r[0]=='resume' for r in rows):
        raise ValueError('Expected fresh completed 20-epoch local training')
    f=finals[0][1];selected=max(epochs,key=lambda r:r[1]['val_val_align_SROCC'])
    if (f['best_epoch']!=selected[2] or ck['selected_epoch']!=selected[2] or f['last_epoch']!=20
            or f['selection_metric']!='val_align_SROCC' or f['frozen_before']!=f['frozen_after']):
        raise ValueError('Wrong selection or frozen state changed')
    meta=base.read_json(d/'run_metadata.json')
    base.require_subset(base.read_json(d/'initial_metadata.json'),meta)
    if (not meta.get('training_complete') or not meta.get('checkpoint_verified_after_load') or meta['frozen_before']!=meta['frozen_after']
            or meta['frozen_before']!=f['frozen_before'] or meta['checkpoint']!=dep
            or meta['variant']!=v or meta['seed']!=42 or meta['split_seed']!=42):
        raise ValueError('Local runtime/source/freeze evidence differs')
    if not meta['trainable_parameters'] or any(not n.startswith(ALLOWED[v]) for n in meta['trainable_parameters']):
        raise ValueError('Unexpected trainable parameters')
    if any(not any(n.startswith(prefix) for n in meta['trainable_parameters']) for prefix in ALLOWED[v]):
        raise ValueError('An intended local module is not trained')
    if not meta['optimizer_initialization']:raise ValueError('Missing optimizer initialization')
    if any(g['name']!='align' or not math.isclose(g['lr'],1e-4) or g['weight_decay']!=.002 for g in meta['optimizer_initialization']):
        raise ValueError('Wrong local optimizer policy')
    import torch
    checkpoint=torch.load(d/'best_local.pth',map_location='cpu',weights_only=False)
    if (checkpoint.get('format')!=LOCAL_FORMAT or checkpoint.get('mode')!='projection_only_v1'
            or checkpoint.get('training_complete') is not True or checkpoint.get('variant')!=v
            or checkpoint.get('run_id')!=c['run_id'] or checkpoint.get('cfg')!=c or checkpoint.get('upstream')!=dep
            or any(checkpoint.get(k)!=value for k,value in f.items())
            or not checkpoint.get('model') or any(not k.startswith(ALLOWED[v]) for k in checkpoint['model'])):
        raise ValueError('Actual local checkpoint contents/provenance differ')
    if not set(meta['trainable_parameters']).issubset(checkpoint['model']):
        raise ValueError('Selected local checkpoint omits trained parameters')
    if v!='old':
        expected={'align_region_word.image_projection.weight':(256,1024),
                  'align_region_word.text_projection.weight':(256,768)}
        if any(k not in checkpoint['model'] or tuple(checkpoint['model'][k].shape)!=shape for k,shape in expected.items()):
            raise ValueError('Incomplete/incompatible selected region projections')
    if v=='warm':
        transfer=meta['initialization'].get('warmup_transfer',{})
        if set(transfer.get('loaded',[]))!=set(expected) or not transfer.get('explicitly_not_transferred'):
            raise ValueError('Missing explicit local pretraining transfer evidence')
    reference=next(r[1] for r in rows if r[0]=='reference_validation')
    from scripts.train_protected_local import fixed_validation
    for _,metrics,_ in epochs:
        for key,value in fixed_validation(reference).items():
            if not math.isclose(metrics['val_'+key],value,rel_tol=0,abs_tol=1e-6):raise ValueError('Frozen validation outputs changed')
    if v=='warmup':
        split=base.read_json(d/'official_split_manifest.json');sets=[set(split['identities'][k]) for k in ('train','val','test')]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)) or f['best_test']:
            raise ValueError('Warmup split overlap or test evaluation')
    else:
        foundation_rows=read_rows(ROOT/foundation_config(ds)['results_csv'])
        foundation_final=next(r[1] for r in foundation_rows if r[0]=='final')
        for key,value in fixed_validation(reference).items():
            if not math.isclose(value,foundation_final['best_val'][key],rel_tol=0,abs_tol=1e-6):
                raise ValueError('Initial frozen outputs differ from mature foundation')
        for key,value in f['best_test'].items():
            if key.startswith('test_quality_') or key in ('test_align_global_SROCC','test_align_global_PLCC'):
                if not math.isclose(value,foundation_final['best_test'][key],rel_tol=0,abs_tol=1e-6):
                    raise ValueError('Frozen test outputs differ from mature foundation')
        a=base.read_json(d/'data_split_manifest.json');b=base.read_json(foundation_path(ds).parent/'data_split_manifest.json')
        if any(a[k]!=b[k] for k in ('indices','dataset_ids_sha256','train_seed','split_seed')):
            raise ValueError('Local/foundation actual memberships differ')
    return f


def run(v,ds,continue_completed=False):
    d,table=paths(v,ds)
    if d.exists() or table.exists():
        if not continue_completed:raise FileExistsError(f'Preserve existing outputs: {d}')
        verify(v,ds);print('Verified complete; skip:',v,ds);return
    dep=dependencies(v,ds)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; cloud CPU training refused')
    d.mkdir(parents=True,exist_ok=False);c=config(v,ds)
    with (d/'launch_config.yaml').open('x') as f:yaml.safe_dump(c,f,sort_keys=False)
    base.write_new(d/'launch_manifest.json',dict(version=VERSION,variant=v,dataset=ds,dependencies=dep,config_sha256=base.digest(d/'launch_config.yaml')))
    command=[sys.executable,'scripts/train_protected_local.py','--config',str(d/'launch_config.yaml')]
    with (d/'console.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        if process.wait():raise RuntimeError('Local trainer failed; partial outputs preserved')
    base.write_new(d/'completion_manifest.json',dict(files_sha256={name:base.digest(d/name) for name in local_files(v)},csv_sha256=base.digest(table)))
    verify(v,ds)


def plan(action,datasets):
    stages=[]
    if action in ('foundation','direct','all'):stages += [('foundation',ds) for ds in datasets]
    if action in ('warmup','warm-chain','all'):stages += [('warmup','richhf')]
    variants=('old','scratch') if action=='direct' else VARIANTS if action in ('downstream','all') else ('warm',) if action=='warm-chain' else (action,) if action in VARIANTS else ()
    stages += [(v,ds) for ds in datasets for v in variants]
    return stages


def wait_for_foundations(datasets,wait_hours=48,poll_seconds=30):
    if not math.isfinite(wait_hours) or wait_hours<=0 or not math.isfinite(poll_seconds) or poll_seconds<=0:
        raise ValueError('Wait timeout/poll interval must be positive and finite')
    deadline=time.monotonic()+wait_hours*3600
    pending=list(datasets)
    while pending:
        for ds in pending[:]:
            marker=foundation_path(ds).parent/'completion_manifest.json'
            if marker.is_file() and marker.stat().st_size>0:
                # An empty file can be the producer's just-created, not-yet-
                # flushed completion record. Leave it pending.
                # Existence alone is insufficient: a corrupt/completed foreign
                # experiment must fail immediately, never be swallowed/retried.
                verify_foundation(ds)
                print('[FOUNDATION READY]',ds,flush=True)
                pending.remove(ds)
        if not pending:return
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise TimeoutError('Foundation wait expired; local pretraining retained; pending: '+', '.join(pending))
        print('[WAIT FOUNDATION]',', '.join(pending),
              '(no model/GPU allocation while waiting; Ctrl-C is safe)',flush=True)
        time.sleep(min(poll_seconds,remaining))


def run_warm_chain(datasets,continue_completed=False,wait_hours=48):
    run('warmup','richhf',continue_completed)
    wait_for_foundations(datasets,wait_hours)
    for ds in datasets:run('warm',ds,continue_completed)


def pack():
    source_record=source();files={ROOT/'docs/protected_local.md',ROOT/'docs/current_tuning_reference.json',ROOT/'docs/current_tuning_reference.md'};results={}
    files.update(ROOT/f'configs/current_tuning_reference/{ds}.yaml' for ds in DATASETS)
    for journal in (ROOT/'run_logs').glob('protected_local_recovery_*.json'):
        recovery=base.read_json(journal)
        if (recovery.get('format')=='protected_local_recovery_v1' and recovery.get('version')==VERSION
                and recovery.get('upstream',{}).get('sha256')==source_record['sha256']):
            files.add(journal)
    for ds in DATASETS:
        verify_foundation(ds);d=foundation_path(ds).parent
        files.update(p for p in d.iterdir() if p.suffix in ('.json','.yaml','.log'))
        table=ROOT/foundation_config(ds)['results_csv'];files.add(table)
        results['foundation/'+ds]=next(r[1] for r in read_rows(table) if r[0]=='final')
    for v,ds in [('warmup','richhf')]+[(v,ds) for ds in DATASETS for v in VARIANTS]:
        results[f'{v}/{ds}']=verify(v,ds);d,t=paths(v,ds);files.add(t)
        files.update(p for p in d.iterdir() if p.suffix in ('.json','.yaml','.log'))
    archive=ROOT/f'{VERSION}_results.tar.gz'
    with archive.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for path in sorted(files):tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        data=json.dumps(dict(source=source_record,results=results),indent=2).encode()
        info=tarfile.TarInfo('summary.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    print(archive)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('foundation','warmup','warm-chain','direct','old','scratch','warm','downstream','all','pack','check-configs'))
    p.add_argument('--dataset',choices=DATASETS);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--continue',dest='continue_completed',action='store_true')
    p.add_argument('--wait-hours',type=float,default=None,help='warm-chain only: foundation timeout, default 48h')
    a=p.parse_args();os.chdir(ROOT);datasets=DATASETS if a.dataset is None else (a.dataset,)
    if a.wait_hours is not None and (a.action!='warm-chain' or not math.isfinite(a.wait_hours) or a.wait_hours<=0):
        p.error('--wait-hours requires warm-chain and a positive finite value')
    if a.action=='pack':
        if a.dataset or a.dry_run or a.continue_completed or a.wait_hours is not None:p.error('pack takes no other arguments')
        pack();return
    if a.action=='warmup' and a.dataset:p.error('warmup is RichHF-only; omit --dataset')
    stages=plan('all' if a.action=='check-configs' else a.action,datasets)
    if a.dry_run or a.action=='check-configs':
        print(json.dumps({'version':VERSION,'configs':[foundation_config(ds) if v=='foundation' else config(v,ds) for v,ds in stages],
                         'note':'Config plan only; CPU training refused; source hashes checked before launch'},indent=2));return
    # Preflight existing runs before starting any new stage. Never reinterpret a
    # partial output as permission to resume/overwrite it.
    for v,ds in stages:
        d,t=(foundation_path(ds).parent,ROOT/foundation_config(ds)['results_csv']) if v=='foundation' else paths(v,ds)
        if d.exists() or t.exists():
            if not a.continue_completed:raise FileExistsError(f'Preserve existing outputs: {d}')
            verify_foundation(ds) if v=='foundation' else verify(v,ds)
    if a.action=='warm-chain':
        run_warm_chain(datasets,a.continue_completed,48 if a.wait_hours is None else a.wait_hours);return
    for v,ds in stages:
        if v=='foundation':foundation_runner().run_all(continue_completed=a.continue_completed,dataset=ds)
        else:run(v,ds,a.continue_completed)

if __name__=='__main__':main()
