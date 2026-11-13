"""Explicit single-interruption NSS recovery; preserves evidence, never resets history."""
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone
import yaml


def rows(path):
    csv.field_size_limit(50_000_000)
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def active_trainer(root,config):
    """Avoid touching an active run. Other variants are allowed on another GPU."""
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            if not any(Path(a).name=='train_cm_pretrain.py' for a in args if a):continue
            if '--config' not in args:continue
            source=Path(args[args.index('--config')+1])
            if not source.is_absolute():source=(proc/'cwd').resolve()/source
            if source.resolve()==config.resolve():return int(proc.name)
        except (OSError,ValueError,UnicodeError):continue
    return None


def verify_training_code(r,launch):
    """Runner changes are allowed; training, model and data code must match."""
    original=launch.get('runtime',{}).get('code_commit')
    if not original:raise ValueError('Original training code commit is missing')
    checked=r.subprocess.run(['git','diff','--name-only',original,'--','models','datasets','losses','utils','train_cm_pretrain.py'],
                            cwd=r.ROOT,capture_output=True,text=True,check=False)
    if checked.returncode!=0:raise ValueError('Cannot verify original training code commit')
    if checked.stdout.strip():raise ValueError('Training/model/data code changed since interrupted run: '+checked.stdout.strip())
    return {'original_code_commit':original,'training_code_matches_original':True}


def preflight(r,variant):
    import torch
    cfg=r.validate('pretrain');d,table=r.paths('pretrain');config=r.CONFIG/'pretrain.yaml'
    if (d/'completion_manifest.json').exists():raise FileExistsError('Already completed; use all to verify/reuse')
    if (d/'recovery_manifest.json').exists() or (d/'recovery_inputs').exists():
        raise FileExistsError('A recovery attempt already exists; preserve it for diagnosis, do not overwrite')
    pid=active_trainer(r.ROOT,config)
    if pid:raise RuntimeError(f'NSS trainer is still active: PID {pid}')
    launch=r.read_json(d/'launch_manifest.json')
    if launch.get('version')!=r.VERSION or launch.get('stage')!='pretrain' or launch.get('source') is not None or launch.get('config_sha256')!=r.digest(config):
        raise ValueError('Original launch provenance mismatch')
    code_evidence=verify_training_code(r,launch)
    snapshot=yaml.safe_load((d/'config_snapshot.yaml').read_text());r.require_subset(cfg,snapshot)
    original=rows(table)
    if any(x['event'] in ('resume','final','early_stop') for x in original):
        raise ValueError('Expected one interrupted fresh run, without previous recovery/final/early-stop')
    if any(x.get('run_id')!=cfg['run_id'] for x in original):raise ValueError('CSV contains another run')
    source=d/'last_pretrain.pth';before=r.digest(source)
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    required=('epoch','model','optimizer','scheduler','cfg','run_id','training_state','auxiliary_bests','scaler','rng')
    if any(k not in checkpoint for k in required):raise ValueError('Full restart checkpoint fields missing')
    if checkpoint['run_id']!=cfg['run_id']:raise ValueError('Checkpoint run identity mismatch')
    r.require_subset(cfg,checkpoint['cfg'])
    snapshot.pop('source_config',None)
    if checkpoint['cfg']!=snapshot:raise ValueError('Effective checkpoint config differs from original snapshot')
    mode=cfg['model']['nss_text_gate']['descriptor'];dim=cfg['model']['nss_text_gate']['nss_dim']
    signature=checkpoint['model'].get('iqa.nss_descriptor_signature')
    code=1 if mode=='mscn_stats6_v1' else 2
    if signature is None or not torch.equal(signature.cpu(),torch.tensor([code,dim])):raise ValueError('NSS restart signature mismatch')
    epoch=int(checkpoint['epoch']);epochs=[x for x in original if x['event']=='epoch']
    if [int(x['epoch']) for x in epochs]!=list(range(1,epoch+1)):raise ValueError('CSV and checkpoint epochs differ')
    if not 0<epoch<int(cfg['train']['epochs']):raise ValueError('Checkpoint is outside the recoverable training horizon')
    state=checkpoint['training_state'];best=max(epochs,key=lambda x:json.loads(x['metrics_json'])['val_component_anchor_SROCC'])
    score=json.loads(best['metrics_json'])['val_component_anchor_SROCC']
    if state['best_epoch']!=int(best['epoch']) or state['best']!=score or state['no_improve']!=epoch-int(best['epoch']):
        raise ValueError('Saved selection / patience state differs from history')
    if state['no_improve']>=cfg['train']['early_stop_patience']:raise ValueError('Saved state already reached early stopping')
    aux=checkpoint['auxiliary_bests']['component_anchor']
    if aux['epoch']!=state['best_epoch'] or aux['value']!=score:raise ValueError('Component selection counter mismatch')
    rng=checkpoint['rng']
    if any(k not in rng for k in ('python','numpy','torch','cuda')):raise ValueError('Incomplete RNG state')
    selected_epoch=state['best_epoch'];saved_state=dict(state)
    del checkpoint;gc.collect()
    if r.digest(source)!=before:raise ValueError('Restart checkpoint changed during preflight')
    selected=d/'best_component_anchor_pretrain.pth';r.verify_checkpoint(selected,cfg,selected_epoch)
    return {**code_evidence,'variant':variant,'version':r.VERSION,'kind':'interruption_recovery_v1',
            'config_sha256':r.digest(config),'launch_sha256':r.digest(d/'launch_manifest.json'),
            'input_checkpoint_sha256':before,'saved_epoch':epoch,'next_epoch':epoch+1,
            'selection_state':saved_state,'selected_checkpoint_before_sha256':r.digest(selected),
            'bitwise_reproduction_claimed':False}


def validate_history(r,stage,history,done):
    if stage!='pretrain':raise ValueError('Only explicit NSS pretraining recovery is allowed')
    d,table=r.paths(stage);manifest=r.read_json(d/'recovery_manifest.json');complete=r.read_json(d/'recovery_complete.json')
    if manifest['kind']!='interruption_recovery_v1' or manifest['version']!=r.VERSION or manifest['config_sha256']!=r.digest(r.CONFIG/'pretrain.yaml') or manifest['launch_sha256']!=r.digest(d/'launch_manifest.json'):
        raise ValueError('NSS recovery provenance mismatch')
    for name in ('recovery_manifest.json','recovery_complete.json'):
        if done['files_sha256'].get(name)!=r.digest(d/name):raise ValueError('Recovery record missing from completion checksums')
    for name,sha in manifest['evidence_sha256'].items():
        if r.digest(d/'recovery_inputs'/name)!=sha:raise ValueError(f'Original interrupted evidence changed: {name}')
    if r.digest(d/'recovery_inputs/last_pretrain.pth')!=manifest['input_checkpoint_sha256']:raise ValueError('Original restart state changed')
    if r.digest(d/'recovery_inputs/best_component_anchor_pretrain.pth')!=manifest['selected_checkpoint_before_sha256']:raise ValueError('Original selected state changed')
    prefix=(d/'recovery_inputs/results.csv').read_bytes()
    with table.open('rb') as f:
        if f.read(len(prefix))!=prefix:raise ValueError('Original CSV prefix was rewritten')
    resumed=[x for x in history if x['event']=='resume'];e=manifest['saved_epoch']
    if len(resumed)!=1 or int(resumed[0]['epoch'])!=e:raise ValueError('Unexpected resume history')
    m=json.loads(resumed[0]['metrics_json'])
    if m['start_epoch']!=e+1 or Path(m['checkpoint']).resolve()!=(d/'recovery_inputs/last_pretrain.pth').resolve():raise ValueError('Resume input differs from recorded state')
    epochs=[x for x in history if x['event']=='epoch'];finals=[x for x in history if x['event']=='final']
    if len(finals)!=1:raise ValueError('Recovered run must finish normally')
    final=json.loads(finals[0]['metrics_json']);last=final['last_epoch']
    if [int(x['epoch']) for x in epochs]!=list(range(1,last+1)) or last<=e:raise ValueError('Recovered epoch sequence is incomplete')
    cfg=r.validate(stage);patience=cfg['train']['early_stop_patience']
    best=max(epochs,key=lambda x:json.loads(x['metrics_json'])['val_component_anchor_SROCC'])
    if final['selection_metric']!='val_component_anchor_SROCC' or final['best_component_anchor_epoch']!=int(best['epoch']):raise ValueError('Recovered component selection is not the history maximum')
    last_metrics=json.loads(epochs[-1]['metrics_json'])
    if last!=cfg['train']['epochs'] and (last_metrics.get('no_improve',0)<patience or not any(x['event']=='early_stop' for x in history)):
        raise ValueError('Recovered run stopped without budget completion or configured early stopping')
    if complete['manifest_sha256']!=r.digest(d/'recovery_manifest.json') or complete['final_csv_sha256']!=r.digest(table) or complete['last_epoch']!=last:
        raise ValueError('Recovered completion record mismatch')


def install(r):
    r.validate_recovery_history=lambda stage,history,done:validate_history(r,stage,history,done)
    original=r.completed
    def completed(stage):
        source=original(stage)
        d,_=r.paths('pretrain')
        if (d/'recovery_manifest.json').exists():
            if 'recovery_manifest.json' not in r.read_json(d/'completion_manifest.json')['files_sha256']:
                raise ValueError('Unverified NSS recovery cannot be a source')
            source.update(training_continuity='interruption_recovery_v1',recovery_manifest_sha256=r.digest(d/'recovery_manifest.json'))
        return source
    r.completed=completed


def recover(r,variant,dry_run=False):
    import torch
    d,table=r.paths('pretrain')
    if (d/'completion_manifest.json').exists():return r.completed('pretrain')
    manifest=preflight(r,variant)
    command=[sys.executable,'train_cm_pretrain.py','--config',str(r.CONFIG/'pretrain.yaml'),
             '--resume',str(d/'recovery_inputs/last_pretrain.pth')]
    if dry_run:
        print(json.dumps({'command':command,'recovery':manifest},indent=2));return
    if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; cloud CPU recovery refused')
    import fcntl
    # flock is released even on process death; separate variants have separate locks.
    with (d/'recovery.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another recovery process owns this NSS run')
        if (d/'recovery_manifest.json').exists():raise FileExistsError('Recovery already attempted; preserving it')
        backup=d/'recovery_inputs';backup.mkdir(exist_ok=False)
        evidence={}
        for src in [table]+[d/name for name in ('console.log','config_snapshot.yaml','launch_manifest.json','run_metadata.json')]:
            name='results.csv' if src==table else src.name
            shutil.copy2(src,backup/name);evidence[name]=r.digest(backup/name)
        # Trainer publishes weights using atomic replace, so these hard links
        # retain the original inodes. Copy if hard links are unavailable.
        for src in d.glob('*.pth'):
            try:os.link(src,backup/src.name)
            except OSError:shutil.copy2(src,backup/src.name)
        if r.digest(backup/'last_pretrain.pth')!=manifest['input_checkpoint_sha256'] or r.digest(backup/'best_component_anchor_pretrain.pth')!=manifest['selected_checkpoint_before_sha256']:
            raise ValueError('Checkpoint changed while preserving interrupted evidence')
        from utils.run_metadata import collect_run_metadata
        manifest.update(evidence_sha256=evidence,command=command,recovery_runtime=collect_run_metadata(None,r.validate('pretrain')),
                        created_utc=datetime.now(timezone.utc).isoformat())
        r.write_new(d/'recovery_manifest.json',manifest)
        with (d/'console.log').open('a') as log:
            log.write('\n[RECOVERY] Original interrupted state preserved; resume at epoch '+str(manifest['next_epoch'])+'\n');log.flush()
            process=r.subprocess.Popen(command,cwd=r.ROOT,stdout=r.subprocess.PIPE,stderr=r.subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if process.wait()!=0:raise RuntimeError('NSS recovery failed; all original and recovery evidence preserved')
        history=rows(table);finals=[x for x in history if x['event']=='final']
        if len(finals)!=1:raise ValueError('Trainer did not finish; no completion will be published')
        final=json.loads(finals[0]['metrics_json'])
        r.verify_checkpoint(d/'best_component_anchor_pretrain.pth',r.validate('pretrain'),final['best_component_anchor_epoch'])
        r.write_new(d/'recovery_complete.json',{'manifest_sha256':r.digest(d/'recovery_manifest.json'),'final_csv_sha256':r.digest(table),'last_epoch':final['last_epoch']})
        files=('config_snapshot.yaml','console.log','run_metadata.json','recovery_manifest.json','recovery_complete.json')
        done={'files_sha256':{name:r.digest(d/name) for name in files},'csv_sha256':r.digest(table),
              'launch_sha256':r.digest(d/'launch_manifest.json'),'checkpoint_sha256':r.digest(d/'best_component_anchor_pretrain.pth')}
        validate_history(r,'pretrain',history,done)
        r.write_new(d/'completion_manifest.json',done)
        return r.completed('pretrain')
