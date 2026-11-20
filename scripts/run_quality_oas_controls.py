"""Independent quality-first and post-residual OAS controls. Never launch on CPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_pretrain_component_strength as pipeline
from utils.pretrain_phases import ALIGN_LOSSES, QUALITY_LOSSES, validate as validate_phase

VARIANTS=('quality_first','residual_oas')
CONFIG=Path('configs/quality_oas_controls')
STAGES=pipeline.STAGES


def expected_config(variant,stage='pretrain'):
    if variant not in (*VARIANTS,'quality_stage') or stage not in STAGES:
        raise ValueError('Unknown quality/OAS variant or stage')
    if variant=='quality_stage' and stage!='pretrain':
        raise ValueError('Quality stage has no independent downstream')
    if stage=='pretrain':
        cfg=copy.deepcopy(pipeline.baseline.validate('pretrain'))
        run_id=f'richhf18k_projection_only_v1_ctx5_{variant}_seed42'
        t=cfg['train']
        if variant in ('quality_stage','quality_first'):
            t['pretraining_phase']='quality' if variant=='quality_stage' else 'alignment'
            t['pcgrad']['enabled']=False
            for key in (ALIGN_LOSSES if variant=='quality_stage' else QUALITY_LOSSES):
                if key in t:t[key]=0.0
            if variant=='quality_first':
                t['quality_auxiliary_supervision']['enabled']=False
                t['quality_stage_checkpoint']=expected_config('quality_stage')['output_dir']+'/best_pretrain.pth'
                t['quality_stage_run_id']=expected_config('quality_stage')['run_id']
                t['pretrain_selection_metric']='val_align_proxy_SROCC'
                cfg['model']['global_encoder']['freeze']=True
                cfg['model']['local_encoder']['freeze']=True
        else:
            cfg['model']['dual_quality_representation']['anchor_global_source']='residual'
        validate_phase(cfg)
    else:
        cfg=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix='agiqa1k_subset1078' if stage=='agiqa1k' else stage
        run_id=f'{prefix}_projection_only_v1_{variant}_train42_split{cfg["split_seed"]}'
        filename='best_align_pretrain.pth' if variant=='quality_first' else 'best_component_anchor_pretrain.pth'
        cfg['train']['pretrained_checkpoint']=expected_config(variant)['output_dir']+'/'+filename
        # Downstream trainability and regression-head initialization retain the
        # original baseline; this route affects only pretraining OAS.
    cfg.update(run_id=run_id,output_dir='outputs/'+run_id,results_csv='outputs/'+run_id+'.csv')
    return cfg


def make_runner(variant):
    r=pipeline.make_runner(variant,config_factory=expected_config,config_dir=CONFIG,
        version='projection_only_v1_ctx5_'+variant)
    original_run,original_completed=r.run,r.completed
    metric='val_align_proxy_SROCC' if variant=='quality_first' else 'val_component_anchor_SROCC'
    filename='best_align_pretrain.pth' if variant=='quality_first' else 'best_component_anchor_pretrain.pth'
    field='align' if variant=='quality_first' else 'component_anchor'

    def completed(stage):
        if stage!='pretrain':return original_completed(stage)
        cfg=r.validate(stage);d,table=r.paths(stage)
        launch=r.read_json(d/'launch_manifest.json');done=r.read_json(d/'completion_manifest.json')
        if launch['version']!=r.VERSION or launch['stage']!=stage or launch['config_sha256']!=r.digest(r.CONFIG/'pretrain.yaml'):
            raise ValueError('Pretraining launch identity mismatch')
        for n,h in done['files_sha256'].items():
            if r.digest(d/n)!=h:raise ValueError('Completed artifact changed: '+n)
        if done['csv_sha256']!=r.digest(table) or done['launch_sha256']!=r.digest(d/'launch_manifest.json'):
            raise ValueError('Pretraining completion changed')
        r.require_subset(cfg,yaml.safe_load((d/'config_snapshot.yaml').read_text()))
        import csv
        csv.field_size_limit(50_000_000)
        with table.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        finals=[json.loads(x['metrics_json']) for x in rows if x['event']=='final']
        if len(finals)!=1 or any(x['event']=='resume' for x in rows):
            raise ValueError('Expected one completed non-resumed pretraining run')
        final=finals[0]
        if final['selection_metric']!=metric:raise ValueError('Wrong phase selection metric')
        epochs=[x for x in rows if x['event']=='epoch']
        if [int(x['epoch']) for x in epochs]!=list(range(1,final['last_epoch']+1)):
            raise ValueError('Missing epoch history')
        selected=max(epochs,key=lambda x:json.loads(x['metrics_json'])[metric])
        epoch=final['best_epoch']
        if int(selected['epoch'])!=epoch or final['best_'+field+'_epoch']!=epoch:
            raise ValueError('Selected epoch differs from maximum validation metric')
        checkpoint=d/filename
        if done['checkpoint_sha256']!=r.digest(checkpoint):raise ValueError('Selected weights changed')
        import torch
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        r.require_subset(cfg,ck['cfg'])
        if (ck.get('run_id')!=cfg['run_id'] or ck.get('checkpoint_scope')!='representation'
                or ck.get('epoch')!=epoch or ck.get('selection_metric')!=metric):
            raise ValueError('Selected checkpoint identity mismatch')
        required=('iqa.backbone.global_encoder.','iqa.backbone.local_encoder.',
            'iqa.backbone.fusion.semantic_proj.','iqa.quality_adapter.','iqa.quality_local_encoder.',
            'iqa.nss_proj.','iqa.nss_gate.','iqa.shared_image_projector.','iqa.shared_text_projector.')
        if any(not any(k.startswith(prefix) for k in ck['model']) for prefix in required):
            raise ValueError('Incomplete selected representations')
        if any(k.startswith('iqa.backbone.fusion.') and not k.startswith('iqa.backbone.fusion.semantic_proj.') for k in ck['model']):
            raise ValueError('Cross-attention checkpoint forbidden')
        del ck
        source={'version':r.VERSION,'config_sha256':r.digest(r.CONFIG/'pretrain.yaml'),
            'checkpoint':str(checkpoint),'sha256':done['checkpoint_sha256'],
            'selected_epoch':epoch,'completion_sha256':r.digest(d/'completion_manifest.json')}
        if variant in ('quality_stage','quality_first'):
            pm=r.read_json(d/'phase_manifest.json')
            if (not pm['frozen_state_unchanged'] or pm['frozen_sha256_before']!=pm['frozen_sha256_after']
                    or pm['phase']!=cfg['train']['pretraining_phase']):
                raise ValueError('Frozen phase failed audit')
        if variant=='quality_stage':
            source.update(full_checkpoint=str(d/'best_pretrain.pth'),full_sha256=done['full_checkpoint_sha256'])
            if r.digest(d/'best_pretrain.pth')!=source['full_sha256']:raise ValueError('Quality full checkpoint changed')
        if variant=='quality_first':
            quality=make_runner('quality_stage').completed('pretrain')
            if r.read_json(d/'quality_stage_source.json')!=quality or launch['source']!=quality:
                raise ValueError('Wrong quality-stage source')
            if pm['source']['sha256']!=quality['full_sha256'] or not pm['strict_full_model_transfer']:
                raise ValueError('Quality full transfer not verified')
            source['quality_stage']=quality
        return source

    def run(stage,dry_run=False):
        if stage!='pretrain':return original_run(stage,dry_run)
        cfg=r.validate(stage);d,table=r.paths(stage)
        if d.exists() or table.exists():raise FileExistsError(f'Preserve existing output: {d} / {table}')
        command=[sys.executable,'train_cm_pretrain.py','--config',str(r.CONFIG/'pretrain.yaml')]
        if dry_run:
            print(json.dumps({'command':command,'phase':cfg['train'].get('pretraining_phase','joint'),
                'selection':metric,'source':'completed quality-stage full checkpoint' if variant=='quality_first' else 'raw CLIP/ResNet initialization'},indent=2));return
        source=make_runner('quality_stage').completed('pretrain') if variant=='quality_first' else None
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; CPU training refused')
        from utils.run_metadata import collect_run_metadata
        d.mkdir(parents=True,exist_ok=False)
        launch={'version':r.VERSION,'stage':stage,'config_sha256':r.digest(r.CONFIG/'pretrain.yaml'),
            'command':command,'source':source,'runtime':collect_run_metadata(None,cfg)}
        r.write_new(d/'launch_manifest.json',launch);r.write_new(d/'run_metadata.json',launch['runtime'])
        if source is not None:r.write_new(d/'quality_stage_source.json',source)
        with (d/'console.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if process.wait()!=0:raise RuntimeError('Pretraining failed; partial results preserved')
        files=['config_snapshot.yaml','console.log','run_metadata.json']
        if variant in ('quality_stage','quality_first'):files.append('phase_manifest.json')
        if source is not None:files.append('quality_stage_source.json')
        done={'files_sha256':{n:r.digest(d/n) for n in files},'csv_sha256':r.digest(table),
            'launch_sha256':r.digest(d/'launch_manifest.json'),'checkpoint_sha256':r.digest(d/filename)}
        if variant=='quality_stage':done['full_checkpoint_sha256']=r.digest(d/'best_pretrain.pth')
        r.write_new(d/'completion_manifest.json',done);completed(stage)
        print('Verified completed pretraining:',variant,flush=True)

    def pack(phase):
        stages=STAGES if phase=='downstream' else ('pretrain',)
        for stage in stages:completed(stage)
        runners=[r]
        if variant=='quality_first':runners.insert(0,make_runner('quality_stage'))
        files=set(r.CONFIG.glob('*.yaml'))
        for rr in runners:
            rr.completed('pretrain');files.update(rr.CONFIG.glob('*.yaml'))
            for stage in (('pretrain',) if rr is not r else stages):
                d,table=rr.paths(stage);files.add(table)
                for n in ('config_snapshot.yaml','console.log','run_metadata.json','launch_manifest.json',
                    'completion_manifest.json','source_fingerprint.json','phase_manifest.json',
                    'quality_stage_source.json','data_split_manifest.json'):
                    if (d/n).is_file():files.add(d/n)
        archive=ROOT/f'{r.VERSION}_{phase}_results.tar.gz'
        with archive.open('xb') as stream:
            with tarfile.open(fileobj=stream,mode='w:gz') as tar:
                for p in sorted(files):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        print(archive)

    r.completed,r.run,r.pack=completed,run,pack
    return r


def run_pipeline(variant,action,dry_run=False):
    r=make_runner(variant)
    if action.startswith('pack'):
        if dry_run:raise ValueError('No dry-run for packaging')
        return r.pack('downstream' if action=='pack-downstream' else 'pretrain')
    quality=make_runner('quality_stage') if variant=='quality_first' else None
    tasks=[]
    if quality and action in ('quality','pretrain','all','continue'):tasks.append((quality,'pretrain'))
    if action in ('pretrain','all','continue'):tasks.append((r,'pretrain'))
    if action in ('all','continue','downstream'):tasks.extend((r,s) for s in STAGES[1:])
    elif action in STAGES[1:]:tasks.append((r,action))
    elif action not in ('pretrain','quality'):raise ValueError('Unknown action')
    if action=='quality' and not quality:raise ValueError('Only quality_first has a quality stage')
    # Preflight the whole plan before launching any stage.
    pending=[]
    for rr,s in tasks:
        rr.validate(s);d,t=rr.paths(s)
        if d.exists() or t.exists():
            rr.completed(s);print('Verified completed; skip:',rr.VERSION,s)
        else:pending.append((rr,s))
    if action=='downstream' or action in STAGES[1:]:r.completed('pretrain')
    if dry_run:
        print(json.dumps({'variant':variant,'pending':[(rr.VERSION,s) for rr,s in pending],
            'configs':[str(rr.CONFIG/f'{s}.yaml') for rr,s in pending],
            'completed':'verify and reuse; partial output refuses; no implicit resume'},indent=2));return
    for rr,s in pending:
        if s=='pretrain':rr.run(s)
        else:pipeline.run_stage(rr,s)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=VARIANTS+('check-configs',))
    p.add_argument('action',nargs='?',default='pretrain',choices=('quality','pretrain','all','continue','downstream',*STAGES[1:],'pack','pack-downstream'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT)
    if a.variant=='check-configs':
        for v in (*VARIANTS,'quality_stage'):
            rr=make_runner(v)
            for s in (('pretrain',) if v=='quality_stage' else STAGES):rr.validate(s)
        print('Validated 3 pretraining phases + 6 downstream configs; no training');return
    run_pipeline(a.variant,a.action,a.dry_run)

if __name__=='__main__':main()
