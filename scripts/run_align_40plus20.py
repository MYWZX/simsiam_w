"""Two new 40+20 downstream runs; reuse validated aprobe/aregionword 60-epoch controls."""
import importlib.util
import io
import json
import math
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from scripts import run_align_representation as representation
from utils.fair_40plus20 import OPTION,VERSION as PHASE_VERSION,PREFIXES
from utils.task_gradient_audit import audit_probe_records
VERSION='projection_only_v1_v2_align_40plus20_v1'
VARIANT='a40local20'


def peer_runner():
    r=representation.make_runner();r.ROOT=ROOT
    return r


def config(ds):
    c=peer_runner().config('aregionword',ds)
    name=f'{ds}_{VERSION}_train42_split42'
    c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
    c['train']['fair_40plus20']=dict(OPTION)
    return c


def verify_lrs(rows,groups):
    for row in rows:
        if row['event']!='epoch':continue
        epoch=int(row['epoch']);m=json.loads(row['metrics_json'])
        for group in groups:
            expected=1e-6+(group['lr']-1e-6)*(1+math.cos(math.pi*epoch/60))/2
            if not math.isclose(m['lr_'+group['name']],expected,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Noncontinuous 60-epoch cosine schedule')


def verify_step_budget(rows,split):
    batches=split['train_batches']
    for row in rows:
        if row['event']!='epoch':continue
        m=json.loads(row['metrics_json'])
        if (m['train_processed_steps']!=batches or m['train_optimizer_updates']<0
                or m['train_amp_skipped_steps']<0
                or m['train_optimizer_updates']+m['train_amp_skipped_steps']!=batches):
            raise ValueError('Actual processed/update/AMP-skip budget differs')


def verify_phase(proof,rows,groups):
    epochs=[(int(r['epoch']),json.loads(r['metrics_json'])) for r in rows if r['event']=='epoch']
    if (proof['version']!=PHASE_VERSION or proof['boundary_epoch']!=40 or proof['start_epoch']!=41
            or proof['last_epoch']!=60 or proof['joint_epochs']!=40 or proof['local_epochs']!=20
            or proof['source_state']!='current_epoch40_not_validation_selected'
            or proof['optimizer_recreated'] or proof['scheduler_restarted'] or proof['warmup_used']
            or proof['scheduler_T_max']!=60 or proof['scheduler_last_epoch']!=60
            or proof['frozen_before']!=proof['frozen_after'] or len(proof['frozen_before'])!=64):
        raise ValueError('Wrong 40+20 boundary/frozen-state/budget evidence')
    names=proof['trainable_parameters']
    if not names or any(not name.startswith(PREFIXES) for name in names):raise ValueError('Wrong local trainable scope')
    if any(not any(name.startswith(prefix) for name in names) for prefix in PREFIXES):raise ValueError('Missing local module')
    boundary=dict(epochs)[40]
    for key,value in proof['boundary_validation'].items():
        if value!=boundary['val_'+key]:raise ValueError('Boundary is not the actual epoch-40 validation state')
        for epoch,m in epochs[40:]:
            if not math.isclose(m['val_'+key],value,rel_tol=0,abs_tol=1e-6):raise ValueError('Frozen validation outputs changed')
    transitions=[r for r in rows if r['event']=='fair_phase_transition']
    if len(transitions)!=1 or int(transitions[0]['epoch'])!=41:raise ValueError('Wrong phase transition history')
    transition=json.loads(transitions[0]['metrics_json'])
    if any(proof[k]!=v for k,v in transition.items()):raise ValueError('Transition/final evidence differs')
    for group in groups:
        expected=1e-6+(group['lr']-1e-6)*(1+math.cos(math.pi*40/60))/2
        if not math.isclose(proof['lrs_at_transition'][group['name']],expected,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Phase-two LR was reset')
        if not math.isclose(proof['lrs_final'][group['name']],1e-6,abs_tol=1e-12):raise ValueError('Wrong final LR')
    updates1=sum(m['train_optimizer_updates'] for e,m in epochs if e<=40)
    updates2=sum(m['train_optimizer_updates'] for e,m in epochs if e>40)
    if set(proof['local_steps_before'])!=set(names) or set(proof['local_steps_after'])!=set(names):raise ValueError('Missing local Adam state evidence')
    for name in names:
        if proof['local_steps_before'][name]!=updates1 or proof['local_steps_after'][name]!=updates1+updates2:
            raise ValueError('Local Adam state was reset or update counts differ')
    verify_lrs(rows,groups)


def make_runner():
    spec=importlib.util.spec_from_file_location('_align_40plus20_protocol',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.ROOT=ROOT
    r.__doc__=__doc__
    r.VERSION=VERSION;r.VARIANTS={VARIANT:('freeze',41)};r.QUEUES={}
    r.PROTOCOL_DOC='docs/align_40plus20.md'
    r.CONFIG_SUMMARY='60 total epochs = 40 joint + 20 frozen-local; continuous optimizer/cosine; existing 60-epoch controls reused'
    r.FILES=(*r.FILES,'task_gradient_probe.jsonl','fair_phase_manifest.json')
    def cfg(v,ds,split=None):
        if v!=VARIANT or split not in (None,42):raise ValueError('Invalid 40+20 variant/split')
        return config(ds)
    r.config=cfg;r.verify_selection_records=representation.verify_comparison
    original_verify=r.verify
    def verify(v,ds,split,source):
        original_verify(v,ds,split,source)
        import csv
        d,table=r.paths(v,ds)
        with table.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        meta=r.base.read_json(d/'run_metadata.json')
        verify_phase(r.base.read_json(d/'fair_phase_manifest.json'),rows,meta['optimizer_initialization'])
        params=meta.get('region_word_initialization',[])
        if (meta.get('region_word_version')!='clip_region_word_v1' or {p['name']:p['shape'] for p in params}!={
                'align_region_word.image_projection.weight':[256,1024],'align_region_word.text_projection.weight':[256,768]}):
            raise ValueError('Missing explicit random region initialization')
        if any(p['initialization']!='new_random' or len(p['sha256'])!=64 for p in params):raise ValueError('Foreign local pretraining initialization')
        probes=[json.loads(line) for line in (d/'task_gradient_probe.jsonl').read_text().splitlines()]
        audit_probe_records(probes,'aregionword')
        peers=peer_runner()
        split_record=r.base.read_json(d/'data_split_manifest.json')
        verify_step_budget(rows,split_record)
        for peer_variant in peers.VARIANTS:
            pd,_=peers.paths(peer_variant,ds)
            if (pd/'data_split_manifest.json').exists():
                actual=r.base.read_json(pd/'data_split_manifest.json')
                if any(actual[k]!=split_record[k] for k in ('indices','dataset_ids_sha256')):
                    raise ValueError('40+20/control memberships differ')
        if any(group['status']!='no_trainable_parameters' for row in probes if row['epoch']==60 for group in row['groups'].values()):
            raise ValueError('Shared encoders/projection remained trainable in phase two')
    r.verify=verify
    original_control=r.control
    def control(ds,split,source):
        result=original_control(ds,split,source)
        peers=peer_runner()
        for v in peers.VARIANTS:
            peers.verify(v,ds,None,source)
            import csv
            pd,pt=peers.paths(v,ds)
            with pt.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
            verify_lrs(records,peers.base.read_json(pd/'run_metadata.json')['optimizer_initialization'])
            verify_step_budget(records,peers.base.read_json(pd/'data_split_manifest.json'))
        return result
    r.control=control
    def pack():
        source=r.source();peers=peer_runner()
        files={ROOT/'docs/align_40plus20.md',ROOT/'docs/current_tuning_reference.json',ROOT/'docs/current_tuning_reference.md'}
        files.update(ROOT/f'configs/current_tuning_reference/{ds}.yaml' for ds in r.DATASETS)
        for journal in (ROOT/'run_logs').glob('protected_local_recovery_*.json'):
            recovery=r.base.read_json(journal)
            if recovery.get('version')==VERSION and recovery.get('upstream',{}).get('sha256')==source['sha256']:
                files.add(journal)
        protocol={'version':VERSION,'total_epochs':60,'joint_epochs':40,'local_epochs':20,
                  'controls':['aprobe','aregionword'],'selection_policies':config(r.DATASETS[0])['train']['selection_policies'],
                  'source':source,'new_run_count':2,'warmup_used':False}
        for ds in r.DATASETS:
            cd,ct=r.control(ds,None,source);files.add(ct)
            files.update(cd/n for n in r.FILES if (cd/n).is_file())
            for v in peers.VARIANTS:
                pd,pt=peers.paths(v,ds);files.add(pt);files.update(pd/n for n in peers.FILES)
            verify(VARIANT,ds,None,source);d,t=r.paths(VARIANT,ds);files.add(t);files.update(d/n for n in r.FILES)
            expected=peers.base.read_json(d/'data_split_manifest.json')
            for v in peers.VARIANTS:
                pd,_=peers.paths(v,ds);actual=peers.base.read_json(pd/'data_split_manifest.json')
                if any(expected[k]!=actual[k] for k in ('indices','dataset_ids_sha256')):raise ValueError('40+20/control memberships differ')
        archive=ROOT/(VERSION+'_results.tar.gz')
        with archive.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in sorted(files):tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
            data=json.dumps(protocol,indent=2).encode();info=tarfile.TarInfo('comparison_protocol.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
        print(archive)
    r.pack=pack
    return r

def main():
    r=make_runner()
    if len(sys.argv)==2 and sys.argv[1]=='check-inputs':
        for ds in r.DATASETS:
            d,t=r.paths(VARIANT,ds)
            if (d.exists() or t.exists()) and not (d/'completion_manifest.json').is_file():
                from scripts.recover_align_40plus20 import make_recovery
                jobs=make_recovery().active_jobs()
                raise RuntimeError('Existing 40+20 output has no completion manifest: '+str(d)
                    +'; active jobs='+json.dumps(jobs)
                    +'. Inspect with scripts/recover_align_40plus20.py. '
                    'If all jobs stopped, preserve incomplete outputs via '
                    'bash scripts/launch_align_40plus20_background.sh --recover-incomplete. '
                    'No new jobs launched or files modified.')
        source=r.source()
        for ds in r.DATASETS:
            r.control(ds,None,source)
            d,t=r.paths(VARIANT,ds)
            if d.exists() or t.exists():r.verify(VARIANT,ds,None,source)
            print('Verified inputs/control results:',ds,flush=True)
        return
    r.main()

if __name__=='__main__':main()
