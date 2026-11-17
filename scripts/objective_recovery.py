"""Original six-channel objective checkpoints: explicit recovery and GPU handoff."""
import os
from pathlib import Path
import time
from scripts import nss_recovery


def install(r):
    def validate(cfg,state):
        ng=cfg['model']['nss_text_gate']
        if ng.get('descriptor','legacy')!='legacy' or ng['nss_dim']!=6 or ng.get('pathway','both')!='both':
            raise ValueError('Objective recovery requires the original six-channel NSS baseline')
        if any(k.endswith(('nss_descriptor_signature','nss_pathway_signature')) for k in state):
            raise ValueError('A versioned NSS checkpoint cannot recover an original objective experiment')
        h=int(ng.get('gate_hidden_dim',cfg['model']['heads'].get('mlp_hidden_dim',256)))
        e=int(cfg['model']['fusion']['embed_dim'])
        shapes={'nss_proj.0.weight':(h,6),'nss_proj.0.bias':(h,),
                'nss_proj.2.weight':(e,h),'nss_proj.2.bias':(e,),
                'nss_gate.0.weight':(h,e+6),'nss_gate.0.bias':(h,),
                'nss_gate.3.weight':(1,h),'nss_gate.3.bias':(1,)}
        for key,shape in shapes.items():
            if 'iqa.'+key not in state or tuple(state['iqa.'+key].shape)!=shape:
                raise ValueError('Original NSS checkpoint missing/mismatched: '+key)
    r.validate_recovery_descriptor=validate
    nss_recovery.install(r)


def recover_queue(study,queue,dry_run=False):
    if queue not in study.QUEUES:raise ValueError('Unknown objective queue')
    plans=[]
    # Inspect the whole remaining queue before starting any expensive work.
    for variant in study.QUEUES[queue]:
        r=study.make_runner(variant);interrupted=False
        for stage in study.STAGES:
            r.validate(stage);d,table=r.paths(stage)
            if not (d.exists() or table.exists()):continue
            if stage=='pretrain' and not (d/'completion_manifest.json').exists():
                interrupted=True;nss_recovery.preflight(r,variant)
            else:
                if interrupted:raise ValueError('Downstream files exist for an incomplete source; preserve for diagnosis')
                r.completed(stage)
        plans.append((variant,r,interrupted))
        print('Recovery plan:',variant,'resume pretraining' if interrupted else 'verify/reuse completed stages; run unstarted stages',flush=True)
    if dry_run:return
    for variant,r,interrupted in plans:
        if interrupted:nss_recovery.recover(r,variant)
        study.pipeline.run_pipeline(r,'continue')


def process_identity(pid):
    try:
        proc=Path('/proc')/str(pid)
        # pid reuse must not extend or redirect the wait to another process.
        fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
        return None if fields[0] in ('Z','X') else fields[19]
    except (OSError,IndexError):return None


def low_queue_processes(root):
    found=[];root=Path(root).resolve()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid!=os.getuid():continue
            argv=[a for a in (proc/'cmdline').read_bytes().decode().split('\0') if a]
            i=next((i for i,a in enumerate(argv) if Path(a).name=='run_pretrain_objectives.py'),None)
            if i is None or argv[i+1:i+3]!=['low','all']:continue
            script=Path(argv[i])
            if not script.is_absolute():script=(proc/'cwd').resolve()/script
            if script.resolve()!=root/'scripts/run_pretrain_objectives.py':continue
            pid=int(proc.name);identity=process_identity(pid)
            if identity is not None:found.append((pid,identity))
        except (OSError,UnicodeError):continue
    return found


def wait_for_low(study,dry_run=False):
    jobs=low_queue_processes(study.ROOT)
    if len(jobs)>1:raise RuntimeError('Multiple low queues found; no automatic handoff')
    if dry_run:
        print('Would wait for low queue:',jobs,'then verify all 20 low-stage completions; no GPU training',flush=True)
        return
    if jobs:
        pid,identity=jobs[0];print(f'Waiting for GPU0 low queue PID={pid}; no training launched yet',flush=True)
        while process_identity(pid)==identity:time.sleep(15)
    # Process exit alone is insufficient: a failure must not trigger new runs.
    for variant in study.QUEUES['low']:
        r=study.make_runner(variant)
        for stage in study.STAGES:
            try:r.completed(stage)
            except Exception as exc:
                raise RuntimeError(f'Low queue has not successfully completed: {variant}/{stage}; GPU0 handoff refused') from exc
    print('Verified all five low experiments including downstreams; continue high queue on GPU0',flush=True)
