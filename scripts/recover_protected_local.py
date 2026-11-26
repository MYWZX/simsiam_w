"""Read-only audit by default; preserve incomplete outputs before phase-level restart.

This is NOT optimizer/RNG checkpoint resume. Completed or ambiguously completed
runs are never moved. No training processes are started or stopped by this tool.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import uuid
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_protected_local as protocol

JOB_SCRIPTS={'run_protected_local.py','run_protected_local.sh','train_protected_local.py','train_cm_iqa.py'}


def active_jobs(proc_root=Path('/proc')):
    result=[]
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            args=[x.decode(errors='replace') for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
            if any(Path(arg).name in JOB_SCRIPTS for arg in args):
                result.append(dict(pid=int(entry.name),args=args))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result


def ensure_idle():
    jobs=active_jobs()
    if jobs:raise RuntimeError('Protected training/queue processes are still active; no files moved: '+json.dumps(jobs))


def stages():
    return protocol.plan('all',protocol.DATASETS)


def stage_paths(variant,ds):
    if variant=='foundation':
        c=protocol.foundation_config(ds)
        return c,protocol.foundation_path(ds).parent,protocol.ROOT/c['results_csv']
    return (protocol.config(variant,ds),*protocol.paths(variant,ds))


def csv_progress(path):
    result=dict(last_recorded_epoch=0,final_seen=False,read_error=None)
    if not path.is_file():return result
    try:
        csv.field_size_limit(50_000_000)
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('event')=='final':result['final_seen']=True
                if row.get('event')=='epoch':
                    json.loads(row['metrics_json'])
                    result['last_recorded_epoch']=max(result['last_recorded_epoch'],int(row['epoch']))
    except (ValueError,csv.Error,KeyError) as error:result['read_error']=str(error)
    return result


def audit():
    results=[]
    for variant,ds in stages():
        c,d,table=stage_paths(variant,ds)
        item=dict(variant=variant,dataset=ds,directory=str(d),csv=str(table))
        if d.is_symlink() or table.is_symlink():
            raise ValueError('Refusing symlink outputs: '+str(d))
        if not d.exists() and not table.exists():
            item['status']='pending';results.append(item);continue
        launch=d/'launch_config.yaml'
        if not launch.is_file() or yaml.safe_load(launch.read_text())!=c:
            raise ValueError('Unrecognized/modified launch config; preserve and inspect: '+str(d))
        item.update(csv_progress(table))
        if (d/'completion_manifest.json').is_file():
            protocol.verify_foundation(ds) if variant=='foundation' else protocol.verify(variant,ds)
            item['status']='completed'
        elif (item['final_seen'] or (d/'selected_checkpoint_manifest.json').exists()
              or (d/'local_checkpoint_manifest.json').exists()
              or (d/'best_selected_downstream.pth').exists() or (d/'best_local.pth').exists()):
            item['status']='unsealed'
        else:item['status']='partial'
        results.append(item)
    return results


def preserve_interrupted():
    ensure_idle()
    # Check all source weights and every completed run BEFORE any rename.
    upstream=protocol.source()
    entries=audit()
    ambiguous=[x for x in entries if x['status']=='unsealed']
    if ambiguous:
        raise RuntimeError('Possibly completed runs lack completion manifests; no files moved. Inspect first: '+json.dumps(ambiguous))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
    moves=[]
    for item in entries:
        if item['status']!='partial':continue
        for key in ('directory','csv'):
            src=Path(item[key])
            if src.exists():moves.append(dict(source=str(src),destination=str(src)+'.interrupted_'+stamp,moved=False))
    if not moves:
        print('No incomplete runs to preserve; completed runs retained.')
        return entries
    for move in moves:
        if os.path.lexists(move['destination']):raise FileExistsError(move['destination'])
    logs=protocol.ROOT/'run_logs';logs.mkdir(exist_ok=True)
    prefix=globals().get('RECOVERY_PREFIX','protected_local')
    journal=logs/f'{prefix}_recovery_{stamp}.json'
    record=dict(format='protected_local_recovery_v1',version=protocol.VERSION,upstream=upstream,stages=entries,moves=moves,
                complete=False,note='Incomplete phases restart from their configured source; no optimizer/RNG resume.')
    with journal.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    for move in moves:
        ensure_idle()
        os.rename(move['source'],move['destination'])
        move['moved']=True
        journal.write_text(json.dumps(record,ensure_ascii=False,indent=2))
        print('Preserved:',move['destination'])
    record['complete']=True;journal.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print('Recovery record:',journal)
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Preserve verified incomplete outputs by renaming; never delete, resume, or start training')
    a=p.parse_args()
    if a.execute:entries=preserve_interrupted()
    else:
        jobs=active_jobs()
        if jobs:print('Active processes:',json.dumps(jobs))
        entries=audit()
    for entry in entries:
        print(entry['variant'],entry['dataset'],entry['status'],
              'last recorded epoch='+str(entry.get('last_recorded_epoch','-')))

if __name__=='__main__':main()
