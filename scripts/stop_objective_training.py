"""Stop only a validated, owned, paused GPU1 objective trainer and its queue/workers."""
import argparse,csv,json,os,signal
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]


def snapshot(pid,proc_root=Path('/proc')):
    p=Path(proc_root)/str(pid);status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (p/'status').read_text().splitlines() if ':' in line}
    argv=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x]
    environment=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
    return {'pid':int(pid),'ppid':int(status['PPid']),'uid':int(status['Uid'].split()[0]),
            'state':status['State'].split()[0],'argv':argv,'environment':environment,
            'starttime':(p/'stat').read_text().rsplit(') ',1)[1].split()[19],'cwd':(p/'cwd').resolve()}


def script_path(record,name):
    script=next((Path(x) for x in record['argv'] if Path(x).name==name),None)
    return None if script is None else (script if script.is_absolute() else record['cwd']/script).resolve()


def inspect(pid,root=ROOT,proc_root=Path('/proc')):
    root=Path(root).resolve();main=snapshot(pid,proc_root)
    if main['uid']!=os.getuid() or main['state'] not in ('T','t') or main['environment'].get('CUDA_VISIBLE_DEVICES')!='1':
        raise ValueError('Target must be your paused GPU1 trainer; nothing stopped')
    if script_path(main,'train_cm_pretrain.py')!=root/'train_cm_pretrain.py':raise ValueError('Not this repository pretrainer')
    argv=main['argv'];config=Path(argv[argv.index('--config')+1]);config=(config if config.is_absolute() else main['cwd']/config).resolve()
    if config.parent.parent!=root/'configs/pretrain_objectives' or config.name!='pretrain.yaml':raise ValueError('Not an objective experiment config')
    cfg=yaml.safe_load(config.read_text());d=root/cfg['output_dir'];table=root/cfg['results_csv']
    if not (d/'last_pretrain.pth').is_file() or not table.is_file():raise ValueError('Saved checkpoint or CSV missing; preserve process for diagnosis')
    with table.open(encoding='utf-8-sig',newline='') as f:
        csv.field_size_limit(50_000_000);events=list(csv.DictReader(f))
    if any(r['event']=='final' for r in events):raise ValueError('Training already has a final record; inspect completion before stopping')
    parent=snapshot(main['ppid'],proc_root)
    if parent['uid']!=os.getuid() or parent['environment'].get('CUDA_VISIBLE_DEVICES')!='1' or script_path(parent,'run_pretrain_objectives.py')!=root/'scripts/run_pretrain_objectives.py':
        raise ValueError('Parent is not your GPU1 objective queue; nothing stopped')
    workers=[]
    for p in Path(proc_root).glob('[0-9]*'):
        try:
            candidate=snapshot(int(p.name),proc_root)
            if candidate['ppid']==main['pid']:
                if candidate['uid']!=os.getuid():raise ValueError('Unexpected worker owner')
                workers.append(candidate)
        except (OSError,UnicodeError,IndexError):continue
    return {'main':main,'parent':parent,'workers':workers,'config':str(config),
            'last_checkpoint':str(d/'last_pretrain.pth'),'completed_csv_epochs':[int(r['epoch']) for r in events if r['event']=='epoch']}


def execute(plan,proc_root=Path('/proc'),send=None):
    send=os.kill if send is None else send
    def guarded(record,sig):
        try:current=snapshot(record['pid'],proc_root)
        except FileNotFoundError:return
        if any(current[k]!=record[k] for k in ('uid','starttime','argv')):raise ValueError('PID identity changed; stop aborted')
        send(record['pid'],sig)
    # Main is paused: it cannot spawn more workers. Kill workers before main,
    # otherwise reparenting makes them invisible to a PPID-only kill command.
    guarded(plan['parent'],signal.SIGTERM)
    for worker in plan['workers']:guarded(worker,signal.SIGKILL)
    guarded(plan['main'],signal.SIGKILL)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pid',type=int);p.add_argument('--execute',action='store_true')
    a=p.parse_args();plan=inspect(a.pid)
    print(json.dumps({k:v for k,v in plan.items() if k not in ('main','parent','workers')}|{'trainer_pid':plan['main']['pid'],'queue_pid':plan['parent']['pid'],'worker_pids':[x['pid'] for x in plan['workers']]},indent=2),flush=True)
    if a.execute:
        execute(plan);print('Stopped only the validated GPU1 objective queue/trainer/workers; saved files unchanged',flush=True)
    else:print('Inspection only; use --execute to stop these processes',flush=True)

if __name__=='__main__':main()
