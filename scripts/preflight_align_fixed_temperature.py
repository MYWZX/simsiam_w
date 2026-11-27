"""Read-only source/control/output preflight; never launch or alter experiments."""
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_align_fixed_temperature import make_runner


def active_jobs(proc_root=Path('/proc')):
    jobs=[]
    for p in proc_root.glob('[0-9]*'):
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            if any(Path(a).name=='run_align_fixed_temperature.py' for a in args):jobs.append(p.name)
            elif any(Path(a).name=='train_cm_iqa.py' for a in args) and '--config' in args:
                config=Path(args[args.index('--config')+1])
                if not config.is_absolute():config=(p/'cwd').resolve()/config
                if 'align_fixed_temperature_v1' in str(config):jobs.append(p.name)
        except (OSError,ValueError):continue
    return jobs


def preflight():
    jobs=active_jobs()
    if jobs:raise RuntimeError('Fixed-temperature jobs already active: '+','.join(jobs))
    r=make_runner();fp=r.source();pending=[]
    for ds in r.DATASETS:
        r.control(ds,None,fp)
        for v in r.VARIANTS:
            d,t=r.paths(v,ds)
            if d.exists() or t.exists():r.verify(v,ds,None,fp);print('Verified completed; skip:',v,ds)
            else:pending.append((v,ds));print('Pending:',v,ds)
    return pending
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    if preflight() and args.require_complete:raise SystemExit(1)
