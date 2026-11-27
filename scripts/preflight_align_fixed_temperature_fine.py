"""Read-only preflight of .1 controls, sources and the four new fine-temperature runs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import preflight_align_fixed_temperature as common
from scripts.run_align_fixed_temperature_fine import make_runner


def active_jobs(proc_root=Path('/proc')):
    return common.active_jobs(proc_root,runner_script='run_align_fixed_temperature_fine.py',
                              profile='align_fixed_temperature_fine_v1')


def preflight():return common.preflight(make_runner(),active_jobs)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    if preflight() and a.require_complete:raise SystemExit(1)
