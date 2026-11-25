"""Audit/recover only the two new 40+20 outputs; no baseline/control files moved."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_align_40plus20 as study


def make_recovery():
    # Isolate the existing conservative recovery engine instead of mutating
    # protected_local's module globals or changing its historical scope.
    spec=importlib.util.spec_from_file_location('_align4020_recovery',ROOT/'scripts/recover_protected_local.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    runner=study.make_runner()
    r.JOB_SCRIPTS=r.JOB_SCRIPTS|{'run_align_40plus20.py','run_align_40plus20.sh'}
    r.protocol=SimpleNamespace(ROOT=study.ROOT,VERSION=study.VERSION,source=runner.source,
        verify=lambda v,ds:runner.verify(v,ds,None,runner.source()))
    r.stages=lambda:[(study.VARIANT,ds) for ds in runner.DATASETS]
    def paths(v,ds):
        if v!=study.VARIANT:raise ValueError('Recovery is restricted to the new 40+20 runs')
        return runner.config(v,ds),*runner.paths(v,ds)
    r.stage_paths=paths
    return r

if __name__=='__main__':make_recovery().main()
