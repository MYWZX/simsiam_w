"""Preserve partial downstream evidence runs; completed runs are verified and kept.

Pretraining is outside this adapter's scope. It has full-state resume via the
separate kurtosis runner. Downstream restarts from its original source at epoch1.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_align_evidence as study


def make_recovery(nss='legacy'):
    runner=study.make_runner(nss)
    spec=importlib.util.spec_from_file_location('_evidence_recovery_'+nss,ROOT/'scripts/recover_protected_local.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.RECOVERY_PREFIX='align_evidence_'+nss
    r.JOB_SCRIPTS=r.JOB_SCRIPTS|{'run_align_evidence.py','run_align_evidence.sh',
        'run_evidence_chain.sh','run_nss_kurtosis.py','run_nss_kurtosis.sh','train_cm_pretrain.py'}
    r.protocol=SimpleNamespace(ROOT=runner.ROOT,VERSION=runner.VERSION,source=runner.source,
        verify=lambda v,ds:runner.verify(v,ds,None,runner.source()))
    r.stages=lambda:[(v,ds) for ds in runner.DATASETS for v in runner.VARIANTS]
    r.stage_paths=lambda v,ds:(runner.config(v,ds),*runner.paths(v,ds))
    return r

if __name__=='__main__':
    # Keep --execute semantics from the conservative existing recovery engine.
    arm='legacy'
    if len(sys.argv)>1 and sys.argv[1] in ('legacy','kurt6'):
        arm=sys.argv.pop(1)
    make_recovery(arm).main()
