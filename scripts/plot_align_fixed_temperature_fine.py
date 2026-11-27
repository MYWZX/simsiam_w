"""Render .05/.1/.2 logged validation curves; .1 is read from its existing run."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import plot_align_fixed_temperature as common
from scripts.run_align_fixed_temperature_fine import make_runner,previous_runner


def render(ds,destination):
    r=make_runner();old=previous_runner(r.ROOT)
    reader=SimpleNamespace(paths=lambda v,ds:old.paths(v,ds) if v=='tau010' else r.paths(v,ds))
    common.render(ds,destination,runner=reader,variants={'tau005':.05,'tau010':.1,'tau020':.2})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for ds in ('agiqa3k','aigciqa2023'):render(ds,a.output/ds)
if __name__=='__main__':main()
