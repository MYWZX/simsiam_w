"""Read-only completion check; do not launch, recover or reconstruct results."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_align_evidence import make_runner,VARIANTS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('nss',choices=('legacy','kurt6'));a=p.parse_args()
    r=make_runner(a.nss)
    try:source=r.source()
    except Exception as e:
        print('Source unavailable/unverified:',type(e).__name__,str(e));raise SystemExit(1)
    ok=0
    for ds in r.DATASETS:
        for v in VARIANTS:
            d,t=r.paths(v,ds)
            if not d.exists() and not t.exists():print(v,ds,'pending');continue
            if not (d/'completion_manifest.json').is_file():
                print(v,ds,'incomplete or still running; preserved');continue
            try:r.verify(v,ds,None,source)
            except Exception as e:print(v,ds,'verification failed:',type(e).__name__,str(e));continue
            ok+=1;print(v,ds,'completed and verified')
    print(f'Verified: {ok}/10')
    if ok!=10:raise SystemExit(1)
if __name__=='__main__':main()
