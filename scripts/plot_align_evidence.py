"""Render logged validation diagnostics; no model inference or test-based selection."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_align_evidence import make_runner,VARIANTS


def render(nss,ds,destination):
    r=make_runner(nss)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    metrics=[('val_val_align_SROCC','Validation alignment SROCC'),
             ('val_val_quality_SROCC','Validation quality SROCC'),
             ('val_val_evidence_spatial_entropy','Spatial entropy (normalized)'),
             ('val_val_evidence_spatial_max','Maximum spatial weight'),
             ('val_val_evidence_temperature','Matching temperature'),
             ('val_val_evidence_s_align_MAE','Validation alignment MAE')]
    for variant in VARIANTS:
        _,table=r.paths(variant,ds)
        if not table.is_file():raise FileNotFoundError(table)
        with table.open(newline='') as f:
            csv.field_size_limit(50_000_000)
            rows=[(int(x['epoch']),json.loads(x['metrics_json'])) for x in csv.DictReader(f) if x['event']=='epoch']
        for ax,(key,title) in zip(axes.flat,metrics):
            points=[(ep,m[key]) for ep,m in rows if key in m]
            if not points:raise ValueError('Missing '+key+' in '+str(table))
            ax.plot(*zip(*points),label=variant,lw=1.5);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(ds+' / '+nss+' / align_evidence_v1')
    for suffix in ('.png','.pdf'):
        fig.savefig(destination.with_suffix(suffix),dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('nss',choices=('legacy','kurt6'))
    p.add_argument('--output',type=Path,required=True,help='New directory for figures; existing directories refused')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for ds in ('agiqa3k','aigciqa2023'):render(a.nss,ds,a.output/ds)
if __name__=='__main__':main()
