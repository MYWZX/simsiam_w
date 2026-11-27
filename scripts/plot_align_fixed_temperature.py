"""Render validation curves for the paired fixed-temperature study; no inference."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_align_fixed_temperature import make_runner,VARIANTS
METRICS=[('val_val_quality_SROCC','Validation quality SROCC'),
         ('val_val_align_SROCC','Validation alignment SROCC'),
         ('val_val_quality_loss','Validation quality MSE'),
         ('val_val_evidence_spatial_entropy','Normalized spatial entropy'),
         ('val_val_evidence_spatial_max','Maximum spatial weight'),
         ('val_val_evidence_temperature','Fixed matching temperature')]


def render(ds,destination, *, runner=None, variants=None):
    r=make_runner() if runner is None else runner
    curves=VARIANTS if variants is None else variants
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for variant,tau in curves.items():
        _,table=r.paths(variant,ds);csv.field_size_limit(50_000_000)
        with table.open(newline='') as f:
            rows=[(int(x['epoch']),json.loads(x['metrics_json'])) for x in csv.DictReader(f) if x['event']=='epoch']
        if not rows:raise ValueError('No epoch records: '+str(table))
        for ax,(key,title) in zip(axes.flat,METRICS):
            if any(key not in m for _,m in rows):raise ValueError('Missing '+key+' in '+str(table))
            ax.plot([e for e,_ in rows],[m[key] for _,m in rows],lw=1.7,label=f'tau={tau:g}')
            ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.suptitle(ds+' / identical region-token mean route / original NSS')
    fig.savefig(destination.with_suffix('.png'),dpi=300);fig.savefig(destination.with_suffix('.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for ds in ('agiqa3k','aigciqa2023'):render(ds,a.output/ds)
if __name__=='__main__':main()
