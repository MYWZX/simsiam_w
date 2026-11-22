"""Read-only branch diagnostics at the existing Quality-selected epoch."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def audit_csv(path):
    path=Path(path);csv.field_size_limit(50_000_000)
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
    epochs=[r for r in rows if r['event']=='epoch']
    if (len(finals)!=1 or finals[0]['last_epoch']!=60 or
            [int(r['epoch']) for r in epochs]!=list(range(1,61)) or
            finals[0]['selection_metric']!='val_quality_SROCC'):
        raise ValueError('Require one completed 60-epoch Quality-selected run')
    key=next(k for k in ('val_val_quality_SROCC','val_quality_SROCC') if k in json.loads(epochs[0]['metrics_json']))
    selected=max(epochs,key=lambda r:json.loads(r['metrics_json'])[key])
    epoch=int(selected['epoch'])
    if epoch!=finals[0]['best_epoch']:raise ValueError('Wrong recorded selection epoch')
    result={'csv':str(path),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_epoch':epoch}
    for event,prefix in [('best_val','val'),('best_test','test')]:
        records=[r for r in rows if r['event']==event]
        if len(records)!=1 or int(records[0]['epoch'])!=epoch:raise ValueError('Branch metrics must come from the same selected epoch')
        m=json.loads(records[0]['metrics_json'])
        if event=='best_test' and m!=finals[0]['best_test']:raise ValueError('Test event differs from final record')
        values={branch:{metric:m[f'{prefix}_align{suffix}_{metric}'] for metric in ('SROCC','PLCC')}
                for branch,suffix in [('mixed',''),('global','_global'),('local','_patch')]}
        gate=m[f'{prefix}_align_gate']
        if not 0<=gate<=1 or any(not math.isfinite(x) for b in values.values() for x in b.values()):
            raise ValueError('Invalid branch metrics/gate')
        result[prefix]={'branches':values,'mean_global_gate':gate,'mean_local_gate':1-gate,
            'mixed_minus_global_srocc':values['mixed']['SROCC']-values['global']['SROCC'],
            'mixed_minus_local_srocc':values['mixed']['SROCC']-values['local']['SROCC']}
    result['note']='Gate is a sample-mean coefficient, not per-sample uniform weighting or importance attribution. No alternate epoch or test-based branch selection.'
    return result


def reference_audit(evidence_root=ROOT):
    import yaml
    reference=json.loads((ROOT/'docs/current_tuning_reference.json').read_text());reports={}
    for ds in ('agiqa3k','aigciqa2023'):
        rec=next(r for r in reference['results'] if r['dataset']==ds)
        cfg=yaml.safe_load((ROOT/rec['config']).read_text());report=audit_csv(Path(evidence_root)/cfg['results_csv'])
        if report['csv_sha256']!=rec['csv_sha256']:raise ValueError('Control CSV differs from pinned reference')
        reports[ds]=report
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,action='append',help='Explicit CSVs; otherwise audit pinned baselines')
    p.add_argument('--evidence-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,help='Optional new JSON file; never overwrite')
    a=p.parse_args();report=[audit_csv(x) for x in a.csv] if a.csv else reference_audit(a.evidence_root)
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if a.output:
        with a.output.open('x') as f:f.write(text+'\n')
    print(text)

if __name__=='__main__':main()
