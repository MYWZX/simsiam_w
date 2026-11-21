"""Read-only validation-epoch audit. Does not infer unrecorded test metrics."""
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
    epochs=[r for r in rows if r['event']=='epoch']
    finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
    if len(finals)!=1 or finals[0]['last_epoch']!=60 or [int(r['epoch']) for r in epochs]!=list(range(1,61)):
        raise ValueError('Require a complete 60-epoch CSV with one final event')
    if finals[0]['selection_metric']!='val_quality_SROCC':raise ValueError('Expected baseline Quality selection')
    records=[(int(r['epoch']),json.loads(r['metrics_json'])) for r in epochs]
    first=records[0][1]
    q=next(k for k in ('val_val_quality_SROCC','val_quality_SROCC') if k in first)
    a=next(k for k in ('val_val_align_SROCC','val_align_SROCC') if k in first)
    if any(not math.isfinite(m[k]) for _,m in records for k in (q,a)):
        raise ValueError('Nonfinite validation metric')
    qe,qm=max(records,key=lambda x:x[1][q]);ae,am=max(records,key=lambda x:x[1][a])
    if qe!=finals[0]['best_epoch']:raise ValueError('Recorded selection differs from validation maximum')
    return {'csv':str(path),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'quality_selected_epoch':qe,'align_validation_best_epoch':ae,
        'quality_selected_val_quality_srocc':qm[q],
        'quality_selected_val_align_srocc':qm[a],
        'align_best_val_align_srocc':am[a],'align_best_val_quality_srocc':am[q],
        'val_align_gap':am[a]-qm[a],
        'note':'Validation diagnosis only; test metrics at the other epoch are not inferred; selection unchanged'}


def reference_audit(evidence_root=ROOT):
    import yaml
    reference=json.loads((ROOT/'docs/current_tuning_reference.json').read_text());result={}
    for ds in ('agiqa3k','aigciqa2023'):
        record=next(x for x in reference['results'] if x['dataset']==ds)
        cfg=yaml.safe_load((ROOT/record['config']).read_text())
        report=audit_csv(Path(evidence_root)/cfg['results_csv'])
        if report['csv_sha256']!=record['csv_sha256']:raise ValueError('Control CSV is not the pinned reference')
        result[ds]=report
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,action='append',help='Explicit CSVs, otherwise use pinned 3K/2023 baseline')
    p.add_argument('--evidence-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,help='Optional new JSON file; existing file is preserved')
    a=p.parse_args()
    report=[audit_csv(x) for x in a.csv] if a.csv else reference_audit(a.evidence_root)
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if a.output:
        with a.output.open('x') as f:f.write(text+'\n')
    print(text)

if __name__=='__main__':main()
