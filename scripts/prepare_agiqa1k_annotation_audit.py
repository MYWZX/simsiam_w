"""Audit paired 1K annotations; optionally write a separate unambiguous subset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True)
    p.add_argument('--xlsx', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--write-subset', action='store_true')
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Use a fresh output directory; original files are never overwritten')
    csv = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    excel = pd.read_excel(args.xlsx, sheet_name=0)
    if not {'name','MOS','prompt','answer'}.issubset(csv.columns):
        raise ValueError('CSV schema mismatch')
    if not {'Image','MOS','Prompt'}.issubset(excel.columns):
        raise ValueError('Excel schema mismatch')
    if len(csv) != len(excel):
        raise ValueError('CSV and Excel have different row counts; manual reconciliation needed')
    names_equal = csv['name'].eq(excel['Image'].astype(str)).to_numpy()
    prompts_equal = csv['prompt'].str.strip().eq(excel['Prompt'].astype(str).str.strip()).to_numpy()
    q_csv = pd.to_numeric(csv['MOS'], errors='raise').to_numpy(dtype=float)
    q_excel = pd.to_numeric(excel['MOS'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(q_csv).all() or not np.isfinite(q_excel).all():
        raise ValueError('Nonfinite MOS')
    diff = np.abs(q_csv-q_excel)
    pairs_match = bool(names_equal.all() and prompts_equal.all() and (diff <= .000050001).all())
    duplicates = sorted(set(csv.loc[csv['name'].duplicated(keep=False), 'name']) |
                        set(excel.loc[excel['Image'].duplicated(keep=False), 'Image'].astype(str)))
    groups = []
    for name in duplicates:
        ci = csv.index[csv['name'] == name].tolist()
        xi = excel.index[excel['Image'].astype(str) == name].tolist()
        groups.append({'name': name, 'csv_row_indices_zero_based': ci,
            'excel_row_indices_zero_based': xi,
            'csv_rows': csv.loc[ci].to_dict('records'),
            'excel_rows': excel.loc[xi].to_dict('records')})
    keep = ~csv['name'].isin(duplicates)
    report = {'source_csv_sha256': sha256(args.csv), 'source_xlsx_sha256': sha256(args.xlsx),
        'csv_rows': len(csv), 'excel_rows': len(excel),
        'csv_unique_names': int(csv['name'].nunique()), 'excel_unique_names': int(excel['Image'].nunique()),
        'name_mismatch_count': int((~names_equal).sum()),
        'trimmed_prompt_mismatch_count': int((~prompts_equal).sum()),
        'max_absolute_mos_difference': float(diff.max()),
        'paired_rows_agree_with_4_decimal_csv_precision': pairs_match,
        'duplicate_groups': groups,
        'candidate_policy': 'exclude every row of every duplicated name; no label/name/answer reassignment',
        'candidate_row_count': int(keep.sum()),
        'kept_original_csv_row_indices': csv.index[keep].tolist(),
        'note': 'Subset is a new nonstandard protocol; it does not repair or certify the full 1080-image benchmark.'}
    if args.write_subset and not pairs_match:
        raise ValueError('Paired annotations differ beyond rounding; no subset written')
    out.mkdir(parents=True)
    if args.write_subset:
        subset = csv.loc[keep].copy()
        if not subset['name'].is_unique:
            raise ValueError('Candidate still has repeated names')
        path = out/'AIGC-1K_answer_unambiguous1078.csv'
        # The output filename must not misdescribe a different input dataset.
        if len(subset) != 1078:
            path = out/f'AIGC-1K_answer_unambiguous{len(subset)}.csv'
        subset.to_csv(path, index=False)
        report['candidate_csv'] = str(path)
        report['candidate_sha256'] = sha256(path)
    (out/'annotation_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: value for key,value in report.items()
                      if key not in ('duplicate_groups','kept_original_csv_row_indices')},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
