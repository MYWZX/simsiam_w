"""Read-only report of duplicate 1K CSV names and nearby image filenames."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='outputs/agiqa1k_allmerged_noce_noneg_dev_seed42/config_snapshot.yaml')
    parser.add_argument('--output', default='outputs/agiqa1k_csv_conflict.json')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.snapshot).read_text())['dataset']
    if cfg['type'] != 'agiqa1k_iqa':
        parser.error('Expected an AGIQA-1K effective snapshot')
    root = Path(cfg['root_dir'])
    csv_path = root/cfg.get('csv_file', 'AIGC-1K_answer.csv')
    image_root = root/cfg.get('image_dir', 'file')
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    duplicated = frame[frame['name'].duplicated(keep=False)]
    files = sorted(p.name for p in image_root.iterdir() if p.is_file())
    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    image_files = sorted(p for p in image_root.rglob('*')
                         if p.is_file() and p.suffix.lower() in image_suffixes)
    referenced_paths = set()
    missing_names = []
    for name in sorted(set(frame['name'])):
        candidates = [image_root/name, root/name, root/'file'/name,
                      root/'image'/name, root/'images'/name]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            missing_names.append(name)
        else:
            referenced_paths.add(found.resolve())
    unreferenced = [str(p.relative_to(image_root)) for p in image_files
                    if p.resolve() not in referenced_paths]
    annotation_tables = sorted(str(p.relative_to(root)) for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {'.csv', '.xlsx', '.xls', '.mat'})
    records = []
    for name, group in duplicated.groupby('name', sort=True):
        stem = Path(name).stem
        prefix = stem.rsplit('_', 1)[0] + '_'
        nearby = [f for f in files if f.startswith(prefix)]
        related_rows = frame[frame['name'].isin(nearby)]
        records.append({'name': name, 'csv_row_indices_zero_based': group.index.tolist(),
                        'rows': group.to_dict('records'), 'nearby_existing_image_files': nearby,
                        'nearby_csv_rows': related_rows.to_dict('records')})
    report = {'snapshot': args.snapshot, 'csv_path': str(csv_path.resolve()),
              'csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
              'csv_rows': len(frame), 'unique_csv_names': frame['name'].nunique(),
              'image_directory_file_count': len(files),
              'image_file_count_recursive': len(image_files),
              'csv_names_with_missing_image': missing_names,
              'image_files_not_referenced_by_csv': unreferenced,
              'annotation_tables_in_dataset_root': annotation_tables,
              'duplicate_groups': records,
              'note': 'Read-only; candidate filenames do not establish the correct MOS mapping.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'CSV rows={len(frame)}, unique names={report["unique_csv_names"]}, duplicate groups={len(records)}')
    print(f'Saved: {output}')

if __name__ == '__main__':
    main()
