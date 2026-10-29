"""Read-only batch overlap diagnosis for completed downstream dev snapshots."""
import argparse
import glob
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from datasets import build_dataloaders
from utils.refit_protocol import split_overlap_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='append', help='Specific effective snapshot; may repeat')
    parser.add_argument('--output', default='outputs/refit_overlap_audit.json')
    args = parser.parse_args()
    snapshots = args.snapshot or sorted(set(
        glob.glob('outputs/*_protocolv2_both_noce_noneg_dev_seed42/config_snapshot.yaml') +
        glob.glob('outputs/*_protocolv2_noce_noneg_dev_seed42/config_snapshot.yaml') +
        glob.glob('outputs/*_allmerged_noce_noneg_dev_seed42/config_snapshot.yaml')))
    if not snapshots:
        parser.error('No matching downstream snapshots found; specify --snapshot')
    reports = []
    for snapshot in snapshots:
        cfg = yaml.safe_load(Path(snapshot).read_text())
        record = {'snapshot': snapshot, 'dataset_type': cfg['dataset']['type']}
        try:
            record.update(split_overlap_report(build_dataloaders(cfg)))
            print(snapshot, {key: value['image_count'] for key,value in record['cross_split'].items()})
        except Exception as error:
            record['audit_error'] = f'{type(error).__name__}: {error}'
            print(snapshot, record['audit_error'])
        reports.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2)+'\n')
    print(f'Saved read-only diagnostic: {output}')

if __name__ == '__main__':
    main()
