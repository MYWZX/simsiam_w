"""Collect PI/PT learning-rate experiment logs/configs/plans without weights."""
import csv
import json
from pathlib import Path
import tarfile


def main():
    csv.field_size_limit(20_000_000)
    output=Path('outputs');files=set()
    for ds in ['agiqa1k_subset1078','agiqa3k','aigciqa2023']:
        for phase in ['dev','refit80']:
            name=f'{ds}_globalpi_ctx5_loss_projlr01_noce_noneg_{phase}_seed42'
            path=output/f'{name}.csv'
            if not path.is_file():raise FileNotFoundError(path)
            with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
            finals=[r for r in rows if r['event']=='final']
            print(f'{name}: final count={len(finals)}')
            files.add(path)
    roots=list(output.glob('*globalpi_ctx5_loss_projlr01*'))+[Path('configs/global_pi_ctx5_loss_projlr01')]
    source=output/'richhf18k_globalpi_ctx5_loss_noce_noneg_dev_seed42'
    roots += [source.with_suffix('.csv'),source/'config_snapshot.yaml']
    for root in roots:
        for path in (root.rglob('*') if root.is_dir() else [root]):
            if path.is_file() and path.suffix.lower() not in {'.pth','.pt','.ckpt','.bin','.safetensors'}:
                files.add(path)
    archive=Path('cliplarge_ctx5_loss_projlr01_results.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(files):tar.add(path,arcname=str(path))
    print(f'{archive.resolve()} ({archive.stat().st_size/1024/1024:.2f} MB)')

if __name__=='__main__':main()
