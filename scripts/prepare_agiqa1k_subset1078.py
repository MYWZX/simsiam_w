"""Prepare the approved independent 1078-image subset without changing sources."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.agiqa1k_subset1078 import (SOURCE_CSV_SHA256,SOURCE_XLSX_SHA256,
    SUBSET_CSV_SHA256,RELATIVE_DIR,CSV_NAME,file_sha256,expected_manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/AGIQA-1K')
    args=p.parse_args();root=Path(args.root)
    source=root/'AIGC-1K_answer.csv';excel=root/'AIGC_MOS_Zscore.xlsx'
    if file_sha256(source)!=SOURCE_CSV_SHA256 or file_sha256(excel)!=SOURCE_XLSX_SHA256:
        raise ValueError('Source annotations differ from the audited upload; original files will not be modified')
    output=root/RELATIVE_DIR
    if output.exists():
        if (file_sha256(output/CSV_NAME)!=SUBSET_CSV_SHA256 or
                json.loads((output/'split_manifest.json').read_text())!=expected_manifest()):
            raise ValueError('Existing revision is incomplete or different; preserve it and investigate')
        print(f'Existing subset1078 verified: {output}');return
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.subset1078-',dir=output.parent) as temp:
        staging=Path(temp)/'candidate'
        subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_agiqa1k_annotation_audit.py')),
            '--csv',str(source),'--xlsx',str(excel),'--output-dir',str(staging),'--write-subset'],check=True)
        if file_sha256(staging/CSV_NAME)!=SUBSET_CSV_SHA256:
            raise ValueError('Candidate contents do not match audited unchanged retained records')
        report_path=staging/'annotation_audit.json';report=json.loads(report_path.read_text())
        report['candidate_csv']=str(output/CSV_NAME)
        report['approved_protocol']='independent subset1078; not the original 1080-image benchmark'
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (staging/'split_manifest.json').write_text(json.dumps(expected_manifest(),indent=2)+'\n')
        staging.rename(output)
    print(f'Prepared subset1078: {output}; split sizes={expected_manifest()["split_sizes"]}')

if __name__=='__main__':main()
