"""Read-only collection of incomplete NSS experiment evidence; never launches training."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('stats6','ggd')
FILES=('console.log','config_snapshot.yaml','launch_manifest.json','run_metadata.json',
       'completion_manifest.json','source_fingerprint.json')
MAX_FILE_BYTES=2*1024*1024


def collect(root,variants):
    """Only explicit metadata/log/CSV allowlist. Weight contents are never read."""
    root=Path(root).resolve();payload={};report={'purpose':'NSS failure diagnosis; not completed-results evidence','variants':{},'files':{}}
    def add(path):
        path=Path(path)
        if not path.is_file():return
        if path.is_symlink() or not path.resolve().is_relative_to(root):return
        size=path.stat().st_size;name=str(path.relative_to(root));truncated=size>MAX_FILE_BYTES
        with path.open('rb') as f:
            if truncated:f.seek(-MAX_FILE_BYTES,2)
            data=f.read(MAX_FILE_BYTES)
        if truncated:name+='.tail.txt'
        payload[name]=data
        report['files'][name]={'original_bytes':size,'tail_only':truncated,'included_sha256':hashlib.sha256(data).hexdigest()}
    for variant in variants:
        if variant not in VARIANTS:raise ValueError('Unknown NSS variant')
        cfgpath=root/f'configs/nss_ablation/{variant}/pretrain.yaml'
        cfg=yaml.safe_load(cfgpath.read_text());directory=root/cfg['output_dir'];csvpath=root/cfg['results_csv']
        if csvpath.suffix != '.csv':raise ValueError('Expected NSS results CSV path')
        if not directory.resolve().is_relative_to(root/'outputs') or not csvpath.resolve().is_relative_to(root/'outputs'):
            raise ValueError('NSS output paths must stay inside repository outputs')
        archived=sorted([p for p in directory.parent.glob(directory.name+'.*') if p.is_dir() and p.name.startswith((directory.name+'.failed_',directory.name+'.incomplete_'))],key=lambda p:p.stat().st_mtime,reverse=True)[:3]
        stage={'output_dir':cfg['output_dir'],'directory_exists':directory.exists(),
               'completion_manifest_exists':(directory/'completion_manifest.json').is_file(),
               'csv_exists':csvpath.exists(),'archived_directories':[],'inventory':[]}
        for d in [directory]+archived:
            if not d.is_dir() or d.is_symlink():continue
            if d!=directory:stage['archived_directories'].append(str(d.relative_to(root)))
            for p in sorted(d.iterdir()):
                if p.is_file():stage['inventory'].append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size})
            for name in FILES:add(d/name)
        add(csvpath)
        for p in sorted(csvpath.parent.glob(csvpath.name+'.*')):
            if p.name.startswith((csvpath.name+'.failed_',csvpath.name+'.incomplete_')):add(p)
        for name in ('pretrain','agiqa1k','agiqa3k','aigciqa2023'):add(root/f'configs/nss_ablation/{variant}/{name}.yaml')
        report['variants'][variant]=stage
    for name in ('scripts/run_nss_ablation.py','scripts/run_nss_ablation.sh',
                 'scripts/run_pretrain_component_strength.py','scripts/run_projection_only_v1_baseline.py',
                 'models/nss.py','models/two_stage_models.py','train_cm_pretrain.py'):
        add(root/name)
    for key,args in (('git_head',['git','rev-parse','HEAD']),('git_status',['git','status','--short'])):
        p=subprocess.run(args,cwd=root,capture_output=True,text=True,check=False)
        report[key]=p.stdout.strip() if p.returncode==0 else 'unavailable'
    payload['diagnosis_summary.json']=json.dumps(report,ensure_ascii=False,indent=2).encode()
    return payload,report


def pack(root,variants):
    root=Path(root).resolve();payload,report=collect(root,variants)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    target=root/f'projection_only_v1_nss_diagnostics_{stamp}.tar.gz'
    with tarfile.open(target,'x:gz') as tar:
        for name,data in sorted(payload.items()):
            entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    return target,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variants',nargs='*',metavar='stats6|ggd')
    args=parser.parse_args()
    if any(v not in VARIANTS for v in args.variants):parser.error('variants must be stats6 or ggd')
    target,report=pack(ROOT,args.variants or VARIANTS)
    for variant,r in report['variants'].items():
        print(f"{variant}: directory={r['directory_exists']}, completion={r['completion_manifest_exists']}, csv={r['csv_exists']}")
    print(f'Diagnostic archive only; no weights or training: {target}')

if __name__=='__main__':main()
