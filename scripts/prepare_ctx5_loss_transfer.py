"""Export this experiment's trusted local training checkpoint for weights-only transfer."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from train_cm_pretrain import _extract_representation_state
from utils.pretrain_resume import atomic_save
from utils.refit_protocol import file_digest
from scripts.run_unmerged_ii_experiments import comparable_cfg

ROOT=Path('outputs/richhf18k_globalpi_ctx5_loss_noce_noneg_dev_seed42')


def export_transfer(root):
    source=root/'best_pretrain.pth';destination=root/'best_loss_representation.pth'
    with root.with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:
        finals=[json.loads(r['metrics_json']) for r in csv.DictReader(stream) if r['event']=='final']
    if len(finals)!=1 or finals[0]['selection_metric']!='val_loss':
        raise ValueError('Expected completed minimum-loss pretraining')
    sha=file_digest(source)
    if destination.exists():
        compact=torch.load(destination,map_location='cpu',weights_only=True)
        if compact.get('source_sha256')!=sha or compact.get('epoch')!=finals[0]['best_epoch']:
            raise ValueError('Existing transfer does not match source; preserve it')
        return destination
    # Explicit trust is granted by --trust-local-training-checkpoint at the CLI.
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    cfg=checkpoint['cfg']
    if (checkpoint['epoch']!=finals[0]['best_epoch'] or
        cfg['train']['pretrain_selection_metric']!='val_loss' or
        cfg['train']['pretrain_selection_mode']!='min' or
        cfg['model']['text_encoder']['learnable_context_tokens']!=5):
        raise ValueError('Source does not match the completed five-context loss-selected run')
    state=_extract_representation_state(checkpoint['model'])
    if not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Representation contains non-tensor values')
    compact={'model':state,'epoch':int(checkpoint['epoch']),'checkpoint_scope':'representation',
             'selection_metric':'val_loss','source_sha256':sha}
    atomic_save(compact,str(destination),torch.save)
    torch.load(destination,map_location='cpu',weights_only=True)
    print(f'Exported SAME loss-selected epoch {compact["epoch"]}: {destination}')
    return destination


def archive_unstarted(config_path):
    cfg=yaml.safe_load(config_path.read_text());output=Path(cfg['output_dir']);csv_path=Path(cfg['results_csv'])
    if not output.exists() and not csv_path.exists():return
    snapshot=output/'config_snapshot.yaml'
    if not snapshot.is_file() or not csv_path.is_file():
        raise ValueError(f'Cannot verify existing output: {output}; preserve it')
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=1 or rows[0]['event']!='run_start':
        raise ValueError(f'Existing output may contain training progress: {output}; preserve it')
    old=yaml.safe_load(snapshot.read_text())
    old['train']['pretrained_checkpoint']=old['train']['pretrained_checkpoint'].replace(
        '/best_pretrain.pth','/best_loss_representation.pth')
    if comparable_cfg(old)!=comparable_cfg(cfg):
        raise ValueError('Existing unstarted output has a different config; preserve it')
    # main() writes runtime provenance BEFORE attempting checkpoint loading.
    # Only an unverified initialization record can belong to this recovery path.
    metadata_path=output/'run_metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        if metadata.get('checkpoint_verified_after_load') is not False:
            raise ValueError(f'Checkpoint may already be loaded in {output}; preserve it')
    allowed={'config_snapshot.yaml','run_metadata.json'}
    unexpected=[str(f.relative_to(output)) for f in output.rglob('*')
                if f.is_file() and str(f.relative_to(output)) not in allowed]
    if unexpected:
        raise ValueError(f'Unexpected files in {output}: {unexpected}; preserve it')
    backup=Path('outputs/failed_before_training')/(output.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    backup.mkdir(parents=True,exist_ok=False)
    shutil.move(str(output),str(backup/output.name));shutil.move(str(csv_path),str(backup/csv_path.name))
    print(f'Backed up unstarted run to {backup}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trust-local-training-checkpoint',action='store_true')
    p.add_argument('--archive-unstarted-config',type=Path)
    args=p.parse_args();csv.field_size_limit(20_000_000)
    if args.archive_unstarted_config:
        archive_unstarted(args.archive_unstarted_config);return
    if not args.trust_local_training_checkpoint:p.error('Explicit local checkpoint trust required')
    export_transfer(ROOT)

if __name__=='__main__':main()
