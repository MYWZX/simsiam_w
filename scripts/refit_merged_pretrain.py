"""Refit RichHF train+val from initial pretrained encoders, using dev budget."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from datasets import build_dataloaders
from datasets.mixed_pretrain_dataset import _build_single_dataset
from models.two_stage_models import CrossModalIQAJointPretrain
from train_cm_pretrain import build_optimizer, train_one_epoch, _extract_representation_state
from utils.experiment_logger import CSVExperimentLogger, dump_config_snapshot
from utils.pretrain_resume import atomic_save
from utils.pretrain_refit import merged_richhf_loader, pretrain_budget, run_pretrain_refit, validate_source
from utils.refit_protocol import canonical_cfg, digest, file_digest
from utils.run_metadata import collect_run_metadata, write_run_metadata
from utils.train_utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    result_csv = Path(str(out) + '.csv')
    if out.exists() or result_csv.exists():
        raise FileExistsError('Merged pretraining output exists; preserve it before a new run')
    snapshot_sha = file_digest(args.snapshot)
    csv_sha = file_digest(args.csv)
    cfg = yaml.safe_load(Path(args.snapshot).read_text())
    cfg.pop('source_config', None)
    validate_source(cfg)
    set_seed(int(cfg.get('seed', 42)))
    loaders = build_dataloaders(cfg)
    budget = pretrain_budget(args.csv, cfg, len(loaders[0]))
    merged, manifest = merged_richhf_loader(cfg, loaders, _build_single_dataset)
    if manifest != cfg['dataset'].get('pretrain_development_manifest'):
        raise ValueError('Original splits/label CSVs changed since development, or manifest is missing')
    plan = {'protocol': 'pretrain_train_val_fixed_budget',
            'reference_snapshot': str(Path(args.snapshot).resolve()), 'snapshot_sha256': snapshot_sha,
            'reference_csv': str(Path(args.csv).resolve()), 'csv_sha256': csv_sha,
            'source_config_sha256': digest(canonical_cfg(cfg)), 'budget': budget,
            'step_unit': 'processed_training_batches', 'original_train_batches': len(loaders[0]),
            'dataset': manifest}
    plan['sha256'] = digest(plan)
    cfg.update(output_dir=str(out), results_csv=str(result_csv), run_id=out.name)
    cfg['train']['merged_pretrain_protocol'] = plan['protocol']
    cfg['dataset']['merged_pretrain_manifest'] = {
        'original_counts': manifest['counts'], 'train_count': len(merged.dataset),
        'val_count': 0, 'test_ids_sha256': manifest['test_ids_sha256'], 'plan_sha256': plan['sha256']}
    out.mkdir(parents=True)
    (out/'refit_plan.json').write_text(json.dumps(plan, indent=2))
    snapshot = dump_config_snapshot(cfg, str(out), config_path=args.snapshot)
    logger = CSVExperimentLogger(csv_path=str(result_csv), run_id=out.name, stage='pretrain_refit',
        cfg=cfg, output_dir=str(out), config_path=snapshot)
    logger.log(event='run_start', epoch=0, metrics={'plan_sha256': plan['sha256'],
        'fixed_batches': budget['total_steps'], 'validation_selected_epoch': budget['epoch'],
        'split': cfg['dataset']['merged_pretrain_manifest']})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Reinitialize from configured CLIP/ResNet initial encoders, not dev weights.
    model = CrossModalIQAJointPretrain(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    meta_path = write_run_metadata(str(out), collect_run_metadata(None, cfg))
    stats = run_pretrain_refit(model=model, loader=merged, optimizer=optimizer, device=device,
        cfg=cfg, budget=budget, original_batches=len(loaders[0]), logger=logger, train_epoch=train_one_epoch)
    if file_digest(args.snapshot) != snapshot_sha or file_digest(args.csv) != csv_sha:
        raise RuntimeError('Reference experiment changed during merged pretraining')
    # Fixed-end representation, deliberately not named "best".
    checkpoint = out/'final_merged_pretrain.pth'
    atomic_save({'model': _extract_representation_state(model.state_dict()), 'cfg': cfg,
        'checkpoint_scope': 'representation', 'protocol': plan['protocol'],
        'plan_sha256': plan['sha256'], **stats}, str(checkpoint), torch.save)
    logger.log(event='final', epoch=stats['epoch'], metrics={**stats,
        'protocol': plan['protocol'], 'checkpoint': str(checkpoint), 'plan_sha256': plan['sha256'],
        'run_metadata_path': meta_path, 'validation_evaluations': 0, 'test_evaluations': 0})
    print(f'Merged pretraining complete: {checkpoint}; batches={stats["training_steps"]}')

if __name__ == '__main__':
    main()
