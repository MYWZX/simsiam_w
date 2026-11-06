"""Four independent fresh pretraining runs, changing Anchor or Ordinal separately."""
import argparse
import copy
import csv
import json
import os
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_projection_only_v1_baseline as baseline

VARIANTS = {'anchor050': ('anchor', 0.5), 'anchor200': ('anchor', 2.0),
            'ordinal050': ('ordinal', 0.5), 'ordinal200': ('ordinal', 2.0)}
CONFIG = Path('configs/pretrain_component_strength')


STAGES = ('pretrain', 'agiqa1k', 'agiqa3k', 'aigciqa2023')


def expected_config(variant, stage='pretrain'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    if stage != 'pretrain':
        import yaml
        cfg = yaml.safe_load((ROOT / f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix = 'agiqa1k_subset1078' if stage == 'agiqa1k' else stage
        run_id = f'{prefix}_projection_only_v1_component_{variant}_train42_split{cfg["split_seed"]}'
        cfg.update(run_id=run_id, output_dir=f'outputs/{run_id}', results_csv=f'outputs/{run_id}.csv')
        source = expected_config(variant)
        cfg['train']['pretrained_checkpoint'] = source['output_dir'] + '/best_component_anchor_pretrain.pth'
        return cfg
    cfg = copy.deepcopy(baseline.validate('pretrain'))
    run_id = f'richhf18k_projection_only_v1_ctx5_component_{variant}_seed42'
    cfg.update(run_id=run_id, output_dir=f'outputs/{run_id}', results_csv=f'outputs/{run_id}.csv')
    loss, multiplier = VARIANTS[variant]
    for component in ('artifact', 'aesthetics'):
        cfg['train'][f'lambda_{component}_{loss}'] *= multiplier
    return cfg


def make_runner(variant):
    """Isolate each runner's globals so parallel runs cannot share paths or identity."""
    expected_config(variant)
    spec = importlib.util.spec_from_file_location(
        f'_component_strength_{variant}', ROOT / 'scripts/run_projection_only_v1_baseline.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = ROOT
    runner.CONFIG = ROOT / CONFIG / variant
    runner.VERSION = f'projection_only_v1_ctx5_component_{variant}'
    runner.STAGES = STAGES

    def validate(stage):
        if stage not in STAGES:
            raise ValueError(f'Unknown stage: {stage}')
        cfg = runner.configuration(stage)
        if cfg != expected_config(variant, stage):
            raise ValueError('Configuration differs from fixed component-strength policy')
        return cfg

    runner.validate = validate
    original_completed = runner.completed

    def completed(stage):
        source = original_completed(stage)
        if stage != 'pretrain':
            directory, table = runner.paths(stage)
            with table.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            finals = [json.loads(r['metrics_json']) for r in rows if r['event'] == 'final']
            if finals[0].get('last_epoch') != 60 or [int(r['epoch']) for r in rows if r['event']=='epoch'] != list(range(1,61)):
                raise ValueError('Expected a complete 60-epoch downstream run')
            from utils.agiqa1k_subset1078 import expected_resplit_manifest
            evidence = runner.read_json(directory / 'data_split_manifest.json')
            cfg = validate(stage)
            if evidence['train_seed'] != 42 or evidence['split_seed'] != cfg['split_seed']:
                raise ValueError('Downstream seed evidence mismatch')
            if stage == 'agiqa1k' and evidence['indices'] != expected_resplit_manifest(3407)['split_indices']:
                raise ValueError('Wrong subset1078 split membership')
        return source

    runner.completed = completed
    return runner


def run_stage(runner, stage, dry_run=False):
    if stage == 'agiqa1k' and not dry_run:
        # Preflight source/output/CUDA before preparing any dataset artifacts.
        runner.run(stage, True)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Server CUDA required; cloud CPU training refused')
        runner.subprocess.run([sys.executable, 'scripts/prepare_agiqa1k_subset1078.py'], cwd=ROOT, check=True)
        from utils.agiqa1k_subset1078 import prepare_resplit_manifest
        prepare_resplit_manifest(ROOT / runner.validate(stage)['dataset']['root_dir'], 3407)
    runner.run(stage, dry_run)


def run_pipeline(runner, action, dry_run=False):
    stages = STAGES if action == 'all' else STAGES[1:]
    # Refuse existing outputs before starting an expensive pipeline.
    for stage in stages:
        runner.validate(stage)
        directory, table = runner.paths(stage)
        if directory.exists() or table.exists():
            raise FileExistsError(f'Preserve existing output: {directory} / {table}')
    if action == 'downstream':
        runner.completed('pretrain')
    if dry_run:
        print(json.dumps({'stages': list(stages), 'configs': [str(runner.CONFIG / f'{s}.yaml') for s in stages]}))
        return
    for stage in stages:
        run_stage(runner, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('action', nargs='?', default='pretrain', choices=STAGES + ('all', 'downstream', 'pack', 'pack-downstream'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    runner = make_runner(args.variant)
    if args.action in ('pack', 'pack-downstream'):
        if args.dry_run:
            parser.error('--dry-run applies only to pretrain')
        runner.pack('pretrain' if args.action == 'pack' else 'downstream')
    elif args.action in ('all', 'downstream'):
        run_pipeline(runner, args.action, args.dry_run)
    else:
        run_stage(runner, args.action, args.dry_run)


if __name__ == '__main__':
    main()
