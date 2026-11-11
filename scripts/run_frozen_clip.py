"""Freeze only the CLIP visual encoder in fresh pretraining and all downstream stages."""
import argparse
import copy
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_pretrain_component_strength as pipeline

VARIANT = 'frozen'
CONFIG = Path('configs/pretrain_frozen_clip')
STAGES = pipeline.STAGES
VERSION = 'projection_only_v1_ctx5_frozen_clip'


def expected_config(variant=VARIANT, stage='pretrain'):
    if variant != VARIANT or stage not in STAGES:
        raise ValueError('Unknown frozen-CLIP variant or stage')
    if stage == 'pretrain':
        cfg = copy.deepcopy(pipeline.baseline.validate(stage))
        run_id = f'richhf18k_{VERSION}_seed42'
    else:
        cfg = yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix = 'agiqa1k_subset1078' if stage == 'agiqa1k' else stage
        run_id = f'{prefix}_projection_only_v1_frozen_clip_train42_split{cfg["split_seed"]}'
        cfg['train']['pretrained_checkpoint'] = expected_config()['output_dir']+'/best_component_anchor_pretrain.pth'
    cfg['model']['global_encoder']['freeze'] = True
    cfg.update(run_id=run_id, output_dir=f'outputs/{run_id}', results_csv=f'outputs/{run_id}.csv')
    return cfg


def make_runner():
    return pipeline.make_runner(VARIANT, config_factory=expected_config, config_dir=CONFIG,
                                version=VERSION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='pretrain',
                        choices=STAGES+('all', 'downstream', 'continue', 'pack', 'pack-downstream'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    runner = make_runner()
    if args.action.startswith('pack'):
        if args.dry_run:
            parser.error('--dry-run applies only to run actions')
        runner.pack('pretrain' if args.action == 'pack' else 'downstream')
    elif args.action in ('all', 'downstream', 'continue'):
        pipeline.run_pipeline(runner, args.action, args.dry_run)
    else:
        pipeline.run_stage(runner, args.action, args.dry_run)


if __name__ == '__main__':
    main()
