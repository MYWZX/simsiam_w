"""Fresh image-Prompt relation loss-weight sweep; optionally chain three downstream datasets."""
import argparse
import copy
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_pretrain_component_strength as pipeline

VARIANTS = {'prompt020': 0.20, 'prompt030': 0.30}
CONFIG = Path('configs/pretrain_img_prompt')
STAGES = pipeline.STAGES


def expected_config(variant, stage='pretrain'):
    if variant not in VARIANTS or stage not in STAGES:
        raise ValueError('Unknown image-Prompt loss-weight variant or stage')
    if stage == 'pretrain':
        cfg = copy.deepcopy(pipeline.baseline.validate(stage))
        run_id = f'richhf18k_projection_only_v1_ctx5_img_{variant}_seed42'
        cfg['train']['lambda_img_prompt'] = VARIANTS[variant]
    else:
        cfg = yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix = 'agiqa1k_subset1078' if stage == 'agiqa1k' else stage
        run_id = f'{prefix}_projection_only_v1_img_{variant}_train42_split{cfg["split_seed"]}'
        cfg['train']['pretrained_checkpoint'] = expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
    cfg.update(run_id=run_id, output_dir=f'outputs/{run_id}', results_csv=f'outputs/{run_id}.csv')
    return cfg


def make_runner(variant):
    return pipeline.make_runner(variant, config_factory=expected_config, config_dir=CONFIG,
                                version=f'projection_only_v1_ctx5_img_{variant}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('action', nargs='?', default='pretrain',
                        choices=STAGES+('all', 'downstream', 'continue', 'pack', 'pack-downstream'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    runner = make_runner(args.variant)
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
