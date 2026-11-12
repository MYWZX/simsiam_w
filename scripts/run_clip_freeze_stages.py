"""Stage-wise CLIP visual freeze controls, sharing verified pretraining sources."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_pretrain_component_strength as pipeline
from scripts import run_frozen_clip as frozen

VARIANTS = ('pretrain_only', 'downstream_only', 'both')
CONFIG = Path('configs/clip_freeze_stages')
STAGES = pipeline.STAGES


def expected_config(variant, stage='pretrain'):
    if variant not in VARIANTS or stage not in STAGES:
        raise ValueError('Unknown CLIP freeze condition or stage')
    if variant == 'both':
        return frozen.expected_config(stage=stage)
    if stage == 'pretrain':
        return (copy.deepcopy(pipeline.baseline.validate(stage)) if variant == 'downstream_only'
                else frozen.expected_config(stage=stage))
    cfg = yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
    cfg['model']['global_encoder']['freeze'] = variant == 'downstream_only'
    prefix = 'agiqa1k_subset1078' if stage == 'agiqa1k' else stage
    run_id = f'{prefix}_projection_only_v1_clip_{variant}_train42_split{cfg["split_seed"]}'
    cfg.update(run_id=run_id, output_dir=f'outputs/{run_id}', results_csv=f'outputs/{run_id}.csv')
    cfg['train']['pretrained_checkpoint'] = expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
    return cfg


def source_runner(variant):
    if variant != 'downstream_only':
        return frozen.make_runner()
    return pipeline.make_runner('projection_only_v1_baseline',
        config_factory=lambda _variant, stage='pretrain': pipeline.baseline.validate(stage),
        config_dir=Path('configs'), version=pipeline.baseline.VERSION)


def make_runner(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown CLIP freeze condition')
    if variant == 'both':
        return frozen.make_runner()
    r = pipeline.make_runner(variant, config_factory=expected_config, config_dir=CONFIG,
                             version=f'projection_only_v1_ctx5_clip_{variant}')
    source = source_runner(variant)
    original_completed, original_run = r.completed, r.run

    def completed(stage):
        if stage != 'pretrain':
            return original_completed(stage)
        fingerprint = source.completed('pretrain')
        if variant == 'downstream_only':
            reference = r.read_json(ROOT/'docs/current_tuning_reference.json')['source']
            for key in ('version', 'config_sha256', 'sha256', 'selected_epoch', 'completion_sha256'):
                if fingerprint[key] != reference[key]:
                    raise ValueError(f'Pinned best-baseline pretraining source mismatch: {key}')
        return fingerprint

    def run(stage, dry_run=False):
        if stage != 'pretrain':
            return original_run(stage, dry_run)
        r.validate(stage)
        if variant == 'downstream_only':
            if dry_run:
                print(json.dumps({'source': str(r.paths(stage)[0]),
                                  'action': 'verify pinned existing baseline; never retrain'}, indent=2))
                return
            completed(stage)
            print('Verified pinned baseline; no pretraining launched')
            return
        return source.run(stage, dry_run)

    r.completed, r.run = completed, run
    return r


def run_pipeline(variant, action='all', dry_run=False):
    r = make_runner(variant)
    if dry_run:
        stages = STAGES if variant != 'downstream_only' and action != 'downstream' else STAGES[1:]
        for stage in STAGES:r.validate(stage)
        print(json.dumps({'condition': variant, 'stages': list(stages),
            'source_action': ('verify pinned existing baseline; never retrain' if variant == 'downstream_only'
                              else 'reuse verified frozen source or train once; never overwrite'),
            'source_checkpoint': r.validate('pretrain')['output_dir']+'/best_component_anchor_pretrain.pth',
            'completed_stages': 'verify and reuse; incomplete stages refuse',
            'configs': [str(r.CONFIG/f'{stage}.yaml') for stage in stages]}, indent=2))
        return
    # all explicitly reuses completed stages, including the earlier both-frozen experiment.
    pipeline.run_pipeline(r, 'continue' if action == 'all' else action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('action', nargs='?', default='all',
                        choices=STAGES+('all', 'downstream', 'continue', 'pack', 'pack-downstream'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args();os.chdir(ROOT)
    if args.action in ('all', 'downstream', 'continue'):
        run_pipeline(args.variant, args.action, args.dry_run)
    else:
        r = make_runner(args.variant)
        if args.action.startswith('pack'):
            if args.dry_run:parser.error('--dry-run applies only to run actions')
            r.pack('pretrain' if args.action == 'pack' else 'downstream')
        else:
            pipeline.run_stage(r, args.action, args.dry_run)


if __name__ == '__main__':main()
