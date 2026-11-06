"""Four independent fresh pretraining runs, changing Anchor or Ordinal separately."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_projection_only_v1_baseline as baseline

VARIANTS = {'anchor050': ('anchor', 0.5), 'anchor200': ('anchor', 2.0),
            'ordinal050': ('ordinal', 0.5), 'ordinal200': ('ordinal', 2.0)}
CONFIG = Path('configs/pretrain_component_strength')


def expected_config(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    cfg = copy.deepcopy(baseline.validate('pretrain'))
    run_id = f'richhf18k_projection_only_v1_ctx5_component_{variant}_seed42'
    cfg.update(run_id=run_id, output_dir=f'outputs/{run_id}', results_csv=f'outputs/{run_id}.csv')
    loss, multiplier = VARIANTS[variant]
    for component in ('artifact', 'aesthetics'):
        cfg['train'][f'lambda_{component}_{loss}'] *= multiplier
    return cfg


def make_runner(variant):
    """Isolate each runner's globals so parallel runs cannot share paths or identity."""
    expected = expected_config(variant)
    spec = importlib.util.spec_from_file_location(
        f'_component_strength_{variant}', ROOT / 'scripts/run_projection_only_v1_baseline.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = ROOT
    runner.CONFIG = ROOT / CONFIG / variant
    runner.VERSION = f'projection_only_v1_ctx5_component_{variant}'
    runner.STAGES = ('pretrain',)

    def validate(stage):
        if stage != 'pretrain':
            raise ValueError('Only fresh pretraining is supported; downstream requires separate source validation')
        cfg = runner.configuration(stage)
        if cfg != expected:
            raise ValueError('Configuration differs from fixed component-strength policy')
        return cfg

    runner.validate = validate
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('action', nargs='?', default='pretrain', choices=('pretrain', 'pack'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    runner = make_runner(args.variant)
    if args.action == 'pack':
        if args.dry_run:
            parser.error('--dry-run applies only to pretrain')
        runner.pack('pretrain')
    else:
        runner.run('pretrain', args.dry_run)


if __name__ == '__main__':
    main()
