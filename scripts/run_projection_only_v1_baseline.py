"""Independent original-split baseline. No stage launches another stage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/projection_only_v1_baseline'
STAGES = ('pretrain', 'agiqa1k', 'agiqa3k', 'aigciqa2023')
VERSION = 'projection_only_v1_baseline_ctx5'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def configuration(stage):
    return yaml.safe_load((CONFIG / f'{stage}.yaml').read_text())


def paths(stage):
    cfg = configuration(stage)
    return ROOT / cfg['output_dir'], ROOT / cfg['results_csv']


def require_subset(expected, actual, location='config'):
    """Snapshots may add measured normalization statistics, never change policy."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f'{location}: expected mapping')
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f'{location}.{key}: missing')
            require_subset(value, actual[key], f'{location}.{key}')
    elif expected != actual:
        raise ValueError(f'{location}: configuration differs')


def validate(stage):
    cfg = configuration(stage)
    # Match every original best-baseline setting, allowing only mode, run paths
    # and the explicitly new downstream checkpoint source.
    original = yaml.safe_load((ROOT / f'configs/global_pi_ctx5_noce_noneg/{stage}.yaml').read_text())
    expected = json.loads(json.dumps(original))
    expected['model']['fusion']['mode'] = 'projection_only_v1'
    for key in ('output_dir', 'results_csv', 'run_id'):
        if key in expected:
            expected[key] = expected[key].replace('globalpi_ctx5_noce_noneg', VERSION)
    if stage != 'pretrain':
        expected['train']['pretrained_checkpoint'] = str(paths('pretrain')[0].relative_to(ROOT) / 'best_component_anchor_pretrain.pth')
    if cfg != expected:
        raise ValueError('Baseline configuration differs from the fixed original policy')
    if stage == 'pretrain' and any(k in cfg['train'] for k in ('resume', 'pretrained_checkpoint', 'refit_plan')):
        raise ValueError('Fresh pretraining cannot load a project checkpoint or refit')
    return cfg


def verify_checkpoint(path, cfg, epoch):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require_subset(cfg, checkpoint['cfg'], 'checkpoint.cfg')
    if (checkpoint.get('run_id') != cfg['run_id']
            or checkpoint.get('checkpoint_scope') != 'representation'
            or checkpoint.get('selection_metric') != 'val_component_anchor_SROCC'
            or checkpoint.get('epoch') != epoch):
        raise ValueError('Checkpoint mode/source/selection identity mismatch')
    keys = tuple(checkpoint['model'])
    required = ('iqa.backbone.fusion.semantic_proj.', 'iqa.quality_adapter.',
                'iqa.shared_image_projector.', 'iqa.shared_text_projector.',
                'iqa.local_patch_proj.', 'iqa.quality_local_encoder.')
    if any(not any(k.startswith(prefix) for k in keys) for prefix in required):
        raise ValueError('Checkpoint lacks required projection-only representations')
    if any(k.startswith('iqa.backbone.fusion.') and not k.startswith('iqa.backbone.fusion.semantic_proj.') for k in keys):
        raise ValueError('Attention-mode checkpoint cannot be a new baseline source')


def completed(stage):
    cfg = validate(stage)
    directory, table = paths(stage)
    launch = read_json(directory / 'launch_manifest.json')
    done = read_json(directory / 'completion_manifest.json')
    config_sha = digest(CONFIG / f'{stage}.yaml')
    if launch['version'] != VERSION or launch['stage'] != stage or launch['config_sha256'] != config_sha:
        raise ValueError('Launch provenance mismatch')
    for name, expected in done['files_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'Completed artifact changed: {name}')
    if done['csv_sha256'] != digest(table) or done['launch_sha256'] != digest(directory / 'launch_manifest.json'):
        raise ValueError('Completion provenance mismatch')
    require_subset(cfg, yaml.safe_load((directory / 'config_snapshot.yaml').read_text()))
    csv.field_size_limit(50_000_000)
    with table.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    finals = [json.loads(r['metrics_json']) for r in rows if r['event'] == 'final']
    if len(finals) != 1 or any(r['event'] == 'resume' for r in rows):
        raise ValueError('Expected exactly one completed, non-resumed run')
    key = 'val_component_anchor_SROCC' if stage == 'pretrain' else 'val_quality_SROCC'
    if finals[0].get('selection_metric') != key:
        raise ValueError('Wrong selection rule')
    if stage == 'pretrain':
        if finals[0].get('best_component_anchor_epoch', 0) <= 0:
            raise ValueError('Missing selected component-anchor checkpoint record')
        checkpoint = directory / 'best_component_anchor_pretrain.pth'
        if Path(finals[0]['best_component_anchor_checkpoint']).resolve() != checkpoint.resolve():
            raise ValueError('Wrong selected checkpoint path')
        if done['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Selected pretraining checkpoint changed')
        verify_checkpoint(checkpoint, cfg, finals[0]['best_component_anchor_epoch'])
        if done['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Checkpoint changed during verification')
        return {'version': VERSION, 'config_sha256': config_sha,
                'checkpoint': str(checkpoint), 'sha256': done['checkpoint_sha256'],
                'selected_epoch': finals[0]['best_component_anchor_epoch'],
                'completion_sha256': digest(directory / 'completion_manifest.json')}
    source = completed('pretrain')
    if read_json(directory / 'source_fingerprint.json') != source or launch['source'] != source:
        raise ValueError('Downstream source mismatch')
    runtime = read_json(directory / 'run_metadata.json')
    if not runtime.get('checkpoint_verified_after_load') or runtime['checkpoint']['sha256'] != source['sha256']:
        raise ValueError('Trainer did not verify the selected source')
    return source


def run(stage, dry_run=False):
    cfg = validate(stage)
    source = None if stage == 'pretrain' else completed('pretrain')
    directory, table = paths(stage)
    if directory.exists() or table.exists():
        raise FileExistsError(f'Preserve existing output: {directory} / {table}')
    trainer = 'train_cm_pretrain.py' if stage == 'pretrain' else 'train_cm_iqa.py'
    command = [sys.executable, trainer, '--config', str(CONFIG / f'{stage}.yaml')]
    if dry_run:
        print(json.dumps({'command': command, 'source': source}, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full baseline requires server CUDA; cloud CPU launch refused')
    if stage == 'agiqa1k':
        subprocess.run([sys.executable, 'scripts/prepare_agiqa1k_subset1078.py'], cwd=ROOT, check=True)
    from utils.run_metadata import collect_run_metadata
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'version': VERSION, 'stage': stage, 'config_sha256': digest(CONFIG / f'{stage}.yaml'),
                'command': command, 'source': source,
                'runtime': collect_run_metadata(None, cfg)}
    write_new(directory / 'launch_manifest.json', manifest)
    if source is None:
        write_new(directory / 'run_metadata.json', manifest['runtime'])
    else:
        write_new(directory / 'source_fingerprint.json', source)
    with (directory / 'console.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        if process.wait() != 0:
            raise RuntimeError(f'{trainer} failed; partial output preserved')
    # The trainer snapshots/checksums are verified again before a run is usable.
    files = ['config_snapshot.yaml', 'console.log', 'run_metadata.json']
    if stage != 'pretrain':
        files += ['source_fingerprint.json']
    done = {'files_sha256': {p: digest(directory / p) for p in files},
            'csv_sha256': digest(table), 'launch_sha256': digest(directory / 'launch_manifest.json')}
    if stage == 'pretrain':
        done['checkpoint_sha256'] = digest(directory / 'best_component_anchor_pretrain.pth')
    write_new(directory / 'completion_manifest.json', done)
    completed(stage)
    print('Verified completed stage:', stage)


def pack(phase):
    stages = ('pretrain',) if phase == 'pretrain' else STAGES
    source = completed('pretrain')
    for stage in stages:
        completed(stage)
    archive = ROOT / f'{VERSION}_{phase}_results.tar.gz'
    files = set(CONFIG.glob('*.yaml'))
    for stage in stages:
        directory, table = paths(stage)
        files.add(table)
        # Explicit allowlist: model weights can never enter the archive.
        for name in ('config_snapshot.yaml', 'console.log', 'launch_manifest.json',
                     'completion_manifest.json', 'source_fingerprint.json', 'run_metadata.json'):
            if (directory / name).is_file():
                files.add(directory / name)
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for path in sorted(files):
                tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
            import io
            data = json.dumps(source, indent=2).encode()
            info = tarfile.TarInfo('source_fingerprint.json'); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    print(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='pretrain', choices=STAGES + ('pack-pretrain', 'pack-downstream'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage.startswith('pack-'):
        if args.dry_run:
            parser.error('--dry-run only applies to run stages')
        pack(args.stage.removeprefix('pack-'))
    else:
        run(args.stage, args.dry_run)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
