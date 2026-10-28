"""Fixed-budget RichHF train+val pretraining; no held-out model selection."""
import math
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from utils.data_protocols import loader_rng_kwargs
from utils.refit_protocol import LimitedLoader, digest, file_digest, step_lrs, validation_budget


def validate_source(cfg):
    items = cfg['dataset'].get('datasets', [])
    if len(items) != 1 or items[0].get('type') != 'richhf18k':
        raise ValueError('Merged pretraining currently supports one RichHF18K source only')
    if items[0].get('split_strategy', cfg['dataset'].get('split_strategy')) != 'official':
        raise ValueError('Merged pretraining requires original official splits')
    train = cfg['train']
    if (cfg['model'].get('quality_text_bank', {}).get('enabled', False)
            or train.get('lambda_quality_text_ce', 0) != 0
            or not train.get('disable_negative_relations', False)
            or train.get('lambda_prompt_answer_gap', 0) != 0):
        raise ValueError('Expected CE/negative-free source config')
    if not train.get('use_joint_iqa_pretraining', False):
        raise ValueError('Merged pretraining requires joint IQA representation')
    for key in ('quality_balanced_sampling', 'quality_relation_siam', 'quality_loss_reweighting'):
        if train.get(key, {}).get('enabled', False):
            raise ValueError(f'Merged pretraining does not support {key}')
    if cfg['dataset'].get('num_samples_per_epoch') is not None:
        raise ValueError('Merged pretraining uses natural combined split length')


def dataset_manifest(datasets):
    """Record filtered image membership and label-source hashes, never scores."""
    identities = {}
    sources = {}
    for name, dataset in zip(('train', 'val', 'test'), datasets):
        paths = []
        for _, row in dataset.df.iterrows():
            path = dataset._find_path(row)
            if path is None:
                raise ValueError(f'Image disappeared in {name} split')
            paths.append(str(path.resolve()))
        if len(set(paths)) != len(paths):
            raise ValueError(f'Repeated image in {name} split')
        identities[name] = paths
        sources[name] = {str(Path(p).resolve()): file_digest(p) for p in dataset.csv_paths}
    for left, right in [('train','val'), ('train','test'), ('val','test')]:
        if set(identities[left]) & set(identities[right]):
            raise ValueError(f'Image overlap between {left} and {right}')
    return {'identities': identities, 'csv_sha256': sources,
            'counts': {k: len(v) for k, v in identities.items()},
            'test_ids_sha256': digest(identities['test'])}


def merged_richhf_loader(cfg, loaders, build_single):
    validate_source(cfg)
    train_loader, val_loader, test_loader = loaders
    original_train = train_loader.dataset.datasets[0]
    original = (original_train.dataset, val_loader.dataset.datasets[0], test_loader.dataset.datasets[0])
    if list(original_train.indices) != list(range(len(original[0]))):
        raise ValueError('Original official training subset was modified')
    manifest = dataset_manifest(original)
    item = cfg['dataset']['datasets'][0]
    train = cfg['train']
    val_cfg = {**item, 'split': item.get('val_split', 'val'),
               'input_size': cfg['dataset'].get('input_size', 224),
               'supervision_view': train.get('human_supervision_view', 'augmented'),
               'quality_loss_reweighting': train.get('quality_loss_reweighting', {}),
               'mirror_augmentation': train.get('mirror_augmentation', item.get('mirror_augmentation', {}))}
    val_cfg['use_local_feedback'] = bool(item.get('use_local_feedback', False)
        or train.get('lambda_img_prompt_local', 0) > 0 or train.get('lambda_prompt_answer_word', 0) > 0)
    val_training = build_single(val_cfg, train=True)
    if dataset_manifest((original[0], val_training, original[2])) != manifest:
        raise ValueError('Training-transform validation dataset changed membership or labels')
    merged = ConcatDataset([original_train, val_training])
    # The single-source weights are uniform; retain replacement sampling used
    # by development, with fresh independent RNG and expanded sample count.
    if not isinstance(train_loader.sampler, WeightedRandomSampler):
        raise ValueError('Expected original weighted replacement sampler')
    weights = train_loader.sampler.weights
    if not train_loader.sampler.replacement or not torch.all(weights == weights[0]):
        raise ValueError('Expected uniform replacement sampling')
    options = cfg['loader']
    sampler = WeightedRandomSampler(torch.ones(len(merged), dtype=torch.double),
        num_samples=len(merged), replacement=True,
        **({'generator': loader_rng_kwargs(cfg, 'train')['generator']}
           if options.get('isolate_rng') else {}))
    result = DataLoader(merged, sampler=sampler,
        batch_size=options.get('batch_size', 8), num_workers=options.get('num_workers', 4),
        pin_memory=options.get('pin_memory', True), drop_last=True,
        **loader_rng_kwargs(cfg, 'train'))
    if not len(result):
        raise ValueError('Merged pretraining split has no full batch')
    return result, manifest


def pretrain_budget(csv_path, cfg, original_batches):
    key = cfg['train'].get('pretrain_selection_metric', 'val_component_anchor_SROCC')
    mode = cfg['train'].get('pretrain_selection_mode', 'max')
    def selector(metrics, _cfg):
        value = float(metrics[key.removeprefix('val_')])
        if not math.isfinite(value):
            raise ValueError('Nonfinite pretraining selection metric')
        return value, mode == 'max', key
    budget = validation_budget(csv_path, cfg, selector)
    if any(s != original_batches for s in budget['epoch_steps']):
        raise ValueError('Reference processed batches disagree with original split')
    return budget


def run_pretrain_refit(*, model, loader, optimizer, device, cfg, budget,
                       original_batches, logger, train_epoch):
    scaler = GradScaler(device.type,
        enabled=bool(cfg['train'].get('amp', True)) and device.type == 'cuda')
    initial = [float(g['lr']) for g in optimizer.param_groups]
    steps = updates = epoch = 0
    def callback(successful):
        nonlocal steps, updates
        steps += 1
        updates += int(successful)
        values = step_lrs(initial, float(cfg.get('scheduler', {}).get('min_lr', 1e-6)),
                          steps, original_batches, int(cfg['train']['epochs']))
        for group, value in zip(optimizer.param_groups, values):
            group['lr'] = value
    while steps < budget['total_steps']:
        epoch += 1
        before = steps
        metrics = train_epoch(model, LimitedLoader(loader, budget['total_steps'] - steps),
            optimizer, scaler, device, cfg, epoch, optimizer_step_callback=callback)
        if steps <= before or steps > budget['total_steps']:
            raise RuntimeError('Invalid merged pretraining step count')
        logger.log(event='refit_epoch', epoch=epoch,
                   metrics={'train': metrics, 'training_steps': steps, 'optimizer_updates': updates})
    return {'epoch': epoch, 'training_steps': steps, 'optimizer_updates': updates,
            'amp_skipped_steps': steps-updates}
