"""Optional CLIP capacity and train-only score calibration experiments."""
import numpy as np
import torch
from torch.utils.data import Subset


def configure_clip_large(cfg):
    """Use matching OpenAI ViT-L/14 visual and text towers, at 224px."""
    model = cfg.setdefault('model', {})
    model.setdefault('global_encoder', {}).update(
        type='open_clip', model_name='ViT-L-14', open_clip_pretrained='openai',
        output_dim=768,
    )
    model.setdefault('text_encoder', {}).update(
        clip_model_name='ViT-L-14', clip_pretrained='openai',
    )


def training_quality_centers(dataset):
    """Read labels without image loading/augmentation or access to held-out rows.

    Supports AIGCIQA2023's raw quality array and its /mos_max normalization.
    Nested Subsets are resolved before selecting any labels.
    """
    indices = np.arange(len(dataset))
    while isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices)[indices]
        dataset = dataset.dataset
    if not hasattr(dataset, 'quality') or not hasattr(dataset, 'mos_max'):
        raise ValueError('Train-quantile calibration requires quality and mos_max arrays (AIGCIQA2023).')
    labels = np.asarray(dataset.quality, dtype=np.float64)[indices]
    labels = np.clip(labels / (float(dataset.mos_max) + 1e-8), 0., 1.)
    if not len(labels) or not np.isfinite(labels).all():
        raise ValueError('Training Quality labels must be nonempty and finite.')
    centers = np.quantile(labels, [.05, .25, .5, .75, .95])
    if not np.all(np.diff(centers) > 0):
        raise ValueError('Training quantiles must produce five strictly increasing centers.')
    return centers.tolist(), len(labels)


def calibrate_quality_centers(model, train_dataset, cfg):
    bank = model.quality_text_bank
    if bank is None:
        raise ValueError('Train-quantile calibration requires --quality-text-blend (use 0).')
    centers, count = training_quality_centers(train_dataset)
    old = bank.centers.detach().cpu().tolist()
    with torch.no_grad():
        bank.centers.copy_(bank.centers.new_tensor(centers))
    cfg['model']['quality_text_bank']['centers'] = centers
    details = dict(source='training_labels_only', quantiles=[.05, .25, .5, .75, .95],
                   count=count, pretrained_centers=old, centers=centers)
    cfg.setdefault('train', {})['quality_center_calibration'] = details
    return details


def configure_local_feature_stage(cfg, args):
    """Record stage and explicit downstream projection reset before snapshots."""
    stage = getattr(args, 'local_feature_stage', None)
    if stage is not None:
        cfg.setdefault('model', {}).setdefault('local_encoder', {})['feature_stage'] = stage
    stage = cfg.get('model', {}).get('local_encoder', {}).get('feature_stage', 'layer4')
    if stage not in {'layer3', 'layer4'}:
        raise ValueError('local_encoder.feature_stage must be layer3 or layer4.')
    if getattr(args, 'reinit_local_projection', False):
        cfg.setdefault('train', {})['reinit_local_projection'] = True
    if cfg.get('train', {}).get('reinit_local_projection', False):
        if stage != 'layer3' or not cfg['train'].get('pretrained_checkpoint'):
            raise ValueError('--reinit-local-projection requires layer3 and a pretrained checkpoint.')
