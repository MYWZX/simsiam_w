"""Explicit, architecture-preserving downstream experiment controls."""
import math


def configure_downstream_controls(cfg, args):
    if getattr(args, 'quality_safe_input', False):
        dataset = cfg.setdefault('dataset', {})
        mirror = dataset.get('mirror_augmentation', {})
        if (getattr(args, 'mirror_augmentation', False)
                or getattr(args, 'mirror_augmentation_probability', None) is not None
                or mirror.get('enabled', False)):
            raise ValueError('Quality-safe input requires mirror augmentation disabled.')
        dataset['train_augmentation_mode'] = 'quality_safe'
        dataset['quality_safe_flip_prob'] = 0.0
        print('[INFO] Quality-safe input: whole-image resize/normalization; no random crop, flip or color jitter.')
    if getattr(args, 'disable_legacy_negative', False):
        cfg.setdefault('train', {})['lambda_neg'] = 0.0
        print('[INFO] Legacy shifted-Prompt negative loss disabled.')
    metric = getattr(args, 'selection_metric', None)
    if metric is not None:
        cfg.setdefault('train', {}).update(selection_metric=metric, selection_mode='max')
    if getattr(args, 'audit_loss_gradients', False):
        cfg.setdefault('train', {})['audit_loss_gradients'] = True


def distinct_prompt_mask(prompts, negatives, device):
    import torch
    def canonical(text):
        return ' '.join(str(text).split()).casefold()
    valid = []
    for positive, negative in zip(prompts, negatives):
        a, b = canonical(positive), canonical(negative)
        valid.append(bool(a and b and a != b))
    return torch.tensor(valid, dtype=torch.float32, device=device)


def audit_fusion_gradients(model, objectives):
    """Unscaled diagnostic norms on Fusion only; does not modify .grad."""
    import torch
    params = [p for p in model.backbone.fusion.parameters() if p.requires_grad]
    if not params:
        return {'gradient_audit_fusion_active': 0.0}
    result = {'gradient_audit_fusion_active': 1.0}
    for name, loss in objectives.items():
        norm2 = 0.0
        if loss.requires_grad:
            gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            norm2 = sum(float(g.detach().float().square().sum()) for g in gradients if g is not None)
        result[f'gradient_fusion_{name}_norm'] = math.sqrt(norm2)
    return result


def validate_local_crops(batch, cfg):
    options = cfg.get('dataset', {}).get('local_multicrop', {})
    if not options.get('enabled', False):
        return
    crops = batch.get('local_crops')
    size = int(options.get('input_size', cfg.get('dataset', {}).get('input_size', 224)))
    if crops is None or crops.ndim != 5 or tuple(crops.shape[-2:]) != (size, size):
        raise ValueError(f'local_multicrop enabled, but batch has no valid [B,K,3,{size},{size}] local_crops; check the dataset loader.')
