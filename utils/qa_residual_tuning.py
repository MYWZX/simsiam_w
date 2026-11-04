"""Explicit downstream scalar override, applied only after representation load."""
import math


def apply_post_load_qa_alpha(model, cfg, checkpoint_path):
    value = cfg.get('train', {}).get('post_load_qa_alpha')
    if value is None:
        return None
    import torch
    alpha = float(value)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('post_load_qa_alpha must be finite and strictly in (0,1)')
    if (not checkpoint_path or not getattr(model.backbone, 'projection_only', False)
            or not getattr(model, 'dual_quality_representation', False)
            or getattr(model, 'quality_adapter_mode', None) != 'residual_feature'
            or getattr(model, 'quality_adapter_residual_fixed', None) is not None
            or model.quality_adapter_gate_logit is None
            or not model.quality_adapter_gate_logit.requires_grad):
        raise ValueError('QA alpha tuning requires loaded projection-only dual residual representation with learnable alpha')
    parameter = model.quality_adapter_gate_logit
    before = float(parameter.detach().sigmoid().item())
    with torch.no_grad():
        parameter.fill_(math.log(alpha / (1-alpha)))
    record = dict(requested=alpha, loaded_alpha=before,
                  applied_alpha=float(parameter.detach().sigmoid().item()), trainable=True)
    print(f'[INFO] Explicit post-load QA alpha initialization: {record}; all other loaded tensors retained.')
    return record
