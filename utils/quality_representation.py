"""Diagnostics for the opt-in global/local quality representation."""
import torch


def dual_quality_diagnostics(model):
    owner = getattr(model, 'module', model)
    owner = getattr(owner, 'iqa', owner)
    if not getattr(owner, 'dual_quality_representation', False):
        return {}
    eta = float(torch.sigmoid(owner.quality_feature_mix_logit.detach().float()).item())
    result = {'quality_feature_local_weight': eta,
              'quality_feature_global_weight': 1.0 - eta}
    if owner.quality_adapter_mode == 'residual_feature':
        result['quality_residual_alpha'] = float(torch.sigmoid(owner.quality_adapter_gate_logit.detach().float()).item())
    if getattr(owner, 'freeze_shared_projection_bn_stats', False):
        result['shared_projection_bn_stats_frozen'] = 1.0
    return result
