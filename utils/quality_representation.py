"""Diagnostics for the opt-in global/local quality representation."""
import torch


def dual_quality_diagnostics(model):
    owner = getattr(model, 'module', model)
    owner = getattr(owner, 'iqa', owner)
    if not getattr(owner, 'dual_quality_representation', False):
        return {}
    eta = float(torch.sigmoid(owner.quality_feature_mix_logit.detach().float()).item())
    return {'quality_feature_local_weight': eta,
            'quality_feature_global_weight': 1.0 - eta}
