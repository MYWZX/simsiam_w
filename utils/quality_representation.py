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
    if getattr(owner, 'shared_simsiam_projection', False) and not getattr(owner, 'shared_projection_project_patches', True):
        result['align_patch_uses_shared_pi'] = 0.0
    if getattr(owner, 'align_control_enabled', False):
        result.update({
            'align_control_global_output': float(owner.align_output_mode == 'global'),
            'align_control_half_mix': float(owner.align_output_mode == 'half'),
            'align_control_uniform_pool': float(owner.align_local_pool == 'uniform'),
            'align_control_gate_trainable': float(any(p.requires_grad for p in owner.align_gate.parameters())),
            'align_control_match_trainable': float(any(p.requires_grad for m in
                (owner.align_patch_img_proj, owner.align_patch_txt_proj) for p in m.parameters())),
        })
    if getattr(owner, 'align_local_temperature', 1.0) != 1.0:
        result['align_local_temperature'] = owner.align_local_temperature
    return result
