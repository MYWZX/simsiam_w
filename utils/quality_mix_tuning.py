"""Observe the actual post-load score-mixing beta; never reinitialize weights."""
import math


def record_quality_mix_start(model, cfg):
    if not cfg.get('train', {}).get('record_quality_mix_initialization', False):
        return None
    import torch
    parameter = model.quality_mix_logit
    if (not getattr(model.backbone, 'projection_only', False)
            or model.quality_global_only or model.quality_sample_gate is not None
            or not parameter.requires_grad):
        raise ValueError('Beta-start control requires learnable scalar mixing of global/local scores')
    logit = float(cfg['model']['nss_text_gate']['quality_mix_init'])
    expected = float(torch.sigmoid(torch.tensor(logit)).item())
    actual = float(parameter.detach().sigmoid().item())
    if not math.isfinite(logit) or not math.isclose(actual, expected, abs_tol=1e-7, rel_tol=0):
        raise ValueError('Quality beta start was overwritten during checkpoint loading')
    record = dict(config_logit=logit, applied_beta=actual, trainable=True)
    print(f'[INFO] Post-load quality score beta initialization: {record}')
    return record
