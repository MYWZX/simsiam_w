"""Observational gradient probe on the SAME training forward, before backward.

No optimizer/PCGrad, additional forward, sampling, or writes to parameter.grad.
"""
import hashlib
import json
from pathlib import Path
import math
import torch

GROUPS = {
    'clip_visual': 'backbone.global_encoder.',
    'resnet_local': 'backbone.local_encoder.',
    'global_projection': 'backbone.fusion.semantic_proj.',
}


def probe_task_gradients(model, quality, alignment, loss_scale=1.0):
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("Gradient probe loss_scale must be finite and positive")
    named = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and any(n.startswith(prefix) for prefix in GROUPS.values())]
    parameters = [p for _, p in named]
    if not parameters:
        return {name: {'status': 'no_trainable_parameters'} for name in GROUPS}
    def gradients(loss):
        return (torch.autograd.grad(loss * loss_scale, parameters, allow_unused=True, retain_graph=True)
                if loss.requires_grad else (None,) * len(parameters))
    q, a = gradients(quality), gradients(alignment)
    result = {}
    for name, prefix in GROUPS.items():
        indices = [i for i, (key, _) in enumerate(named) if key.startswith(prefix)]
        if not indices:
            result[name] = {'status': 'no_trainable_parameters'}
            continue
        # Sum tensorwise rather than materializing one enormous concatenation.
        q2 = a2 = dot = 0.0
        shared = q_only = a_only = absent = 0
        for i in indices:
            x, y = q[i], a[i]
            x = None if x is None else x.detach().double() / loss_scale
            y = None if y is None else y.detach().double() / loss_scale
            if x is not None: q2 += float(x.square().sum())
            if y is not None: a2 += float(y.square().sum())
            if x is not None and y is not None:
                dot += float((x * y).sum()); shared += 1
            elif x is not None: q_only += 1
            elif y is not None: a_only += 1
            else: absent += 1
        finite = all(math.isfinite(v) for v in (q2, a2, dot))
        qn, an = math.sqrt(q2), math.sqrt(a2)
        result[name] = dict(status='ok' if finite else 'nonfinite',
            quality_norm=qn if finite else None, align_norm=an if finite else None,
            cosine=max(-1., min(1., dot / (qn*an))) if finite and qn and an else None,
            align_over_quality=an/qn if finite and qn else None,
            shared_tensors=shared, quality_only_tensors=q_only,
            align_only_tensors=a_only, absent_tensors=absent)
    return result


def maybe_record_probe(model, losses, cfg, epoch, step, loss_scale=1.0, scaler=None):
    probe = cfg.get('train', {}).get('task_gradient_probe', {})
    if not probe: return
    if (set(probe) - {'version', 'epochs', 'batches'} or probe.get('version') != 'same_forward_v1'
            or not isinstance(probe.get('batches'), int) or not 1 <= probe['batches'] <= 4
            or not isinstance(probe.get('epochs'), list)
            or any(type(e) is not int or e < 1 for e in probe['epochs'])):
        raise ValueError('Invalid bounded same_forward_v1 gradient probe')
    if epoch not in probe['epochs'] or step >= probe['batches']: return
    if scaler is not None: loss_scale = float(scaler.get_scale())
    stats = probe_task_gradients(model, losses['loss_quality_task'], losses['loss_align_task'], loss_scale=loss_scale)
    path = Path(cfg['output_dir']) / 'task_gradient_probe.jsonl'
    record = dict(version='same_forward_v1', epoch=epoch, batch=step+1,
                  model_mode='train', loss_scale=loss_scale, objective='weighted_quality_and_alignment_tasks', groups=stats)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    print('[TASK GRADIENT]', json.dumps(record, allow_nan=False))


def new_region_parameter_record(model):
    """Explicitly declare the only added trainable parameters, never inherited."""
    records = []
    for name, p in model.named_parameters():
        if name.startswith('align_region_word.'):
            data = p.detach().float().cpu().contiguous().numpy().tobytes()
            records.append(dict(name=name, shape=list(p.shape), initialization='new_random',
                                sha256=hashlib.sha256(data).hexdigest()))
    return records
