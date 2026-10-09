"""Two-task PCGrad on the complete shared-parameter gradient vector."""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torch.amp import GradScaler


PCGRAD_DIAGNOSTIC_KEYS = (
    "active", "conflict", "cosine", "quality_grad_norm", "align_grad_norm",
    "grad_norm_ratio", "update_norm_before", "update_norm_after",
    "update_norm_ratio", "nonfinite", "shared_numel",
)


def global_shared_pcgrad_backward(
    loss_quality: torch.Tensor,
    loss_align: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    *,
    quality_active: bool = True,
    align_active: bool = True,
    projection_strength: float = 1.0,
) -> Dict[str, float]:
    """Set scaled gradients; the caller still unscales/clips/steps as usual.

    All parameters reached by BOTH tasks form one vector. Reductions across
    tensors are equivalent to concatenation, without allocating that vector.
    Task-private gradients are summed without projection. Diagnostics refer
    to the unscaled shared gradient, BEFORE clipping and the optimizer step.
    No extra forward pass is performed, so Dropout and BatchNorm are untouched.
    Strength eta blends summed and projected shared gradients:
    (1-eta)*(gQ+gA) + eta*gPCGrad. eta=1 preserves the original algorithm;
    eta=0 performs ordinary summed-loss backward, without geometry diagnostics.
    """
    projection_strength = float(projection_strength)
    if not math.isfinite(projection_strength) or not 0.0 <= projection_strength <= 1.0:
        raise ValueError("PCGrad projection_strength must be finite and in [0, 1].")
    stats = dict.fromkeys(PCGRAD_DIAGNOSTIC_KEYS, 0.0)
    optimizer.zero_grad(set_to_none=True)
    if projection_strength == 0.0 or not (quality_active and align_active):
        # Includes 1K and batches with missing labels. Preserve the ordinary
        # summed objective exactly; validity flags only decide whether to project.
        total = loss_quality + loss_align
        if total.requires_grad:
            scaler.scale(total).backward()
        return stats
    if not (loss_quality.requires_grad and loss_align.requires_grad):
        total = loss_quality + loss_align
        if total.requires_grad:
            scaler.scale(total).backward()
        return stats

    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    params = [p for p in model.parameters() if p.requires_grad and id(p) in optimizer_ids]
    scaler.scale(loss_quality).backward(retain_graph=True)
    qgrads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss_align).backward()
    # Reuse parameter.grad rather than cloning a second complete gradient set.
    agrads = [None if p.grad is None else p.grad.detach() for p in params]
    shared = [i for i, (q, a) in enumerate(zip(qgrads, agrads)) if q is not None and a is not None]
    scale = float(scaler.get_scale())
    for grad in qgrads + agrads:
        if grad is not None:
            grad.mul_(1.0 / scale)

    project = False
    q_factor = a_factor = 0.0
    if shared:
        stats["shared_numel"] = float(sum(params[i].numel() for i in shared))
        # FP64 reductions and removing the AMP scale keep diagnostics and
        # projection coefficients independent of the loss-scaling magnitude.
        dot = torch.stack([(qgrads[i] * agrads[i]).sum(dtype=torch.float64) for i in shared]).sum()
        q2 = torch.stack([qgrads[i].square().sum(dtype=torch.float64) for i in shared]).sum()
        a2 = torch.stack([agrads[i].square().sum(dtype=torch.float64) for i in shared]).sum()
        finite = bool(torch.isfinite(torch.stack([dot, q2, a2])).all().item())
        if finite:
            dot_value, q2_value, a2_value = dot.item(), q2.item(), a2.item()
            qnorm, anorm = math.sqrt(q2_value), math.sqrt(a2_value)
            before = math.sqrt(max(0.0, q2_value + a2_value + 2.0 * dot_value))
            stats.update(quality_grad_norm=qnorm, align_grad_norm=anorm, update_norm_before=before)
            if qnorm > 0.0 and anorm > 0.0:
                stats["active"] = 1.0
                stats["cosine"] = max(-1.0, min(1.0, dot_value / (qnorm * anorm)))
                stats["grad_norm_ratio"] = qnorm / anorm
                project = dot_value < 0.0
                stats["conflict"] = float(project)
                if project:
                    q_factor = dot_value / a2_value
                    a_factor = dot_value / q2_value
        else:
            # Do not hide overflow with a zero/fake finite gradient. Leave the
            # nonfinite sum for GradScaler.unscale_/step to detect and skip.
            stats["nonfinite"] = 1.0

    for i, (parameter, qgrad, agrad) in enumerate(zip(params, qgrads, agrads)):
        if qgrad is None and agrad is None:
            parameter.grad = None
        elif qgrad is None:
            parameter.grad = agrad
        elif agrad is None:
            parameter.grad = qgrad
        elif project:
            # (1-eta)*(q+a) + eta*(q'+a')
            # = (1-eta*dot/|q|^2)q + (1-eta*dot/|a|^2)a.
            # Coefficients are the SAME for every shared parameter tensor.
            agrad.mul_(1.0 - projection_strength * q_factor).add_(
                qgrad, alpha=1.0 - projection_strength * a_factor
            )
            parameter.grad = agrad
        else:
            parameter.grad = agrad.add_(qgrad)

    if shared and not stats["nonfinite"]:
        after2 = torch.stack([params[i].grad.square().sum(dtype=torch.float64) for i in shared]).sum()
        after = math.sqrt(max(0.0, after2.item()))
        stats["update_norm_after"] = after
        before = stats["update_norm_before"]
        # Exact cancellation has no meaningful ratio. Zero marks that case.
        stats["update_norm_ratio"] = after / before if before > 0.0 else 0.0
    for parameter in params:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    return stats
