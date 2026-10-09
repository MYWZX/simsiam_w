# 扰动排序 loss、MOS label ranking loss
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_soft_ranking_loss(score_clean, score_pert, tau=0.15, reduction="mean"):
    if score_clean.dim() > 1:
        score_clean = score_clean.view(-1)
    if score_pert.dim() > 1:
        score_pert = score_pert.view(-1)

    logits = (score_clean - score_pert) / tau
    target = torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)


def multi_perturbation_ranking_loss(score_clean, pert_scores, tau=0.15):
    if len(pert_scores) == 0:
        return score_clean.new_tensor(0.0)
    losses = [pairwise_soft_ranking_loss(score_clean, s_pert, tau=tau, reduction="mean") for s_pert in pert_scores]
    return sum(losses) / len(losses)



def masked_pairwise_label_ranking_loss(pred, target, mask=None, margin: float = 0.02):
    """Pairwise label ranking loss for IQA metrics such as SROCC/KROCC.

    If target_i > target_j + margin, encourage pred_i > pred_j + margin.

    Args:
        pred:   Tensor [B], predicted score.
        target: Tensor [B], ground-truth score.
        mask:   Optional Tensor [B], 1 means this sample has valid label.
        margin: Minimum label/prediction gap used by hinge ranking.

    Returns:
        Scalar loss. Returns 0 if no valid pair exists.
    """
    pred = pred.view(-1)
    target = target.view(-1).to(pred.device).float()

    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool, device=pred.device)
    else:
        valid = mask.view(-1).to(pred.device).float() > 0.5

    pair_valid = valid.unsqueeze(1) & valid.unsqueeze(0)

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)  # target_i - target_j
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)        # pred_i - pred_j

    pair_mask = pair_valid & (target_diff > margin)
    if pair_mask.sum() == 0:
        return pred.new_tensor(0.0)

    return F.relu(margin - pred_diff[pair_mask]).mean()


def _soft_rank_1d(x: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """Differentiable approximate ascending rank.

    rank_i = 1 + sum_j sigmoid((x_j - x_i)/tau)
    Lower x has larger contribution from other samples and therefore higher
    numerical rank; correlation is invariant to affine rank scaling.
    """
    x = x.view(-1)
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # x_j - x_i
    return 1.0 + torch.sigmoid(diff / max(float(tau), 1e-6)).sum(dim=1)


def masked_soft_spearman_loss(pred, target, mask=None, tau: float = 0.02, eps: float = 1e-6):
    """Soft Spearman loss for directly improving SROCC.

    It computes differentiable soft ranks for prediction and target inside the
    mini-batch, then minimizes 1 - PearsonCorr(soft_rank(pred), soft_rank(target)).
    Use it as a small auxiliary loss together with MSE and pairwise ranking.
    """
    pred = pred.view(-1)
    target = target.view(-1).to(pred.device).float()
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool, device=pred.device)
    else:
        valid = mask.view(-1).to(pred.device).float() > 0.5
    if valid.sum() < 3:
        return pred.new_tensor(0.0)

    p = pred[valid]
    t = target[valid]
    pr = _soft_rank_1d(p, tau=tau)
    # The target rank can be hard through argsort, but soft target rank is stable
    # when labels contain ties/noise.
    tr = _soft_rank_1d(t, tau=tau)

    pr = pr - pr.mean()
    tr = tr - tr.mean()
    corr = (pr * tr).sum() / (torch.sqrt((pr ** 2).sum() + eps) * torch.sqrt((tr ** 2).sum() + eps))
    return 1.0 - corr


def masked_hard_pairwise_label_ranking_loss(pred, target, mask=None, margin: float = 0.02, topk_ratio: float = 0.35):
    """Hard-pair ranking loss for SROCC/KROCC.

    This is a stronger version of masked_pairwise_label_ranking_loss. It first
    computes all valid pairwise hinge losses, then only keeps the hardest top-k
    pairs. It focuses training on order mistakes that most affect SROCC.
    """
    pred = pred.view(-1)
    target = target.view(-1).to(pred.device).float()
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool, device=pred.device)
    else:
        valid = mask.view(-1).to(pred.device).float() > 0.5
    pair_valid = valid.unsqueeze(1) & valid.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    pair_mask = pair_valid & (target_diff > margin)
    if pair_mask.sum() == 0:
        return pred.new_tensor(0.0)
    losses = F.relu(margin - pred_diff[pair_mask])
    if losses.numel() == 0:
        return pred.new_tensor(0.0)
    k = max(1, int(losses.numel() * float(topk_ratio)))
    if k < losses.numel():
        losses = torch.topk(losses, k=k, largest=True).values
    return losses.mean()


def align_prompt_ranking_loss(
    s_pos: torch.Tensor,
    s_hard: torch.Tensor,
    s_neg: torch.Tensor,
    mask=None,
    margin_pos_hard: float = 0.05,
    margin_hard_neg: float = 0.05,
):
    """Prompt ranking loss for the alignment branch.

    For the same image, encourage:
      S(image, positive_prompt) > S(image, hard_prompt) > S(image, negative_prompt)

    This supervises image-text matching order directly, instead of relying only
    on noisy scalar align MOS regression.
    """
    s_pos = s_pos.view(-1)
    s_hard = s_hard.view(-1).to(s_pos.device)
    s_neg = s_neg.view(-1).to(s_pos.device)

    if mask is None:
        weight = torch.ones_like(s_pos, dtype=s_pos.dtype, device=s_pos.device)
    else:
        weight = mask.view(-1).to(s_pos.device).float()

    if weight.sum() <= 0:
        return s_pos.new_tensor(0.0)

    l_pos_hard = F.relu(float(margin_pos_hard) - (s_pos - s_hard))
    l_hard_neg = F.relu(float(margin_hard_neg) - (s_hard - s_neg))
    loss_vec = l_pos_hard + l_hard_neg
    return (loss_vec * weight).sum() / (weight.sum() + 1e-6)


def align_prompt_pair_ranking_loss(
    s_pos: torch.Tensor,
    s_hard: torch.Tensor,
    mask=None,
    margin: float = 0.03,
):
    """Weak prompt ranking loss: positive prompt should score above hard prompt."""
    s_pos = s_pos.view(-1)
    s_hard = s_hard.view(-1).to(s_pos.device)

    if mask is None:
        weight = torch.ones_like(s_pos, dtype=s_pos.dtype, device=s_pos.device)
    else:
        weight = mask.view(-1).to(s_pos.device).float()

    if weight.sum() <= 0:
        return s_pos.new_tensor(0.0)

    loss_vec = F.relu(float(margin) - (s_pos - s_hard))
    return (loss_vec * weight).sum() / (weight.sum() + 1e-6)
