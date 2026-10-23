# Cross-modal pretraining entry: four SimSiam relations plus optional IQA supervision.
from __future__ import annotations

import argparse
import json
import math
import os
import re
import random
import time
import numpy as np
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import build_dataloaders
from models.two_stage_models import CrossModalIQAJointPretrain, CrossModalSimSiamPretrain, simsiam_pair_loss
from utils.train_utils import AverageMeter, ensure_dir, load_yaml, set_seed
from utils.experiment_logger import CSVExperimentLogger, dump_config_snapshot, make_run_id, resolve_results_csv
from utils.losses import masked_pairwise_label_ranking_loss, masked_soft_spearman_loss, masked_hard_pairwise_label_ranking_loss


from utils.clip_experiments import configure_clip_large, configure_local_feature_stage, local_stage_diagnostics


from utils.pretrain_resume import atomic_save, legacy_resume_records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local-feature-stage", choices=("layer3", "layer4", "layer3_4"), default=None, help="Select ResNet output stage for new pretraining; otherwise use YAML (legacy default: layer4).")
    p.add_argument("--disable-pretrain-pcgrad", action="store_true", help="Explicitly disable pretraining PCGrad for a NEW controlled pretraining run.")
    p.add_argument("--resume", type=str, default=None, help="Restore a complete pretraining checkpoint and continue at epoch+1.")
    p.add_argument("--max-runtime-hours", type=float, default=None, help="Soft runtime budget: stop between epochs, leaving a complete resumable checkpoint.")
    p.add_argument("--clip-large", action="store_true", help="Use matching OpenAI ViT-L/14 visual/text encoders; requires new pretraining.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--results-csv", type=str, default=None)
    p.add_argument(
        "--text-context-tokens", type=int, default=None,
        help="Train N learnable CLIP text-prefix tokens (0 disables; recommended N=4).",
    )
    p.add_argument(
        "--quality-text-levels", action="store_true",
        help="Train a persistent five-word CLIP quality bank on Clean View only.",
    )
    p.add_argument(
        "--align-text-levels", action="store_true",
        help="Train a separate five-word image-Prompt matching bank on Clean View.",
    )
    p.add_argument(
        "--isolate-text-context-rng", action="store_true",
        help="Isolate prefix initialization RNG for matched no-prefix vs prefix ablations.",
    )
    p.add_argument(
        "--disable-regular-simsiam", action="store_true",
        help="Diagnostic only: zero four regular SimSiam weights, preserving human/local/anchor supervision.",
    )
    p.add_argument(
        "--fixed-epochs", type=int, default=None,
        help="Train exactly N epochs without early stopping; preserve the YAML scheduler horizon.",
    )
    p.add_argument(
        "--quality-balanced-sampling",
        action="store_true",
        help="Enable Artifact/Aesthetics label-balanced replacement sampling for training rows.",
    )
    p.add_argument(
        "--quality-balanced-sampling-alpha",
        type=float,
        default=None,
        help="Mixing strength for quality-balanced sampling: 0=uniform, 1=full balanced.",
    )
    p.add_argument(
        "--quality-loss-reweighting",
        action="store_true",
        help="Keep uniform sampling and reweight only Artifact/Aesthetics anchor losses.",
    )
    p.add_argument(
        "--quality-loss-reweighting-alpha",
        type=float,
        default=None,
        help="Mixing strength for quality-only loss reweighting: 0=uniform, 1=full balanced.",
    )
    p.add_argument(
        "--mirror-augmentation",
        action="store_true",
        help="Mirror directional training samples and swap left/right words in Prompt/Answer.",
    )
    p.add_argument(
        "--mirror-augmentation-probability",
        type=float,
        default=None,
        help="Probability of mirroring an eligible directional sample (default: 0.5).",
    )
    p.add_argument(
        "--disable-final-pretrain-loss",
        action="store_true",
        help=(
            "Ablation: disable only the pretraining supervision on the fused final "
            "score; keep Quality/Align supervision, the Final head, and downstream "
            "architecture unchanged."
        ),
    )
    p.add_argument(
        "--disable-quality-auxiliary",
        action="store_true",
        help=(
            "Ablation: disable the dataset-specific Quality Auxiliary heads and "
            "their losses; keep Component Anchor supervision as the only "
            "Artifact/Aesthetics path."
        ),
    )
    p.add_argument(
        "--quality-aux-scale",
        type=float,
        default=None,
        help=(
            "Multiply every Quality Auxiliary component loss weight by this "
            "factor (for example 0.1 or 0.5); 0 disables the auxiliary task."
        ),
    )
    p.add_argument(
        "--disable-component-anchor",
        action="store_true",
        help="Ablation: disable Component Anchor/Ordinal supervision and heads.",
    )
    p.add_argument(
        "--disable-quality-align-reg-rank",
        action="store_true",
        help="Ablation: disable pretraining Quality/Align regression and ranking losses only.",
    )
    p.add_argument(
        "--pretrain-selection-metric", type=str, default=None,
        help="Override train.pretrain_selection_metric for an ablation (e.g. val_quality_proxy_SROCC).",
    )
    p.add_argument(
        "--pretrain-selection-mode", choices=("max", "min"), default=None,
        help="Override train.pretrain_selection_mode for an ablation.",
    )
    return p.parse_args()


REGULAR_SIMSIAM_WEIGHT_KEYS = (
    "lambda_img_siam", "lambda_img_answer", "lambda_img_prompt", "lambda_prompt_answer",
)


def configure_research_controls(cfg: Dict, args: argparse.Namespace) -> None:
    """Apply explicit diagnostic overrides before recording the effective config."""
    if getattr(args, "disable_pretrain_pcgrad", False):
        cfg.setdefault("train", {}).setdefault("pcgrad", {})["enabled"] = False
    print(f"[INFO] Pretraining PCGrad enabled={bool(cfg.get('train', {}).get('pcgrad', {}).get('enabled', False))}.")
    fixed_epochs = args.fixed_epochs
    planned = fixed_epochs if fixed_epochs is not None else cfg.get("train", {}).get("fixed_training_epochs")
    if planned is not None:
        scheduler_epochs = int(cfg.get("train", {}).get("epochs", 40))
        if isinstance(planned, bool) or not isinstance(planned, int) or not 1 <= planned <= scheduler_epochs:
            raise ValueError(f"Fixed training epochs must be an integer in [1, {scheduler_epochs}].")
    if args.disable_regular_simsiam:
        train_cfg = cfg.setdefault("train", {})
        for key in REGULAR_SIMSIAM_WEIGHT_KEYS:
            train_cfg[key] = 0.0
        # Some legacy YAMLs override the image-image coefficient with this
        # alias. It is the same objective, not an additional quality loss.
        if "lambda_quality_view_siam" in train_cfg:
            train_cfg["lambda_quality_view_siam"] = 0.0
        print("[INFO] Diagnostic: four regular SimSiam weights=0; modules, views, human, local and anchor supervision unchanged.")
    if args.disable_quality_auxiliary:
        train_cfg = cfg.setdefault("train", {})
        train_cfg.setdefault("quality_auxiliary_supervision", {})["enabled"] = False
        cfg.setdefault("model", {}).setdefault("quality_auxiliary_heads", {})["enabled"] = False
        print("[INFO] Ablation: Artifact/Aesthetics auxiliary heads and losses disabled.")
    if args.disable_component_anchor:
        train_cfg = cfg.setdefault("train", {})
        train_cfg.setdefault("quality_component_anchor_ordinal", {})["enabled"] = False
        print("[INFO] Ablation: Component Anchor/Ordinal supervision disabled.")
    if args.disable_quality_align_reg_rank:
        train_cfg = cfg.setdefault("train", {})
        for key in ("lambda_quality_reg", "lambda_quality_rank", "lambda_align_pretrain_reg", "lambda_align_pretrain_rank"):
            train_cfg[key] = 0.0
        print("[INFO] Ablation: pretraining Quality/Align regression and ranking losses disabled.")
    if args.pretrain_selection_metric is not None:
        cfg.setdefault("train", {})["pretrain_selection_metric"] = str(args.pretrain_selection_metric)
    if args.pretrain_selection_mode is not None:
        cfg.setdefault("train", {})["pretrain_selection_mode"] = str(args.pretrain_selection_mode)
    if fixed_epochs is not None:
        cfg.setdefault("train", {})["fixed_training_epochs"] = int(fixed_epochs)
    if planned is not None:
        print(f"[INFO] Fixed training budget: {planned} epochs, early stopping disabled, cosine T_max={scheduler_epochs} unchanged.")


def unpack_loaders(loaders):
    return loaders if len(loaders) == 3 else (loaders[0], loaders[1], None)



def _extract_representation_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only encoder/Fusion/text projection weights used by downstream IQA."""
    prefixes = (
        "iqa.backbone.global_encoder.",
        "iqa.backbone.local_encoder.",
        "iqa.backbone.fusion.",
        "iqa.backbone.text_encoder.context_vectors",
        "iqa.text_to_fused.",
        "iqa.shared_image_projector.",
        "iqa.shared_text_projector.",
        "iqa.quality_adapter.",
        "iqa.quality_text_bank.",
        "iqa.align_text_bank.",
        "iqa.quality_image_projector.",
        "iqa.align_global_adapter.",
        "iqa.align_text_adapter.",
        "iqa.align_patch_adapter.",
        "iqa.align_vmamba.",
        "backbone.global_encoder.",
        "backbone.local_encoder.",
        "backbone.fusion.",
        "backbone.text_encoder.context_vectors",
        "text_to_fused.",
        "shared_image_projector.",
        "shared_text_projector.",
        "quality_adapter.",
        "quality_text_bank.",
        "align_text_bank.",
        "quality_image_projector.",
        "align_global_adapter.",
        "align_text_adapter.",
        "align_patch_adapter.",
        "align_vmamba.",
    )
    compact = {key: value for key, value in state.items() if key.startswith(prefixes)}
    if not compact:
        raise RuntimeError("No downstream representation weights found in pretraining state.")
    return compact
def build_optimizer(cfg: Dict, model: torch.nn.Module):
    tcfg = cfg["train"]
    base_lr = float(tcfg.get("lr", 2e-5))
    weight_decay = float(tcfg.get("weight_decay", 1e-3))
    lrcfg = tcfg.get("layerwise_lr", {})
    if not bool(lrcfg.get("enabled", False)):
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    groups = {
        "global_encoder": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("global_encoder_mult", lrcfg.get("backbone_mult", 0.1))),
        },
        "local_encoder": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("local_encoder_mult", lrcfg.get("backbone_mult", 0.1))),
        },
        "fusion": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("fusion_mult", 0.5)),
        },
        "shared_projection": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("shared_projection_mult", 0.5)),
        },
        "align_features": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("align_feature_mult", 1.0)),
        },
        "align_patch": {
            "params": [],
            "lr": base_lr * float(
                lrcfg.get("align_patch_mult", lrcfg.get("align_feature_mult", 1.0))
            ),
        },
        "align_gate": {
            "params": [],
            "lr": base_lr * float(
                lrcfg.get("align_gate_mult", lrcfg.get("iqa_head_mult", 5.0))
            ),
        },
        "iqa_heads": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("iqa_head_mult", 5.0)),
        },
        "simsiam_heads": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("simsiam_head_mult", 5.0)),
        },
        "other": {
            "params": [],
            "lr": base_lr * float(lrcfg.get("other_mult", 1.0)),
        },
    }

    iqa_head_prefixes = (
        "local_patch_proj",
        "nss_proj",
        "nss_gate",
        "text_gate",
        "patch_quality_head",
        "global_quality_head",
        "quality_mix_logit",
        "quality_aux_heads",
        "align_global_head",
        "align_patch_head",
        "fusion_logit",
        "quality_probe",
        "quality_adapter",
        "quality_text_bank",
        "align_text_bank",
    )
    align_feature_prefixes = (
        "align_global_adapter",
        "align_text_adapter",
    )
    align_patch_prefixes = (
        "align_patch_img_proj",
        "align_patch_txt_proj",
        "align_patch_adapter",
        "align_vmamba",
    )
    simsiam_prefixes = (
        "shared_image_projector",
        "shared_text_projector",
        "image_projector",
        "text_projector",
        "image_predictor",
        "text_predictor",
        "quality_image_projector",
        "quality_image_predictor",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Joint pretraining stores the persistent model below ``iqa`` while
        # legacy pure-SimSiam pretraining exposes ``backbone`` at the root.
        canonical_name = name.removeprefix("module.").removeprefix("iqa.")
        if canonical_name.startswith("backbone.global_encoder"):
            group_name = "global_encoder"
        elif canonical_name.startswith("backbone.local_encoder"):
            group_name = "local_encoder"
        elif canonical_name.startswith("backbone.fusion"):
            group_name = "fusion"
        elif canonical_name.startswith("text_to_fused"):
            group_name = "shared_projection"
        elif canonical_name.startswith("align_gate"):
            group_name = "align_gate"
        elif canonical_name.startswith(align_patch_prefixes):
            group_name = "align_patch"
        elif canonical_name.startswith(align_feature_prefixes):
            group_name = "align_features"
        elif canonical_name.startswith(iqa_head_prefixes):
            group_name = "iqa_heads"
        elif canonical_name.startswith(simsiam_prefixes):
            group_name = "simsiam_heads"
        else:
            group_name = "other"
        groups[group_name]["params"].append(param)

    param_groups = []
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grouped_count = 0
    summary = []
    for group_name, group in groups.items():
        if not group["params"]:
            continue
        parameter_count = sum(p.numel() for p in group["params"])
        grouped_count += parameter_count
        param_groups.append({
            "params": group["params"],
            "lr": float(group["lr"]),
            "weight_decay": weight_decay,
            "name": group_name,
        })
        summary.append(f"{group_name}={group['lr']:.2e} ({parameter_count:,})")

    if grouped_count != trainable_count:
        raise RuntimeError(
            f"Layerwise optimizer grouped {grouped_count:,} parameters, "
            f"but the model has {trainable_count:,} trainable parameters."
        )
    if not param_groups:
        raise RuntimeError("Layerwise optimizer found no trainable parameters.")

    print("[INFO] Pretrain layerwise LR enabled: " + ", ".join(summary))
    return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)




def _safe_corrcoef_np(y_true, y_pred, kind: str = "spearman") -> float:
    """Small dependency-light correlation helper for pretraining validation."""
    try:
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) < 3:
            return 0.0
        fn = spearmanr if kind == "spearman" else pearsonr
        v = fn(y_true, y_pred)[0]
        return 0.0 if np.isnan(v) else float(v)
    except Exception:
        return 0.0


def _legacy_pretrain_proxy(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Legacy joint proxy retained for backward compatibility.

    It combines image-prompt and image-answer cosine similarities. The score is
    not a final IQA output; it is a lightweight ordering signal for MOS-based
    ranking and soft-SROCC auxiliary losses.
    """
    img_feat = F.normalize(out["img1_z"], dim=1)
    prompt_feat = F.normalize(out["prompt_z"], dim=1)
    answer_feat = F.normalize(out["answer_z"], dim=1)
    return 0.5 * (
        F.cosine_similarity(img_feat, prompt_feat, dim=1)
        + F.cosine_similarity(img_feat, answer_feat, dim=1)
    )


def _pretrain_align_proxy(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Image-prompt alignment proxy averaged across the two image views."""
    img1 = F.normalize(out["img1_z"], dim=1)
    img2 = F.normalize(out["img2_z"], dim=1)
    prompt = F.normalize(out["prompt_z"], dim=1)
    return 0.5 * (
        F.cosine_similarity(img1, prompt, dim=1)
        + F.cosine_similarity(img2, prompt, dim=1)
    )


def _pretrain_rank_loss(
    proxy: torch.Tensor,
    target: torch.Tensor,
    tcfg: Dict,
    margin: float | None = None,
) -> torch.Tensor:
    margin = float(tcfg.get("rank_margin", 0.05) if margin is None else margin)
    if bool(tcfg.get("pretrain_use_hard_rank", False)):
        return masked_hard_pairwise_label_ranking_loss(
            proxy,
            target,
            mask=torch.ones_like(target),
            margin=margin,
            topk_ratio=float(tcfg.get("pretrain_hard_rank_ratio", 0.35)),
        )
    return masked_pairwise_label_ranking_loss(
        proxy,
        target,
        mask=torch.ones_like(target),
        margin=margin,
    )


def _weighted_mean(loss_vec: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.view(-1).to(loss_vec.device).float()
    if weight.sum() <= 0:
        return loss_vec.new_tensor(0.0)
    return (loss_vec.view(-1) * weight).sum() / (weight.sum() + 1e-6)


def _prompt_answer_gap_loss(out: Dict[str, torch.Tensor], weight: torch.Tensor, max_similarity: float = 0.20) -> torch.Tensor:
    """Push low-confidence prompt-answer pairs away in projected text space."""
    prompt_z = F.normalize(out["prompt_z"], dim=1)
    answer_z = F.normalize(out["answer_z"], dim=1)
    similarity = F.cosine_similarity(prompt_z, answer_z, dim=1)
    return _weighted_mean(F.relu(similarity - float(max_similarity)), weight)


_RICH_HF_WORD_SPLIT = re.compile(r'[,\.?!":;\s]+')


def _parse_richhf_prompt_phrases(prompts, serialized_labels, device):
    """Split each prompt into human-labeled aligned and misaligned phrases."""
    aligned_phrases = []
    misaligned_phrases = []
    aligned_valid = []
    misaligned_valid = []
    for prompt, raw_labels in zip(prompts, serialized_labels):
        prompt_words = [w for w in _RICH_HF_WORD_SPLIT.split(str(prompt)) if w]
        try:
            labels = json.loads(str(raw_labels))
        except (TypeError, ValueError, json.JSONDecodeError):
            labels = [x for x in re.split(r"[\s,]+", str(raw_labels).strip("[] ")) if x]
        if not isinstance(labels, list):
            labels = []
        try:
            labels = [int(x) for x in labels]
        except (TypeError, ValueError):
            labels = []
        usable = min(len(prompt_words), len(labels))
        aligned = [word for word, label in zip(prompt_words[:usable], labels[:usable]) if label == 1]
        misaligned = [word for word, label in zip(prompt_words[:usable], labels[:usable]) if label == 0]
        fallback = str(prompt).strip() or "image"
        aligned_phrases.append(" ".join(aligned) if aligned else fallback)
        misaligned_phrases.append(" ".join(misaligned) if misaligned else fallback)
        aligned_valid.append(float(bool(aligned)))
        misaligned_valid.append(float(bool(misaligned)))
    return (
        aligned_phrases,
        misaligned_phrases,
        torch.tensor(aligned_valid, dtype=torch.float32, device=device),
        torch.tensor(misaligned_valid, dtype=torch.float32, device=device),
    )


def _map_to_patch_weights(target_map: torch.Tensor, n: int) -> torch.Tensor:
    h = int(round(n ** 0.5))
    w = n // max(1, h)
    if h * w != n:
        h, w = 1, n
    target = F.interpolate(target_map.float(), size=(h, w), mode="bilinear", align_corners=False)
    return target.flatten(1).clamp(0.0, 1.0)


def _simsiam_push_apart(
    p_a: torch.Tensor,
    z_b: torch.Tensor,
    p_b: torch.Tensor,
    z_a: torch.Tensor,
    weight: torch.Tensor,
    max_similarity: float,
) -> torch.Tensor:
    similarity = 0.5 * (
        F.cosine_similarity(F.normalize(p_a, dim=-1), F.normalize(z_b.detach(), dim=-1), dim=-1)
        + F.cosine_similarity(F.normalize(p_b, dim=-1), F.normalize(z_a.detach(), dim=-1), dim=-1)
    )
    return _weighted_mean(F.relu(similarity - float(max_similarity)), weight)


def _symmetric_simsiam_similarity(
    p_a: torch.Tensor,
    z_b: torch.Tensor,
    p_b: torch.Tensor,
    z_a: torch.Tensor,
) -> torch.Tensor:
    """Per-sample SimSiam similarity with stop-gradient on both targets."""
    return 0.5 * (
        F.cosine_similarity(F.normalize(p_a, dim=-1), F.normalize(z_b.detach(), dim=-1), dim=-1)
        + F.cosine_similarity(F.normalize(p_b, dim=-1), F.normalize(z_a.detach(), dim=-1), dim=-1)
    )


def _simsiam_relative_rank(
    positive_a: Dict[str, torch.Tensor],
    negative_a: Dict[str, torch.Tensor],
    reference_z: torch.Tensor,
    reference_p: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Require the positive relation to exceed the negative by the margin."""
    positive_similarity = _symmetric_simsiam_similarity(
        positive_a["p"], reference_z, reference_p, positive_a["z_proj"]
    )
    negative_similarity = _symmetric_simsiam_similarity(
        negative_a["p"], reference_z, reference_p, negative_a["z_proj"]
    )
    loss_vec = F.relu(float(margin) - positive_similarity + negative_similarity)
    if scale is not None:
        loss_vec = loss_vec * scale.view(-1).to(loss_vec).float()
    return _weighted_mean(loss_vec, mask)


def _compute_richhf_relational_losses(model, out, batch, prompts, device, tcfg):
    zero = out["img1_z"].new_tensor(0.0)
    img_prompt_local = zero
    prompt_answer_word = zero
    core_model = model.module if hasattr(model, "module") else model
    relation_mode = str(tcfg.get("local_relation_loss", "pull_push")).strip().lower()
    if relation_mode not in {"pull_push", "relative_rank"}:
        raise ValueError(
            f"Unknown local_relation_loss={relation_mode!r}. "
            "Choose from: pull_push, relative_rank."
        )

    if float(tcfg.get("lambda_img_prompt_local", 0.0)) > 0.0:
        supervision_view = str(
            tcfg.get("human_supervision_view", "augmented")
        ).strip().lower()
        use_three_views = supervision_view in {"three_view", "separate_clean"}
        required = {"misalignment_map1", "misalignment_map2", "misalignment_map_valid"}
        if use_three_views:
            required.add("misalignment_map_clean")
        if not required.issubset(batch):
            raise RuntimeError(
                "Local map supervision is enabled but the RichHF batch has no maps. "
                "Export them with tools/prepare_richhf18k.py first."
            )
        map1 = batch["misalignment_map1"].to(device, non_blocking=True)
        map2 = batch["misalignment_map2"].to(device, non_blocking=True)
        valid = batch["misalignment_map_valid"].to(device, non_blocking=True)
        if use_three_views:
            map_clean = batch["misalignment_map_clean"].to(device, non_blocking=True)
            view_pairs = ((out["align_patch_features_clean"], map_clean),)
        else:
            view_pairs = (
                (out["align_patch_features1"], map1),
                (out["align_patch_features2"], map2),
            )
        if supervision_view == "clean":
            view_pairs = view_pairs[:1]
        view_losses = []
        for patch_features, spatial_map in view_pairs:
            misaligned_weight = _map_to_patch_weights(spatial_map, patch_features.shape[1])
            aligned_weight = 1.0 - misaligned_weight
            aligned_valid = valid * (aligned_weight.sum(dim=1) > 1e-4).float()
            active_threshold = float(tcfg.get("misalignment_map_active_threshold", 0.05))
            misaligned_valid = valid * (misaligned_weight.amax(dim=1) > active_threshold).float()
            aligned_region = core_model.project_patch_regions(patch_features, aligned_weight)
            misaligned_region = core_model.project_patch_regions(patch_features, misaligned_weight)
            if relation_mode == "relative_rank":
                view_losses.append(_simsiam_relative_rank(
                    aligned_region,
                    misaligned_region,
                    out["align_prompt_z"],
                    out["align_prompt_p"],
                    mask=aligned_valid * misaligned_valid,
                    margin=float(tcfg.get(
                        "img_prompt_local_rank_margin",
                        tcfg.get("img_prompt_misaligned_max_similarity", 0.10),
                    )),
                ))
            else:
                pull = simsiam_pair_loss(
                    aligned_region["p"], out["align_prompt_z"],
                    out["align_prompt_p"], aligned_region["z_proj"],
                    weight=aligned_valid,
                )
                push = _simsiam_push_apart(
                    misaligned_region["p"], out["align_prompt_z"],
                    out["align_prompt_p"], misaligned_region["z_proj"],
                    weight=misaligned_valid,
                    max_similarity=float(tcfg.get("img_prompt_misaligned_max_similarity", 0.10)),
                )
                view_losses.append(0.5 * (pull + push))
        img_prompt_local = sum(view_losses) / len(view_losses)

    if float(tcfg.get("lambda_prompt_answer_word", 0.0)) > 0.0:
        if "prompt_word_labels" not in batch:
            raise RuntimeError(
                "Prompt-word supervision is enabled but prompt_word_labels is missing from the RichHF batch."
            )
        aligned_phrases, misaligned_phrases, aligned_valid, misaligned_valid = _parse_richhf_prompt_phrases(
            prompts, batch["prompt_word_labels"], device
        )
        aligned_prompt = core_model.project_aux_texts(aligned_phrases, device)
        misaligned_prompt = core_model.project_aux_texts(misaligned_phrases, device)
        answer_conf = batch.get("answer_confidence", torch.ones_like(aligned_valid)).to(
            device, non_blocking=True
        ).float().clamp(0.0, 1.0)
        if relation_mode == "relative_rank":
            prompt_answer_word = _simsiam_relative_rank(
                aligned_prompt,
                misaligned_prompt,
                out["answer_z"],
                out["answer_p"],
                mask=aligned_valid * misaligned_valid,
                scale=answer_conf,
                margin=float(tcfg.get(
                    "prompt_answer_word_rank_margin",
                    tcfg.get("prompt_answer_misaligned_max_similarity", 0.10),
                )),
            )
        else:
            pull = simsiam_pair_loss(
                aligned_prompt["p"], out["answer_z"],
                out["answer_p"], aligned_prompt["z_proj"],
                weight=aligned_valid * answer_conf,
            )
            push = _simsiam_push_apart(
                misaligned_prompt["p"], out["answer_z"],
                out["answer_p"], misaligned_prompt["z_proj"],
                weight=misaligned_valid * answer_conf,
                max_similarity=float(tcfg.get("prompt_answer_misaligned_max_similarity", 0.10)),
            )
            prompt_answer_word = 0.5 * (pull + push)
    return img_prompt_local, prompt_answer_word



def _quality_view_simsiam_loss(out, weight):
    """Quality-safe view invariance in the quality projection space."""
    return simsiam_pair_loss(
        out.get("quality_img1_p", out["img1_p"]),
        out.get("quality_img2_z", out["img2_z"]),
        out.get("quality_img2_p", out["img2_p"]),
        out.get("quality_img1_z", out["img1_z"]),
        weight=weight,
    )


def _compute_quality_text_loss(
    out: Dict[str, torch.Tensor],
    quality_target: torch.Tensor,
    sample_weight: torch.Tensor,
    centers: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Softly assign continuous quality to adjacent five-word CLIP prototypes."""
    zero = quality_target.new_zeros(())
    if centers is None:
        return {"quality_text_ce": zero, "quality_text_mae": zero}
    if "quality_text_logits" not in out or "quality_text_score" not in out:
        raise RuntimeError("Quality text bank is enabled but Clean View predictions are missing.")
    logits = out["quality_text_logits"].float()
    centers = centers.to(device=logits.device, dtype=torch.float32)
    target = quality_target.float().view(-1).clamp(0.0, 1.0)
    weight = sample_weight.float().view(-1).clamp_min(0.0)
    high = torch.searchsorted(centers.contiguous(), target.contiguous()).clamp(1, centers.numel() - 1)
    low = high - 1
    fraction = ((target - centers[low]) / (centers[high] - centers[low])).clamp(0.0, 1.0)
    soft_target = torch.zeros_like(logits).scatter_add(1, low[:, None], (1.0 - fraction)[:, None])
    soft_target.scatter_add_(1, high[:, None], fraction[:, None])
    ce = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    score = out["quality_text_score"].float()
    denominator = weight.sum().clamp_min(1e-6)
    return {
        "quality_text_ce": (ce * weight).sum() / denominator,
        "quality_text_mae": ((score - target).abs() * weight).sum() / denominator,
    }


def _compute_align_text_loss(
    out: Dict[str, torch.Tensor],
    align_target: torch.Tensor,
    centers: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Supervise all human Align labels, including low-match examples equally."""
    if centers is None:
        zero = align_target.new_zeros(())
        return {"align_text_ce": zero, "align_text_mae": zero}
    if "align_text_logits" not in out or "align_text_score" not in out:
        raise RuntimeError("Align text bank is enabled but Clean View predictions are missing.")
    result = _compute_quality_text_loss(
        {
            "quality_text_logits": out["align_text_logits"],
            "quality_text_score": out["align_text_score"],
        },
        align_target,
        torch.ones_like(align_target),
        centers,
    )
    return {
        "align_text_ce": result["quality_text_ce"],
        "align_text_mae": result["quality_text_mae"],
    }


def _compute_quality_anchor_ordinal_losses(
    out: Dict[str, torch.Tensor],
    quality_target: torch.Tensor,
    sample_weight: torch.Tensor,
    tcfg: Dict,
    *,
    anchor_cfg: Optional[Dict] = None,
    metric_prefix: str = "quality",
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Impose a five-level ordered geometry on the quality SimSiam image space.

    Every labelled image contributes, so this objective does not depend on
    finding same-prompt Anchor/Near/Far examples.  Fixed prototypes lie on an
    ordered arc.  A continuous score is linearly assigned to its two adjacent
    score anchors, while cumulative probabilities provide the ordinal loss.
    The anchor head is intentionally parameter-free: gradients must reshape
    the existing image projector and the persistent image encoder instead of
    being absorbed by another disposable prediction branch.
    """
    # In strict three-view pretraining, human quality labels must supervise
    # the clean view. SafeAug View1/View2 remain dedicated to image-image
    # SimSiam. Keep the old two-view fallback for backward compatibility.
    feature = out.get(
        "quality_clean_z",
        out.get("quality_img1_z", out["img1_z"]),
    )
    zero = feature.new_tensor(0.0)
    anchor_key = f"{metric_prefix}_anchor_ce"
    ordinal_key = f"{metric_prefix}_ordinal_bce"
    anchor_accuracy_key = f"{metric_prefix}_anchor_accuracy"
    ordinal_accuracy_key = f"{metric_prefix}_ordinal_accuracy"
    anchor_mae_key = f"{metric_prefix}_anchor_mae"
    prediction_key = f"{metric_prefix}_anchor_score"
    losses = {
        anchor_key: zero,
        ordinal_key: zero,
        anchor_accuracy_key: zero,
        ordinal_accuracy_key: zero,
        anchor_mae_key: zero,
    }
    acfg = tcfg.get("quality_anchor_ordinal", {}) if anchor_cfg is None else anchor_cfg
    if not bool(acfg.get("enabled", False)):
        return losses, {}

    # Keep prototype probabilities and soft labels in FP32. The surrounding
    # training step may run under autocast, where matmul would otherwise emit
    # FP16 while the score-derived interpolation weights remain FP32.
    feature = F.normalize(feature.float(), dim=-1)
    target = quality_target.to(device=feature.device, dtype=torch.float32).view(-1).clamp(0.0, 1.0)
    weight = sample_weight.to(device=feature.device, dtype=torch.float32).view(-1).clamp_min(0.0)
    if feature.shape[0] != target.numel() or target.numel() != weight.numel():
        raise ValueError(
            f"{metric_prefix}_anchor_ordinal expects one feature, target and weight per image."
        )
    if feature.shape[-1] < 2:
        raise ValueError(f"{metric_prefix}_anchor_ordinal needs a feature dimension of at least 2.")

    num_levels = int(acfg.get("num_levels", 5))
    if num_levels < 3:
        raise ValueError(f"{metric_prefix}_anchor_ordinal num_levels must be at least 3.")
    configured_centers = acfg.get("centers")
    if configured_centers is None:
        centers = torch.linspace(0.0, 1.0, num_levels, device=feature.device)
    else:
        if len(configured_centers) != num_levels:
            raise ValueError(
                f"{metric_prefix}_anchor_ordinal centers length must equal num_levels: "
                f"{len(configured_centers)} != {num_levels}."
            )
        centers = feature.new_tensor([float(value) for value in configured_centers])
    if not bool(torch.all((centers >= 0.0) & (centers <= 1.0)).item()):
        raise ValueError(f"{metric_prefix}_anchor_ordinal centers must lie in [0, 1].")
    if not bool(torch.all(centers[1:] > centers[:-1]).item()):
        raise ValueError(f"{metric_prefix}_anchor_ordinal centers must be strictly increasing.")

    temperature = float(acfg.get("anchor_temperature", 0.20))
    if temperature <= 0.0:
        raise ValueError(f"{metric_prefix}_anchor_ordinal anchor_temperature must be positive.")
    arc_degrees = float(acfg.get("anchor_arc_degrees", 90.0))
    if not 0.0 < arc_degrees <= 180.0:
        raise ValueError(f"{metric_prefix}_anchor_ordinal anchor_arc_degrees must be in (0, 180].")

    # The aggregate-quality anchor retains its legacy dense basis. Component
    # anchors use distinct Fourier frequencies, which form orthogonal 2-D
    # planes inside the same shared SimSiam projection.
    feature_dim = feature.shape[-1]
    basis_frequency = int(acfg.get("basis_frequency", 0))
    if basis_frequency <= 0:
        basis0 = torch.ones(feature_dim, device=feature.device, dtype=feature.dtype)
        basis1 = torch.linspace(
            -1.0, 1.0, feature_dim, device=feature.device, dtype=feature.dtype
        )
    else:
        coordinate = (
            torch.arange(feature_dim, device=feature.device, dtype=feature.dtype) + 0.5
        ) / float(feature_dim)
        angle = 2.0 * math.pi * float(basis_frequency) * coordinate
        basis0 = torch.cos(angle)
        basis1 = torch.sin(angle)
    basis0 = F.normalize(basis0, dim=0)
    basis1 = basis1 - torch.dot(basis1, basis0) * basis0
    basis1 = F.normalize(basis1, dim=0)
    half_arc = 0.5 * math.radians(arc_degrees)
    angles = torch.linspace(-half_arc, half_arc, num_levels, device=feature.device)
    anchors = (
        torch.cos(angles).unsqueeze(1) * basis0.unsqueeze(0)
        + torch.sin(angles).unsqueeze(1) * basis1.unsqueeze(0)
    )
    anchors = F.normalize(anchors, dim=-1)
    anchor_feature = feature
    if bool(acfg.get("normalize_in_anchor_plane", False)):
        # Mean-residual training reserves orthogonal directions for residual
        # information. Normalize only the 2-D anchor-plane projection so that
        # adding such information cannot lower the mean-anchor confidence.
        anchor_feature = (
            (feature @ basis0).unsqueeze(1) * basis0.unsqueeze(0)
            + (feature @ basis1).unsqueeze(1) * basis1.unsqueeze(0)
        )
        anchor_feature = F.normalize(anchor_feature, dim=-1)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        anchor_logits = anchor_feature.float() @ anchors.float().t() / temperature

    # Softly interpolate between neighbouring score anchors. Scores outside
    # the first/last center are clipped to the corresponding endpoint anchor.
    upper = torch.searchsorted(centers, target, right=False).clamp(max=num_levels - 1)
    lower = (upper - 1).clamp(min=0)
    lower_center = centers[lower]
    upper_center = centers[upper]
    alpha = torch.where(
        upper_center > lower_center,
        (target - lower_center) / (upper_center - lower_center).clamp_min(1e-6),
        torch.zeros_like(target),
    ).clamp(0.0, 1.0)
    soft_target = torch.zeros_like(anchor_logits)
    soft_target.scatter_add_(1, lower.unsqueeze(1), (1.0 - alpha).unsqueeze(1))
    soft_target.scatter_add_(1, upper.unsqueeze(1), alpha.unsqueeze(1))

    log_prob = F.log_softmax(anchor_logits, dim=-1)
    anchor_loss_vec = -(soft_target * log_prob).sum(dim=-1)
    anchor_prob = anchor_logits.softmax(dim=-1)

    # P(level > k), k=0..L-2. This construction is monotonically decreasing by
    # definition and penalizes a two-level error more than an adjacent error.
    ordinal_target = 1.0 - soft_target.cumsum(dim=-1)[:, :-1]
    ordinal_prob = (1.0 - anchor_prob.cumsum(dim=-1)[:, :-1]).clamp(1e-6, 1.0 - 1e-6)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        ordinal_loss_vec = F.binary_cross_entropy(
            ordinal_prob.float(), ordinal_target.float(), reduction="none"
        ).mean(dim=-1)

    anchor_score = (anchor_prob * centers.unsqueeze(0)).sum(dim=-1)
    target_level = (target.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=-1)
    predicted_level = anchor_prob.argmax(dim=-1)
    losses.update({
        anchor_key: _weighted_mean(anchor_loss_vec, weight),
        ordinal_key: _weighted_mean(ordinal_loss_vec, weight),
        anchor_accuracy_key: _weighted_mean(
            (predicted_level == target_level).float(), weight
        ).detach(),
        ordinal_accuracy_key: _weighted_mean(
            ((ordinal_prob >= 0.5) == (ordinal_target >= 0.5)).float().mean(dim=-1),
            weight,
        ).detach(),
        anchor_mae_key: _weighted_mean((anchor_score - target).abs(), weight).detach(),
    })
    return losses, {prediction_key: anchor_score}


def _compute_quality_component_anchor_ordinal_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict,
    image_weight: torch.Tensor,
    tcfg: Dict,
    quality_loss_weight: Optional[torch.Tensor] = None,
    artifact_loss_weight: Optional[torch.Tensor] = None,
    aesthetics_loss_weight: Optional[torch.Tensor] = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Supervise Artifact and Aesthetics on orthogonal quality-projector planes."""
    component_cfg = tcfg.get("quality_component_anchor_ordinal", {})
    globally_enabled = bool(component_cfg.get("enabled", False))
    common_cfg = {
        key: value
        for key, value in component_cfg.items()
        if key not in {"artifact", "aesthetics"}
    }
    all_losses: Dict[str, torch.Tensor] = {}
    all_predictions: Dict[str, torch.Tensor] = {}
    if quality_loss_weight is None:
        quality_loss_weight = torch.ones_like(image_weight)
    if artifact_loss_weight is None:
        artifact_loss_weight = quality_loss_weight
    if aesthetics_loss_weight is None:
        aesthetics_loss_weight = quality_loss_weight
    component_loss_weights = {
        "artifact": artifact_loss_weight,
        "aesthetics": aesthetics_loss_weight,
    }
    for component in ("artifact", "aesthetics"):
        component_reweight = component_loss_weights[component].to(
            image_weight.device, non_blocking=True
        ).float().view(-1).clamp_min(0.0)
        component_reweight = component_reweight / component_reweight.mean().clamp_min(1e-6)
        if component_reweight.numel() != image_weight.numel():
            raise ValueError(
                f"{component}_loss_reweighting expects one weight per image in the batch."
            )
        child_cfg = dict(component_cfg.get(component, {}))
        child_enabled = globally_enabled and bool(child_cfg.get("enabled", True))
        anchor_cfg = {**common_cfg, **child_cfg, "enabled": child_enabled}
        score_key = f"{component}_score"
        weight_key = f"{component}_weight"
        if child_enabled and score_key not in batch:
            raise RuntimeError(
                f"Component anchor supervision is enabled but batch lacks {score_key!r}."
            )
        target = batch.get(score_key, torch.zeros_like(image_weight)).to(
            image_weight.device, non_blocking=True
        )
        component_weight = batch.get(weight_key, image_weight).to(
            image_weight.device, non_blocking=True
        )
        component_weight = (
            component_weight.float().clamp(0.0, 1.0)
            * image_weight
            * component_reweight
        )
        losses, predictions = _compute_quality_anchor_ordinal_losses(
            out,
            target,
            component_weight,
            tcfg,
            anchor_cfg=anchor_cfg,
            metric_prefix=component,
        )
        all_losses.update(losses)
        all_predictions.update(predictions)
    return all_losses, all_predictions


def _append_component_anchor_predictions(
    storage: Dict[str, Dict[str, list]],
    predictions: Dict[str, torch.Tensor],
    batch: Dict,
) -> None:
    for component in ("artifact", "aesthetics"):
        prediction_key = f"{component}_anchor_score"
        target_key = f"{component}_score"
        weight_key = f"{component}_weight"
        if prediction_key not in predictions or target_key not in batch:
            continue
        pred = predictions[prediction_key].detach().cpu().float().view(-1)
        target = batch[target_key].detach().cpu().float().view(-1)
        weight = batch.get(weight_key, torch.ones_like(batch[target_key]))
        valid = weight.detach().cpu().float().view(-1) > 0.0
        component_storage = storage.setdefault(component, {"pred": [], "target": []})
        component_storage["pred"].extend(pred[valid].tolist())
        component_storage["target"].extend(target[valid].tolist())


def _add_component_anchor_correlations(
    metrics: Dict[str, float],
    storage: Dict[str, Dict[str, list]],
) -> None:
    available = []
    for component in ("artifact", "aesthetics"):
        component_storage = storage.get(component, {"pred": [], "target": []})
        pred = component_storage["pred"]
        target = component_storage["target"]
        if not pred:
            continue
        metrics[f"{component}_anchor_PLCC"] = _safe_corrcoef_np(
            target, pred, kind="pearson"
        )
        metrics[f"{component}_anchor_SROCC"] = _safe_corrcoef_np(
            target, pred, kind="spearman"
        )
        metrics[f"{component}_anchor_N"] = len(pred)
        available.append(component)
    if len(available) == 2:
        metrics["component_anchor_PLCC"] = 0.5 * sum(
            metrics[f"{component}_anchor_PLCC"] for component in available
        )
        metrics["component_anchor_SROCC"] = 0.5 * sum(
            metrics[f"{component}_anchor_SROCC"] for component in available
        )


def _build_mean_residual_axis(
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a fixed residual axis orthogonal to the mean-quality anchor plane."""
    if feature_dim < 3:
        raise ValueError("mean_residual_quality needs a feature dimension of at least 3.")

    mean_basis0 = torch.ones(feature_dim, device=device, dtype=dtype)
    mean_basis0 = F.normalize(mean_basis0, dim=0)
    mean_basis1 = torch.linspace(-1.0, 1.0, feature_dim, device=device, dtype=dtype)
    mean_basis1 = mean_basis1 - torch.dot(mean_basis1, mean_basis0) * mean_basis0
    mean_basis1 = F.normalize(mean_basis1, dim=0)

    coordinate = (
        torch.arange(feature_dim, device=device, dtype=dtype) + 0.5
    ) / float(feature_dim)
    residual_axis = torch.cos(2.0 * math.pi * coordinate)
    residual_axis = residual_axis - torch.dot(residual_axis, mean_basis0) * mean_basis0
    residual_axis = residual_axis - torch.dot(residual_axis, mean_basis1) * mean_basis1
    residual_norm = residual_axis.norm()
    if float(residual_norm.item()) < 1e-6:
        raise RuntimeError("Failed to construct an axis orthogonal to the mean-quality plane.")
    return residual_axis / residual_norm


def _compute_mean_residual_quality_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict,
    image_weight: torch.Tensor,
    tcfg: Dict,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Encode Artifact-Aesthetics as a continuous direction in shared SimSiam space.

    The aggregate ``quality_score`` is handled by the existing ordered anchors.
    This parameter-free objective uses one fixed direction, orthogonal to that
    anchor plane, for the signed residual ``artifact_score-aesthetics_score``.
    Therefore the same shared projector must retain both overall quality and the
    source of the quality difference; no disposable regression head is added.
    """
    feature = out.get("quality_img1_z", out["img1_z"])
    zero = feature.new_tensor(0.0)
    losses = {
        "quality_residual_huber": zero,
        "quality_residual_mae": zero,
        "quality_residual_sign_accuracy": zero,
        "quality_residual_clipped_rate": zero,
    }
    rcfg = tcfg.get("mean_residual_quality", {})
    if not bool(rcfg.get("enabled", False)):
        return losses, {}
    for key in ("artifact_score", "aesthetics_score"):
        if key not in batch:
            raise RuntimeError(
                f"mean_residual_quality is enabled but batch lacks {key!r}."
            )

    feature = F.normalize(feature.float(), dim=-1)
    artifact = batch["artifact_score"].to(
        device=feature.device, dtype=torch.float32, non_blocking=True
    ).view(-1)
    aesthetics = batch["aesthetics_score"].to(
        device=feature.device, dtype=torch.float32, non_blocking=True
    ).view(-1)
    weight = image_weight.to(device=feature.device, dtype=torch.float32).view(-1)
    artifact_weight = batch.get("artifact_weight", torch.ones_like(batch["artifact_score"]))
    aesthetics_weight = batch.get(
        "aesthetics_weight", torch.ones_like(batch["aesthetics_score"])
    )
    weight = (
        weight.clamp_min(0.0)
        * artifact_weight.to(feature.device, non_blocking=True).float().view(-1).clamp(0.0, 1.0)
        * aesthetics_weight.to(feature.device, non_blocking=True).float().view(-1).clamp(0.0, 1.0)
    )
    if not (feature.shape[0] == artifact.numel() == aesthetics.numel() == weight.numel()):
        raise ValueError(
            "mean_residual_quality expects one feature, two component scores and weight per image."
        )

    residual_scale = float(rcfg.get("residual_scale", 0.50))
    temperature = float(rcfg.get("residual_temperature", 0.20))
    huber_beta = float(rcfg.get("huber_beta", 0.20))
    neutral_threshold = float(rcfg.get("near_equal_threshold", 1.0 / 24.0))
    if residual_scale <= 0.0:
        raise ValueError("mean_residual_quality residual_scale must be positive.")
    if temperature <= 0.0:
        raise ValueError("mean_residual_quality residual_temperature must be positive.")
    if huber_beta <= 0.0:
        raise ValueError("mean_residual_quality huber_beta must be positive.")
    if neutral_threshold < 0.0:
        raise ValueError("mean_residual_quality near_equal_threshold cannot be negative.")

    axis = _build_mean_residual_axis(
        feature.shape[-1], feature.device, feature.dtype
    )
    with torch.autocast(device_type=feature.device.type, enabled=False):
        residual_logit = feature.float() @ axis.float()
        residual_pred_norm = torch.tanh(residual_logit / temperature)

    residual_target = artifact - aesthetics
    residual_target_clipped = residual_target.clamp(-residual_scale, residual_scale)
    residual_target_norm = residual_target_clipped / residual_scale
    residual_pred = residual_pred_norm * residual_scale
    loss_vec = F.smooth_l1_loss(
        residual_pred_norm,
        residual_target_norm,
        reduction="none",
        beta=huber_beta,
    )

    target_sign = torch.where(
        residual_target > neutral_threshold,
        torch.ones_like(residual_target),
        torch.where(
            residual_target < -neutral_threshold,
            -torch.ones_like(residual_target),
            torch.zeros_like(residual_target),
        ),
    )
    pred_sign = torch.where(
        residual_pred > neutral_threshold,
        torch.ones_like(residual_pred),
        torch.where(
            residual_pred < -neutral_threshold,
            -torch.ones_like(residual_pred),
            torch.zeros_like(residual_pred),
        ),
    )
    losses.update({
        "quality_residual_huber": _weighted_mean(loss_vec, weight),
        "quality_residual_mae": _weighted_mean(
            (residual_pred - residual_target_clipped).abs(), weight
        ).detach(),
        "quality_residual_sign_accuracy": _weighted_mean(
            (pred_sign == target_sign).float(), weight
        ).detach(),
        "quality_residual_clipped_rate": _weighted_mean(
            (residual_target.abs() > residual_scale).float(), weight
        ).detach(),
    })
    return losses, {"quality_residual_score": residual_pred}


def _append_mean_residual_predictions(
    storage: Dict[str, Dict[str, list]],
    mean_predictions: Dict[str, torch.Tensor],
    residual_predictions: Dict[str, torch.Tensor],
    batch: Dict,
) -> None:
    """Collect mean/residual and reconstructed component scores for correlations."""
    if (
        "quality_anchor_score" not in mean_predictions
        or "quality_residual_score" not in residual_predictions
        or "artifact_score" not in batch
        or "aesthetics_score" not in batch
    ):
        return

    mean_pred = mean_predictions["quality_anchor_score"].detach().cpu().float().view(-1)
    residual_pred = residual_predictions["quality_residual_score"].detach().cpu().float().view(-1)
    artifact_target = batch["artifact_score"].detach().cpu().float().view(-1)
    aesthetics_target = batch["aesthetics_score"].detach().cpu().float().view(-1)
    artifact_weight = batch.get("artifact_weight", torch.ones_like(batch["artifact_score"]))
    aesthetics_weight = batch.get(
        "aesthetics_weight", torch.ones_like(batch["aesthetics_score"])
    )
    valid = (
        artifact_weight.detach().cpu().float().view(-1) > 0.0
    ) & (
        aesthetics_weight.detach().cpu().float().view(-1) > 0.0
    )
    mean_target = 0.5 * (artifact_target + aesthetics_target)
    residual_target = artifact_target - aesthetics_target
    derived_artifact = (mean_pred + 0.5 * residual_pred).clamp(0.0, 1.0)
    derived_aesthetics = (mean_pred - 0.5 * residual_pred).clamp(0.0, 1.0)

    values = {
        "quality_mean": (mean_pred, mean_target),
        "quality_residual": (residual_pred, residual_target),
        "derived_artifact": (derived_artifact, artifact_target),
        "derived_aesthetics": (derived_aesthetics, aesthetics_target),
    }
    for name, (pred, target) in values.items():
        target_storage = storage.setdefault(name, {"pred": [], "target": []})
        target_storage["pred"].extend(pred[valid].tolist())
        target_storage["target"].extend(target[valid].tolist())


def _add_mean_residual_correlations(
    metrics: Dict[str, float],
    storage: Dict[str, Dict[str, list]],
) -> None:
    available = []
    for name in (
        "quality_mean",
        "quality_residual",
        "derived_artifact",
        "derived_aesthetics",
    ):
        values = storage.get(name, {"pred": [], "target": []})
        if not values["pred"]:
            continue
        metrics[f"{name}_PLCC"] = _safe_corrcoef_np(
            values["target"], values["pred"], kind="pearson"
        )
        metrics[f"{name}_SROCC"] = _safe_corrcoef_np(
            values["target"], values["pred"], kind="spearman"
        )
        metrics[f"{name}_N"] = len(values["pred"])
        available.append(name)
    if "derived_artifact" in available and "derived_aesthetics" in available:
        metrics["derived_component_PLCC"] = 0.5 * (
            metrics["derived_artifact_PLCC"] + metrics["derived_aesthetics_PLCC"]
        )
        metrics["derived_component_SROCC"] = 0.5 * (
            metrics["derived_artifact_SROCC"] + metrics["derived_aesthetics_SROCC"]
        )


def _compute_quality_relation_siam_losses(out, batch, tcfg):
    """Quality-aware relations in the image-only quality projection space."""
    quality_img1_z = out.get("quality_img1_z", out["img1_z"])
    quality_img1_p = out.get("quality_img1_p", out["img1_p"])
    quality_img2_z = out.get("quality_img2_z", out["img2_z"])
    quality_img2_p = out.get("quality_img2_p", out["img2_p"])
    zero = quality_img1_z.new_tensor(0.0)
    result = {
        "quality_near_siam": zero,
        "quality_order_siam": zero,
        "quality_order_similarity": zero,
        "quality_order_score": zero,
        "quality_near_pairs": zero,
        "quality_order_pairs": zero,
        "quality_triplet_siam": zero,
        "quality_triplet_accuracy": zero,
        "quality_triplet_gap": zero,
        "quality_triplets": zero,
    }
    rcfg = tcfg.get("quality_relation_siam", {})
    if not bool(rcfg.get("enabled", False)):
        return result
    if "split_group" not in batch or "quality_score" not in batch:
        raise RuntimeError(
            "quality_relation_siam requires split_group and quality_score in every batch."
        )

    prompt_groups = [str(value) for value in batch["split_group"]]
    dataset_ids = [str(value) for value in batch.get("dataset_id", [""] * len(prompt_groups))]
    quality = batch["quality_score"].to(quality_img1_z).view(-1)
    align = batch.get("align_score", torch.zeros_like(quality)).to(quality).view(-1)
    count = quality.numel()
    if count < 2:
        return result

    # Directed masks are required because every sample may serve as Anchor.
    same_prompt = torch.zeros((count, count), dtype=torch.bool, device=quality.device)
    for left in range(count):
        for right in range(count):
            is_richhf = dataset_ids[left] == dataset_ids[right] == "richhf18k"
            if left != right and is_richhf and prompt_groups[left] == prompt_groups[right]:
                same_prompt[left, right] = True

    quality_gap = (quality[:, None] - quality[None, :]).abs()
    align_gap = (align[:, None] - align[None, :]).abs()
    align_ok = align_gap <= float(rcfg.get("align_max_gap", 0.05))
    near_directed = (
        same_prompt
        & align_ok
        & (quality_gap <= float(rcfg.get("near_quality_max_gap", 0.03)))
    )
    far_directed = (
        same_prompt
        & align_ok
        & (quality_gap >= float(rcfg.get("far_quality_min_gap", 0.15)))
    )

    # Legacy pair relations remain available for old experiment configs.
    near_mask = torch.triu(near_directed, diagonal=1)
    near_left, near_right = torch.where(near_mask)
    if near_left.numel() > 0:
        near_vec = -0.5 * (
            F.cosine_similarity(
                quality_img1_p[near_left], quality_img1_z[near_right].detach(), dim=-1
            )
            + F.cosine_similarity(
                quality_img1_p[near_right], quality_img1_z[near_left].detach(), dim=-1
            )
        )
        result["quality_near_siam"] = near_vec.mean()
        result["quality_near_pairs"] = zero.new_tensor(float(near_left.numel()))

    far_mask = torch.triu(far_directed, diagonal=1)
    far_left, far_right = torch.where(far_mask)
    if far_left.numel() > 0:
        same_similarity = 0.5 * (
            F.cosine_similarity(quality_img1_p, quality_img2_z.detach(), dim=-1)
            + F.cosine_similarity(quality_img2_p, quality_img1_z.detach(), dim=-1)
        )
        cross_similarity = 0.5 * (
            F.cosine_similarity(
                quality_img1_p[far_left], quality_img1_z[far_right].detach(), dim=-1
            )
            + F.cosine_similarity(
                quality_img1_p[far_right], quality_img1_z[far_left].detach(), dim=-1
            )
        )
        pair_positive_similarity = 0.5 * (
            same_similarity[far_left] + same_similarity[far_right]
        )
        gap_over_threshold = (
            quality_gap[far_left, far_right]
            - float(rcfg.get("far_quality_min_gap", 0.15))
        ).clamp_min(0.0)
        similarity_margin = (
            float(rcfg.get("similarity_margin", 0.05))
            + float(rcfg.get("similarity_gap_scale", 0.10)) * gap_over_threshold
        )
        similarity_loss = F.relu(
            similarity_margin + cross_similarity - pair_positive_similarity
        ).mean()

        quality_pred = out.get("quality_pred1", out.get("quality_pred"))
        if quality_pred is None:
            raise RuntimeError(
                "quality_relation_siam score ordering requires a quality prediction head."
            )
        direction = torch.sign(quality[far_left] - quality[far_right])
        ordered_prediction_gap = direction * (
            quality_pred[far_left] - quality_pred[far_right]
        )
        score_loss = F.relu(
            float(rcfg.get("score_margin", 0.05)) - ordered_prediction_gap
        ).mean()
        similarity_weight = float(rcfg.get("order_similarity_weight", 0.5))
        result["quality_order_similarity"] = similarity_loss
        result["quality_order_score"] = score_loss
        result["quality_order_siam"] = (
            similarity_weight * similarity_loss
            + (1.0 - similarity_weight) * score_loss
        )
        result["quality_order_pairs"] = zero.new_tensor(float(far_left.numel()))

    # True quality triplets compare two different images: Anchor/Near must be
    # more similar than Anchor/Far. They never use the trivial same-image pair.
    triplet_anchor = []
    triplet_near = []
    triplet_far = []
    for anchor in range(count):
        near_indices = torch.where(near_directed[anchor])[0].tolist()
        far_indices = torch.where(far_directed[anchor])[0].tolist()
        for near in near_indices:
            for far in far_indices:
                triplet_anchor.append(anchor)
                triplet_near.append(near)
                triplet_far.append(far)

    if triplet_anchor:
        anchor_idx = torch.tensor(triplet_anchor, dtype=torch.long, device=quality.device)
        near_idx = torch.tensor(triplet_near, dtype=torch.long, device=quality.device)
        far_idx = torch.tensor(triplet_far, dtype=torch.long, device=quality.device)
        near_similarity = 0.5 * (
            F.cosine_similarity(
                quality_img1_p[anchor_idx], quality_img1_z[near_idx].detach(), dim=-1
            )
            + F.cosine_similarity(
                quality_img1_p[near_idx], quality_img1_z[anchor_idx].detach(), dim=-1
            )
        )
        far_similarity = 0.5 * (
            F.cosine_similarity(
                quality_img1_p[anchor_idx], quality_img1_z[far_idx].detach(), dim=-1
            )
            + F.cosine_similarity(
                quality_img1_p[far_idx], quality_img1_z[anchor_idx].detach(), dim=-1
            )
        )
        far_gap_over_threshold = (
            quality_gap[anchor_idx, far_idx]
            - float(rcfg.get("far_quality_min_gap", 0.15))
        ).clamp_min(0.0)
        triplet_margin = (
            float(rcfg.get("triplet_margin", 0.05))
            + float(rcfg.get("triplet_gap_scale", 0.0)) * far_gap_over_threshold
        )
        similarity_gap = near_similarity - far_similarity
        result["quality_triplet_siam"] = F.relu(
            triplet_margin - similarity_gap
        ).mean()
        result["quality_triplet_accuracy"] = (
            similarity_gap.detach() > 0.0
        ).float().mean()
        result["quality_triplet_gap"] = similarity_gap.detach().mean()
        result["quality_triplets"] = zero.new_tensor(float(anchor_idx.numel()))

    return result

@torch.no_grad()
def _compute_quality_triplet_diagnostics(
    quality_z,
    quality_p,
    prompt_groups,
    dataset_ids,
    quality,
    align,
    relation_cfg,
):
    """Measure held-out quality ordering without affecting validation loss."""
    empty = {
        "quality_triplet_diag_accuracy": 0.0,
        "quality_triplet_diag_margin_accuracy": 0.0,
        "quality_triplet_diag_gap": 0.0,
        "quality_triplet_diag_count": 0.0,
    }
    if quality_z.numel() == 0:
        return empty

    near_max = float(relation_cfg.get(
        "diagnostic_near_quality_max_gap",
        relation_cfg.get("near_quality_max_gap", 0.03),
    ))
    far_min = float(relation_cfg.get(
        "diagnostic_far_quality_min_gap",
        relation_cfg.get("far_quality_min_gap", 0.15),
    ))
    align_max = float(relation_cfg.get(
        "diagnostic_align_max_gap",
        relation_cfg.get("align_max_gap", 0.05),
    ))
    margin = float(relation_cfg.get("triplet_margin", 0.05))
    groups = {}
    for index, (dataset_id, prompt_group) in enumerate(zip(dataset_ids, prompt_groups)):
        if str(dataset_id) == "richhf18k":
            groups.setdefault(str(prompt_group), []).append(index)

    anchors = []
    nears = []
    fars = []
    for indices in groups.values():
        for anchor in indices:
            near_indices = []
            far_indices = []
            for other in indices:
                if anchor == other or abs(float(align[anchor]) - float(align[other])) > align_max:
                    continue
                gap = abs(float(quality[anchor]) - float(quality[other]))
                if gap <= near_max:
                    near_indices.append(other)
                if gap >= far_min:
                    far_indices.append(other)
            for near in near_indices:
                for far in far_indices:
                    anchors.append(anchor)
                    nears.append(near)
                    fars.append(far)
    if not anchors:
        return empty

    quality_z = F.normalize(quality_z.float(), dim=-1)
    quality_p = F.normalize(quality_p.float(), dim=-1)
    anchor_idx = torch.tensor(anchors, dtype=torch.long)
    near_idx = torch.tensor(nears, dtype=torch.long)
    far_idx = torch.tensor(fars, dtype=torch.long)
    near_similarity = 0.5 * (
        (quality_p[anchor_idx] * quality_z[near_idx]).sum(dim=-1)
        + (quality_p[near_idx] * quality_z[anchor_idx]).sum(dim=-1)
    )
    far_similarity = 0.5 * (
        (quality_p[anchor_idx] * quality_z[far_idx]).sum(dim=-1)
        + (quality_p[far_idx] * quality_z[anchor_idx]).sum(dim=-1)
    )
    gap = near_similarity - far_similarity
    return {
        "quality_triplet_diag_accuracy": float((gap > 0.0).float().mean().item()),
        "quality_triplet_diag_margin_accuracy": float((gap >= margin).float().mean().item()),
        "quality_triplet_diag_gap": float(gap.mean().item()),
        "quality_triplet_diag_count": float(gap.numel()),
    }

PRETRAIN_METER_KEYS = [
    "loss", "img", "img_answer", "img_prompt", "prompt_answer", "prompt_answer_gap",
    "img_prompt_local", "prompt_answer_word",
    "quality_near_siam", "quality_order_siam",
    "quality_order_similarity", "quality_order_score",
    "quality_near_pairs", "quality_order_pairs",
    "quality_triplet_siam", "quality_triplet_accuracy",
    "quality_triplet_gap", "quality_triplets",
    "quality_anchor_ce", "quality_ordinal_bce",
    "quality_text_ce", "quality_text_mae",
    "align_text_ce", "align_text_mae",
    "quality_anchor_accuracy", "quality_ordinal_accuracy", "quality_anchor_mae",
    "artifact_anchor_ce", "artifact_ordinal_bce",
    "artifact_anchor_accuracy", "artifact_ordinal_accuracy", "artifact_anchor_mae",
    "aesthetics_anchor_ce", "aesthetics_ordinal_bce",
    "aesthetics_anchor_accuracy", "aesthetics_ordinal_accuracy", "aesthetics_anchor_mae",
    "quality_residual_huber", "quality_residual_mae",
    "quality_residual_sign_accuracy", "quality_residual_clipped_rate",
    "rank", "srocc", "quality_reg", "quality_rank", "quality_srocc",
    "align_reg", "align_rank", "align_srocc",
    "final_reg", "final_rank", "final_srocc",
]


def _compute_score_supervision(
    out: Dict[str, torch.Tensor],
    quality_target: torch.Tensor,
    align_target: torch.Tensor,
    final_target: torch.Tensor,
    tcfg: Dict,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Build legacy, dual-proxy, or persistent-IQA-head supervision."""
    zero = out["img1_z"].new_tensor(0.0)
    losses = {
        "rank": zero,
        "srocc": zero,
        "quality_reg": zero,
        "quality_rank": zero,
        "quality_srocc": zero,
        "align_reg": zero,
        "align_rank": zero,
        "align_srocc": zero,
        "final_reg": zero,
        "final_rank": zero,
        "final_srocc": zero,
    }
    proxies: Dict[str, torch.Tensor] = {}
    use_joint_iqa = bool(tcfg.get("use_joint_iqa_pretraining", False))
    use_dual_proxy = bool(tcfg.get("use_dual_proxy_supervision", False))
    if use_joint_iqa and use_dual_proxy:
        raise ValueError("Enable only one of use_joint_iqa_pretraining and use_dual_proxy_supervision.")

    if use_joint_iqa:
        quality_pred = out["quality_pred"]
        align_pred = out["align_pred"]
        final_pred = out["final_pred"]
        losses["quality_reg"] = F.mse_loss(quality_pred, quality_target)
        losses["quality_rank"] = _pretrain_rank_loss(
            quality_pred,
            quality_target,
            tcfg,
            margin=float(tcfg.get("quality_rank_margin", 0.05)),
        )
        losses["quality_srocc"] = masked_soft_spearman_loss(
            quality_pred,
            quality_target,
            mask=torch.ones_like(quality_target),
            tau=float(tcfg.get("quality_srocc_tau", tcfg.get("pretrain_srocc_tau", 0.02))),
        )
        losses["align_reg"] = F.mse_loss(align_pred, align_target)
        losses["align_rank"] = _pretrain_rank_loss(
            align_pred,
            align_target,
            tcfg,
            margin=float(tcfg.get("align_rank_margin", 0.05)),
        )
        losses["align_srocc"] = masked_soft_spearman_loss(
            align_pred,
            align_target,
            mask=torch.ones_like(align_target),
            tau=float(tcfg.get("align_srocc_tau", tcfg.get("pretrain_srocc_tau", 0.02))),
        )
        losses["final_reg"] = F.mse_loss(final_pred, final_target)
        losses["final_rank"] = _pretrain_rank_loss(
            final_pred,
            final_target,
            tcfg,
            margin=float(tcfg.get("final_rank_margin", 0.05)),
        )
        losses["final_srocc"] = masked_soft_spearman_loss(
            final_pred,
            final_target,
            mask=torch.ones_like(final_target),
            tau=float(tcfg.get("final_srocc_tau", tcfg.get("pretrain_srocc_tau", 0.02))),
        )
        proxies.update({"quality": quality_pred, "align": align_pred, "final": final_pred})
    elif use_dual_proxy:
        quality_pred = out["quality_pred"]
        align_pred = _pretrain_align_proxy(out)
        losses["quality_reg"] = F.mse_loss(quality_pred, quality_target)
        losses["quality_rank"] = _pretrain_rank_loss(
            quality_pred,
            quality_target,
            tcfg,
            margin=float(tcfg.get("quality_rank_margin", 0.05)),
        )
        losses["quality_srocc"] = masked_soft_spearman_loss(
            quality_pred,
            quality_target,
            mask=torch.ones_like(quality_target),
            tau=float(tcfg.get("quality_srocc_tau", tcfg.get("pretrain_srocc_tau", 0.02))),
        )
        losses["align_rank"] = _pretrain_rank_loss(
            align_pred,
            align_target,
            tcfg,
            margin=float(tcfg.get("align_rank_margin", 0.05)),
        )
        losses["align_srocc"] = masked_soft_spearman_loss(
            align_pred,
            align_target,
            mask=torch.ones_like(align_target),
            tau=float(tcfg.get("align_srocc_tau", tcfg.get("pretrain_srocc_tau", 0.02))),
        )
        proxies.update({"quality": quality_pred, "align": align_pred})
    else:
        legacy_proxy = _legacy_pretrain_proxy(out)
        losses["rank"] = _pretrain_rank_loss(legacy_proxy, final_target, tcfg)
        losses["srocc"] = masked_soft_spearman_loss(
            legacy_proxy,
            final_target,
            mask=torch.ones_like(final_target),
            tau=float(tcfg.get("pretrain_srocc_tau", tcfg.get("srocc_tau", 0.02))),
        )
        proxies["legacy"] = legacy_proxy
    return losses, proxies


def _get_quality_supervision_target(batch, mos, tcfg, device):
    """Select which RichHF quality signal supervises the persistent quality path."""
    source = str(tcfg.get("quality_supervision_target", "quality_score")).strip().lower()
    aliases = {"quality": "quality_score", "artifact": "artifact_score", "aesthetics": "aesthetics_score"}
    source = aliases.get(source, source)
    if source == "mos":
        target = mos
    else:
        if source not in batch:
            raise KeyError(f"quality_supervision_target={source!r} is missing from the batch")
        target = batch[source]
    return target.to(device, non_blocking=True).float().clamp(0, 1)


def _compute_quality_auxiliary_supervision(
    out: Dict[str, torch.Tensor],
    batch: Dict,
    tcfg: Dict,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """Supervise optional quality factors only where their labels are valid."""
    zero = out["quality_pred"].new_tensor(0.0)
    losses = {"quality_aux_total": zero}
    predictions: Dict[str, Dict[str, torch.Tensor]] = {}
    aux_cfg = tcfg.get("quality_auxiliary_supervision", {})
    if not bool(aux_cfg.get("enabled", False)):
        return losses, predictions

    component_cfgs = aux_cfg.get("components", {})
    if isinstance(component_cfgs, (list, tuple)):
        component_cfgs = {str(component): {} for component in component_cfgs}
    if not isinstance(component_cfgs, dict) or not component_cfgs:
        raise ValueError(
            "train.quality_auxiliary_supervision.components must be a non-empty mapping"
        )

    total = zero
    for raw_component, raw_spec in component_cfgs.items():
        component = str(raw_component).strip().lower()
        spec = {} if raw_spec is None else raw_spec
        if not isinstance(spec, dict):
            raise TypeError(
                f"Auxiliary quality spec for {component!r} must be a mapping"
            )
        if not bool(spec.get("enabled", True)):
            continue

        prediction_key = str(
            spec.get("prediction_key", f"quality_aux_{component}_pred")
        )
        if prediction_key not in out:
            raise KeyError(
                f"{prediction_key!r} is missing from model output. Add {component!r} "
                "to model.quality_auxiliary_heads.components."
            )

        target_key = str(spec.get("target_key", f"{component}_score"))
        if target_key not in batch:
            if bool(spec.get("required", False)):
                raise KeyError(
                    f"Required auxiliary quality target {target_key!r} is missing"
                )
            continue

        prediction = out[prediction_key].float().view(-1)
        target = batch[target_key].to(
            device, non_blocking=True
        ).float().view(-1)
        if prediction.numel() != target.numel():
            raise ValueError(
                f"{component} auxiliary prediction/target size mismatch: "
                f"{prediction.numel()} vs {target.numel()}"
            )

        weight_key = str(spec.get("weight_key", f"{component}_weight"))
        if weight_key and weight_key in batch:
            weight = batch[weight_key].to(
                device, non_blocking=True
            ).float().view(-1)
        else:
            weight = torch.ones_like(target)
        if weight.numel() != target.numel():
            raise ValueError(
                f"{component} auxiliary weight/target size mismatch: "
                f"{weight.numel()} vs {target.numel()}"
            )

        finite = (
            torch.isfinite(prediction)
            & torch.isfinite(target)
            & torch.isfinite(weight)
        )
        target = torch.where(finite, target, torch.zeros_like(target)).clamp(0, 1)
        valid_weight = torch.where(
            finite, weight.clamp(0, 1), torch.zeros_like(weight)
        )
        valid_mask = (valid_weight > 0).float()

        regression = _weighted_mean((prediction - target) ** 2, valid_weight)
        margin = float(spec.get("rank_margin", 0.05))
        if bool(spec.get("use_hard_rank", tcfg.get("pretrain_use_hard_rank", False))):
            ranking = masked_hard_pairwise_label_ranking_loss(
                prediction,
                target,
                mask=valid_mask,
                margin=margin,
                topk_ratio=float(
                    spec.get("hard_rank_ratio", tcfg.get("pretrain_hard_rank_ratio", 0.35))
                ),
            )
        else:
            ranking = masked_pairwise_label_ranking_loss(
                prediction, target, mask=valid_mask, margin=margin
            )
        srocc = masked_soft_spearman_loss(
            prediction,
            target,
            mask=valid_mask,
            tau=float(spec.get("srocc_tau", tcfg.get("quality_srocc_tau", 0.02))),
        )

        reg_weight = float(spec.get("lambda_reg", 0.0))
        rank_weight = float(spec.get("lambda_rank", 0.0))
        srocc_weight = float(spec.get("lambda_srocc", 0.0))
        total = (
            total
            + reg_weight * regression
            + rank_weight * ranking
            + srocc_weight * srocc
        )
        losses[f"{component}_aux_reg"] = regression
        losses[f"{component}_aux_rank"] = ranking
        losses[f"{component}_aux_srocc"] = srocc
        losses[f"{component}_aux_valid"] = valid_mask.sum()
        predictions[component] = {
            "pred": prediction,
            "target": target,
            "mask": valid_mask,
        }

    losses["quality_aux_total"] = total
    return losses, predictions


def _append_quality_auxiliary_predictions(storage, predictions):
    for component, values in predictions.items():
        valid = values["mask"].detach().cpu().bool()
        if not bool(valid.any()):
            continue
        component_storage = storage.setdefault(
            component, {"pred": [], "target": []}
        )
        component_storage["pred"].extend(
            values["pred"].detach().cpu()[valid].tolist()
        )
        component_storage["target"].extend(
            values["target"].detach().cpu()[valid].tolist()
        )


def _add_quality_auxiliary_correlations(ret, storage):
    component_srocc = []
    component_plcc = []
    for component, values in storage.items():
        if len(values["target"]) < 2:
            continue
        plcc = _safe_corrcoef_np(
            values["target"], values["pred"], kind="pearson"
        )
        srocc = _safe_corrcoef_np(
            values["target"], values["pred"], kind="spearman"
        )
        ret[f"{component}_aux_PLCC"] = plcc
        ret[f"{component}_aux_SROCC"] = srocc
        ret[f"{component}_aux_N"] = len(values["target"])
        component_plcc.append(plcc)
        component_srocc.append(srocc)
    if component_srocc:
        ret["quality_aux_mean_SROCC"] = sum(component_srocc) / len(component_srocc)
        ret["quality_aux_mean_PLCC"] = sum(component_plcc) / len(component_plcc)


def _pcgrad_backward(
    loss_quality: torch.Tensor,
    loss_align: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> None:
    """Backpropagate quality and alignment objectives with gradient surgery.

    Private branch parameters naturally receive only their own task gradient.
    For parameters shared by both tasks, conflicting per-parameter gradients
    are projected before they are summed. The same AMP scale is used for both
    losses, so the projection is valid before scaler.unscale_.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)

    if loss_quality.requires_grad:
        scaler.scale(loss_quality).backward(retain_graph=True)
    quality_grads = [
        None if p.grad is None else p.grad.detach().clone()
        for p in params
    ]
    optimizer.zero_grad(set_to_none=True)

    if loss_align.requires_grad:
        scaler.scale(loss_align).backward()
    align_grads = [
        None if p.grad is None else p.grad.detach().clone()
        for p in params
    ]

    for parameter, g_quality, g_align in zip(params, quality_grads, align_grads):
        if g_quality is None and g_align is None:
            continue
        if g_quality is None:
            parameter.grad = g_align
            continue
        if g_align is None:
            parameter.grad = g_quality
            continue

        dot = torch.sum(g_quality * g_align)
        if dot < 0:
            q_norm = torch.sum(g_quality * g_quality).clamp_min(1e-12)
            a_norm = torch.sum(g_align * g_align).clamp_min(1e-12)
            g_quality_original = g_quality
            g_quality = g_quality - dot / a_norm * g_align
            g_align = g_align - dot / q_norm * g_quality_original
        parameter.grad = g_quality + g_align



def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int):
    model.train()
    tcfg = cfg["train"]
    supervision_view = str(tcfg.get("human_supervision_view", "augmented")).strip().lower()
    if supervision_view not in {
        "augmented", "clean", "quality_safe", "three_view", "separate_clean"
    }:
        raise ValueError(
            f"Unknown human_supervision_view={supervision_view!r}. "
            "Choose from: augmented, clean, quality_safe, three_view."
        )
    clean_supervision = supervision_view == "clean"
    use_three_views = supervision_view in {"three_view", "separate_clean"}
    amp_enabled = bool(tcfg.get("amp", True)) and device.type == "cuda"
    meters = {k: AverageMeter() for k in PRETRAIN_METER_KEYS}
    all_quality_anchor_score = []
    all_quality_anchor_target = []
    all_quality_text_score = []
    all_quality_text_target = []
    all_align_text_score = []
    all_align_text_target = []
    component_anchor_storage = {
        component: {"pred": [], "target": []}
        for component in ("artifact", "aesthetics")
    }
    mean_residual_storage = {}
    quality_aux_storage = {}
    for step, batch in enumerate(loader):
        image1 = batch["image1"].to(device, non_blocking=True)
        image_clean = None
        if use_three_views:
            if "image_clean" not in batch:
                raise RuntimeError("three_view mode requires batch['image_clean']")
            image_clean = batch["image_clean"].to(device, non_blocking=True)
        image2 = batch["image2"].to(device, non_blocking=True)
        prompts = batch["prompt"]
        answers = batch["answer"]
        align = batch.get("align_weight", batch["align_score"]).to(device, non_blocking=True).clamp(0, 1)
        answer_weight = batch.get("answer_weight", torch.ones_like(align)).to(device, non_blocking=True).clamp(0, 1)
        prompt_answer_weight = batch.get("prompt_answer_weight", align).to(device, non_blocking=True).clamp(0, 1)
        prompt_answer_gap_weight = batch.get("prompt_answer_gap_weight", torch.zeros_like(align)).to(device, non_blocking=True).clamp(0, 1)
        image_weight = batch.get("image_weight", torch.ones_like(align)).to(device, non_blocking=True).clamp(0, 1)
        artifact_loss_weight = batch.get(
            "artifact_loss_weight", torch.ones_like(image_weight)
        ).to(device, non_blocking=True).float().clamp_min(0.0)
        aesthetics_loss_weight = batch.get(
            "aesthetics_loss_weight", torch.ones_like(image_weight)
        ).to(device, non_blocking=True).float().clamp_min(0.0)
        mos = batch.get("mos", batch.get("align_score", None))
        if mos is None:
            mos = torch.zeros_like(align)
        else:
            mos = mos.to(device, non_blocking=True).float().clamp(0, 1)
        quality_target = _get_quality_supervision_target(batch, mos, tcfg, device)
        align_target = batch.get("align_score", mos).to(device, non_blocking=True).float().clamp(0, 1)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            out = model.forward_views(
                image1, image2, prompts, answers, image_clean=image_clean
            )
            l_img = _quality_view_simsiam_loss(out, image_weight)
            img1_answer = simsiam_pair_loss(
                out["img1_p"], out["answer_z"], out["answer_p"], out["img1_z"], weight=answer_weight
            )
            img1_prompt = simsiam_pair_loss(
                out["img1_p"], out["prompt_z"], out["prompt_p"], out["img1_z"], weight=align
            )
            if use_three_views:
                l_img_answer = simsiam_pair_loss(
                    out["clean_p"], out["answer_z"], out["answer_p"], out["clean_z"], weight=answer_weight
                )
                l_img_prompt = simsiam_pair_loss(
                    out["clean_p"], out["prompt_z"], out["prompt_p"], out["clean_z"], weight=align
                )
                supervision_out = out
            elif clean_supervision:
                l_img_answer = img1_answer
                l_img_prompt = img1_prompt
                supervision_out = out
                if "quality_pred1" in out:
                    supervision_out = {
                        **out,
                        "quality_pred": out["quality_pred1"],
                        "align_pred": out["align_pred1"],
                        "final_pred": out["final_pred1"],
                    }
                    supervision_out.update({
                        key[:-1]: value for key, value in out.items()
                        if key.startswith("quality_aux_")
                        and key.endswith("_pred1")
                    })
            else:
                l_img_answer = 0.5 * (
                    img1_answer
                    + simsiam_pair_loss(
                        out["img2_p"], out["answer_z"], out["answer_p"], out["img2_z"], weight=answer_weight
                    )
                )
                l_img_prompt = 0.5 * (
                    img1_prompt
                    + simsiam_pair_loss(
                        out["img2_p"], out["prompt_z"], out["prompt_p"], out["img2_z"], weight=align
                    )
                )
                supervision_out = out
            l_prompt_answer = simsiam_pair_loss(out["prompt_p"], out["answer_z"], out["answer_p"], out["prompt_z"], weight=prompt_answer_weight)
            l_prompt_answer_gap = _prompt_answer_gap_loss(
                out,
                prompt_answer_gap_weight,
                max_similarity=float(tcfg.get("prompt_answer_gap_max_similarity", 0.20)),
            )

            supervised, _ = _compute_score_supervision(
                supervision_out,
                quality_target=quality_target,
                align_target=align_target,
                final_target=mos,
                tcfg=tcfg,
            )
            quality_aux, quality_aux_predictions = (
                _compute_quality_auxiliary_supervision(supervision_out, batch, tcfg, device)
            )
            l_img_prompt_local, l_prompt_answer_word = _compute_richhf_relational_losses(
                model, out, batch, prompts, device, tcfg
            )
            quality_relation = _compute_quality_relation_siam_losses(out, batch, tcfg)
            quality_anchor, quality_anchor_predictions = _compute_quality_anchor_ordinal_losses(
                out, quality_target, image_weight, tcfg
            )
            quality_text = _compute_quality_text_loss(
                out, quality_target, image_weight,
                model.iqa.quality_text_bank.centers
                if isinstance(model, CrossModalIQAJointPretrain) and model.iqa.quality_text_bank is not None
                else None,
            )
            align_text = _compute_align_text_loss(
                out, align_target,
                model.iqa.align_text_bank.centers
                if isinstance(model, CrossModalIQAJointPretrain) and model.iqa.align_text_bank is not None
                else None,
            )
            component_anchor, component_anchor_predictions = (
                _compute_quality_component_anchor_ordinal_losses(
                    out,
                    batch,
                    image_weight,
                    tcfg,
                    artifact_loss_weight=artifact_loss_weight,
                    aesthetics_loss_weight=aesthetics_loss_weight,
                )
            )
            residual_quality, residual_quality_predictions = (
                _compute_mean_residual_quality_losses(
                    out, batch, image_weight, tcfg
                )
            )

            loss_quality = (
                float(tcfg.get("lambda_quality_view_siam", tcfg.get("lambda_img_siam", 1.0))) * l_img
                + float(tcfg.get("lambda_rank", 0.0)) * supervised["rank"]
                + float(tcfg.get("lambda_pretrain_srocc", 0.0)) * supervised["srocc"]
                + float(tcfg.get("lambda_quality_reg", 0.0)) * supervised["quality_reg"]
                + float(tcfg.get("lambda_quality_rank", 0.0)) * supervised["quality_rank"]
                + float(tcfg.get("lambda_quality_srocc", 0.0)) * supervised["quality_srocc"]
                + float(tcfg.get("lambda_quality_neighbor_siam", 0.0)) * quality_relation["quality_near_siam"]
                + float(tcfg.get("lambda_quality_order_siam", 0.0)) * quality_relation["quality_order_siam"]
                + float(tcfg.get("lambda_quality_triplet_siam", 0.0)) * quality_relation["quality_triplet_siam"]
                + float(tcfg.get("lambda_quality_text_ce", 0.0)) * quality_text["quality_text_ce"]
                + float(tcfg.get("lambda_quality_anchor", 0.0)) * quality_anchor["quality_anchor_ce"]
                + float(tcfg.get("lambda_quality_ordinal", 0.0)) * quality_anchor["quality_ordinal_bce"]
                + float(tcfg.get("lambda_artifact_anchor", 0.0)) * component_anchor["artifact_anchor_ce"]
                + float(tcfg.get("lambda_artifact_ordinal", 0.0)) * component_anchor["artifact_ordinal_bce"]
                + float(tcfg.get("lambda_aesthetics_anchor", 0.0)) * component_anchor["aesthetics_anchor_ce"]
                + float(tcfg.get("lambda_aesthetics_ordinal", 0.0)) * component_anchor["aesthetics_ordinal_bce"]
                + float(tcfg.get("lambda_residual_huber", 0.0)) * residual_quality["quality_residual_huber"]
                + quality_aux["quality_aux_total"]
            )
            loss_align = (
                float(tcfg.get("lambda_img_answer", 1.0)) * l_img_answer
                + float(tcfg.get("lambda_img_prompt", 1.0)) * l_img_prompt
                + float(tcfg.get("lambda_prompt_answer", 0.5)) * l_prompt_answer
                + float(tcfg.get("lambda_prompt_answer_gap", 0.0)) * l_prompt_answer_gap
                + float(tcfg.get("lambda_align_pretrain_reg", 0.0)) * supervised["align_reg"]
                + float(tcfg.get("lambda_align_pretrain_rank", 0.0)) * supervised["align_rank"]
                + float(tcfg.get("lambda_align_pretrain_srocc", 0.0)) * supervised["align_srocc"]
                + float(tcfg.get("lambda_align_text_ce", 0.0)) * align_text["align_text_ce"]
                + float(tcfg.get("lambda_img_prompt_local", 0.0)) * l_img_prompt_local
                + float(tcfg.get("lambda_prompt_answer_word", 0.0)) * l_prompt_answer_word
            )
            loss_final = (
                (0.0 if bool(tcfg.get("two_output_only", False)) else float(tcfg.get("lambda_final_pretrain_reg", 0.0))) * supervised["final_reg"]
                + (0.0 if bool(tcfg.get("two_output_only", False)) else float(tcfg.get("lambda_final_pretrain_rank", 0.0))) * supervised["final_rank"]
                + (0.0 if bool(tcfg.get("two_output_only", False)) else float(tcfg.get("lambda_final_pretrain_srocc", 0.0))) * supervised["final_srocc"]
            )
            # A synthetic final target couples both tasks. If explicitly enabled
            # for a legacy run, split its gradient evenly; the new decoupled
            # configuration keeps this term at zero.
            loss_quality = loss_quality + 0.5 * loss_final
            loss_align = loss_align + 0.5 * loss_final
            loss = loss_quality + loss_align

        if bool(tcfg.get("pcgrad", {}).get("enabled", False)):
            _pcgrad_backward(loss_quality, loss_align, model, optimizer, scaler)
        else:
            scaler.scale(loss).backward()
        grad_clip = float(tcfg.get("grad_clip", 1.0))
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = image1.size(0)
        for k, v in {
            "loss": loss, "img": l_img, "img_answer": l_img_answer,
            "img_prompt": l_img_prompt, "prompt_answer": l_prompt_answer,
            "prompt_answer_gap": l_prompt_answer_gap,
            "img_prompt_local": l_img_prompt_local,
            "prompt_answer_word": l_prompt_answer_word,
            **supervised,
            **quality_relation,
            **quality_anchor,
            **quality_text,
            **align_text,
            **component_anchor,
            **residual_quality,
            **quality_aux,
        }.items():
            if k not in meters:
                meters[k] = AverageMeter()
            meters[k].update(float(v.item()), bs)
        if "quality_anchor_score" in quality_anchor_predictions:
            all_quality_anchor_score.extend(
                quality_anchor_predictions["quality_anchor_score"].detach().cpu().tolist()
            )
            all_quality_anchor_target.extend(quality_target.detach().cpu().tolist())
        if "quality_text_score" in out:
            all_quality_text_score.extend(out["quality_text_score"].detach().cpu().tolist())
            all_quality_text_target.extend(quality_target.detach().cpu().tolist())
        if "align_text_score" in out:
            all_align_text_score.extend(out["align_text_score"].detach().cpu().tolist())
            all_align_text_target.extend(align_target.detach().cpu().tolist())
        _append_component_anchor_predictions(
            component_anchor_storage, component_anchor_predictions, batch
        )
        _append_mean_residual_predictions(
            mean_residual_storage,
            quality_anchor_predictions,
            residual_quality_predictions,
            batch,
        )
        _append_quality_auxiliary_predictions(
            quality_aux_storage, quality_aux_predictions
        )
        if step % int(tcfg.get("log_interval", 20)) == 0:
            print(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] loss={meters['loss'].avg:.4f} "
                f"img={meters['img'].avg:.4f} img_ans={meters['img_answer'].avg:.4f} "
                f"img_prompt={meters['img_prompt'].avg:.4f} p_ans={meters['prompt_answer'].avg:.4f} "
                f"q_reg={meters['quality_reg'].avg:.4f} q_rank={meters['quality_rank'].avg:.4f} "
                f"q_srocc={meters['quality_srocc'].avg:.4f} a_reg={meters['align_reg'].avg:.4f} "
                f"a_rank={meters['align_rank'].avg:.4f} a_srocc={meters['align_srocc'].avg:.4f} "
                f"f_reg={meters['final_reg'].avg:.4f} f_rank={meters['final_rank'].avg:.4f} "
                f"img_prompt_local={meters['img_prompt_local'].avg:.4f} "
                f"prompt_answer_word={meters['prompt_answer_word'].avg:.4f} "
                f"q_near={meters['quality_near_siam'].avg:.4f}/"
                f"{meters['quality_near_pairs'].avg:.1f}pairs "
                f"q_order={meters['quality_order_siam'].avg:.4f}/"
                f"{meters['quality_order_pairs'].avg:.1f}pairs "
                f"q_triplet={meters['quality_triplet_siam'].avg:.4f}/"
                f"{meters['quality_triplets'].avg:.1f}triplets "
                f"acc={meters['quality_triplet_accuracy'].avg:.3f} "
                f"gap={meters['quality_triplet_gap'].avg:.4f} "
                f"q_anchor={meters['quality_anchor_ce'].avg:.4f} "
                f"q_text={meters['quality_text_ce'].avg:.4f} "
                f"a_text={meters['align_text_ce'].avg:.4f} "
                f"q_ordinal={meters['quality_ordinal_bce'].avg:.4f} "
                f"anchor_acc={meters['quality_anchor_accuracy'].avg:.3f} "
                f"artifact={meters['artifact_anchor_ce'].avg:.4f}/"
                f"{meters['artifact_ordinal_bce'].avg:.4f} "
                f"aesthetics={meters['aesthetics_anchor_ce'].avg:.4f}/"
                f"{meters['aesthetics_ordinal_bce'].avg:.4f} "
                f"residual={meters['quality_residual_huber'].avg:.4f} "
                f"residual_mae={meters['quality_residual_mae'].avg:.4f}"
            )
    ret = {k: m.avg for k, m in meters.items()}
    if all_quality_anchor_score:
        ret["quality_anchor_PLCC"] = _safe_corrcoef_np(
            all_quality_anchor_target, all_quality_anchor_score, kind="pearson"
        )
        ret["quality_anchor_SROCC"] = _safe_corrcoef_np(
            all_quality_anchor_target, all_quality_anchor_score, kind="spearman"
        )
        ret["quality_anchor_N"] = len(all_quality_anchor_score)
    if all_quality_text_score:
        ret["quality_text_PLCC"] = _safe_corrcoef_np(
            all_quality_text_target, all_quality_text_score, kind="pearson"
        )
        ret["quality_text_SROCC"] = _safe_corrcoef_np(
            all_quality_text_target, all_quality_text_score, kind="spearman"
        )
        ret["quality_text_N"] = len(all_quality_text_score)
    if all_align_text_score:
        ret["align_text_PLCC"] = _safe_corrcoef_np(
            all_align_text_target, all_align_text_score, kind="pearson"
        )
        ret["align_text_SROCC"] = _safe_corrcoef_np(
            all_align_text_target, all_align_text_score, kind="spearman"
        )
        ret["align_text_N"] = len(all_align_text_score)
    _add_component_anchor_correlations(ret, component_anchor_storage)
    _add_mean_residual_correlations(ret, mean_residual_storage)
    _add_quality_auxiliary_correlations(ret, quality_aux_storage)
    ret.update(local_stage_diagnostics(model))
    return ret


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    tcfg = cfg["train"]
    supervision_view = str(tcfg.get("human_supervision_view", "augmented")).strip().lower()
    if supervision_view not in {
        "augmented", "clean", "quality_safe", "three_view", "separate_clean"
    }:
        raise ValueError(
            f"Unknown human_supervision_view={supervision_view!r}. "
            "Choose from: augmented, clean, quality_safe, three_view."
        )
    clean_supervision = supervision_view == "clean"
    use_three_views = supervision_view in {"three_view", "separate_clean"}
    meters = {k: AverageMeter() for k in PRETRAIN_METER_KEYS}
    use_joint_iqa = bool(tcfg.get("use_joint_iqa_pretraining", False))
    use_dual_proxy = bool(tcfg.get("use_dual_proxy_supervision", False))
    all_proxy = []
    all_mos = []
    all_quality_proxy = []
    all_quality_target = []
    all_align_proxy = []
    all_align_target = []
    all_final_proxy = []
    all_final_target = []
    relation_cfg = tcfg.get("quality_relation_siam", {})
    relation_diag_enabled = bool(relation_cfg.get("enabled", False)) and bool(
        relation_cfg.get("diagnostic_enabled", True)
    )
    all_quality_diag_z = []
    all_quality_diag_p = []
    all_relation_prompt_groups = []
    all_relation_dataset_ids = []
    all_relation_quality = []
    all_relation_align = []
    all_quality_anchor_score = []
    all_quality_anchor_target = []
    all_quality_text_score = []
    all_quality_text_target = []
    all_align_text_score = []
    all_align_text_target = []
    component_anchor_storage = {}
    mean_residual_storage = {}
    quality_aux_storage = {}
    for batch in loader:
        image1 = batch["image1"].to(device, non_blocking=True)
        image2 = batch["image2"].to(device, non_blocking=True)
        image_clean = None
        if use_three_views:
            if "image_clean" not in batch:
                raise RuntimeError("three_view mode requires batch['image_clean']")
            image_clean = batch["image_clean"].to(device, non_blocking=True)
        prompts = batch["prompt"]
        answers = batch["answer"]
        align = batch.get("align_weight", batch["align_score"]).to(device, non_blocking=True).clamp(0, 1)
        answer_weight = batch.get("answer_weight", torch.ones_like(align)).to(device, non_blocking=True).clamp(0, 1)
        prompt_answer_weight = batch.get("prompt_answer_weight", align).to(device, non_blocking=True).clamp(0, 1)
        prompt_answer_gap_weight = batch.get("prompt_answer_gap_weight", torch.zeros_like(align)).to(device, non_blocking=True).clamp(0, 1)
        image_weight = batch.get("image_weight", torch.ones_like(align)).to(device, non_blocking=True).clamp(0, 1)
        artifact_loss_weight = batch.get(
            "artifact_loss_weight", torch.ones_like(image_weight)
        ).to(device, non_blocking=True).float().clamp_min(0.0)
        aesthetics_loss_weight = batch.get(
            "aesthetics_loss_weight", torch.ones_like(image_weight)
        ).to(device, non_blocking=True).float().clamp_min(0.0)
        mos = batch.get("mos", batch.get("align_score", None))
        if mos is None:
            mos = torch.zeros_like(align)
        else:
            mos = mos.to(device, non_blocking=True).float().clamp(0, 1)
        quality_target = _get_quality_supervision_target(batch, mos, tcfg, device)
        align_target = batch.get("align_score", mos).to(device, non_blocking=True).float().clamp(0, 1)
        out = model.forward_views(
            image1, image2, prompts, answers, image_clean=image_clean
        )
        if relation_diag_enabled:
            all_quality_diag_z.append(
                out.get("quality_img1_z", out["img1_z"]).detach().cpu().float()
            )
            all_quality_diag_p.append(
                out.get("quality_img1_p", out["img1_p"]).detach().cpu().float()
            )
            all_relation_prompt_groups.extend(str(value) for value in batch["split_group"])
            all_relation_dataset_ids.extend(str(value) for value in batch["dataset_id"])
            all_relation_quality.extend(quality_target.detach().cpu().tolist())
            all_relation_align.extend(align_target.detach().cpu().tolist())
        l_img = _quality_view_simsiam_loss(out, image_weight)
        img1_answer = simsiam_pair_loss(out["img1_p"], out["answer_z"], out["answer_p"], out["img1_z"], weight=answer_weight)
        img1_prompt = simsiam_pair_loss(out["img1_p"], out["prompt_z"], out["prompt_p"], out["img1_z"], weight=align)
        if use_three_views:
            l_img_answer = simsiam_pair_loss(
                out["clean_p"], out["answer_z"], out["answer_p"], out["clean_z"], weight=answer_weight
            )
            l_img_prompt = simsiam_pair_loss(
                out["clean_p"], out["prompt_z"], out["prompt_p"], out["clean_z"], weight=align
            )
            supervision_out = out
        elif clean_supervision:
            l_img_answer = img1_answer
            l_img_prompt = img1_prompt
            supervision_out = out
            if "quality_pred1" in out:
                supervision_out = {
                    **out,
                    "quality_pred": out["quality_pred1"],
                    "align_pred": out["align_pred1"],
                    "final_pred": out["final_pred1"],
                }
                supervision_out.update({
                    key[:-1]: value for key, value in out.items()
                    if key.startswith("quality_aux_")
                    and key.endswith("_pred1")
                })
        else:
            l_img_answer = 0.5 * (img1_answer + simsiam_pair_loss(out["img2_p"], out["answer_z"], out["answer_p"], out["img2_z"], weight=answer_weight))
            l_img_prompt = 0.5 * (img1_prompt + simsiam_pair_loss(out["img2_p"], out["prompt_z"], out["prompt_p"], out["img2_z"], weight=align))
            supervision_out = out
        l_prompt_answer = simsiam_pair_loss(out["prompt_p"], out["answer_z"], out["answer_p"], out["prompt_z"], weight=prompt_answer_weight)
        l_prompt_answer_gap = _prompt_answer_gap_loss(
            out,
            prompt_answer_gap_weight,
            max_similarity=float(tcfg.get("prompt_answer_gap_max_similarity", 0.20)),
        )
        supervised, proxies = _compute_score_supervision(
            supervision_out,
            quality_target=quality_target,
            align_target=align_target,
            final_target=mos,
            tcfg=tcfg,
        )
        quality_aux, quality_aux_predictions = (
            _compute_quality_auxiliary_supervision(supervision_out, batch, tcfg, device)
        )
        l_img_prompt_local, l_prompt_answer_word = _compute_richhf_relational_losses(
            model, out, batch, prompts, device, tcfg
        )
        quality_relation = _compute_quality_relation_siam_losses(out, batch, tcfg)
        quality_anchor, quality_anchor_predictions = _compute_quality_anchor_ordinal_losses(
            out, quality_target, image_weight, tcfg
        )
        quality_text = _compute_quality_text_loss(
            out, quality_target, image_weight,
            model.iqa.quality_text_bank.centers
            if isinstance(model, CrossModalIQAJointPretrain) and model.iqa.quality_text_bank is not None
            else None,
        )
        align_text = _compute_align_text_loss(
            out, align_target,
            model.iqa.align_text_bank.centers
            if isinstance(model, CrossModalIQAJointPretrain) and model.iqa.align_text_bank is not None
            else None,
        )
        component_anchor, component_anchor_predictions = (
            _compute_quality_component_anchor_ordinal_losses(
                out,
                batch,
                image_weight,
                tcfg,
                artifact_loss_weight=artifact_loss_weight,
                aesthetics_loss_weight=aesthetics_loss_weight,
            )
        )
        residual_quality, residual_quality_predictions = (
            _compute_mean_residual_quality_losses(
                out, batch, image_weight, tcfg
            )
        )
        loss = (
            float(tcfg.get("lambda_quality_view_siam", tcfg.get("lambda_img_siam", 1.0))) * l_img
            + float(tcfg.get("lambda_img_answer", 1.0)) * l_img_answer
            + float(tcfg.get("lambda_img_prompt", 1.0)) * l_img_prompt
            + float(tcfg.get("lambda_prompt_answer", 0.5)) * l_prompt_answer
            + float(tcfg.get("lambda_prompt_answer_gap", 0.0)) * l_prompt_answer_gap
            + float(tcfg.get("lambda_rank", 0.0)) * supervised["rank"]
            + float(tcfg.get("lambda_pretrain_srocc", 0.0)) * supervised["srocc"]
            + float(tcfg.get("lambda_quality_reg", 0.0)) * supervised["quality_reg"]
            + float(tcfg.get("lambda_quality_rank", 0.0)) * supervised["quality_rank"]
            + float(tcfg.get("lambda_quality_srocc", 0.0)) * supervised["quality_srocc"]
            + float(tcfg.get("lambda_align_pretrain_reg", 0.0)) * supervised["align_reg"]
            + float(tcfg.get("lambda_align_pretrain_rank", 0.0)) * supervised["align_rank"]
            + float(tcfg.get("lambda_align_pretrain_srocc", 0.0)) * supervised["align_srocc"]
            + float(tcfg.get("lambda_align_text_ce", 0.0)) * align_text["align_text_ce"]
            + (0.0 if bool(tcfg.get("two_output_only", False)) else float(tcfg.get("lambda_final_pretrain_reg", 0.0))) * supervised["final_reg"]
            + (0.0 if bool(tcfg.get("two_output_only", False)) else float(tcfg.get("lambda_final_pretrain_rank", 0.0))) * supervised["final_rank"]
            + (0.0 if bool(tcfg.get("two_output_only", False)) else float(tcfg.get("lambda_final_pretrain_srocc", 0.0))) * supervised["final_srocc"]
            + float(tcfg.get("lambda_img_prompt_local", 0.0)) * l_img_prompt_local
            + float(tcfg.get("lambda_prompt_answer_word", 0.0)) * l_prompt_answer_word
            + float(tcfg.get("lambda_quality_neighbor_siam", 0.0)) * quality_relation["quality_near_siam"]
            + float(tcfg.get("lambda_quality_order_siam", 0.0)) * quality_relation["quality_order_siam"]
            + float(tcfg.get("lambda_quality_triplet_siam", 0.0)) * quality_relation["quality_triplet_siam"]
            + float(tcfg.get("lambda_quality_text_ce", 0.0)) * quality_text["quality_text_ce"]
            + float(tcfg.get("lambda_quality_anchor", 0.0)) * quality_anchor["quality_anchor_ce"]
            + float(tcfg.get("lambda_quality_ordinal", 0.0)) * quality_anchor["quality_ordinal_bce"]
            + float(tcfg.get("lambda_artifact_anchor", 0.0)) * component_anchor["artifact_anchor_ce"]
            + float(tcfg.get("lambda_artifact_ordinal", 0.0)) * component_anchor["artifact_ordinal_bce"]
            + float(tcfg.get("lambda_aesthetics_anchor", 0.0)) * component_anchor["aesthetics_anchor_ce"]
            + float(tcfg.get("lambda_aesthetics_ordinal", 0.0)) * component_anchor["aesthetics_ordinal_bce"]
            + float(tcfg.get("lambda_residual_huber", 0.0)) * residual_quality["quality_residual_huber"]
            + quality_aux["quality_aux_total"]
        )
        bs = image1.size(0)
        for k, v in {
            "loss": loss, "img": l_img, "img_answer": l_img_answer,
            "img_prompt": l_img_prompt, "prompt_answer": l_prompt_answer,
            "prompt_answer_gap": l_prompt_answer_gap,
            "img_prompt_local": l_img_prompt_local,
            "prompt_answer_word": l_prompt_answer_word,
            **supervised,
            **quality_relation,
            **quality_anchor,
            **quality_text,
            **align_text,
            **component_anchor,
            **residual_quality,
            **quality_aux,
        }.items():
            if k not in meters:
                meters[k] = AverageMeter()
            meters[k].update(float(v.item()), bs)
        if "quality_anchor_score" in quality_anchor_predictions:
            all_quality_anchor_score.extend(
                quality_anchor_predictions["quality_anchor_score"].detach().cpu().tolist()
            )
            all_quality_anchor_target.extend(quality_target.detach().cpu().tolist())
        if "quality_text_score" in out:
            all_quality_text_score.extend(out["quality_text_score"].detach().cpu().tolist())
            all_quality_text_target.extend(quality_target.detach().cpu().tolist())
        if "align_text_score" in out:
            all_align_text_score.extend(out["align_text_score"].detach().cpu().tolist())
            all_align_text_target.extend(align_target.detach().cpu().tolist())
        _append_component_anchor_predictions(
            component_anchor_storage, component_anchor_predictions, batch
        )
        _append_mean_residual_predictions(
            mean_residual_storage,
            quality_anchor_predictions,
            residual_quality_predictions,
            batch,
        )
        _append_quality_auxiliary_predictions(
            quality_aux_storage, quality_aux_predictions
        )
        if use_joint_iqa or use_dual_proxy:
            all_quality_proxy.extend(proxies["quality"].detach().cpu().float().numpy().reshape(-1).tolist())
            all_quality_target.extend(quality_target.detach().cpu().float().numpy().reshape(-1).tolist())
            all_align_proxy.extend(proxies["align"].detach().cpu().float().numpy().reshape(-1).tolist())
            all_align_target.extend(align_target.detach().cpu().float().numpy().reshape(-1).tolist())
            if use_joint_iqa and not bool(tcfg.get("two_output_only", False)):
                all_final_proxy.extend(proxies["final"].detach().cpu().float().numpy().reshape(-1).tolist())
                all_final_target.extend(mos.detach().cpu().float().numpy().reshape(-1).tolist())
        else:
            all_proxy.extend(proxies["legacy"].detach().cpu().float().numpy().reshape(-1).tolist())
            all_mos.extend(mos.detach().cpu().float().numpy().reshape(-1).tolist())
    ret = {k: m.avg for k, m in meters.items()}
    if use_joint_iqa or use_dual_proxy:
        ret["quality_proxy_PLCC"] = _safe_corrcoef_np(all_quality_target, all_quality_proxy, kind="pearson")
        ret["quality_proxy_SROCC"] = _safe_corrcoef_np(all_quality_target, all_quality_proxy, kind="spearman")
        ret["align_proxy_PLCC"] = _safe_corrcoef_np(all_align_target, all_align_proxy, kind="pearson")
        ret["align_proxy_SROCC"] = _safe_corrcoef_np(all_align_target, all_align_proxy, kind="spearman")
        if use_joint_iqa and not bool(tcfg.get("two_output_only", False)):
            ret["final_proxy_PLCC"] = _safe_corrcoef_np(all_final_target, all_final_proxy, kind="pearson")
            ret["final_proxy_SROCC"] = _safe_corrcoef_np(all_final_target, all_final_proxy, kind="spearman")
            ret["joint_SROCC"] = (
                ret["quality_proxy_SROCC"] + ret["align_proxy_SROCC"] + ret["final_proxy_SROCC"]
            ) / 3.0
            ret["joint_PLCC"] = (
                ret["quality_proxy_PLCC"] + ret["align_proxy_PLCC"] + ret["final_proxy_PLCC"]
            ) / 3.0
        else:
            ret["joint_SROCC"] = 0.5 * (ret["quality_proxy_SROCC"] + ret["align_proxy_SROCC"])
            ret["joint_PLCC"] = 0.5 * (ret["quality_proxy_PLCC"] + ret["align_proxy_PLCC"])
            if bool(tcfg.get("two_output_only", False)):
                ret["final_output_disabled"] = 1.0
    else:
        ret["proxy_PLCC"] = _safe_corrcoef_np(all_mos, all_proxy, kind="pearson")
        ret["proxy_SROCC"] = _safe_corrcoef_np(all_mos, all_proxy, kind="spearman")
    if all_quality_anchor_score:
        ret["quality_anchor_PLCC"] = _safe_corrcoef_np(
            all_quality_anchor_target, all_quality_anchor_score, kind="pearson"
        )
        ret["quality_anchor_SROCC"] = _safe_corrcoef_np(
            all_quality_anchor_target, all_quality_anchor_score, kind="spearman"
        )
        ret["quality_anchor_N"] = len(all_quality_anchor_score)
    if all_quality_text_score:
        ret["quality_text_PLCC"] = _safe_corrcoef_np(
            all_quality_text_target, all_quality_text_score, kind="pearson"
        )
        ret["quality_text_SROCC"] = _safe_corrcoef_np(
            all_quality_text_target, all_quality_text_score, kind="spearman"
        )
        ret["quality_text_N"] = len(all_quality_text_score)
    if all_align_text_score:
        ret["align_text_PLCC"] = _safe_corrcoef_np(
            all_align_text_target, all_align_text_score, kind="pearson"
        )
        ret["align_text_SROCC"] = _safe_corrcoef_np(
            all_align_text_target, all_align_text_score, kind="spearman"
        )
        ret["align_text_N"] = len(all_align_text_score)
    _add_component_anchor_correlations(ret, component_anchor_storage)
    _add_mean_residual_correlations(ret, mean_residual_storage)
    _add_quality_auxiliary_correlations(ret, quality_aux_storage)
    if relation_diag_enabled and all_quality_diag_z:
        ret.update(_compute_quality_triplet_diagnostics(
            torch.cat(all_quality_diag_z, dim=0),
            torch.cat(all_quality_diag_p, dim=0),
            all_relation_prompt_groups,
            all_relation_dataset_ids,
            all_relation_quality,
            all_relation_align,
            relation_cfg,
        ))
    ret.update(local_stage_diagnostics(model))
    return ret




def get_pretrain_selection_value(metrics: Dict, cfg: Dict) -> tuple[float, bool, str]:
    """Select best pretraining checkpoint by val loss or proxy SROCC."""
    tcfg = cfg.get("train", {})
    key = str(tcfg.get("pretrain_selection_metric", "val_loss"))
    mode = str(tcfg.get("pretrain_selection_mode", "min")).lower()
    if key in {"val_quality_align_hmean_SROCC", "val_component_hmean_SROCC"}:
        # RichHF supplies separate quality and alignment labels. In the
        # two-output protocol, select a checkpoint from those labels rather
        # than from a synthetic weighted Final score.
        q = float(metrics.get("val_quality_proxy_SROCC", metrics.get("val_quality_SROCC", 0.0)))
        a = float(metrics.get("val_align_proxy_SROCC", metrics.get("val_align_SROCC", 0.0)))
        q01 = min(1.0, max(0.0, 0.5 * (q + 1.0)))
        a01 = min(1.0, max(0.0, 0.5 * (a + 1.0)))
        h01 = 2.0 * q01 * a01 / (q01 + a01 + 1e-8)
        value = 2.0 * h01 - 1.0
        metrics[key] = value
        return value, True, key
    if key == "val_loss":
        value = float(metrics.get("val_loss", 0.0))
    elif key == "val_proxy_SROCC":
        value = float(metrics.get("val_proxy_SROCC", 0.0))
        mode = "max" if "pretrain_selection_mode" not in tcfg else mode
    elif key == "val_proxy_PLCC":
        value = float(metrics.get("val_proxy_PLCC", 0.0))
        mode = "max" if "pretrain_selection_mode" not in tcfg else mode
    elif key in {
        "val_joint_SROCC",
        "val_joint_PLCC",
        "val_quality_proxy_SROCC",
        "val_quality_proxy_PLCC",
        "val_align_proxy_SROCC",
        "val_align_proxy_PLCC",
        "val_final_proxy_SROCC",
        "val_final_proxy_PLCC",
        "val_quality_anchor_SROCC",
        "val_quality_anchor_PLCC",
        "val_component_anchor_SROCC",
        "val_component_anchor_PLCC",
        "val_artifact_anchor_SROCC",
        "val_artifact_anchor_PLCC",
        "val_aesthetics_anchor_SROCC",
        "val_aesthetics_anchor_PLCC",
        "val_quality_mean_SROCC",
        "val_quality_mean_PLCC",
        "val_quality_residual_SROCC",
        "val_quality_residual_PLCC",
        "val_derived_artifact_SROCC",
        "val_derived_artifact_PLCC",
        "val_derived_aesthetics_SROCC",
        "val_derived_aesthetics_PLCC",
        "val_derived_component_SROCC",
        "val_derived_component_PLCC",
    }:
        value = float(metrics.get(key, 0.0))
        mode = "max" if "pretrain_selection_mode" not in tcfg else mode
    else:
        print(f"[WARN] pretrain_selection_metric={key} not found; fallback to val_loss")
        key = "val_loss"
        value = float(metrics.get("val_loss", 0.0))
        mode = "min"
    return value, mode == "max", key

def main():
    args = parse_args()
    session_started = time.monotonic()
    if args.max_runtime_hours is not None and (not math.isfinite(args.max_runtime_hours) or args.max_runtime_hours <= 0):
        raise ValueError("--max-runtime-hours must be finite and positive.")
    cfg = load_yaml(args.config)
    if args.clip_large:
        configure_clip_large(cfg)
    if args.seed is not None:
        cfg.setdefault("split_seed", int(cfg.get("seed", 42)))
        cfg["seed"] = int(args.seed)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.results_csv:
        cfg["results_csv"] = str(args.results_csv)
    configure_research_controls(cfg, args)
    configure_local_feature_stage(cfg, args)
    local_stage = cfg.get("model", {}).get("local_encoder", {}).get("feature_stage", "layer4")
    print(f"[INFO] Pretraining ResNet local feature stage={local_stage} (recorded in config/checkpoints).")
    if args.align_text_levels:
        if not bool(cfg.get("train", {}).get("use_joint_iqa_pretraining", False)):
            raise ValueError("--align-text-levels requires joint IQA pretraining.")
        if str(cfg.get("train", {}).get("human_supervision_view", "")).lower() not in {"three_view", "separate_clean"}:
            raise ValueError("--align-text-levels requires a separate Clean View.")
        bank_cfg = cfg.setdefault("model", {}).setdefault("align_text_bank", {})
        bank_cfg["enabled"] = True
        bank_cfg["score_blend"] = 0.0
        bank_cfg.setdefault("temperature", 0.2)
        bank_cfg.setdefault("class_context_tokens", max(1, int(args.text_context_tokens or 4)))
        # Cover the full human Align range, including rare exact-zero labels.
        # Fixed centers are dataset-independent and use no val/downstream labels.
        bank_cfg.setdefault("centers", [0.0, 0.25, 0.5, 0.75, 1.0])
        cfg.setdefault("train", {})["lambda_align_text_ce"] = 0.005
        print(f"[INFO] Clean View five-word Align bank enabled; centers={bank_cfg['centers']}.")
    if args.quality_text_levels:
        if not bool(cfg.get("train", {}).get("use_joint_iqa_pretraining", False)):
            raise ValueError("--quality-text-levels requires joint IQA pretraining.")
        if str(cfg.get("train", {}).get("human_supervision_view", "")).lower() not in {"three_view", "separate_clean"}:
            raise ValueError("--quality-text-levels requires a separate Clean View.")
        if not bool(cfg.get("model", {}).get("simsiam", {}).get("separate_quality_projection", False)):
            raise ValueError("--quality-text-levels requires a separate quality SimSiam projector.")
        bank_cfg = cfg.setdefault("model", {}).setdefault("quality_text_bank", {})
        bank_cfg["enabled"] = True
        bank_cfg["score_blend"] = 0.0
        bank_cfg.setdefault("temperature", 0.2)
        bank_cfg.setdefault("class_context_tokens", max(1, int(args.text_context_tokens or 4)))
        # 5/25/50/75/95 percentiles of the filtered official RichHF train CSV.
        bank_cfg.setdefault("centers", [0.375, 0.54166667, 0.66666669, 0.79166666, 0.95833334])
        cfg.setdefault("train", {})["lambda_quality_text_ce"] = 0.005
        print(f"[INFO] Clean View five-word quality bank enabled; centers={bank_cfg['centers']}.")
    if args.text_context_tokens is not None:
        if not 0 <= args.text_context_tokens <= 16:
            raise ValueError("--text-context-tokens must be in [0, 16].")
        cfg.setdefault("model", {}).setdefault("text_encoder", {})[
            "learnable_context_tokens"
        ] = args.text_context_tokens
        print(f"[INFO] Learnable CLIP text context: {args.text_context_tokens} tokens.")
    if args.isolate_text_context_rng:
        text_cfg = cfg.setdefault("model", {}).setdefault("text_encoder", {})
        if int(text_cfg.get("learnable_context_tokens", 0)) < 1:
            raise ValueError("--isolate-text-context-rng requires positive --text-context-tokens.")
        text_cfg["isolate_context_init_rng"] = True
        print("[INFO] Text-context initialization RNG isolated from later modules.")
    if args.disable_final_pretrain_loss:
        train_cfg = cfg.setdefault("train", {})
        train_cfg["lambda_final_pretrain_reg"] = 0.0
        train_cfg["lambda_final_pretrain_rank"] = 0.0
        train_cfg["lambda_final_pretrain_srocc"] = 0.0
        print(
            "[INFO] Final-score pretraining supervision disabled: "
            "reg=0, rank=0, soft-SROCC=0. Quality/Align supervision and "
            "the downstream Final head remain unchanged."
        )
    if args.disable_quality_auxiliary:
        model_cfg = cfg.setdefault("model", {})
        quality_aux_heads_cfg = dict(model_cfg.get("quality_auxiliary_heads", {}))
        quality_aux_heads_cfg["enabled"] = False
        model_cfg["quality_auxiliary_heads"] = quality_aux_heads_cfg
        train_cfg = cfg.setdefault("train", {})
        quality_aux_supervision_cfg = dict(
            train_cfg.get("quality_auxiliary_supervision", {})
        )
        quality_aux_supervision_cfg["enabled"] = False
        train_cfg["quality_auxiliary_supervision"] = quality_aux_supervision_cfg
        print(
            "[INFO] Quality Auxiliary disabled: model auxiliary heads and their "
            "losses are off; Component Anchor supervision remains enabled."
        )
    if args.quality_aux_scale is not None:
        scale = float(args.quality_aux_scale)
        if scale < 0.0:
            raise ValueError("--quality-aux-scale must be non-negative.")
        model_cfg = cfg.setdefault("model", {})
        train_cfg = cfg.setdefault("train", {})
        aux_cfg = dict(train_cfg.get("quality_auxiliary_supervision", {}))
        components = aux_cfg.get("components", {})
        if isinstance(components, dict):
            scaled_components = {}
            for component, raw_spec in components.items():
                spec = dict(raw_spec or {})
                for key in ("lambda_reg", "lambda_rank", "lambda_srocc"):
                    if key in spec:
                        spec[key] = float(spec[key]) * scale
                scaled_components[component] = spec
            aux_cfg["components"] = scaled_components
        if scale == 0.0:
            aux_cfg["enabled"] = False
            heads_cfg = dict(model_cfg.get("quality_auxiliary_heads", {}))
            heads_cfg["enabled"] = False
            model_cfg["quality_auxiliary_heads"] = heads_cfg
        train_cfg["quality_auxiliary_supervision"] = aux_cfg
        print(f"[INFO] Quality Auxiliary loss scale set to {scale:g}.")
    if args.quality_balanced_sampling:
        train_cfg = cfg.setdefault("train", {})
        sampling_cfg = dict(train_cfg.get("quality_balanced_sampling", {}))
        balance_strength = sampling_cfg.get("balance_strength", 1.0)
        if args.quality_balanced_sampling_alpha is not None:
            balance_strength = float(args.quality_balanced_sampling_alpha)
        if not 0.0 <= float(balance_strength) <= 1.0:
            raise ValueError("--quality-balanced-sampling-alpha must be in [0, 1].")
        sampling_cfg.update({
            "enabled": True,
            "bin_edges": sampling_cfg.get(
                "bin_edges", [0.0, 0.5, 0.6, 0.7, 0.85, 1.000001]
            ),
            "weight_power": float(sampling_cfg.get("weight_power", 0.5)),
            "min_weight": float(sampling_cfg.get("min_weight", 0.5)),
            "max_weight": float(sampling_cfg.get("max_weight", 3.0)),
            "balance_strength": float(balance_strength),
        })
        train_cfg["quality_balanced_sampling"] = sampling_cfg
    if args.quality_loss_reweighting or args.quality_loss_reweighting_alpha is not None:
        train_cfg = cfg.setdefault("train", {})
        loss_cfg = dict(train_cfg.get("quality_loss_reweighting", {}))
        balance_strength = loss_cfg.get("balance_strength", 0.3)
        if args.quality_loss_reweighting_alpha is not None:
            balance_strength = float(args.quality_loss_reweighting_alpha)
        if not 0.0 <= float(balance_strength) <= 1.0:
            raise ValueError("--quality-loss-reweighting-alpha must be in [0, 1].")
        loss_cfg.update({
            "enabled": True,
            "bin_edges": loss_cfg.get(
                "bin_edges", [0.0, 0.5, 0.6, 0.7, 0.85, 1.000001]
            ),
            "weight_power": float(loss_cfg.get("weight_power", 0.5)),
            "min_weight": float(loss_cfg.get("min_weight", 0.5)),
            "max_weight": float(loss_cfg.get("max_weight", 3.0)),
            "balance_strength": float(balance_strength),
        })
        train_cfg["quality_loss_reweighting"] = loss_cfg
    if args.mirror_augmentation or args.mirror_augmentation_probability is not None:
        train_cfg = cfg.setdefault("train", {})
        mirror_cfg = dict(train_cfg.get("mirror_augmentation", {}))
        probability = (
            float(args.mirror_augmentation_probability)
            if args.mirror_augmentation_probability is not None
            else float(mirror_cfg.get("probability", 0.5))
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("--mirror-augmentation-probability must be in [0, 1].")
        mirror_cfg.update({
            "enabled": True,
            "probability": probability,
            "directional_only": bool(mirror_cfg.get("directional_only", True)),
            "swap_text": bool(mirror_cfg.get("swap_text", True)),
            "disable_random_horizontal_flip": bool(
                mirror_cfg.get("disable_random_horizontal_flip", True)
            ),
        })
        train_cfg["mirror_augmentation"] = mirror_cfg
    set_seed(int(cfg.get("seed", 42)))
    print(
        f"[INFO] Seeds: train={int(cfg.get('seed', 42))}, "
        f"split={int(cfg.get('split_seed', cfg.get('seed', 42)))}"
    )
    output_dir = cfg.get("output_dir", "./outputs/cm_pretrain")
    ensure_dir(output_dir)

    run_id = cfg.get("run_id", make_run_id("pretrain", cfg))
    if args.output_dir:
        run_id = os.path.basename(os.path.normpath(output_dir))
        cfg["run_id"] = run_id
    config_snapshot = dump_config_snapshot(cfg, output_dir, config_path=args.config)
    logger = CSVExperimentLogger(
        csv_path=resolve_results_csv(cfg),
        run_id=run_id,
        stage="pretrain",
        cfg=cfg,
        output_dir=output_dir,
        config_path=config_snapshot,
    )
    logger.log(event="run_start", epoch=0, metrics={"config_file": args.config})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = unpack_loaders(build_dataloaders(cfg))
    if bool(cfg.get("train", {}).get("use_joint_iqa_pretraining", False)):
        model = CrossModalIQAJointPretrain(cfg).to(device)
        print("[INFO] Joint IQA pretraining enabled: persistent IQA heads will be saved for downstream loading.")
    else:
        model = CrossModalSimSiamPretrain(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"].get("epochs", 40)),
        eta_min=float(cfg.get("scheduler", {}).get("min_lr", 1e-6)),
    )
    scaler = GradScaler(device.type, enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    best = None
    best_epoch = -1
    no_improve = 0
    patience = int(cfg["train"].get("early_stop_patience", 10))
    auxiliary_best_specs = {
        "joint": ("val_joint_SROCC", "best_joint_pretrain.pth"),
        "align": ("val_align_proxy_SROCC", "best_align_pretrain.pth"),
        "quality": ("val_quality_proxy_SROCC", "best_quality_pretrain.pth"),
        "quality_anchor": ("val_quality_anchor_SROCC", "best_quality_anchor_pretrain.pth"),
        "component_anchor": ("val_component_anchor_SROCC", "best_component_anchor_pretrain.pth"),
        "artifact_anchor": ("val_artifact_anchor_SROCC", "best_artifact_anchor_pretrain.pth"),
        "aesthetics_anchor": ("val_aesthetics_anchor_SROCC", "best_aesthetics_anchor_pretrain.pth"),
        "quality_residual": ("val_quality_residual_SROCC", "best_quality_residual_pretrain.pth"),
        "derived_artifact": ("val_derived_artifact_SROCC", "best_derived_artifact_pretrain.pth"),
        "derived_aesthetics": ("val_derived_aesthetics_SROCC", "best_derived_aesthetics_pretrain.pth"),
        "derived_component": ("val_derived_component_SROCC", "best_derived_component_pretrain.pth"),
    }
    if args.align_text_levels:
        # The new Clean View bank must not be selected by a quality-component
        # metric or by the SafeAug image-text proxy.
        auxiliary_best_specs["align_text"] = (
            "val_align_text_SROCC", "best_align_text_pretrain.pth"
        )
    auxiliary_bests = {
        name: {"value": None, "epoch": -1}
        for name in auxiliary_best_specs
    }

    fixed_training_epochs = cfg["train"].get("fixed_training_epochs")
    training_epochs = int(fixed_training_epochs or cfg["train"].get("epochs", 40))
    start_epoch = 1
    if args.resume:
        # Old full checkpoints contain optimizer/scheduler but not RNG/scaler
        # or selection counters. Recover their counters from the experiment CSV.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        for required in ("epoch", "model", "optimizer", "scheduler", "cfg"):
            if required not in checkpoint:
                raise ValueError(f"Resume requires a full checkpoint; missing {required}.")
        for section in ("model", "dataset", "train", "optimizer", "scheduler", "seed"):
            if checkpoint["cfg"].get(section) != cfg.get(section):
                raise ValueError(f"Resume configuration differs in {section}; use the original config/flags.")
        saved_split = checkpoint["cfg"].get("split_seed", checkpoint["cfg"].get("seed", 42))
        if saved_split != cfg.get("split_seed", cfg.get("seed", 42)):
            raise ValueError("Resume split_seed differs from checkpoint.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > training_epochs:
            raise ValueError("Checkpoint already reached the configured training horizon.")
        historical_main, historical_aux = legacy_resume_records(
            resolve_results_csv(cfg), checkpoint.get("run_id", run_id),
            start_epoch - 1, output_dir, auxiliary_best_specs,
        )
        state = checkpoint.get("training_state", historical_main)
        if state is None:
            restored_val = evaluate(model, val_loader, device, cfg)
            restored_metrics = {f"val_{key}": value for key, value in restored_val.items()}
            restored_best, _, _ = get_pretrain_selection_value(restored_metrics, cfg)
            state = dict(best=restored_best, best_epoch=start_epoch-1, no_improve=0)
            print("[WARN] No saved selection history; initialize selection from restored model validation.")
        best, best_epoch, no_improve = state["best"], state["best_epoch"], state["no_improve"]
        auxiliary_bests.update(checkpoint.get("auxiliary_bests", {}))
        for name, record in historical_aux.items():
            existing = auxiliary_bests[name]
            if existing["value"] is None or record["value"] > existing["value"]:
                auxiliary_bests[name] = record
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "rng" in checkpoint:
            rng = checkpoint["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if device.type == "cuda" and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            print("[WARN] Legacy checkpoint has no RNG/scaler snapshot; continuation is not bitwise identical.")
        del checkpoint
        logger.log(event="resume", epoch=start_epoch-1,
                   metrics={"checkpoint": args.resume, "start_epoch": start_epoch}, best_epoch=best_epoch)
        print(f"[INFO] Resumed full training state; next epoch={start_epoch}, best epoch={best_epoch}.")

    epoch = start_epoch - 1
    previous_epoch_seconds = 0.0
    for epoch in range(start_epoch, training_epochs + 1):
        if (args.max_runtime_hours is not None and
            time.monotonic() - session_started + previous_epoch_seconds >= args.max_runtime_hours * 3600):
            print("[INFO] Runtime budget reached; stop before another epoch. Resume from last_pretrain.pth (or original source if no new epoch ran).")
            logger.log(event="paused", epoch=epoch-1, metrics={"reason": "runtime_budget"}, best_epoch=best_epoch)
            return
        epoch_started = time.monotonic()
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        va = evaluate(model, val_loader, device, cfg)
        scheduler.step()

        current_lrs = {
            str(group.get("name", f"group{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        metrics = {
            "lr": optimizer.param_groups[0]["lr"],
            **{f"lr_{name}": value for name, value in current_lrs.items()},
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        if bool(cfg["train"].get("use_joint_iqa_pretraining", False)):
            print(
                f"Epoch {epoch}: TrainLoss={tr['loss']:.4f} | ValLoss={va['loss']:.4f} "
                f"| ValQualitySROCC={va.get('quality_proxy_SROCC', 0.0):.4f} "
                f"| ValAlignSROCC={va.get('align_proxy_SROCC', 0.0):.4f} "
                f"| ValFinalSROCC={va.get('final_proxy_SROCC', 0.0):.4f} "
                f"| ValJointSROCC={va.get('joint_SROCC', 0.0):.4f} "
                f"| ValAnchorSROCC={va.get('quality_anchor_SROCC', 0.0):.4f} "
                f"| ValArtifactAnchorSROCC={va.get('artifact_anchor_SROCC', 0.0):.4f} "
                f"| ValAestheticsAnchorSROCC={va.get('aesthetics_anchor_SROCC', 0.0):.4f} "
                f"| ValComponentAnchorSROCC={va.get('component_anchor_SROCC', 0.0):.4f} "
                f"| ValResidualSROCC={va.get('quality_residual_SROCC', 0.0):.4f} "
                f"| ValDerivedArtifactSROCC={va.get('derived_artifact_SROCC', 0.0):.4f} "
                f"| ValDerivedAestheticsSROCC={va.get('derived_aesthetics_SROCC', 0.0):.4f} "
                f"| ValArtifactAuxSROCC={va.get('artifact_aux_SROCC', 0.0):.4f} "
                f"| ValAestheticsAuxSROCC={va.get('aesthetics_aux_SROCC', 0.0):.4f}"
            )
        elif bool(cfg["train"].get("use_dual_proxy_supervision", False)):
            print(
                f"Epoch {epoch}: TrainLoss={tr['loss']:.4f} | ValLoss={va['loss']:.4f} "
                f"| ValQualitySROCC={va.get('quality_proxy_SROCC', 0.0):.4f} "
                f"| ValAlignSROCC={va.get('align_proxy_SROCC', 0.0):.4f} "
                f"| ValJointSROCC={va.get('joint_SROCC', 0.0):.4f}"
            )
        else:
            print(f"Epoch {epoch}: TrainLoss={tr['loss']:.4f} | ValLoss={va['loss']:.4f} | ValProxySROCC={va.get('proxy_SROCC', 0.0):.4f}")

        if float(va.get("quality_triplet_diag_count", 0.0)) > 0:
            print(
                "  QualityTripletDiag: "
                f"count={int(va['quality_triplet_diag_count'])} "
                f"acc={va['quality_triplet_diag_accuracy']:.4f} "
                f"margin_acc={va['quality_triplet_diag_margin_accuracy']:.4f} "
                f"mean_gap={va['quality_triplet_diag_gap']:.4f}"
            )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": cfg,
            "run_id": run_id,
        }

        cur_value, maximize, sel_key = get_pretrain_selection_value(metrics, cfg)
        is_best = (best is None) or (cur_value > best if maximize else cur_value < best)
        if is_best:
            best = cur_value
            best_epoch = epoch
            no_improve = 0
            print(f"Saved best pretrain checkpoint: epoch={best_epoch}, best_{sel_key}={best:.6f}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s). Best epoch={best_epoch}, best_{sel_key}={best:.6f}")

        # Save each joint-IQA validation view independently so checkpoint
        # diagnosis is not tied to early stopping's single selection metric.
        # These transfer checkpoints omit optimizer state to reduce disk use.
        for name, (metric_key, filename) in auxiliary_best_specs.items():
            if metric_key not in metrics:
                continue
            value = float(metrics[metric_key])
            record = auxiliary_bests[name]
            if record["value"] is None or value > record["value"]:
                record.update({"value": value, "epoch": epoch})
                auxiliary_ckpt = {
                    "epoch": epoch,
                    "model": _extract_representation_state(ckpt["model"]),
                    "checkpoint_scope": "representation",
                    "cfg": cfg,
                    "run_id": run_id,
                    "selection_metric": metric_key,
                    "selection_value": value,
                }
                atomic_save(auxiliary_ckpt, os.path.join(output_dir, filename), torch.save)
                print(
                    f"Saved best {name} pretrain checkpoint: "
                    f"epoch={epoch}, {metric_key}={value:.6f}"
                )

        ckpt.update({
            "scaler": scaler.state_dict(),
            "training_state": dict(best=best, best_epoch=best_epoch, no_improve=no_improve),
            "auxiliary_bests": auxiliary_bests,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
        })
        if is_best:
            atomic_save(ckpt, os.path.join(output_dir, "best_pretrain.pth"), torch.save)
        atomic_save(ckpt, os.path.join(output_dir, "last_pretrain.pth"), torch.save)
        previous_epoch_seconds = time.monotonic() - epoch_started

        auxiliary_metrics = {}
        for name, record in auxiliary_bests.items():
            if record["value"] is not None:
                auxiliary_metrics[f"best_{name}_selection_value"] = record["value"]
                auxiliary_metrics[f"best_{name}_epoch"] = record["epoch"]
        metrics.update({
            "best_selection_value": best,
            "selection_metric": cfg.get("train", {}).get("pretrain_selection_metric", "val_loss"),
            "no_improve": no_improve,
            **auxiliary_metrics,
        })
        logger.log(event="epoch", epoch=epoch, metrics=metrics, best_epoch=best_epoch, is_best=is_best)

        if fixed_training_epochs is None and no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            logger.log(
                event="early_stop",
                epoch=epoch,
                metrics={"best_epoch": best_epoch, "best_selection_value": best, "patience": patience},
                best_epoch=best_epoch,
                is_best=False,
            )
            break

    # Save final summary row into the same CSV.
    final_metrics = {
        "best_epoch": best_epoch,
        "best_selection_value": best,
        "selection_metric": cfg.get("train", {}).get("pretrain_selection_metric", "val_loss"),
        "last_epoch": epoch,
        "best_checkpoint": os.path.join(output_dir, "best_pretrain.pth"),
        "last_checkpoint": os.path.join(output_dir, "last_pretrain.pth"),
    }
    for name, (_, filename) in auxiliary_best_specs.items():
        record = auxiliary_bests[name]
        if record["value"] is not None:
            final_metrics[f"best_{name}_epoch"] = record["epoch"]
            final_metrics[f"best_{name}_selection_value"] = record["value"]
            final_metrics[f"best_{name}_checkpoint"] = os.path.join(output_dir, filename)
    final_metrics["planned_training_epochs"] = training_epochs
    final_metrics["fixed_training_budget"] = fixed_training_epochs is not None
    logger.log(event="final", epoch=epoch, metrics=final_metrics, best_epoch=best_epoch, is_best=False)


if __name__ == "__main__":
    main()
