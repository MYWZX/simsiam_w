# 用1k、3k、2023进行微调
from __future__ import annotations

import argparse
import copy
import math
import os
import re
from typing import Dict, List, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import build_dataloaders
from models.two_stage_models import CrossModalIQAFullModel, load_pretrained_backbone
from utils.experiment_logger import CSVExperimentLogger, dump_config_snapshot, make_run_id, resolve_results_csv
from utils.losses import (
    align_prompt_pair_ranking_loss,
    align_prompt_ranking_loss,
    masked_pairwise_label_ranking_loss,
    masked_soft_spearman_loss,
    masked_hard_pairwise_label_ranking_loss,
)
from utils.train_utils import AverageMeter, ensure_dir, load_yaml, set_seed
from utils.pcgrad import PCGRAD_DIAGNOSTIC_KEYS, global_shared_pcgrad_backward


from utils.selection_policies import configure_selection_comparison, SelectionComparison
from utils.run_metadata import checkpoint_fingerprint, collect_run_metadata, write_run_metadata

from utils.clip_experiments import configure_clip_large, calibrate_quality_centers, configure_local_feature_stage, local_stage_diagnostics


from utils.downstream_controls import configure_downstream_controls, distinct_prompt_mask, audit_fusion_gradients, validate_local_crops


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local-feature-stage", choices=("layer3", "layer4", "layer3_4"), default=None, help="Select ResNet layer3, layer4, or layer3_4 fusion; default follows YAML (legacy layer4).")
    p.add_argument("--reinit-local-projection", action="store_true", help="Explicitly reset only local projection weight/bias for layer4 checkpoint -> layer3 downstream transfer.")
    p.add_argument("--compare-selection-policies", action="store_true", help="Compare Quality and joint validation selection on one fixed 60-epoch trajectory, no early stopping or disk weights.")
    p.add_argument("--quality-safe-input", action="store_true", help="Use whole-image resize only for training on all downstream datasets; no crop, flip or color jitter.")
    p.add_argument("--disable-legacy-negative", action="store_true", help="Set lambda_neg=0 without changing other losses.")
    p.add_argument("--selection-metric", choices=("val_quality_SROCC", "val_align_SROCC", "val_component_hmean_SROCC"), default=None)
    p.add_argument("--audit-loss-gradients", action="store_true", help="Log first-batch Fusion gradient norms per epoch (additional backward probes; no optimizer update).")
    p.add_argument("--clip-large", action="store_true", help="Use ViT-L/14; supply a matching Large pretraining checkpoint.")
    p.add_argument("--quality-centers-train-quantiles", action="store_true", help="Calibrate five centers using only AIGCIQA2023 training labels after checkpoint loading.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--results-csv", type=str, default=None, help="Write results to a dedicated experiment CSV.")
    p.add_argument(
        "--text-context-tokens", type=int, default=None,
        help="Load/fine-tune N pretrained CLIP text-prefix tokens (use the pretraining value).",
    )
    p.add_argument(
        "--isolate-text-context-rng", action="store_true",
        help="Isolate prefix initialization RNG for matched downstream ablations.",
    )
    p.add_argument(
        "--quality-text-blend", type=float, default=None,
        help="Load five-word quality bank; 0=representation-only, 0.10=blend 10%% into Quality.",
    )
    p.add_argument(
        "--align-text-blend", type=float, default=None,
        help="Load five-word image-Prompt bank; 0=representation-only, 0.10=blend 10%% into Align.",
    )
    p.add_argument(
        "--quality-text-lambda", type=float, default=None,
        help="Downstream Quality five-level auxiliary CE weight (default 0.005 when enabled).",
    )
    p.add_argument(
        "--align-text-lambda", type=float, default=None,
        help="Downstream Align five-level auxiliary CE weight (default 0.005 when enabled).",
    )
    p.add_argument(
        "--quality-text-anchor-lambda", type=float, default=None,
        help="Quality class-context initialization anchor regularization weight (default 0).",
    )
    p.add_argument(
        "--align-text-anchor-lambda", type=float, default=None,
        help="Align class-context initialization anchor regularization weight (default 0).",
    )
    p.add_argument(
        "--disable-perturbation", action="store_true",
        help="Disable synthetic downstream perturbation augmentation for this run.",
    )
    p.add_argument(
        "--quality-distribution", action="store_true",
        help="Enable an optional five-bin Quality distribution head and auxiliary loss.",
    )
    p.add_argument(
        "--quality-highres-crops", action="store_true",
        help="Use high-resolution local crops only in the Quality branch (Align stays base-resolution).",
    )
    p.add_argument(
        "--quality-attention-pooling", action="store_true",
        help="Use global-query attention pooling over Quality local patch/crop tokens.",
    )
    p.add_argument(
        "--align-prompt-cross-attention", action="store_true",
        help="Use Prompt token-to-visual-patch cross-attention in the Align branch.",
    )
    p.add_argument(
        "--quality-global-feature", choices=("z", "g"), default=None,
        help="Quality global input: fused z or existing projected CLIP g; Align always keeps z.",
    )
    p.add_argument(
        "--quality-no-local-branch", action="store_true",
        help="Strict global-only Quality: skip/freeze NSS and patch scoring; retain Fusion and Align.",
    )
    p.add_argument(
        "--quality-global-only", action="store_true",
        help="Use only the global Quality score, bypassing explicit patch-score fusion; keep ResNet/Fusion and Align.",
    )
    p.add_argument(
        "--quality-adapter-mode", choices=("replace", "parallel_score", "residual_feature"), default=None,
        help="Change only the enabled quality Adapter's scoring interface.",
    )
    p.add_argument(
        "--quality-adapter-residual-init", type=float, default=None,
        help="Initial feature residual weight in (0,1), used only with residual_feature (default 0.1).",
    )
    p.add_argument(
        "--quality-adapter-residual-fixed", type=float, default=None,
        help="Fix feature residual alpha in [0,1] instead of learning it; 0 keeps Adapter auxiliary training but removes direct residual scoring.",
    )
    p.add_argument(
        "--quality-adapter-freeze-bn-stats", action="store_true",
        help="Use pretrained Quality Adapter BatchNorm running statistics; keep affine/Adapter parameters trainable and Dropout active.",
    )
    p.add_argument(
        "--quality-adapter-init", choices=("pretrained", "random"), default=None,
        help="Keep identical Adapter architecture; choose checkpoint weights or seeded random initialization.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in the YAML config for repeated runs.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir so repeated seeds do not overwrite checkpoints.",
    )
    p.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Override train.pretrained_checkpoint without changing the fine-tuning YAML.",
    )
    p.add_argument(
        "--disable-quality-adapter",
        action="store_true",
        help=(
            "Disable the persistent quality adapter for a strict downstream "
            "ablation. The pretrained checkpoint and all other losses remain unchanged."
        ),
    )
    p.add_argument(
        "--quality-sample-gate", action="store_true",
        help="Use image-conditioned Quality patch/global mixing; zero-initialized correction to baseline beta.",
    )
    p.add_argument(
        "--visual-backbone-lr-mult", type=float, default=None,
        help="Override only CLIP visual and ResNet layerwise LR multipliers relative to train.lr.",
    )
    p.add_argument(
        "--quality-adapter-lr-mult",
        type=float,
        default=None,
        help=(
            "Override only the Quality Adapter layerwise LR multiplier "
            "(relative to train.lr); 0.10 gives initial LR 1e-5 for train.lr=1e-4."
        ),
    )
    p.add_argument(
        "--enable-pcgrad",
        action="store_true",
        help="Enable downstream PCGrad (global_shared for new CLI runs).",
    )
    p.add_argument(
        "--pcgrad-mode",
        choices=("global_shared", "per_parameter"),
        default=None,
        help="Enable the selected PCGrad mode; per_parameter reproduces the legacy variant.",
    )
    p.add_argument(
        "--pcgrad-projection-strength",
        type=float,
        default=None,
        help=(
            "Enable global_shared PCGrad with strength eta in [0, 1]: "
            "0=ordinary backward, 0.5=half correction, 1=full PCGrad (default)."
        ),
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
    return p.parse_args()


def configure_quality_adapter_experiment(cfg: Dict, args: argparse.Namespace) -> None:
    """Keep architecture/initialization controls separate from optimizer controls."""
    residual_init = getattr(args, "quality_adapter_residual_init", None)
    residual_fixed = getattr(args, "quality_adapter_residual_fixed", None)
    freeze_bn_stats = getattr(args, "quality_adapter_freeze_bn_stats", False)
    if args.quality_adapter_mode is None and args.quality_adapter_init is None and residual_init is None and residual_fixed is None and not freeze_bn_stats:
        return
    qcfg = cfg.get("model", {}).get("quality_adapter", {})
    if args.disable_quality_adapter or not bool(qcfg.get("enabled", False)):
        raise ValueError("Quality Adapter mode/init controls require an enabled Adapter.")
    if freeze_bn_stats:
        if bool(qcfg.get("freeze", False)):
            raise ValueError("BN-statistics-only ablation requires quality_adapter.freeze=false.")
        qcfg["freeze_bn_stats"] = True
        print("[INFO] Quality Adapter BatchNorm running statistics frozen; affine and other Adapter parameters remain trainable.")
    if args.quality_adapter_mode is not None:
        qcfg["mode"] = args.quality_adapter_mode
    if residual_init is not None:
        if qcfg.get("mode") != "residual_feature":
            raise ValueError("--quality-adapter-residual-init requires --quality-adapter-mode residual_feature.")
        if not math.isfinite(residual_init) or not 0.0 < residual_init < 1.0:
            raise ValueError("--quality-adapter-residual-init must be finite and in (0,1).")
        qcfg["residual_init"] = float(residual_init)
    if residual_fixed is not None:
        if qcfg.get("mode") != "residual_feature":
            raise ValueError("--quality-adapter-residual-fixed requires --quality-adapter-mode residual_feature.")
        if not math.isfinite(residual_fixed) or not 0.0 <= residual_fixed <= 1.0:
            raise ValueError("--quality-adapter-residual-fixed must be finite and in [0,1].")
        qcfg["residual_fixed"] = float(residual_fixed)
        print(f"[INFO] Quality feature residual alpha fixed at {residual_fixed:g}; Adapter and auxiliary supervision retained.")
    if args.quality_adapter_init is not None:
        cfg.setdefault("train", {})["skip_pretrained_quality_adapter"] = args.quality_adapter_init == "random"
    if qcfg.get("mode") == "residual_feature":
        qcfg.setdefault("residual_init", 0.1)
    print(
        f"[INFO] Quality Adapter experiment: mode={qcfg.get('mode', 'replace')}, "
        f"initialization={'random' if cfg.get('train', {}).get('skip_pretrained_quality_adapter', False) else 'pretrained'}; "
        "all other checkpoint loading, targets and losses unchanged."
    )


def configure_visual_backbone_lr(cfg: Dict, args: argparse.Namespace) -> None:
    """Set both visual encoder LRs; preserve Fusion, text and task groups."""
    if args.visual_backbone_lr_mult is None:
        return
    multiplier = float(args.visual_backbone_lr_mult)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Visual backbone LR multiplier must be finite and positive.")
    tcfg = cfg.get("train", {})
    lrcfg = tcfg.get("layerwise_lr", {})
    if not bool(lrcfg.get("enabled", False)):
        raise ValueError("--visual-backbone-lr-mult requires train.layerwise_lr.enabled=true.")
    lrcfg["global_encoder_mult"] = multiplier
    lrcfg["local_encoder_mult"] = multiplier
    print(f"[INFO] Visual backbone LR override: multiplier={multiplier:g}, "
          f"initial_lr={float(tcfg.get('lr', 2e-5)) * multiplier:.2e}; "
          "Fusion, text and task-group LRs unchanged.")


def configure_quality_adapter_lr(cfg: Dict, args: argparse.Namespace) -> None:
    """Override one optimizer-group LR without changing losses or trainability."""
    if args.quality_adapter_lr_mult is None:
        return
    multiplier = float(args.quality_adapter_lr_mult)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("Quality Adapter LR multiplier must be finite and positive.")
    tcfg = cfg.get("train", {})
    lrcfg = tcfg.get("layerwise_lr", {})
    if not bool(lrcfg.get("enabled", False)):
        raise ValueError("--quality-adapter-lr-mult requires train.layerwise_lr.enabled=true.")
    qcfg = cfg.get("model", {}).get("quality_adapter", {})
    if not bool(qcfg.get("enabled", False)) or bool(qcfg.get("freeze", False)):
        raise ValueError("--quality-adapter-lr-mult requires an enabled, unfrozen Quality Adapter.")
    lrcfg["quality_adapter_mult"] = multiplier
    initial_lr = float(tcfg.get("lr", 2e-5)) * multiplier
    print(
        f"[INFO] Quality Adapter LR override: multiplier={multiplier:g}, "
        f"initial_lr={initial_lr:.2e}; other optimizer-group LRs unchanged."
    )


def configure_downstream_pcgrad(cfg: Dict, args: argparse.Namespace) -> None:
    """Apply CLI overrides and validate before data/model construction."""
    if (
        args.enable_pcgrad
        or args.pcgrad_mode is not None
        or args.pcgrad_projection_strength is not None
    ):
        pcfg = cfg.setdefault("train", {}).setdefault("pcgrad", {})
        pcfg["enabled"] = True
        pcfg["mode"] = args.pcgrad_mode or pcfg.get("mode", "global_shared")
        if args.pcgrad_projection_strength is not None:
            pcfg["projection_strength"] = float(args.pcgrad_projection_strength)
    if not bool(cfg.get("train", {}).get("pcgrad", {}).get("enabled", False)):
        return

    pcfg = cfg["train"]["pcgrad"]
    # Historical snapshots with no mode retain their per-tensor algorithm.
    pcfg.setdefault("mode", "per_parameter")
    if pcfg["mode"] not in {"global_shared", "per_parameter"}:
        raise ValueError(f"Unknown downstream PCGrad mode: {pcfg['mode']!r}")
    if pcfg["mode"] == "per_parameter":
        if "projection_strength" in pcfg:
            raise ValueError("projection_strength is supported only by global_shared PCGrad.")
    else:
        strength = float(pcfg.get("projection_strength", 1.0))
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("PCGrad projection_strength must be finite and in [0, 1].")
        pcfg["projection_strength"] = strength
        if (
            not bool(cfg["train"].get("two_output_only", False))
            or bool(cfg["train"].get("final_auxiliary_enabled", False))
            or float(cfg["train"].get("lambda_fusion_gate", 0.0)) > 0
        ):
            raise ValueError(
                "global_shared PCGrad requires the two-output Quality/Align protocol "
                "without Final auxiliary or Final fusion-gate supervision."
            )
    strength_info = (
        f", eta={pcfg['projection_strength']:g}"
        if pcfg["mode"] == "global_shared" else ""
    )
    print(f"[INFO] Downstream PCGrad enabled (mode={pcfg['mode']}{strength_info}).")


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return {"N": int(len(y_true)), "PLCC": 0.0, "SROCC": 0.0, "KROCC": 0.0, "RMSE": 0.0, "MAE": 0.0}

    def safe(fn):
        try:
            v = fn()[0]
            return 0.0 if np.isnan(v) else float(v)
        except Exception:
            return 0.0

    return {
        "N": int(len(y_true)),
        "PLCC": safe(lambda: pearsonr(y_true, y_pred)),
        "SROCC": safe(lambda: spearmanr(y_true, y_pred)),
        "KROCC": safe(lambda: kendalltau(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


def unpack_loaders(loaders):
    return loaders if len(loaders) == 3 else (loaders[0], loaders[1], None)


def build_iqa_model(cfg: Dict, pretrained_checkpoint: str = "") -> CrossModalIQAFullModel:
    """Build locally restored modules without first downloading their base weights."""
    model_cfg = copy.deepcopy(cfg)
    if not pretrained_checkpoint:
        return CrossModalIQAFullModel(model_cfg)

    load_scope = str(
        model_cfg.get("train", {}).get("pretrained_load_scope", "auto") or "auto"
    ).strip().lower()
    if load_scope not in {"auto", "full", "representation", "backbone"}:
        raise ValueError(
            f"Unknown pretrained load scope: {load_scope}. "
            "Choose from: auto, full, representation, backbone."
        )

    # All supported checkpoint scopes restore the two visual encoders.
    mcfg = model_cfg.setdefault("model", {})
    mcfg.setdefault("global_encoder", {})["pretrained"] = False
    mcfg.setdefault("local_encoder", {})["pretrained"] = False

    # A full joint checkpoint also contains backbone.text_encoder.*. Selective
    # loading keeps the YAML CLIP bootstrap because it does not restore CLIP.
    disabled = ["global_encoder", "local_encoder"]
    if load_scope in {"auto", "full"}:
        mcfg.setdefault("text_encoder", {})["clip_pretrained"] = None
        disabled.append("text_encoder")

    print(
        "[INFO] Local checkpoint supplied; skip redundant online initialization "
        f"for {', '.join(disabled)} (scope={load_scope})."
    )
    return CrossModalIQAFullModel(model_cfg)


def build_optimizer(cfg: Dict, model: nn.Module):
    tcfg = cfg["train"]
    base_lr = float(tcfg.get("lr", 2e-5))
    weight_decay = float(tcfg.get("weight_decay", 2e-3))
    lrcfg = tcfg.get("layerwise_lr", {})
    if not bool(lrcfg.get("enabled", False)):
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    groups = {
        "global_encoder": {"params": [], "lr": base_lr * float(lrcfg.get("global_encoder_mult", lrcfg.get("backbone_mult", 0.1)))},
        "local_encoder": {"params": [], "lr": base_lr * float(lrcfg.get("local_encoder_mult", lrcfg.get("backbone_mult", 0.1)))},
        "fusion": {"params": [], "lr": base_lr * float(lrcfg.get("fusion_mult", 0.5))},
        "text_context": {"params": [], "lr": base_lr * float(lrcfg.get("text_context_mult", 0.10))},
        "fusion_gate": {"params": [], "lr": base_lr * float(lrcfg.get("fusion_gate_mult", lrcfg.get("fusion_mult", 0.5)))},
        "quality_adapter": {"params": [], "lr": base_lr * float(lrcfg.get("quality_adapter_mult", lrcfg.get("quality_mult", 1.0)))},
        "quality": {"params": [], "lr": base_lr * float(lrcfg.get("quality_mult", 1.0))},
        "align": {"params": [], "lr": base_lr * float(lrcfg.get("align_mult", 1.0))},
        "other": {"params": [], "lr": base_lr * float(lrcfg.get("other_mult", 1.0))},
    }

    quality_prefixes = (
        "local_patch_proj", "nss_proj", "nss_gate", "text_gate",
        "patch_quality_head", "global_quality_head", "quality_mix_logit", "quality_sample_gate",
        "quality_distribution_head",
        "quality_adapter_head", "quality_adapter_gate_logit",
    )
    align_prefixes = (
        "text_to_fused", "align_global_head", "align_patch_img_proj",
        "align_patch_txt_proj", "align_patch_head", "align_gate",
        "align_global_adapter", "align_patch_adapter", "align_text_adapter",
        "align_vmamba",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.global_encoder"):
            groups["global_encoder"]["params"].append(param)
        elif name.startswith("backbone.local_encoder"):
            groups["local_encoder"]["params"].append(param)
        elif name.startswith("backbone.fusion"):
            groups["fusion"]["params"].append(param)
        elif name == "backbone.text_encoder.context_vectors":
            groups["text_context"]["params"].append(param)
        elif name.startswith(("fusion_gate", "fusion_logit")):
            groups["fusion_gate"]["params"].append(param)
        elif name.startswith("quality_adapter."):
            groups["quality_adapter"]["params"].append(param)
        elif name.startswith("quality_text_bank."):
            groups["quality"]["params"].append(param)
        elif name.startswith("align_text_bank."):
            groups["align"]["params"].append(param)
        elif name.startswith(quality_prefixes):
            groups["quality"]["params"].append(param)
        elif name.startswith(align_prefixes):
            groups["align"]["params"].append(param)
        else:
            groups["other"]["params"].append(param)

    param_groups = []
    for group_name, group in groups.items():
        if not group["params"]:
            continue
        param_groups.append({
            "params": group["params"],
            "lr": float(group["lr"]),
            "weight_decay": weight_decay,
            "name": group_name,
        })

    summary = ", ".join(f"{g['name']}={g['lr']:.2e}" for g in param_groups)
    print(f"[INFO] Layerwise LR enabled: {summary}")
    return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def get_progressive_finetune_stage(cfg: Dict, epoch: int) -> str:
    """Return the active downstream fine-tuning stage for a 1-indexed epoch."""
    pcfg = cfg.get("train", {}).get("progressive_unfreeze", {})
    if not bool(pcfg.get("enabled", False)):
        return "full"

    fusion_start = int(pcfg.get("fusion_start_epoch", 6))
    encoder_start = int(pcfg.get("encoder_start_epoch", 16))
    if int(epoch) < fusion_start:
        return "head_only"
    if int(epoch) < encoder_start:
        return "fusion"
    return "full"


def validate_progressive_finetune_config(cfg: Dict) -> None:
    """Fail early when a progressive schedule cannot be applied safely."""
    tcfg = cfg.get("train", {})
    pcfg = tcfg.get("progressive_unfreeze", {})
    if not bool(pcfg.get("enabled", False)):
        return

    if not bool(tcfg.get("layerwise_lr", {}).get("enabled", False)):
        raise ValueError(
            "train.progressive_unfreeze requires train.layerwise_lr.enabled=true "
            "so parameter groups can be frozen and restored unambiguously."
        )

    epochs = int(tcfg.get("epochs", 60))
    fusion_start = int(pcfg.get("fusion_start_epoch", 6))
    encoder_start = int(pcfg.get("encoder_start_epoch", 16))
    if not (1 <= fusion_start < encoder_start <= epochs):
        raise ValueError(
            "Expected 1 <= fusion_start_epoch < encoder_start_epoch <= epochs, "
            f"got fusion_start_epoch={fusion_start}, "
            f"encoder_start_epoch={encoder_start}, epochs={epochs}."
        )


def apply_progressive_finetune_stage(
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    epoch: int,
    base_trainable: Mapping[int, bool],
) -> tuple[str, Dict[str, int]]:
    """Freeze optimizer groups while preserving original trainability.

    All parameter groups are added to the optimizer before this function is
    called. Consequently, later stages can unfreeze a group without rebuilding
    the optimizer or losing optimizer state. The base_trainable mapping prevents
    the schedule from overriding modules frozen permanently by model config.
    """
    stage = get_progressive_finetune_stage(cfg, epoch)
    allowed_groups = {
        "head_only": {"quality", "fusion_gate"},
        "fusion": {"quality", "fusion", "fusion_gate"},
        "full": None,
    }[stage]

    trainable_counts: Dict[str, int] = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group{index}"))
        group_allowed = allowed_groups is None or group_name in allowed_groups
        count = 0
        for parameter in group["params"]:
            should_train = bool(base_trainable.get(id(parameter), False)) and group_allowed
            parameter.requires_grad_(should_train)
            if should_train:
                count += parameter.numel()
        trainable_counts[group_name] = count
    return stage, trainable_counts


def enforce_progressive_frozen_eval(model: nn.Module, cfg: Dict, epoch: int) -> None:
    """Keep frozen feature modules in eval mode after model.train().

    Disabling gradients alone would still let ResNet BatchNorm running
    statistics and Fusion dropout change during the frozen stages.
    """
    if not bool(cfg.get("train", {}).get("progressive_unfreeze", {}).get("enabled", False)):
        return

    stage = get_progressive_finetune_stage(cfg, epoch)
    base_model = _unwrap_model(model)
    if stage in {"head_only", "fusion"}:
        base_model.backbone.global_encoder.eval()
        base_model.backbone.local_encoder.eval()
    if stage == "head_only":
        base_model.backbone.fusion.eval()
    if hasattr(base_model, "quality_adapter") and not any(
        parameter.requires_grad for parameter in base_model.quality_adapter.parameters()
    ):
        base_model.quality_adapter.eval()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.view(-1).to(pred.device).float()
    pred = pred.view(-1)
    target = target.view(-1).to(pred.device).float()
    if mask.sum() <= 0:
        return pred.new_tensor(0.0)
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + 1e-6)


def make_dynamic_align_gate_target(
    s_global: torch.Tensor,
    s_patch: torch.Tensor,
    align_target: torch.Tensor,
    tau: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a soft target that favors the expert with lower Align MOS error.

    The returned tensors are detached so experts cannot improve this auxiliary
    objective by changing their predictions instead of improving the gate.
    """
    tau = max(float(tau), 1e-4)
    target = align_target.view(-1).to(s_global.device).float()
    global_error = torch.abs(s_global.view(-1).detach() - target)
    patch_error = torch.abs(s_patch.view(-1).detach() - target)
    gate_target = torch.sigmoid((patch_error - global_error) / tau)
    oracle_global = (global_error <= patch_error).to(gate_target.dtype)
    return gate_target.detach(), oracle_global.detach()


def make_dynamic_fusion_gate_target(
    s_quality: torch.Tensor,
    s_align: torch.Tensor,
    final_target: torch.Tensor,
    tau: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a MOS-error-aware soft target for the final fusion gate.

    A target near one means the quality expert has lower detached error and
    should receive more weight; a target near zero means the alignment expert
    is better for this sample. The target is computed only on training labels
    and is detached before BCE, so experts cannot game the gate auxiliary loss.
    """
    tau = max(float(tau), 1e-4)
    target = final_target.view(-1).to(s_quality.device).float()
    quality_error = torch.abs(s_quality.view(-1).detach() - target)
    align_error = torch.abs(s_align.view(-1).detach() - target)
    gate_target = torch.sigmoid((align_error - quality_error) / tau)
    oracle_quality = (quality_error <= align_error).to(gate_target.dtype)
    return gate_target.detach(), oracle_quality.detach()


def extend_valid(dst_pred: List[float], dst_tgt: List[float], pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    m = mask.detach().cpu().numpy().reshape(-1) > 0.5
    p = pred.detach().cpu().numpy().reshape(-1)
    t = target.detach().cpu().numpy().reshape(-1)
    if m.sum() > 0:
        dst_pred.extend(p[m].tolist())
        dst_tgt.extend(t[m].tolist())


def perturb_rank_loss(scores: List[torch.Tensor], margin: float = 0.05) -> torch.Tensor:
    if len(scores) < 2:
        return scores[0].new_tensor(0.0)
    loss = scores[0].new_tensor(0.0)
    for a, b in zip(scores[:-1], scores[1:]):
        loss = loss + F.relu(margin - (a.view(-1) - b.view(-1))).mean()
    return loss


def make_shifted_prompts(prompts):
    """Random/easy negative fallback: cyclically shift prompts in the mini-batch."""
    prompts = [str(p) for p in prompts]
    if len(prompts) < 2:
        return prompts
    return prompts[1:] + prompts[:1]


def _unwrap_model(model):
    return getattr(model, "module", model)


def _pcgrad_backward(
    loss_quality: torch.Tensor,
    loss_align: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> None:
    """Legacy per-parameter-tensor PCGrad, retained for old-run reproduction.

    The two task losses are differentiated separately. For parameters shared
    by both tasks, a negative gradient dot product is projected away before
    the gradients are summed. Parameters used by only one task keep their
    original gradient. Both losses are scaled by the same AMP scaler, so the
    projection is valid before ``unscale_``.
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


def _normalize_prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt).strip().lower())


def _prompt_from_dataset_index(dataset, idx: int) -> str | None:
    idx = int(idx)
    if hasattr(dataset, "_prompt"):
        return str(dataset._prompt(idx)).strip()
    if hasattr(dataset, "_make_prompt") and hasattr(dataset, "df"):
        return str(dataset._make_prompt(dataset.df.iloc[idx])).strip()
    if hasattr(dataset, "df") and "prompt" in dataset.df.columns:
        return str(dataset.df.iloc[idx]["prompt"]).strip()
    return None


def collect_prompts_from_dataset(dataset) -> List[str]:
    """Collect prompts from a Dataset/Subset without reading image tensors."""
    if isinstance(dataset, torch.utils.data.Subset):
        prompts = []
        for idx in dataset.indices:
            prompt = _prompt_from_dataset_index(dataset.dataset, int(idx))
            if prompt:
                prompts.append(prompt)
        return prompts

    if isinstance(dataset, torch.utils.data.ConcatDataset):
        prompts = []
        for sub_dataset in dataset.datasets:
            prompts.extend(collect_prompts_from_dataset(sub_dataset))
        return prompts

    prompts = []
    for idx in range(len(dataset)):
        prompt = _prompt_from_dataset_index(dataset, idx)
        if prompt:
            prompts.append(prompt)
    return prompts


@torch.no_grad()
def build_prompt_bank(model, dataset, device, batch_size: int = 256):
    """Encode all unique training prompts once for full-dataset hard mining."""
    prompts = collect_prompts_from_dataset(dataset)
    unique_prompts = []
    seen = set()
    for prompt in prompts:
        key = _normalize_prompt_key(prompt)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_prompts.append(str(prompt))

    if len(unique_prompts) < 2:
        print("[WARN] prompt bank disabled: fewer than 2 unique prompts in training set.")
        return None

    base_model = _unwrap_model(model)
    chunks = []
    for start in range(0, len(unique_prompts), int(batch_size)):
        chunk_prompts = unique_prompts[start:start + int(batch_size)]
        chunk_feat = base_model.backbone.encode_text(chunk_prompts, device=device)
        chunks.append(F.normalize(chunk_feat.float(), dim=-1))
    embeddings = torch.cat(chunks, dim=0)
    print(f"[INFO] Built prompt bank: total={len(prompts)}, unique={len(unique_prompts)}")
    return {"prompts": unique_prompts, "keys": [_normalize_prompt_key(p) for p in unique_prompts], "embeddings": embeddings}


@torch.no_grad()
def make_prompt_bank_hard_prompts(model, prompts, prompt_bank, device, topk: int = 20):
    """Select real training prompts that are CLIP-similar to each positive prompt."""
    prompts = [str(p) for p in prompts]
    if prompt_bank is None:
        shifted = make_shifted_prompts(prompts)
        return shifted, torch.zeros(len(prompts), dtype=torch.float32, device=device)

    bank_prompts = prompt_bank["prompts"]
    bank_keys = prompt_bank["keys"]
    bank_feat = prompt_bank["embeddings"].to(device)
    base_model = _unwrap_model(model)
    text_feat = base_model.backbone.encode_text(prompts, device=device)
    text_feat = F.normalize(text_feat.float(), dim=-1)
    sim = text_feat @ bank_feat.t()

    hard_prompts = list(prompts)
    valid = torch.zeros(len(prompts), dtype=torch.float32, device=device)
    for i, prompt in enumerate(prompts):
        row = sim[i].clone()
        pos_key = _normalize_prompt_key(prompt)
        for j, bank_key in enumerate(bank_keys):
            if bank_key == pos_key:
                row[j] = -1e4

        candidate_idx = torch.nonzero(row > -9999, as_tuple=False).view(-1)
        if candidate_idx.numel() == 0:
            continue

        candidate_scores = row[candidate_idx]
        order = torch.argsort(candidate_scores, descending=True)
        sorted_idx = candidate_idx[order]
        k = max(1, min(int(topk), int(sorted_idx.numel())))
        hard_pool = sorted_idx[:k]
        pick = hard_pool[torch.randint(k, (1,), device=device)].item()
        hard_prompts[i] = bank_prompts[int(pick)]
        valid[i] = 1.0
    return hard_prompts, valid


_PROMPT_REPLACEMENT_GROUPS = [
    ["dog", "cat", "horse", "bird", "corgi", "puppy", "rabbit", "tiger", "lion", "bear", "cow", "sheep", "elephant", "zebra", "giraffe"],
    ["man", "woman", "boy", "girl", "person", "child", "baby"],
    ["car", "truck", "bus", "train", "boat", "airplane", "bicycle", "motorcycle", "aircraft"],
    ["chair", "bench", "table", "bed", "sofa", "lamp", "vase", "clock", "bag", "bottle", "cup"],
    ["pizza", "sushi", "cake", "bread", "burger", "sandwich", "apple", "banana", "flower"],
    ["red", "blue", "green", "yellow", "black", "white", "purple", "pink", "orange", "brown"],
    ["grass", "snow", "beach", "street", "forest", "desert", "mountain", "ocean", "room", "city", "sky", "space"],
    ["sitting", "standing", "running", "walking", "flying", "sleeping", "jumping", "dancing", "riding"],
    ["anime", "realistic", "cartoon", "sketch", "oil painting", "watercolor", "digital art", "photorealistic"],
]

_PROMPT_CROSS_REPLACEMENTS = {
    "dog": "car", "cat": "truck", "horse": "bench", "bird": "boat", "corgi": "bicycle", "puppy": "chair",
    "man": "dog", "woman": "cat", "boy": "bird", "girl": "horse", "person": "car", "child": "truck",
    "car": "dog", "truck": "cat", "bus": "horse", "train": "bird", "boat": "chair", "airplane": "bench",
    "pizza": "vase", "sushi": "clock", "cake": "truck", "bread": "bird", "burger": "lamp", "sandwich": "boat",
    "grass": "room", "snow": "beach", "beach": "forest", "street": "ocean", "forest": "city", "space": "kitchen",
}

_FALLBACK_NEGATIVE_PROMPTS = [
    "an abstract geometric pattern",
    "a close up of a plain texture",
    "an empty indoor room",
    "a simple object on a table",
]


def _contains_term(text_lower: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text_lower) is not None


def _replace_term(text: str, src: str, dst: str) -> str:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(src)}(?![A-Za-z0-9])"
    return re.sub(pattern, dst, text, count=1, flags=re.IGNORECASE)


def _make_same_group_negative(prompt: str) -> tuple[str, bool]:
    text_lower = prompt.lower()
    candidates = []
    for group in _PROMPT_REPLACEMENT_GROUPS:
        for src in group:
            if not _contains_term(text_lower, src):
                continue
            for dst in group:
                if dst == src or _contains_term(text_lower, dst):
                    continue
                candidates.append((src, dst))
    if not candidates:
        return prompt, False
    src, dst = candidates[torch.randint(len(candidates), (1,)).item()]
    return _replace_term(prompt, src, dst), True


def _make_cross_group_negative(prompt: str) -> tuple[str, bool]:
    text_lower = prompt.lower()
    candidates = [(src, dst) for src, dst in _PROMPT_CROSS_REPLACEMENTS.items() if _contains_term(text_lower, src)]
    if candidates:
        src, dst = candidates[torch.randint(len(candidates), (1,)).item()]
        return _replace_term(prompt, src, dst), True

    hard_prompt, ok = _make_same_group_negative(prompt)
    if ok:
        second_prompt, second_ok = _make_same_group_negative(hard_prompt)
        return (second_prompt, True) if second_ok and second_prompt != hard_prompt else (hard_prompt, True)

    fallback = _FALLBACK_NEGATIVE_PROMPTS[torch.randint(len(_FALLBACK_NEGATIVE_PROMPTS), (1,)).item()]
    return fallback, False


@torch.no_grad()
def make_constructed_align_rank_prompts(prompts, device):
    """Construct prompt negatives with rule-based local semantic edits.

    Hard prompt changes one matched word within the same semantic group, such as
    dog -> cat or grass -> snow. Negative prompt uses a stronger cross-group edit
    when possible. This avoids relying on the current mini-batch for negatives.
    """
    hard_prompts, neg_prompts, valid = [], [], []
    for prompt in [str(p) for p in prompts]:
        hard_prompt, hard_ok = _make_same_group_negative(prompt)
        neg_prompt, neg_ok = _make_cross_group_negative(prompt)
        hard_prompts.append(hard_prompt)
        neg_prompts.append(neg_prompt)
        valid.append(1.0 if hard_ok and (hard_prompt.strip().lower() != prompt.strip().lower()) else 0.0)
    return hard_prompts, neg_prompts, torch.tensor(valid, dtype=torch.float32, device=device)


@torch.no_grad()
def make_clip_inbatch_hard_negative_prompts(model, prompts, device):
    """Build semantic hard negative prompts from the current mini-batch.

    For each prompt, select the most CLIP-similar *different* prompt in the
    mini-batch. This is harder than a random shifted prompt and encourages the
    alignment branch to learn fine-grained prompt-image evidence instead of only
    separating obviously unrelated prompts.

    If a batch contains identical prompts or has no valid alternative, it falls
    back to the shifted prompt.
    """
    prompts = [str(p) for p in prompts]
    b = len(prompts)
    if b < 2:
        return prompts

    shifted = make_shifted_prompts(prompts)
    try:
        text_feat = model.backbone.encode_text(prompts, device=device)  # [B,D], already normalized
        sim = text_feat @ text_feat.t()
        # Exclude itself and exact duplicated prompts.
        for i in range(b):
            sim[i, i] = -1e4
            for j in range(b):
                if i != j and prompts[i].strip().lower() == prompts[j].strip().lower():
                    sim[i, j] = -1e4
        idx = sim.argmax(dim=1).detach().cpu().tolist()
        hard = []
        for i, j in enumerate(idx):
            if j < 0 or j >= b or sim[i, j].item() < -999:
                hard.append(shifted[i])
            else:
                hard.append(prompts[j])
        return hard
    except Exception:
        return shifted


@torch.no_grad()
def make_clip_inbatch_align_rank_prompts(model, prompts, device, topk: int = 5):
    """Build prompt triplets for align ranking using CLIP text similarity.

    For each positive prompt in the mini-batch:
      - hard prompt: sampled from the top-k most CLIP-similar different prompts.
      - negative prompt: the least CLIP-similar different prompt in the batch.

    A per-sample validity mask is returned because batches can contain repeated
    prompts or too few distinct alternatives.
    """
    prompts = [str(p) for p in prompts]
    b = len(prompts)
    shifted = make_shifted_prompts(prompts)
    if b < 3:
        return shifted, shifted, torch.zeros(b, dtype=torch.float32, device=device)

    hard_prompts = list(shifted)
    neg_prompts = list(shifted)
    valid = torch.zeros(b, dtype=torch.float32, device=device)
    try:
        base_model = _unwrap_model(model)
        text_feat = base_model.backbone.encode_text(prompts, device=device)
        sim = text_feat @ text_feat.t()
        norm_prompts = [p.strip().lower() for p in prompts]

        for i in range(b):
            row = sim[i].clone()
            row[i] = -1e4
            for j in range(b):
                if i != j and norm_prompts[i] == norm_prompts[j]:
                    row[j] = -1e4

            candidate_idx = torch.nonzero(row > -9999, as_tuple=False).view(-1)
            if candidate_idx.numel() < 2:
                continue

            candidate_scores = row[candidate_idx]
            order = torch.argsort(candidate_scores, descending=True)
            sorted_idx = candidate_idx[order]

            k = max(1, min(int(topk), int(sorted_idx.numel()) - 1))
            hard_pool = sorted_idx[:k]
            hard_pick = hard_pool[torch.randint(k, (1,), device=device)].item()
            neg_pick = sorted_idx[-1].item()

            if int(hard_pick) == int(neg_pick):
                continue

            hard_prompts[i] = prompts[int(hard_pick)]
            neg_prompts[i] = prompts[int(neg_pick)]
            valid[i] = 1.0
    except Exception:
        valid.zero_()

    return hard_prompts, neg_prompts, valid


def hard_negative_prompt_ranking_loss(
    true_align: torch.Tensor,
    hard_align: torch.Tensor,
    random_align: torch.Tensor,
    mask: torch.Tensor,
    margin_true_hard: float = 0.08,
    margin_hard_random: float = 0.04,
    random_weight: float = 0.5,
    direct_random_weight: float = 0.25,
):
    """Triplet-style prompt ranking for the alignment branch.

    Desired order:
      S_align(image, true_prompt)
        > S_align(image, hard_negative_prompt)
        > S_align(image, random_negative_prompt)

    The direct true-random term is kept with a small weight as a stabilizer.
    """
    mask = mask.view(-1).to(true_align.device).float()
    if mask.sum() <= 0:
        return true_align.new_tensor(0.0)
    t = true_align.view(-1)
    h = hard_align.view(-1)
    r = random_align.view(-1)
    l_th = F.relu(float(margin_true_hard) - (t - h))
    l_hr = F.relu(float(margin_hard_random) - (h - r))
    l_tr = F.relu(float(margin_true_hard + margin_hard_random) - (t - r))
    loss_vec = l_th + float(random_weight) * l_hr + float(direct_random_weight) * l_tr
    return (loss_vec * mask).sum() / (mask.sum() + 1e-6)


def _compute_text_bank_aux_loss(logits, score, target, mask, centers):
    """Soft ordinal five-level loss used by downstream auxiliary training."""
    zero = target.new_zeros(())
    if logits is None or score is None or centers is None:
        return zero, zero
    logits = logits.float()
    centers = centers.to(device=logits.device, dtype=torch.float32)
    target = target.float().view(-1).clamp(0.0, 1.0)
    mask = mask.float().view(-1).clamp_min(0.0)
    high = torch.searchsorted(centers.contiguous(), target.contiguous()).clamp(1, centers.numel() - 1)
    low = high - 1
    fraction = ((target - centers[low]) / (centers[high] - centers[low])).clamp(0.0, 1.0)
    soft_target = torch.zeros_like(logits)
    soft_target.scatter_add_(1, low[:, None], (1.0 - fraction)[:, None])
    soft_target.scatter_add_(1, high[:, None], fraction[:, None])
    ce = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    denominator = mask.sum().clamp_min(1e-6)
    return ((ce * mask).sum() / denominator,
            ((score.float().view(-1) - target).abs() * mask).sum() / denominator)


def compute_losses(model, batch, device, cfg, prompt_bank=None, epoch: int | None = None):
    tcfg = cfg["train"]
    dcfg = cfg.get("dataset", {})
    image = batch["image"].to(device, non_blocking=True)
    validate_local_crops(batch, cfg)
    local_crops = batch.get("local_crops")
    if local_crops is not None:
        local_crops = local_crops.to(device, non_blocking=True)
    quality = batch["quality_score"].to(device, non_blocking=True)
    align = batch["align_score"].to(device, non_blocking=True)
    final_score = batch["score"].to(device, non_blocking=True)
    qmask = batch.get("quality_mask", torch.ones_like(quality)).to(device, non_blocking=True).float()
    amask = batch.get("align_mask", torch.ones_like(align)).to(device, non_blocking=True).float()
    fmask = batch.get("final_mask", torch.ones_like(final_score)).to(device, non_blocking=True).float()
    final_auxiliary_enabled = bool(tcfg.get("final_auxiliary_enabled", False))
    # In the two-output protocol `score` is only a backward-compatible
    # dataset field. It is never evaluated as an output. A run may opt into
    # a very small regression-only auxiliary term to stabilize the shared
    # representation; ranking and Soft-SROCC on this synthetic mixture stay
    # disabled.
    if bool(tcfg.get("two_output_only", False)) and not final_auxiliary_enabled:
        fmask = torch.zeros_like(final_score)
    prompts = batch["prompt"]

    out = model.forward_single(image, prompts=prompts, local_crops=local_crops)
    if bool(tcfg.get("quality_only", False)):
        # Datasets without alignment MOS use the quality prediction as the
        # final output. This removes every gradient path through Align/Final.
        out["s_final"] = out["s_quality"]
        out["fusion_w_img"] = out["s_quality"].new_tensor(1.0)

    l_quality = masked_mse(out["s_quality"], quality, qmask)
    l_align = masked_mse(out["s_align"], align, amask)
    l_final = masked_mse(out["s_final"], final_score, fmask)
    l_align_g = masked_mse(out["s_align_global"], align, amask)
    l_align_p = masked_mse(out["s_align_patch"], align, amask)
    l_quality_adapter_aux = masked_mse(
        out["s_quality_global_adapter"], quality, qmask
    )
    base_model = _unwrap_model(model)
    quality_bank = getattr(base_model, "quality_text_bank", None)
    align_bank = getattr(base_model, "align_text_bank", None)
    quality_text_ce, quality_text_mae = _compute_text_bank_aux_loss(
        out.get("quality_text_logits"), out.get("quality_text_score"), quality, qmask,
        quality_bank.centers if quality_bank is not None else None,
    )
    align_text_ce, align_text_mae = _compute_text_bank_aux_loss(
        out.get("align_text_logits"), out.get("align_text_score"), align, amask,
        align_bank.centers if align_bank is not None else None,
    )
    quality_text_anchor = quality_bank.context_anchor_loss() if quality_bank is not None else image.new_zeros(())
    align_text_anchor = align_bank.context_anchor_loss() if align_bank is not None else image.new_zeros(())
    quality_distribution_ce, quality_distribution_mae = _compute_text_bank_aux_loss(
        out.get("quality_distribution_logits"), out.get("quality_distribution_score"),
        quality, qmask, getattr(base_model, "quality_distribution_centers", None),
    )

    rank_margin = float(tcfg.get("rank_margin", 0.02))
    if bool(tcfg.get("use_hard_rank", False)):
        hard_ratio = float(tcfg.get("hard_rank_ratio", 0.35))
        l_rank_quality = masked_hard_pairwise_label_ranking_loss(out["s_quality"], quality, qmask, margin=rank_margin, topk_ratio=hard_ratio)
        l_rank_align = masked_hard_pairwise_label_ranking_loss(out["s_align"], align, amask, margin=rank_margin, topk_ratio=hard_ratio)
        l_rank_final = masked_hard_pairwise_label_ranking_loss(out["s_final"], final_score, fmask, margin=rank_margin, topk_ratio=hard_ratio)
    else:
        l_rank_quality = masked_pairwise_label_ranking_loss(out["s_quality"], quality, qmask, margin=rank_margin)
        l_rank_align = masked_pairwise_label_ranking_loss(out["s_align"], align, amask, margin=rank_margin)
        l_rank_final = masked_pairwise_label_ranking_loss(out["s_final"], final_score, fmask, margin=rank_margin)

    # Soft-SROCC auxiliary loss. This directly optimizes a differentiable
    # Spearman-like rank correlation inside each mini-batch. Keep weights small
    # because MSE still stabilizes PLCC.
    srocc_tau = float(tcfg.get("srocc_tau", 0.02))
    l_srocc_quality = masked_soft_spearman_loss(out["s_quality"], quality, qmask, tau=srocc_tau)
    l_srocc_align = masked_soft_spearman_loss(out["s_align"], align, amask, tau=srocc_tau)
    l_srocc_final = masked_soft_spearman_loss(out["s_final"], final_score, fmask, tau=srocc_tau)

    scores = [out["s_quality"]]
    for strength in [str(s) for s in (dcfg.get("perturbation_strengths", []) or [])]:
        key = f"image_{strength}"
        if key in batch:
            pimg = batch[key].to(device, non_blocking=True)
            pout = model.forward_single(pimg, prompts=prompts)
            scores.append(pout["s_quality"])
    l_pert = perturb_rank_loss(scores, margin=float(tcfg.get("pert_margin", 0.05))) if qmask.sum() > 0 else image.new_tensor(0.0)

    # Easy in-batch negative prompt ranking (old version):
    # S_align(image, true_prompt) > S_align(image, shifted_prompt).
    if float(tcfg.get("lambda_neg", 0.0)) > 0 and amask.sum() > 0:
        neg_prompts = make_shifted_prompts(prompts)
        negative_mask = amask.view(-1) * distinct_prompt_mask(prompts, neg_prompts, device)
        if bool((negative_mask > 0).any()):
            # Keep full batch size for BN; exclude invalid pairs from the loss.
            neg_out = model.forward_single(image, prompts=neg_prompts, local_crops=local_crops)
            neg_vec = F.relu(float(tcfg.get("neg_margin", 0.10)) - (out["s_align"] - neg_out["s_align"]))
            l_neg = (neg_vec * negative_mask).sum() / (negative_mask.sum() + 1e-6)
        else:
            l_neg = image.new_tensor(0.0)
    else:
        l_neg = image.new_tensor(0.0)

    # Hard negative prompt ranking for SROCC improvement:
    # true prompt > CLIP-similar hard negative prompt > random shifted prompt.
    if float(tcfg.get("lambda_hard_neg", 0.0)) > 0 and amask.sum() > 0:
        hard_prompts = make_clip_inbatch_hard_negative_prompts(model, prompts, image.device)
        random_prompts = make_shifted_prompts(prompts)
        hard_out = model.forward_single(
            image, prompts=hard_prompts, local_crops=local_crops
        )
        random_out = model.forward_single(
            image, prompts=random_prompts, local_crops=local_crops
        )
        l_hard_neg = hard_negative_prompt_ranking_loss(
            out["s_align"],
            hard_out["s_align"],
            random_out["s_align"],
            amask,
            margin_true_hard=float(tcfg.get("hard_neg_margin_true_hard", 0.08)),
            margin_hard_random=float(tcfg.get("hard_neg_margin_hard_random", 0.04)),
            random_weight=float(tcfg.get("hard_neg_random_weight", 0.5)),
            direct_random_weight=float(tcfg.get("hard_neg_direct_random_weight", 0.25)),
        )
    else:
        l_hard_neg = image.new_tensor(0.0)

    # New prompt ranking loss:
    # pair mode:    positive prompt > hard prompt
    # triplet mode: positive prompt > hard prompt > negative prompt
    if bool(tcfg.get("use_align_rank", False)) and float(tcfg.get("lambda_align_rank", 0.0)) > 0 and amask.sum() > 0:
        align_rank_mode = str(tcfg.get("align_rank_mode", "triplet")).lower()
        neg_source = str(tcfg.get("align_negative_source", "constructed" if bool(tcfg.get("use_constructed_align_negatives", False)) else "batch")).lower()

        if neg_source == "prompt_bank":
            hard_rank_prompts, rank_valid = make_prompt_bank_hard_prompts(
                model,
                prompts,
                prompt_bank,
                image.device,
                topk=int(tcfg.get("prompt_bank_topk", tcfg.get("hard_negative_topk", 20))),
            )
            neg_rank_prompts = hard_rank_prompts
        elif bool(tcfg.get("use_constructed_align_negatives", False)) or neg_source == "constructed":
            hard_rank_prompts, neg_rank_prompts, rank_valid = make_constructed_align_rank_prompts(prompts, image.device)
        else:
            hard_rank_prompts, neg_rank_prompts, rank_valid = make_clip_inbatch_align_rank_prompts(
                model,
                prompts,
                image.device,
                topk=int(tcfg.get("hard_negative_topk", 5)),
            )
        rank_mask = amask.view(-1) * rank_valid.to(image.device).view(-1)
        if rank_mask.sum() > 0:
            hard_rank_out = model.forward_single(
                image, prompts=hard_rank_prompts, local_crops=local_crops
            )
            score_key = str(tcfg.get("align_rank_score_key", "s_align"))
            if score_key not in out or score_key not in hard_rank_out:
                score_key = "s_align"
            if align_rank_mode == "pair":
                l_align_prompt_rank = align_prompt_pair_ranking_loss(
                    out[score_key],
                    hard_rank_out[score_key],
                    mask=rank_mask,
                    margin=float(tcfg.get("align_rank_margin", tcfg.get("align_rank_margin_pos_hard", 0.03))),
                )
            else:
                neg_rank_out = model.forward_single(
                    image, prompts=neg_rank_prompts, local_crops=local_crops
                )
                if score_key not in neg_rank_out:
                    score_key = "s_align"
                l_align_prompt_rank = align_prompt_ranking_loss(
                    out[score_key],
                    hard_rank_out[score_key],
                    neg_rank_out[score_key],
                    mask=rank_mask,
                    margin_pos_hard=float(tcfg.get("align_rank_margin_pos_hard", 0.05)),
                    margin_hard_neg=float(tcfg.get("align_rank_margin_hard_neg", 0.05)),
                )
        else:
            l_align_prompt_rank = image.new_tensor(0.0)
    else:
        l_align_prompt_rank = image.new_tensor(0.0)

    # Prevent the alignment fusion from collapsing onto the global expert.
    # The alignment mask keeps this auxiliary objective disabled for datasets
    # without alignment labels, such as AGIQA-1K.
    if float(tcfg.get("lambda_align_gate_reg", 0.0)) > 0 and amask.sum() > 0:
        gate_target = float(tcfg.get("align_gate_max", 0.85))
        gate_excess = F.relu(out["align_gate"].view(-1) - gate_target)
        gate_mask = amask.view(-1)
        l_align_gate_reg = (gate_excess * gate_mask).sum() / (gate_mask.sum() + 1e-6)
    else:
        l_align_gate_reg = image.new_tensor(0.0)

    # Performance-aware dynamic gate supervision. A second gate forward uses
    # detached inputs, so this auxiliary BCE updates only the Align Gate head;
    # shared visual/text features and the two experts are still trained by the
    # original regression/ranking objectives.
    dynamic_gate_weight = float(tcfg.get("lambda_dynamic_align_gate", 0.0))
    dynamic_gate_start = int(tcfg.get("dynamic_align_gate_start_epoch", 0))
    dynamic_gate_active = (
        bool(tcfg.get("use_dynamic_align_gate", False))
        and dynamic_gate_weight > 0
        and amask.sum() > 0
        and (epoch is None or int(epoch) >= dynamic_gate_start)
    )
    if dynamic_gate_active:
        gate_target, _ = make_dynamic_align_gate_target(
            out["s_align_global"],
            out["s_align_patch"],
            align,
            tau=float(tcfg.get("dynamic_align_gate_tau", 0.05)),
        )
        if bool(tcfg.get("dynamic_align_gate_detach_inputs", True)):
            gate_logits = _unwrap_model(model).align_gate(out["align_gate_input"].detach()).squeeze(-1)
        else:
            gate_logits = out["align_gate_logit"]
        gate_bce = F.binary_cross_entropy_with_logits(
            gate_logits.float(), gate_target.float(), reduction="none"
        )
        gate_mask = amask.view(-1)
        l_dynamic_align_gate = (gate_bce * gate_mask).sum() / (gate_mask.sum() + 1e-6)
        dynamic_gate_target_mean = (gate_target * gate_mask).sum() / (gate_mask.sum() + 1e-6)
        dynamic_gate_pred_mean = (torch.sigmoid(gate_logits).view(-1) * gate_mask).sum() / (gate_mask.sum() + 1e-6)
    else:
        l_dynamic_align_gate = image.new_tensor(0.0)
        dynamic_gate_target_mean = image.new_tensor(0.0)
        dynamic_gate_pred_mean = image.new_tensor(0.0)

    # Supervise the final Quality/Align fusion gate with the detached MOS error
    # of both experts. This is separate from the Align global/patch gate above
    # and is disabled for quality-only datasets.
    fusion_gate_weight = float(tcfg.get("lambda_fusion_gate", 0.0))
    fusion_gate_active = (
        fusion_gate_weight > 0
        and not bool(tcfg.get("quality_only", False))
        and fmask.sum() > 0
        and "fusion_gate_logit" in out
        and bool(getattr(_unwrap_model(model), "use_fusion_gate", False))
        and (epoch is None or int(epoch) >= int(tcfg.get("fusion_gate_start_epoch", 0)))
    )
    if fusion_gate_active:
        fusion_gate_target, fusion_gate_oracle = make_dynamic_fusion_gate_target(
            out["s_quality"],
            out["s_align"],
            final_score,
            tau=float(tcfg.get("fusion_gate_tau", 0.05)),
        )
        gate_mask = fmask.view(-1)
        gate_bce = F.binary_cross_entropy_with_logits(
            out["fusion_gate_logit"].view(-1).float(),
            fusion_gate_target.float(),
            reduction="none",
        )
        l_fusion_gate = (gate_bce * gate_mask).sum() / (gate_mask.sum() + 1e-6)
        fusion_gate_target_mean = (fusion_gate_target * gate_mask).sum() / (gate_mask.sum() + 1e-6)
        fusion_gate_pred_mean = (
            torch.sigmoid(out["fusion_gate_logit"].view(-1)) * gate_mask
        ).sum() / (gate_mask.sum() + 1e-6)
        fusion_gate_oracle_mean = (fusion_gate_oracle * gate_mask).sum() / (gate_mask.sum() + 1e-6)
    else:
        l_fusion_gate = image.new_tensor(0.0)
        fusion_gate_target_mean = image.new_tensor(0.0)
        fusion_gate_pred_mean = image.new_tensor(0.0)
        fusion_gate_oracle_mean = image.new_tensor(0.0)

    # Keep the original total objective exactly decomposable into two task
    # objectives. A legacy Final auxiliary (when explicitly enabled) is split
    # equally so Quality+Align still equals the old total loss. In the current
    # two-output protocol all Final terms are zero.
    lambda_quality_adapter_aux = float(
        tcfg.get(
            "lambda_quality_adapter_aux",
            cfg.get("model", {}).get("quality_adapter", {}).get(
                "auxiliary_loss_weight", 0.0
            ),
        )
    )
    lambda_final = (
        float(tcfg.get("lambda_final_aux", 0.0))
        if final_auxiliary_enabled
        else float(tcfg.get("lambda_final", 1.0))
    )
    lambda_rank_final = (
        0.0 if final_auxiliary_enabled else float(tcfg.get("lambda_rank_final", 0.0))
    )
    lambda_srocc_final = (
        0.0 if final_auxiliary_enabled else float(tcfg.get("lambda_srocc_final", 0.0))
    )

    loss_quality_task = (
        float(tcfg.get("lambda_quality", 1.0)) * l_quality
        + float(tcfg.get("lambda_quality_text_ce", 0.0)) * quality_text_ce
        + float(tcfg.get("lambda_quality_text_anchor", 0.0)) * quality_text_anchor
        + float(tcfg.get("lambda_quality_distribution_ce", 0.0)) * quality_distribution_ce
        + lambda_quality_adapter_aux * l_quality_adapter_aux
        + 0.5 * lambda_final * l_final
        + float(tcfg.get("lambda_rank_quality", 0.0)) * l_rank_quality
        + 0.5 * lambda_rank_final * l_rank_final
        + float(tcfg.get("lambda_srocc_quality", 0.0)) * l_srocc_quality
        + 0.5 * lambda_srocc_final * l_srocc_final
        + float(tcfg.get("lambda_pert", 0.02)) * l_pert
        + 0.5 * fusion_gate_weight * l_fusion_gate
    )
    loss_align_task = (
        float(tcfg.get("lambda_align", 1.0)) * l_align
        + float(tcfg.get("lambda_align_text_ce", 0.0)) * align_text_ce
        + float(tcfg.get("lambda_align_text_anchor", 0.0)) * align_text_anchor
        + 0.5 * lambda_final * l_final
        + float(tcfg.get("lambda_align_aux", 0.3)) * (l_align_g + l_align_p)
        + float(tcfg.get("lambda_rank_align", 0.0)) * l_rank_align
        + 0.5 * lambda_rank_final * l_rank_final
        + float(tcfg.get("lambda_srocc_align", 0.0)) * l_srocc_align
        + 0.5 * lambda_srocc_final * l_srocc_final
        + float(tcfg.get("lambda_neg", 0.05)) * l_neg
        + float(tcfg.get("lambda_hard_neg", 0.0)) * l_hard_neg
        + float(tcfg.get("lambda_align_rank", 0.0)) * l_align_prompt_rank
        + float(tcfg.get("lambda_align_gate_reg", 0.0)) * l_align_gate_reg
        + dynamic_gate_weight * l_dynamic_align_gate
        + 0.5 * fusion_gate_weight * l_fusion_gate
    )
    loss = loss_quality_task + loss_align_task
    losses = {
        "loss": loss,
        "objective_quality_mse": float(tcfg.get("lambda_quality", 1.0)) * l_quality,
        "objective_quality_text": float(tcfg.get("lambda_quality_text_ce", 0.0)) * quality_text_ce,
        "loss_quality_task": loss_quality_task,
        "loss_align_task": loss_align_task,
        "quality": l_quality.detach(),
        "weighted_quality_mse": (float(tcfg.get("lambda_quality", 1.0)) * l_quality).detach(),
        "weighted_quality_text": (float(tcfg.get("lambda_quality_text_ce", 0.0)) * quality_text_ce).detach(),
        "weighted_quality_rank": (float(tcfg.get("lambda_rank_quality", 0.0)) * l_rank_quality).detach(),
        "weighted_quality_srocc": (float(tcfg.get("lambda_srocc_quality", 0.0)) * l_srocc_quality).detach(),
        "weighted_align_patch": (float(tcfg.get("lambda_align_aux", 0.3)) * l_align_p).detach(),
        "weighted_legacy_negative": (float(tcfg.get("lambda_neg", 0.05)) * l_neg).detach(),
        "quality_text_ce": quality_text_ce.detach(),
        "quality_text_mae": quality_text_mae.detach(),
        "quality_text_anchor": quality_text_anchor.detach(),
        "quality_distribution_ce": quality_distribution_ce.detach(),
        "quality_distribution_mae": quality_distribution_mae.detach(),
        "quality_adapter_aux": l_quality_adapter_aux.detach(),
        "align": l_align.detach(),
        "align_text_ce": align_text_ce.detach(),
        "align_text_mae": align_text_mae.detach(),
        "align_text_anchor": align_text_anchor.detach(),
        "final": l_final.detach(),
        "align_global": l_align_g.detach(),
        "align_patch": l_align_p.detach(),
        "rank_quality": l_rank_quality.detach(),
        "rank_align": l_rank_align.detach(),
        "rank_final": l_rank_final.detach(),
        "srocc_quality": l_srocc_quality.detach(),
        "srocc_align": l_srocc_align.detach(),
        "srocc_final": l_srocc_final.detach(),
        "pert": l_pert.detach(),
        "neg": l_neg.detach(),
        "hard_neg": l_hard_neg.detach(),
        "align_prompt_rank": l_align_prompt_rank.detach(),
        "align_gate_reg": l_align_gate_reg.detach(),
        "dynamic_align_gate": l_dynamic_align_gate.detach(),
        "dynamic_gate_target": dynamic_gate_target_mean.detach(),
        "dynamic_gate_pred": dynamic_gate_pred_mean.detach(),
        "fusion_gate_loss": l_fusion_gate.detach(),
        "fusion_gate_target": fusion_gate_target_mean.detach(),
        "fusion_gate_pred": fusion_gate_pred_mean.detach(),
        "fusion_gate_oracle": fusion_gate_oracle_mean.detach(),
        "fusion_w_img": out["fusion_w_img"].detach().mean(),
        "fusion_w_img_std": out["fusion_w_img"].detach().std(unbiased=False),
        "quality_adapter_gate": out["quality_adapter_gate"].detach(),
        "align_gate": out["align_gate"].detach().mean(),
    }
    return out, losses


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch, prompt_bank=None):
    model.train()
    enforce_progressive_frozen_eval(model, cfg, epoch)
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    keys = [
        "loss", "quality", "quality_text_ce", "quality_text_mae", "quality_text_anchor",
        "weighted_quality_mse", "weighted_quality_text", "weighted_quality_rank", "weighted_quality_srocc",
        "weighted_align_patch", "weighted_legacy_negative",
        "quality_distribution_ce", "quality_distribution_mae", "quality_adapter_aux",
        "align", "align_text_ce", "align_text_mae", "align_text_anchor", "final", "align_global", "align_patch",
        "rank_quality", "rank_align", "rank_final",
        "srocc_quality", "srocc_align", "srocc_final",
        "pert", "neg", "hard_neg", "align_prompt_rank", "align_gate_reg",
        "dynamic_align_gate", "dynamic_gate_target", "dynamic_gate_pred",
        "fusion_gate_loss", "fusion_gate_target", "fusion_gate_pred", "fusion_gate_oracle",
        "fusion_w_img", "fusion_w_img_std", "quality_adapter_gate", "align_gate",
    ]
    meters = {k: AverageMeter() for k in keys}
    pcfg = cfg.get("train", {}).get("pcgrad", {})
    pcgrad_enabled = bool(pcfg.get("enabled", False))
    # Old snapshots recorded only enabled=true. Keep them reproducible rather
    # than silently interpreting them as the new algorithm.
    pcgrad_mode = str(pcfg.get("mode", "per_parameter"))
    if pcgrad_enabled and pcgrad_mode not in {"global_shared", "per_parameter"}:
        raise ValueError(f"Unknown downstream PCGrad mode: {pcgrad_mode!r}")
    global_pcgrad = pcgrad_enabled and pcgrad_mode == "global_shared"
    projection_strength = float(pcfg.get("projection_strength", 1.0))
    if global_pcgrad and (
        not math.isfinite(projection_strength) or not 0.0 <= projection_strength <= 1.0
    ):
        raise ValueError("PCGrad projection_strength must be finite and in [0, 1].")
    pcgrad_meters = {k: AverageMeter() for k in PCGRAD_DIAGNOSTIC_KEYS}
    pcgrad_steps = pcgrad_active_steps = pcgrad_conflict_steps = pcgrad_nonfinite_steps = 0
    all_f, all_ft, all_q, all_qt, all_a, all_at = [], [], [], [], [], []
    all_ag, all_agt, all_ap, all_apt = [], [], [], []
    all_qgs, all_qgst, all_qga, all_qgat = [], [], [], []
    gradient_audit = {}
    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            out, losses = compute_losses(model, batch, device, cfg, prompt_bank=prompt_bank, epoch=epoch)
            loss = losses["loss"]
        if step == 0 and bool(cfg["train"].get("audit_loss_gradients", False)):
            gradient_audit = audit_fusion_gradients(_unwrap_model(model), {
                "quality_mse": losses["objective_quality_mse"],
                "quality_text": losses["objective_quality_text"],
                "quality_task": losses["loss_quality_task"],
                "align_task": losses["loss_align_task"],
            })
            print(f"[INFO] Fusion gradient audit: {gradient_audit}")
        if global_pcgrad:
            quality_mask = batch.get("quality_mask")
            align_mask = batch.get("align_mask")
            quality_active = quality_mask is None or bool((quality_mask > 0).any().item())
            align_active = (
                not bool(cfg["train"].get("quality_only", False))
                and (align_mask is None or bool((align_mask > 0).any().item()))
            )
            stats = global_shared_pcgrad_backward(
                losses["loss_quality_task"], losses["loss_align_task"],
                model, optimizer, scaler,
                quality_active=quality_active, align_active=align_active,
                projection_strength=projection_strength,
            )
            pcgrad_steps += 1
            pcgrad_active_steps += int(stats["active"])
            pcgrad_conflict_steps += int(stats["conflict"])
            pcgrad_nonfinite_steps += int(stats["nonfinite"])
            if stats["active"]:
                for key, value in stats.items():
                    pcgrad_meters[key].update(value)
        elif pcgrad_enabled:
            _pcgrad_backward(
                losses["loss_quality_task"],
                losses["loss_align_task"],
                model,
                optimizer,
                scaler,
            )
        else:
            scaler.scale(loss).backward()
        if float(cfg["train"].get("grad_clip", 1.0)) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()

        bs = batch["image"].size(0)
        for k in keys:
            meters[k].update(float(losses[k].item()), bs)
        final_mask = (
            torch.zeros_like(batch["score"])
            if bool(cfg.get("train", {}).get("two_output_only", False))
            else batch.get("final_mask", torch.ones_like(batch["score"]))
        )
        extend_valid(all_f, all_ft, out["s_final"], batch["score"], final_mask)
        extend_valid(all_q, all_qt, out["s_quality"], batch["quality_score"], batch.get("quality_mask", torch.ones_like(batch["quality_score"])))
        extend_valid(all_a, all_at, out["s_align"], batch["align_score"], batch.get("align_mask", torch.ones_like(batch["align_score"])))
        extend_valid(all_ag, all_agt, out["s_align_global"], batch["align_score"], batch.get("align_mask", torch.ones_like(batch["align_score"])))
        extend_valid(all_qgs, all_qgst, out["s_quality_global_shared"], batch["quality_score"], batch.get("quality_mask", torch.ones_like(batch["quality_score"])))
        extend_valid(all_qga, all_qgat, out["s_quality_global_adapter"], batch["quality_score"], batch.get("quality_mask", torch.ones_like(batch["quality_score"])))
        extend_valid(all_ap, all_apt, out["s_align_patch"], batch["align_score"], batch.get("align_mask", torch.ones_like(batch["align_score"])))

        if step % int(cfg["train"].get("log_interval", 20)) == 0:
            print(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"loss={meters['loss'].avg:.4f} final={meters['final'].avg:.4f} "
                f"quality={meters['quality'].avg:.4f} align={meters['align'].avg:.4f} "
                f"rankF={meters['rank_final'].avg:.4f} rankQ={meters['rank_quality'].avg:.4f} "
                f"rankA={meters['rank_align'].avg:.4f} "
                f"sroccF={meters['srocc_final'].avg:.4f} sroccA={meters['srocc_align'].avg:.4f} "
                f"qAux={meters['quality_adapter_aux'].avg:.4f} "
                f"gQ={meters['quality_adapter_gate'].avg:.4f} "
                f"promptRankA={meters['align_prompt_rank'].avg:.4f} gateRegA={meters['align_gate_reg'].avg:.4f} "
                f"dynGateA={meters['dynamic_align_gate'].avg:.4f} "
                f"gTarget={meters['dynamic_gate_target'].avg:.4f} "
                f"fusionGate={meters['fusion_gate_loss'].avg:.4f} "
                f"fTarget={meters['fusion_gate_target'].avg:.4f} "
                f"fPred={meters['fusion_gate_pred'].avg:.4f} "
                f"w_img={meters['fusion_w_img'].avg:.4f} "
                f"w_std={meters['fusion_w_img_std'].avg:.4f}"
            )
            if global_pcgrad:
                print(
                    f"  PCGrad[global_shared eta={projection_strength:g}] "
                    f"active={pcgrad_active_steps}/{pcgrad_steps} "
                    f"conflict_rate={pcgrad_conflict_steps / max(1, pcgrad_active_steps):.3f} "
                    f"cos={pcgrad_meters['cosine'].avg:.3f} "
                    f"|gQ|={pcgrad_meters['quality_grad_norm'].avg:.3e} "
                    f"|gA|={pcgrad_meters['align_grad_norm'].avg:.3e} "
                    f"update_ratio={pcgrad_meters['update_norm_ratio'].avg:.3f} "
                    f"nonfinite={pcgrad_nonfinite_steps}"
                )

    two_output_only = bool(cfg.get("train", {}).get("two_output_only", False))
    m = {} if two_output_only else compute_regression_metrics(all_ft, all_f)
    if two_output_only:
        m["final_output_disabled"] = 1.0
    qm = compute_regression_metrics(all_qt, all_q)
    am = compute_regression_metrics(all_at, all_a)
    agm = compute_regression_metrics(all_agt, all_ag)
    apm = compute_regression_metrics(all_apt, all_ap)
    qgsm = compute_regression_metrics(all_qgst, all_qgs)
    qgam = compute_regression_metrics(all_qgat, all_qga)
    m["train_loss"] = meters["loss"].avg
    for k, v in qm.items():
        m[f"train_quality_{k}"] = v
    for k, v in am.items():
        m[f"train_align_{k}"] = v
    for k, v in agm.items():
        m[f"train_align_global_{k}"] = v
    for k, v in apm.items():
        m[f"train_align_patch_{k}"] = v
    for k, v in qgsm.items():
        m[f"train_quality_global_shared_{k}"] = v
    for k, v in qgam.items():
        m[f"train_quality_global_adapter_{k}"] = v
    for k in keys:
        m[f"train_{k}"] = meters[k].avg
    if global_pcgrad:
        # main() adds the train_ prefix when saving epoch metrics_json.
        # Per-step means use ONLY finite, nonzero two-task shared gradients.
        m["pcgrad_projection_strength"] = projection_strength
        for key in PCGRAD_DIAGNOSTIC_KEYS:
            if key not in {"active", "conflict", "nonfinite"}:
                m[f"pcgrad_{key}"] = pcgrad_meters[key].avg
        m["pcgrad_steps"] = pcgrad_steps
        m["pcgrad_active_steps"] = pcgrad_active_steps
        m["pcgrad_conflict_steps"] = pcgrad_conflict_steps
        m["pcgrad_nonfinite_steps"] = pcgrad_nonfinite_steps
        m["pcgrad_conflict_rate"] = pcgrad_conflict_steps / max(1, pcgrad_active_steps)
        m["pcgrad_active_rate"] = pcgrad_active_steps / max(1, pcgrad_steps)
    m.update(gradient_audit)
    return m


@torch.no_grad()
def evaluate(model, loader, device, cfg, split="val"):
    model.eval()
    quality_global, quality_patch, quality_targets, quality_betas = [], [], [], []
    losses = AverageMeter()
    qloss = AverageMeter()
    aloss = AverageMeter()
    qag = AverageMeter()
    fw = AverageMeter()
    fw_std = AverageMeter()
    ag = AverageMeter()
    all_qgs, all_qgst, all_qga, all_qgat = [], [], [], []
    all_qvisual, all_qvisual_target, all_qtext, all_qtext_target = [], [], [], []
    all_abase, all_abase_target, all_atext, all_atext_target = [], [], [], []
    all_f, all_ft, all_q, all_qt, all_a, all_at = [], [], [], [], [], []
    all_ag, all_agt, all_ap, all_apt = [], [], [], []
    oracle_global_sum = 0.0
    align_valid_count = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        validate_local_crops(batch, cfg)
        local_crops = batch.get("local_crops")
        if local_crops is not None:
            local_crops = local_crops.to(device, non_blocking=True)
        score = batch["score"].to(device, non_blocking=True)
        quality = batch["quality_score"].to(device, non_blocking=True)
        align = batch["align_score"].to(device, non_blocking=True)
        fmask = batch.get("final_mask", torch.ones_like(score)).to(device, non_blocking=True).float()
        qmask = batch.get("quality_mask", torch.ones_like(quality)).to(device, non_blocking=True).float()
        amask = batch.get("align_mask", torch.ones_like(align)).to(device, non_blocking=True).float()
        if bool(cfg.get("train", {}).get("two_output_only", False)):
            fmask = torch.zeros_like(score)
        out = model.forward_single(
            image, prompts=batch["prompt"], local_crops=local_crops
        )
        if bool(cfg.get("train", {}).get("quality_only", False)):
            out["s_final"] = out["s_quality"]
            out["fusion_w_img"] = out["s_quality"].new_tensor(1.0)
        bs = image.size(0)
        q_mse = float(masked_mse(out["s_quality"], quality, qmask).item())
        a_mse = float(masked_mse(out["s_align"], align, amask).item())
        if qmask.sum() > 0:
            qloss.update(q_mse, float(qmask.sum()))
        if amask.sum() > 0:
            aloss.update(a_mse, float(amask.sum()))
        if not bool(cfg.get("train", {}).get("two_output_only", False)):
            losses.update(float(masked_mse(out["s_final"], score, fmask).item()), bs)
        qag.update(float(out["quality_adapter_gate"].item()), bs)
        fw.update(float(out["fusion_w_img"].mean().item()), bs)
        fw_std.update(float(out["fusion_w_img"].std(unbiased=False).item()), bs)
        ag.update(float(out["align_gate"].mean().item()), bs)
        if not bool(cfg.get("train", {}).get("two_output_only", False)):
            extend_valid(all_f, all_ft, out["s_final"], score, fmask)
        valid = qmask.view(-1) > 0
        quality_global.extend(out["s_quality_global"].view(-1)[valid].detach().cpu().tolist())
        if out["s_quality_patch"] is not None:
            quality_patch.extend(out["s_quality_patch"].view(-1)[valid].detach().cpu().tolist())
        quality_targets.extend(quality.view(-1)[valid].detach().cpu().tolist())
        quality_betas.extend(out["quality_mix"].expand_as(quality.view(-1))[valid].cpu().tolist())
        extend_valid(all_q, all_qt, out["s_quality"], quality, qmask)
        if "quality_text_score" in out:
            extend_valid(all_qvisual, all_qvisual_target, out["s_quality_visual"], quality, qmask)
            extend_valid(all_qtext, all_qtext_target, out["quality_text_score"], quality, qmask)
        extend_valid(all_qgs, all_qgst, out["s_quality_global_shared"], quality, qmask)
        extend_valid(all_qga, all_qgat, out["s_quality_global_adapter"], quality, qmask)
        extend_valid(all_a, all_at, out["s_align"], align, amask)
        if "align_text_score" in out:
            extend_valid(all_abase, all_abase_target, out["s_align_base"], align, amask)
            extend_valid(all_atext, all_atext_target, out["align_text_score"], align, amask)
        extend_valid(all_ag, all_agt, out["s_align_global"], align, amask)
        extend_valid(all_ap, all_apt, out["s_align_patch"], align, amask)
        _, oracle_global = make_dynamic_align_gate_target(
            out["s_align_global"],
            out["s_align_patch"],
            align,
            tau=float(cfg.get("train", {}).get("dynamic_align_gate_tau", 0.05)),
        )
        oracle_global_sum += float((oracle_global * amask.view(-1)).sum().item())
        align_valid_count += float(amask.sum().item())

    two_output_only = bool(cfg.get("train", {}).get("two_output_only", False))
    m = {} if two_output_only else compute_regression_metrics(all_ft, all_f)
    if two_output_only:
        m["final_output_disabled"] = 1.0
    m[f"{split}_quality_local_enabled"] = float(not getattr(_unwrap_model(model), "quality_no_local_branch", False))
    m[f"{split}_quality_global_uses_g"] = float(getattr(_unwrap_model(model), "quality_global_feature", "z") == "g")
    for branch, predictions in [("global", quality_global), ("patch", quality_patch)]:
        if branch == "patch" and not m[f"{split}_quality_local_enabled"]:
            continue
        for key, value in compute_regression_metrics(quality_targets, predictions).items():
            m[f"{split}_quality_{branch}_{key}"] = value
    if quality_betas:
        m[f"{split}_quality_mix_mean"] = float(np.mean(quality_betas))
        m[f"{split}_quality_mix_std"] = float(np.std(quality_betas))
        m[f"{split}_quality_mix_min"] = float(np.min(quality_betas))
        m[f"{split}_quality_mix_max"] = float(np.max(quality_betas))
    if len(quality_targets) >= 2 and len(quality_patch) == len(quality_targets):
        eg = np.asarray(quality_global) - np.asarray(quality_targets)
        ep = np.asarray(quality_patch) - np.asarray(quality_targets)
        # Undefined correlation is omitted rather than reported as zero.
        if np.std(eg) > 0 and np.std(ep) > 0:
            m[f"{split}_quality_branch_error_PLCC"] = float(np.corrcoef(eg, ep)[0, 1])
    qm = compute_regression_metrics(all_qt, all_q)
    am = compute_regression_metrics(all_at, all_a)
    agm = compute_regression_metrics(all_agt, all_ag)
    apm = compute_regression_metrics(all_apt, all_ap)
    qgsm = compute_regression_metrics(all_qgst, all_qgs)
    qgam = compute_regression_metrics(all_qgat, all_qga)
    if all_qtext:
        for key, value in compute_regression_metrics(all_qtext_target, all_qtext).items():
            m[f"{split}_quality_text_{key}"] = value
        for key, value in compute_regression_metrics(all_qvisual_target, all_qvisual).items():
            m[f"{split}_quality_visual_{key}"] = value
    if all_atext:
        for key, value in compute_regression_metrics(all_atext_target, all_atext).items():
            m[f"{split}_align_text_{key}"] = value
        for key, value in compute_regression_metrics(all_abase_target, all_abase).items():
            m[f"{split}_align_base_{key}"] = value
    m[f"{split}_reg_loss"] = (
        float(cfg["train"].get("lambda_quality", 1.0)) * qloss.avg
        + float(cfg["train"].get("lambda_align", 1.0)) * aloss.avg
        if two_output_only else losses.avg
    )
    m[f"{split}_quality_adapter_gate"] = qag.avg
    m[f"{split}_quality_loss"] = qloss.avg
    m[f"{split}_align_loss"] = aloss.avg
    m[f"{split}_fusion_w_img_std"] = fw_std.avg
    m[f"{split}_fusion_w_img"] = fw.avg
    m[f"{split}_align_gate"] = ag.avg
    m[f"{split}_align_oracle_global_rate"] = oracle_global_sum / max(align_valid_count, 1e-6)
    for k, v in qm.items():
        m[f"{split}_quality_{k}"] = v
    for k, v in am.items():
        m[f"{split}_align_{k}"] = v
    for k, v in qgsm.items():
        m[f"{split}_quality_global_shared_{k}"] = v
    for k, v in qgam.items():
        m[f"{split}_quality_global_adapter_{k}"] = v
    for k, v in agm.items():
        m[f"{split}_align_global_{k}"] = v
    for k, v in apm.items():
        m[f"{split}_align_patch_{k}"] = v
    m.update({f"{split}_{key}": value for key, value in local_stage_diagnostics(model).items()})
    return m


def format_metrics(prefix, metrics):
    split = prefix.lower().replace("best-", "")
    parts = []
    loss_key = f"{split}_reg_loss"
    if loss_key in metrics:
        parts.append(f"{prefix}Loss={metrics[loss_key]:.4f}")
    elif "train_loss" in metrics:
        parts.append(f"{prefix}Loss={metrics['train_loss']:.4f}")
    for k in ["PLCC", "SROCC", "KROCC", "RMSE", "MAE"]:
        if k in metrics and metrics.get("N", 0) >= 2:
            parts.append(f"{k}={metrics[k]:.4f}")
    if metrics.get(f"{split}_quality_N", 0) >= 2:
        parts.append(f"Quality[PLCC={metrics[f'{split}_quality_PLCC']:.4f},SROCC={metrics[f'{split}_quality_SROCC']:.4f}]")
    if metrics.get(f"{split}_quality_global_shared_N", 0) >= 2:
        parts.append(
            f"QualityGlobalSub[Shared:SROCC="
            f"{metrics[f'{split}_quality_global_shared_SROCC']:.4f};"
            f"Adapter:SROCC="
            f"{metrics[f'{split}_quality_global_adapter_SROCC']:.4f}]"
        )
    if metrics.get(f"{split}_align_N", 0) >= 2:
        parts.append(f"Align[PLCC={metrics[f'{split}_align_PLCC']:.4f},SROCC={metrics[f'{split}_align_SROCC']:.4f}]")
    if metrics.get(f"{split}_align_global_N", 0) >= 2:
        parts.append(
            f"AlignSub[G:PLCC={metrics[f'{split}_align_global_PLCC']:.4f},"
            f"SROCC={metrics[f'{split}_align_global_SROCC']:.4f};"
            f"P:PLCC={metrics[f'{split}_align_patch_PLCC']:.4f},"
            f"SROCC={metrics[f'{split}_align_patch_SROCC']:.4f}]"
        )
    if f"{split}_quality_adapter_gate" in metrics:
        parts.append(f"g_quality_adapter={metrics[f'{split}_quality_adapter_gate']:.4f}")
    if f"{split}_fusion_w_img_std" in metrics:
        parts.append(f"w_std={metrics[f'{split}_fusion_w_img_std']:.4f}")
    if f"{split}_fusion_w_img" in metrics:
        parts.append(f"w_img={metrics[f'{split}_fusion_w_img']:.4f}")
    if f"{split}_align_gate" in metrics:
        parts.append(f"g_align={metrics[f'{split}_align_gate']:.4f}")
    if f"{split}_align_oracle_global_rate" in metrics:
        parts.append(f"oracleG={metrics[f'{split}_align_oracle_global_rate']:.4f}")
    return " | ".join(parts)




def get_selection_value(metrics: Dict, cfg: Dict) -> tuple[float, bool, str]:
    """Return value used for best-checkpoint selection.

    Default is the old behavior: minimize val_reg_loss. For SROCC-oriented
    experiments, set for example:
      selection_metric: val_SROCC
      selection_mode: max
    or:
      selection_metric: val_align_SROCC
      selection_mode: max
    """
    tcfg = cfg.get("train", {})
    key = str(tcfg.get("selection_metric", "val_reg_loss"))
    mode = str(tcfg.get("selection_mode", "min")).lower()
    if key in {"val_component_hmean_SROCC", "val_quality_align_hmean_SROCC"}:
        # No synthetic Final-MOS checkpoint selection. Combine the two
        # independently labelled tasks only for model selection.
        if "val_quality_SROCC" not in metrics:
            raise ValueError("Joint selection requires val_quality_SROCC.")
        q = float(metrics["val_quality_SROCC"])
        a = float(metrics.get("val_align_SROCC", 0.0))
        if not math.isfinite(q) or not math.isfinite(a):
            raise ValueError("Joint selection requires finite Quality/Align SROCC.")
        if float(metrics.get("val_align_N", 0.0)) >= 2.0 and "val_align_SROCC" not in metrics:
            raise ValueError("Joint selection has Align labels but no val_align_SROCC.")
        if float(metrics.get("val_align_N", 0.0)) >= 2.0:
            # Map correlations from [-1, 1] to [0, 1] before applying the
            # harmonic mean, so early negative correlations remain well-defined.
            q01 = min(1.0, max(0.0, 0.5 * (q + 1.0)))
            a01 = min(1.0, max(0.0, 0.5 * (a + 1.0)))
            h01 = 2.0 * q01 * a01 / (q01 + a01 + 1e-8)
            value = 2.0 * h01 - 1.0
        else:
            value = q
        metrics[key] = value
        return value, True, key
    if key not in metrics:
        # evaluate(split="val") returns final-score metrics as plain PLCC/SROCC,
        # while branch metrics are already prefixed, e.g. val_align_SROCC.
        # Accept the common config spelling val_SROCC by mapping it to SROCC.
        stripped_key = key[4:] if key.startswith("val_") else ""
        if stripped_key and stripped_key in metrics:
            key = stripped_key
        else:
            raise ValueError(f"selection_metric={key!r} is absent from validation metrics; refusing a silent fallback.")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"selection_metric={key!r} is nonfinite: {value}")
    maximize = mode == "max"
    return value, maximize, key


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.clip_large:
        configure_clip_large(cfg)
    if args.seed is not None:
        # Preserve the YAML seed as the data-split seed while varying only
        # initialization, shuffling and augmentation across repeated runs.
        cfg.setdefault("split_seed", int(cfg.get("seed", 42)))
        cfg["seed"] = int(args.seed)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.results_csv:
        cfg["results_csv"] = str(args.results_csv)
    if args.pretrained_checkpoint:
        cfg.setdefault("train", {})["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
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
    if args.align_text_blend is not None:
        align_blend = float(args.align_text_blend)
        if not math.isfinite(align_blend) or not 0.0 <= align_blend <= 1.0:
            raise ValueError("--align-text-blend must be finite and in [0, 1].")
        if args.align_text_lambda is not None and (not math.isfinite(args.align_text_lambda) or args.align_text_lambda < 0):
            raise ValueError("--align-text-lambda must be finite and nonnegative.")
        if args.align_text_anchor_lambda is not None and (not math.isfinite(args.align_text_anchor_lambda) or args.align_text_anchor_lambda < 0):
            raise ValueError("--align-text-anchor-lambda must be finite and nonnegative.")
        align_bank_cfg = cfg.setdefault("model", {}).setdefault("align_text_bank", {})
        align_bank_cfg["enabled"] = True
        align_bank_cfg["score_blend"] = align_blend
        align_bank_cfg.setdefault("temperature", 0.2)
        align_bank_cfg.setdefault("class_context_tokens", max(1, int(args.text_context_tokens or 4)))
        align_bank_cfg.setdefault("centers", [0.0, 0.25, 0.5, 0.75, 1.0])
        cfg.setdefault("train", {})["lambda_align_text_ce"] = (
            0.005 if args.align_text_lambda is None else float(args.align_text_lambda)
        )
        cfg.setdefault("train", {})["lambda_align_text_anchor"] = (
            0.0 if args.align_text_anchor_lambda is None else float(args.align_text_anchor_lambda)
        )
        print(f"[INFO] Five-word image-Prompt matching bank enabled; score_blend={align_blend:.2f}.")
    if args.quality_text_blend is not None:
        blend = float(args.quality_text_blend)
        if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise ValueError("--quality-text-blend must be finite and in [0, 1].")
        if args.disable_quality_adapter:
            raise ValueError("--quality-text-blend requires an enabled Quality Adapter.")
        if args.quality_text_lambda is not None and (not math.isfinite(args.quality_text_lambda) or args.quality_text_lambda < 0):
            raise ValueError("--quality-text-lambda must be finite and nonnegative.")
        if args.quality_text_anchor_lambda is not None and (not math.isfinite(args.quality_text_anchor_lambda) or args.quality_text_anchor_lambda < 0):
            raise ValueError("--quality-text-anchor-lambda must be finite and nonnegative.")
        bank_cfg = cfg.setdefault("model", {}).setdefault("quality_text_bank", {})
        bank_cfg["enabled"] = True
        bank_cfg["score_blend"] = blend
        bank_cfg.setdefault("temperature", 0.2)
        bank_cfg.setdefault("class_context_tokens", max(1, int(args.text_context_tokens or 4)))
        bank_cfg.setdefault("centers", [0.375, 0.54166667, 0.66666669, 0.79166666, 0.95833334])
        cfg.setdefault("train", {})["lambda_quality_text_ce"] = (
            0.005 if args.quality_text_lambda is None else float(args.quality_text_lambda)
        )
        cfg.setdefault("train", {})["lambda_quality_text_anchor"] = (
            0.0 if args.quality_text_anchor_lambda is None else float(args.quality_text_anchor_lambda)
        )
        print(f"[INFO] Five-word quality bank enabled; score_blend={blend:.2f}.")
    if args.disable_perturbation:
        dataset_cfg = cfg.setdefault("dataset", {})
        dataset_cfg["use_perturb"] = False
        dataset_cfg["perturbation_strengths"] = []
        print("[INFO] Synthetic downstream perturbation disabled.")
    if args.quality_distribution:
        qdcfg = cfg.setdefault("model", {}).setdefault("quality_distribution", {})
        qdcfg["enabled"] = True
        qdcfg.setdefault("centers", [0.0, 0.25, 0.5, 0.75, 1.0])
        cfg.setdefault("train", {})["lambda_quality_distribution_ce"] = 0.005
        print("[INFO] Optional five-bin Quality distribution head enabled.")
    if args.quality_highres_crops:
        crop_cfg = cfg.setdefault("dataset", {}).setdefault("local_multicrop", {})
        crop_cfg.update({"enabled": True, "input_size": 336, "train_num_crops": 5, "eval_num_crops": 5})
        print("[INFO] 336px local crops enabled for Quality only; Align remains base-resolution.")
    if args.quality_attention_pooling:
        qlcfg = cfg.setdefault("model", {}).setdefault("quality_local_attention", {})
        qlcfg["enabled"] = True
        qlcfg.setdefault("num_heads", cfg.get("model", {}).get("fusion", {}).get("num_heads", 4))
        qlcfg.setdefault("mix_init_logit", -1.1)
        print("[INFO] Quality global-query local attention pooling enabled.")
    if args.align_prompt_cross_attention:
        atcfg = cfg.setdefault("model", {}).setdefault("align_prompt_cross_attention", {})
        atcfg["enabled"] = True
        atcfg.setdefault("num_heads", cfg.get("model", {}).get("fusion", {}).get("num_heads", 4))
        atcfg.setdefault("residual_scale", 1.0)
        print("[INFO] Align Prompt-token cross-attention enabled.")
    if args.quality_global_feature is not None:
        cfg.setdefault("model", {})["quality_global_feature"] = args.quality_global_feature
        print(f"[INFO] Quality global feature={args.quality_global_feature}; Align continues to use z.")
    if args.quality_no_local_branch:
        cfg.setdefault("model", {})["quality_no_local_branch"] = True
        print("[INFO] Strict Quality global-only: local Quality forward disabled; Fusion/Align retained.")
    if args.quality_global_only or args.quality_no_local_branch:
        cfg.setdefault("model", {})["quality_global_only"] = True
        if float(cfg["model"].get("quality_text_bank", {}).get("score_blend", 0.0)) != 0.0:
            raise ValueError("--quality-global-only requires --quality-text-blend 0.")
        print("[INFO] Global-only Quality scoring enabled: patch score fusion bypassed; ResNet/Fusion and Align retained.")
    configure_local_feature_stage(cfg, args)
    configure_downstream_controls(cfg, args)
    configure_quality_adapter_experiment(cfg, args)
    if args.disable_quality_adapter:
        # Keep this as a command-line override rather than a second YAML copy.
        # The effective value is written to config_snapshot.yaml, so each
        # ablation remains auditable and cannot silently reuse the enabled path.
        cfg.setdefault("model", {}).setdefault("quality_adapter", {})["enabled"] = False
        print("[INFO] Persistent quality adapter disabled for this downstream ablation.")
    if args.quality_sample_gate:
        cfg.setdefault("model", {}).setdefault("quality_sample_gate", {})["enabled"] = True
    configure_visual_backbone_lr(cfg, args)
    configure_quality_adapter_lr(cfg, args)
    configure_downstream_pcgrad(cfg, args)
    if args.mirror_augmentation or args.mirror_augmentation_probability is not None:
        dataset_cfg = cfg.setdefault("dataset", {})
        mirror_cfg = dict(dataset_cfg.get("mirror_augmentation", {}))
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
        dataset_cfg["mirror_augmentation"] = mirror_cfg
    compare_selection = configure_selection_comparison(cfg, args)
    validate_progressive_finetune_config(cfg)
    set_seed(int(cfg.get("seed", 42)))
    print(
        f"[INFO] Seeds: train={int(cfg.get('seed', 42))}, "
        f"split={int(cfg.get('split_seed', cfg.get('seed', 42)))}"
    )
    outdir = cfg.get("output_dir", "./outputs/cm_iqa")
    ensure_dir(outdir)

    run_id = cfg.get("run_id", make_run_id("iqa", cfg))
    if args.output_dir:
        # Repeated-seed commands normally override only output_dir. Use that
        # unique directory name as the run id instead of retaining seed42 from
        # the source YAML.
        run_id = os.path.basename(os.path.normpath(outdir))
        cfg["run_id"] = run_id
    config_snapshot = dump_config_snapshot(cfg, outdir, config_path=args.config)
    logger = CSVExperimentLogger(
        csv_path=resolve_results_csv(cfg),
        run_id=run_id,
        stage="iqa",
        cfg=cfg,
        output_dir=outdir,
        config_path=config_snapshot,
    )
    logger.log(event="run_start", epoch=0, metrics={"config_file": args.config})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_path = cfg["train"].get("pretrained_checkpoint", "")
    if pretrain_path and not os.path.isfile(pretrain_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_path}")

    fingerprint = checkpoint_fingerprint(pretrain_path) if compare_selection else None
    metadata = None
    if compare_selection:
        metadata = collect_run_metadata(fingerprint, cfg)
        metadata_path = write_run_metadata(outdir, metadata)
        print("[INFO] Comparing validation selection on one 60-epoch trajectory; early stopping disabled.")
    train_loader, val_loader, test_loader = unpack_loaders(build_dataloaders(cfg))
    if compare_selection and test_loader is None:
        raise ValueError("Selection comparison requires a held-out test loader for final reporting.")
    model = build_iqa_model(cfg, pretrain_path).to(device)
    if pretrain_path:
        load_pretrained_backbone(
            model,
            pretrain_path,
            load_scope=cfg["train"].get("pretrained_load_scope", "auto"),
            skip_quality_adapter=bool(cfg["train"].get("skip_pretrained_quality_adapter", False)),
            reinit_local_projection=bool(cfg["train"].get("reinit_local_projection", False)),
        )

    if compare_selection:
        # Hash again after both model construction and checkpoint loading. A
        # replaced input must not silently acquire an unrelated provenance hash.
        if checkpoint_fingerprint(pretrain_path) != fingerprint:
            raise RuntimeError("Input checkpoint changed during model loading; aborting run.")
        metadata["checkpoint_verified_after_load"] = bool(pretrain_path)
        write_run_metadata(outdir, metadata)
        logger.log(event="run_metadata", epoch=0, metrics={"metadata_path": metadata_path, **metadata})

    if args.quality_centers_train_quantiles:
        details = calibrate_quality_centers(model, train_loader.dataset, cfg)
        dump_config_snapshot(cfg, outdir, config_path=args.config)
        logger.log(event="quality_center_calibration", epoch=0, metrics=details)
        print(f"[INFO] Training-only Quality centers: {details['centers']} (N={details['count']})")

    prompt_bank = None
    tcfg = cfg.get("train", {})
    if bool(tcfg.get("use_align_rank", False)) and str(tcfg.get("align_negative_source", "")).lower() == "prompt_bank":
        prompt_bank = build_prompt_bank(
            model,
            train_loader.dataset,
            device,
            batch_size=int(tcfg.get("prompt_bank_encode_batch_size", 256)),
        )

    optimizer = build_optimizer(cfg, model)
    # Capture model-config trainability before progressive stages alter it.
    # The optimizer already contains every parameter that may be unfrozen later.
    base_trainable = {
        id(parameter): bool(parameter.requires_grad)
        for parameter in model.parameters()
    }
    progressive_enabled = bool(
        cfg.get("train", {}).get("progressive_unfreeze", {}).get("enabled", False)
    )
    encoder_start_epoch = int(
        cfg.get("train", {})
        .get("progressive_unfreeze", {})
        .get("encoder_start_epoch", 1)
    )
    scheduler_epochs = int(cfg["train"].get("epochs", 60))
    if progressive_enabled:
        # Keep the original group learning rates throughout frozen stages.
        # Cosine decay starts at the first fully unfrozen epoch.
        scheduler_epochs = max(
            1,
            int(cfg["train"].get("epochs", 60)) - encoder_start_epoch + 1,
        )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scheduler_epochs,
        eta_min=float(cfg.get("scheduler", {}).get("min_lr", 1e-6)),
    )
    scaler = GradScaler(device.type, enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    # Best epoch is selected by validation loss or validation SROCC.  Downstream
    # runs intentionally do not write model checkpoints: keep the selected
    # weights in memory for the final validation/test report and retain only
    # the config snapshot plus the experiment CSV.
    comparison = SelectionComparison() if compare_selection else None
    best = None
    best_epoch = -1
    best_model_state = None
    no_improve = 0
    patience = int(cfg["train"].get("early_stop_patience", 15))
    last_epoch = 0

    previous_finetune_stage = None
    for epoch in range(1, int(cfg["train"].get("epochs", 60)) + 1):
        last_epoch = epoch
        finetune_stage, trainable_counts = apply_progressive_finetune_stage(
            optimizer,
            cfg,
            epoch,
            base_trainable,
        )
        if finetune_stage != previous_finetune_stage:
            if previous_finetune_stage is not None:
                # Give every newly unfrozen stage its own early-stop window.
                # Otherwise an early head-only plateau could stop before encoders train.
                no_improve = 0
            trainable_summary = ", ".join(
                f"{name}={count / 1e6:.2f}M"
                for name, count in trainable_counts.items()
            )
            print(
                f"[INFO] Fine-tune stage -> {finetune_stage} at epoch {epoch}; "
                f"trainable parameters: {trainable_summary}"
            )
            logger.log(
                event="finetune_stage",
                epoch=epoch,
                metrics={
                    "stage": finetune_stage,
                    "trainable_parameters": int(sum(trainable_counts.values())),
                    "trainable_by_group": trainable_counts,
                },
                best_epoch=best_epoch,
                is_best=False,
            )
            previous_finetune_stage = finetune_stage
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch, prompt_bank=prompt_bank)
        va = evaluate(model, val_loader, device, cfg, split="val")
        if not progressive_enabled or epoch >= encoder_start_epoch:
            scheduler.step()

        current_lrs = {str(g.get("name", f"group{i}")): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)}
        print(f"Epoch {epoch}: lr={max(current_lrs.values()):.6e} | {format_metrics('Train', tr)} | {format_metrics('Val', va)}")
        policy_status = None
        if comparison is not None:
            policy_status = comparison.update(model, va, epoch, get_selection_value)
            quality = policy_status["quality"]
            best, best_epoch, is_best = quality["value"], quality["epoch"], quality["is_best"]
            for name, selected in policy_status.items():
                print(f"[Selection:{name}] best epoch={selected['epoch']}, value={selected['value']:.6f}")
        else:
            cur, maximize, sel_key = get_selection_value(va, cfg)
            is_best = (best is None) or (cur > best if maximize else cur < best)
            if is_best:
                best = cur
                best_epoch = epoch
                no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"Selected epoch={epoch}, best_{sel_key}={best:.6f}")
            else:
                no_improve += 1
                print(f"No improvement for {no_improve} epoch(s). Best epoch so far: {best_epoch}, best_{sel_key}={best:.6f}")
        metrics = {
            "lr": max(current_lrs.values()),
            **{f"lr_{k}": v for k, v in current_lrs.items()},
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "best_selection_value": best,
            "selection_metric": "val_quality_SROCC" if compare_selection else cfg.get("train", {}).get("selection_metric", "val_reg_loss"),
            **({"selection_policies": policy_status} if compare_selection else {}),
            "no_improve": no_improve,
            "finetune_stage": finetune_stage,
            "trainable_parameters": int(sum(trainable_counts.values())),
        }
        logger.log(event="epoch", epoch=epoch, metrics=metrics, best_epoch=best_epoch, is_best=is_best)

        if not compare_selection and no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            logger.log(
                event="early_stop",
                epoch=epoch,
                metrics={"best_epoch": best_epoch, "best_selection_value": best, "patience": patience},
                best_epoch=best_epoch,
                is_best=False,
            )
            break

    if comparison is not None:
        def evaluate_policy():
            val = evaluate(model, val_loader, device, cfg, split="val")
            test = evaluate(model, test_loader, device, cfg, split="test") if test_loader is not None else None
            return val, test

        results = comparison.evaluate_selected(model, evaluate_policy)
        for name, result in results.items():
            selected_epoch = result["best_epoch"]
            print(f"[Selection:{name}] epoch={selected_epoch}, {result['selection_metric']}={result['best_selection_value']:.6f}")
            print(format_metrics(f"{name}-Val", result["best_val"]))
            print(format_metrics(f"{name}-Test", result["best_test"]))
            logger.log(event=f"{name}_selected", epoch=selected_epoch,
                       metrics=result, best_epoch=selected_epoch, is_best=True)
        logger.log(event="final", epoch=last_epoch, metrics={
            "last_epoch": last_epoch, "comparison_complete": True,
            "selection_policies": results, "run_metadata_path": metadata_path,
            "best_checkpoint": "", "last_checkpoint": "",
        })
        return

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored selected in-memory model state from epoch {best_epoch}.")

    print("Evaluating the selected in-memory model on validation set...")
    best_val = evaluate(model, val_loader, device, cfg, split="val")
    print(f"Best epoch: {best_epoch}")
    print(format_metrics("Best-Val", best_val))
    logger.log(event="best_val", epoch=best_epoch, metrics=best_val, best_epoch=best_epoch, is_best=True)

    best_test = None
    if test_loader is not None:
        print("Evaluating the selected in-memory model on test set...")
        best_test = evaluate(model, test_loader, device, cfg, split="test")
        print(format_metrics("Best-Test", best_test))
        logger.log(event="best_test", epoch=best_epoch, metrics=best_test, best_epoch=best_epoch, is_best=True)

    final_metrics = {
        "best_epoch": best_epoch,
        "best_selection_value": best,
        "selection_metric": cfg.get("train", {}).get("selection_metric", "val_reg_loss"),
        "last_epoch": last_epoch,
        "best_checkpoint": "",
        "last_checkpoint": "",
        "best_val": best_val,
        "best_test": best_test or {},
    }
    logger.log(event="final", epoch=best_epoch, metrics=final_metrics, best_epoch=best_epoch, is_best=False)


if __name__ == "__main__":
    main()
