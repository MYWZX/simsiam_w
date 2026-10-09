#!/usr/bin/env python3
"""Diagnose whether a pretraining SimSiam projector transfers to AGIQA-1K.

The script is intentionally read-only with respect to checkpoints.  For every
checkpoint it extracts deterministic evaluation-view features on one fixed
AGIQA-1K split and compares:

* the shared ViT/ResNet fusion representation;
* the shared SimSiam image-projector representation;
* their concatenation;
* optional low-dimensional quality coordinates defined by the checkpoint.

Ridge regularization is selected on the validation split.  The selected model
is then refit on train+validation and evaluated once on the test split.  This is
a diagnostic linear probe, not a replacement for the downstream IQA result.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.multi_iqa_dataset import AGIQA1KIQADataset  # noqa: E402
from models.backbones import GlobalVisualEncoder, ResNet50LocalEncoder  # noqa: E402
from models.fusion import SemanticStructureFusion  # noqa: E402
from models.two_stage_models import MLP  # noqa: E402
from utils.train_utils import load_yaml, set_seed  # noqa: E402


DEFAULT_CHECKPOINTS = (
    (
        "old_artifact_anchor",
        "outputs/richhf18k_cm_pretrain_local_feedback_quality_components_ordinal_matched_clip_balanced_lr/"
        "best_artifact_anchor_pretrain.pth",
    ),
    (
        "mean_residual_derived_artifact",
        "outputs/richhf18k_cm_pretrain_local_feedback_mean_residual_matched_clip_balanced_lr/"
        "best_derived_artifact_pretrain.pth",
    ),
    (
        "mean_residual_best_quality",
        "outputs/richhf18k_cm_pretrain_local_feedback_mean_residual_matched_clip_balanced_lr/"
        "best_quality_pretrain.pth",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen fusion and SimSiam-projector features on AGIQA-1K."
    )
    parser.add_argument(
        "--downstream-config",
        default="configs/agiqa1k_cm_iqa_mean_residual_derived_artifact_repr_low_lr_srocc.yaml",
        help="AGIQA-1K config used only for dataset paths, image size and split ratios.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Checkpoint to diagnose. Repeat this option; defaults to the three project checkpoints.",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="'auto' uses CUDA when available without selecting a GPU index.",
    )
    parser.add_argument(
        "--alphas",
        default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000",
        help="Comma-separated Ridge alpha grid.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/projector_transfer_diagnosis_split42.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output CSV.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Smoke-test only: truncate each split to this many examples.",
    )
    return parser.parse_args()


def _parse_checkpoints(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        checkpoints = list(DEFAULT_CHECKPOINTS)
    else:
        checkpoints = []
        for value in values:
            if "=" not in value:
                raise ValueError(
                    f"Invalid --checkpoint {value!r}; expected NAME=PATH."
                )
            name, path = value.split("=", 1)
            name, path = name.strip(), path.strip()
            if not name or not path:
                raise ValueError(
                    f"Invalid --checkpoint {value!r}; NAME and PATH must be non-empty."
                )
            checkpoints.append((name, path))
    names = [name for name, _ in checkpoints]
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint names must be unique.")
    for _, path in checkpoints:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    return checkpoints


def _parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas or any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("--alphas must contain at least one positive value.")
    return alphas


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(value)


class VisualProjectorExtractor(nn.Module):
    """Minimal checkpoint-compatible visual backbone plus shared projector."""

    def __init__(self, cfg: Dict):
        super().__init__()
        mcfg = cfg["model"]
        gcfg = mcfg["global_encoder"]
        lcfg = mcfg["local_encoder"]
        fcfg = mcfg["fusion"]
        pcfg = mcfg.get("simsiam", {})
        hcfg = mcfg.get("heads", {})

        if bool(pcfg.get("separate_quality_projection", False)):
            raise ValueError(
                "This diagnostic expects the shared image_projector. "
                "The supplied checkpoint uses separate_quality_projection=true."
            )

        # Checkpoint weights are loaded immediately below, so online pretrained
        # initialization would be redundant and would break offline diagnosis.
        self.global_encoder = GlobalVisualEncoder(
            encoder_type=gcfg.get("type", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
            pretrained=False,
            output_dim=int(gcfg.get("output_dim", 768)),
            img_size=int(cfg.get("dataset", {}).get("input_size", 224)),
            open_clip_model_name=gcfg.get("model_name", "ViT-B-16"),
            open_clip_pretrained=gcfg.get("open_clip_pretrained", "openai"),
            source_normalization=gcfg.get("source_normalization", "imagenet"),
            normalize_output=bool(gcfg.get("normalize_output", False)),
        )
        self.local_encoder = ResNet50LocalEncoder(
            pretrained=False,
            output_dim=int(lcfg.get("output_dim", 256)),
            model_name=lcfg.get("type", "resnet50.a1_in1k"),
        )
        self.fusion = SemanticStructureFusion(
            semantic_dim=int(gcfg.get("output_dim", 768)),
            local_dim=int(lcfg.get("output_dim", 256)),
            embed_dim=int(fcfg.get("embed_dim", 256)),
            num_heads=int(fcfg.get("num_heads", 4)),
            dropout=float(fcfg.get("dropout", 0.1)),
        )
        fused_dim = int(fcfg.get("embed_dim", 256))
        self.image_projector = MLP(
            fused_dim,
            int(pcfg.get("proj_hidden_dim", 512)),
            int(pcfg.get("proj_dim", 256)),
            dropout=float(hcfg.get("dropout", 0.1)),
            last_bn=True,
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = self.global_encoder(image)
        local_map = self.local_encoder(image)
        fused = self.fusion(semantic, local_map)
        projected = F.normalize(self.image_projector(fused), dim=-1)
        return F.normalize(fused.float(), dim=-1), projected.float()


def _load_visual_projector(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[VisualProjectorExtractor, Dict, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "cfg" not in checkpoint:
        raise RuntimeError(f"Checkpoint lacks its training config: {checkpoint_path}")
    cfg = copy.deepcopy(checkpoint["cfg"])
    print(
        "[INFO] Online pretrained initialization is intentionally disabled; "
        "all extractor states will now be replaced by checkpoint weights."
    )
    model = VisualProjectorExtractor(cfg)
    source_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    target_state = model.state_dict()
    loadable = {}
    bad_shapes = []

    for raw_key, value in source_state.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key.startswith("iqa.backbone."):
            target_key = key[len("iqa.backbone."):]
        elif key.startswith("backbone."):
            target_key = key[len("backbone."):]
        elif key.startswith("image_projector."):
            target_key = key
        else:
            continue
        if target_key not in target_state:
            continue
        if target_state[target_key].shape != value.shape:
            bad_shapes.append(
                (target_key, tuple(value.shape), tuple(target_state[target_key].shape))
            )
            continue
        loadable[target_key] = value

    missing = sorted(set(target_state) - set(loadable))
    if bad_shapes or missing:
        raise RuntimeError(
            "Checkpoint is not compatible with the visual-projector extractor. "
            f"missing={missing[:8]}, bad_shapes={bad_shapes[:5]}"
        )
    model.load_state_dict(loadable, strict=True)
    model.eval().to(device)
    metadata = {
        "checkpoint_epoch": checkpoint.get("epoch", -1),
        "checkpoint_selection_metric": checkpoint.get("selection_metric", ""),
        "checkpoint_selection_value": checkpoint.get("selection_value", ""),
        "loaded_keys": len(loadable),
    }
    return model, cfg, metadata


def _build_dataset_and_indices(cfg: Dict, split_seed: int):
    dcfg = cfg["dataset"]
    dataset = AGIQA1KIQADataset(
        root_dir=dcfg["root_dir"],
        image_dir=dcfg.get("image_dir", "file"),
        csv_file=dcfg.get("csv_file", "AIGC-1K_answer.csv"),
        input_size=int(dcfg.get("input_size", 224)),
        train=False,
        mos_max=dcfg.get("mos_max"),
        use_perturb=False,
        perturbation_prob=0.0,
        perturbation_strengths=[],
        saliency_sampling_ratio=float(dcfg.get("saliency_sampling_ratio", 0.7)),
        final_quality_weight=float(dcfg.get("final_quality_weight", 1.0)),
        final_align_weight=float(dcfg.get("final_align_weight", 0.0)),
        train_augmentation_mode="quality_safe",
        quality_safe_flip_prob=0.0,
        local_multicrop={"enabled": False},
    )
    total = len(dataset)
    train_n = int(total * float(dcfg.get("train_ratio", 0.7)))
    val_n = int(total * float(dcfg.get("val_ratio", 0.1)))
    generator = torch.Generator().manual_seed(int(split_seed))
    order = torch.randperm(total, generator=generator).tolist()
    indices = {
        "train": order[:train_n],
        "val": order[train_n:train_n + val_n],
        "test": order[train_n + val_n:],
    }
    return dataset, indices


def _truncate_indices(
    indices: Dict[str, list[int]],
    maximum: int | None,
) -> Dict[str, list[int]]:
    if maximum is None:
        return indices
    if maximum < 2:
        raise ValueError("--max-samples-per-split must be at least 2.")
    return {name: values[:maximum] for name, values in indices.items()}


def _make_loaders(
    dataset,
    indices: Dict[str, list[int]],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Dict[str, DataLoader]:
    return {
        name: DataLoader(
            Subset(dataset, split_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        for name, split_indices in indices.items()
    }


def _fixed_anchor_score(feature: torch.Tensor, cfg: Dict) -> torch.Tensor:
    feature = F.normalize(feature.float(), dim=-1)
    levels = int(cfg.get("num_levels", 5))
    centers_cfg = cfg.get("centers")
    centers = (
        torch.linspace(0.0, 1.0, levels, device=feature.device)
        if centers_cfg is None
        else feature.new_tensor([float(value) for value in centers_cfg])
    )
    dimension = feature.shape[-1]
    frequency = int(cfg.get("basis_frequency", 0))
    if frequency <= 0:
        basis0 = torch.ones(dimension, device=feature.device, dtype=feature.dtype)
        basis1 = torch.linspace(
            -1.0, 1.0, dimension, device=feature.device, dtype=feature.dtype
        )
    else:
        coordinate = (
            torch.arange(dimension, device=feature.device, dtype=feature.dtype) + 0.5
        ) / float(dimension)
        angle = 2.0 * math.pi * float(frequency) * coordinate
        basis0, basis1 = torch.cos(angle), torch.sin(angle)
    basis0 = F.normalize(basis0, dim=0)
    basis1 = basis1 - torch.dot(basis1, basis0) * basis0
    basis1 = F.normalize(basis1, dim=0)

    half_arc = 0.5 * math.radians(float(cfg.get("anchor_arc_degrees", 90.0)))
    angles = torch.linspace(-half_arc, half_arc, levels, device=feature.device)
    anchors = F.normalize(
        torch.cos(angles).unsqueeze(1) * basis0.unsqueeze(0)
        + torch.sin(angles).unsqueeze(1) * basis1.unsqueeze(0),
        dim=-1,
    )
    anchor_feature = feature
    if bool(cfg.get("normalize_in_anchor_plane", False)):
        anchor_feature = (
            (feature @ basis0).unsqueeze(1) * basis0.unsqueeze(0)
            + (feature @ basis1).unsqueeze(1) * basis1.unsqueeze(0)
        )
        anchor_feature = F.normalize(anchor_feature, dim=-1)
    temperature = float(cfg.get("anchor_temperature", 0.20))
    probability = (anchor_feature @ anchors.t() / temperature).softmax(dim=-1)
    return (probability * centers.unsqueeze(0)).sum(dim=-1)


def _mean_residual_axis(
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    basis0 = F.normalize(torch.ones(dimension, device=device, dtype=dtype), dim=0)
    basis1 = torch.linspace(-1.0, 1.0, dimension, device=device, dtype=dtype)
    basis1 = F.normalize(basis1 - torch.dot(basis1, basis0) * basis0, dim=0)
    coordinate = (
        torch.arange(dimension, device=device, dtype=dtype) + 0.5
    ) / float(dimension)
    axis = torch.cos(2.0 * math.pi * coordinate)
    axis = axis - torch.dot(axis, basis0) * basis0
    axis = axis - torch.dot(axis, basis1) * basis1
    return F.normalize(axis, dim=0)


def _quality_coordinate_features(
    projected: torch.Tensor,
    pretrain_cfg: Dict,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    tcfg = pretrain_cfg.get("train", {})
    feature_sets: Dict[str, torch.Tensor] = {}
    zero_shot: Dict[str, torch.Tensor] = {}

    mean_cfg = tcfg.get("quality_anchor_ordinal", {})
    residual_cfg = tcfg.get("mean_residual_quality", {})
    if bool(mean_cfg.get("enabled", False)) and bool(residual_cfg.get("enabled", False)):
        mean_score = _fixed_anchor_score(projected, mean_cfg)
        axis = _mean_residual_axis(
            projected.shape[-1], projected.device, projected.dtype
        )
        scale = float(residual_cfg.get("residual_scale", 0.50))
        temperature = float(residual_cfg.get("residual_temperature", 0.20))
        residual_score = scale * torch.tanh((projected @ axis) / temperature)
        derived_artifact = (mean_score + 0.5 * residual_score).clamp(0.0, 1.0)
        derived_aesthetics = (mean_score - 0.5 * residual_score).clamp(0.0, 1.0)
        feature_sets["mean_residual_coordinates"] = torch.stack(
            [mean_score, residual_score], dim=-1
        )
        zero_shot.update({
            "mean_anchor": mean_score,
            "derived_artifact": derived_artifact,
            "derived_aesthetics": derived_aesthetics,
        })

    component_cfg = tcfg.get("quality_component_anchor_ordinal", {})
    if bool(component_cfg.get("enabled", False)):
        common = {
            key: value
            for key, value in component_cfg.items()
            if key not in {"artifact", "aesthetics"}
        }
        scores = {}
        for component in ("artifact", "aesthetics"):
            child = {**common, **dict(component_cfg.get(component, {}))}
            if not bool(child.get("enabled", True)):
                continue
            scores[component] = _fixed_anchor_score(projected, child)
            zero_shot[f"{component}_anchor"] = scores[component]
        if len(scores) == 2:
            feature_sets["component_anchor_coordinates"] = torch.stack(
                [scores["artifact"], scores["aesthetics"]], dim=-1
            )
    return feature_sets, zero_shot


@torch.inference_mode()
def _extract_features(
    model: VisualProjectorExtractor,
    loaders: Dict[str, DataLoader],
    pretrain_cfg: Dict,
    device: torch.device,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    features: Dict[str, Dict[str, np.ndarray]] = {}
    zero_shot: Dict[str, Dict[str, np.ndarray]] = {}
    targets: Dict[str, np.ndarray] = {}
    amp_enabled = device.type == "cuda"

    for split, loader in loaders.items():
        split_features: Dict[str, list[torch.Tensor]] = {}
        split_zero: Dict[str, list[torch.Tensor]] = {}
        split_targets = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                fused, projected = model(image)
            coordinate_features, coordinate_scores = _quality_coordinate_features(
                projected.float(), pretrain_cfg
            )
            current_features = {
                "fusion": fused.float(),
                "projector": projected.float(),
                "fusion_projector": torch.cat([fused.float(), projected.float()], dim=-1),
                **coordinate_features,
            }
            for name, value in current_features.items():
                split_features.setdefault(name, []).append(value.cpu())
            for name, value in coordinate_scores.items():
                split_zero.setdefault(name, []).append(value.float().cpu())
            split_targets.append(batch["quality_score"].float().view(-1).cpu())

        features[split] = {
            name: torch.cat(chunks, dim=0).numpy()
            for name, chunks in split_features.items()
        }
        zero_shot[split] = {
            name: torch.cat(chunks, dim=0).numpy()
            for name, chunks in split_zero.items()
        }
        targets[split] = torch.cat(split_targets, dim=0).numpy()
    return features, zero_shot, targets


def _safe_correlation(function, target: np.ndarray, prediction: np.ndarray) -> float:
    try:
        value = function(target, prediction)[0]
        return 0.0 if not np.isfinite(value) else float(value)
    except Exception:
        return 0.0


def _metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    return {
        "N": int(target.size),
        "PLCC": _safe_correlation(pearsonr, target, prediction),
        "SROCC": _safe_correlation(spearmanr, target, prediction),
        "KROCC": _safe_correlation(kendalltau, target, prediction),
        "RMSE": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "MAE": float(np.mean(np.abs(target - prediction))),
    }


def _prepare_ridge(
    features: np.ndarray,
    target: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (features - mean) / scale
    target_mean = float(target.mean())
    centered_target = target - target_mean
    u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    return {
        "mean": mean,
        "scale": scale,
        "target_mean": target_mean,
        "singular": singular,
        "vt": vt,
        "projected_target": u.T @ centered_target,
    }


def _ridge_state(
    prepared: Dict[str, np.ndarray | float],
    alpha: float,
) -> Dict[str, np.ndarray | float]:
    singular = np.asarray(prepared["singular"])
    vt = np.asarray(prepared["vt"])
    coefficient = vt.T @ (
        (singular / (singular * singular + float(alpha)))
        * np.asarray(prepared["projected_target"])
    )
    return {
        "mean": prepared["mean"],
        "scale": prepared["scale"],
        "target_mean": prepared["target_mean"],
        "coefficient": coefficient,
    }


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> Dict[str, np.ndarray | float]:
    return _ridge_state(_prepare_ridge(features, target), alpha)


def _predict_ridge(state: Dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    normalized = (
        np.asarray(features, dtype=np.float64) - np.asarray(state["mean"])
    ) / np.asarray(state["scale"])
    return float(state["target_mean"]) + normalized @ np.asarray(state["coefficient"])


def _select_and_evaluate_ridge(
    split_features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    alphas: Iterable[float],
) -> tuple[float, Dict[str, Dict[str, float]]]:
    candidates = []
    prepared = _prepare_ridge(split_features["train"], targets["train"])
    for alpha in alphas:
        state = _ridge_state(prepared, alpha)
        train_prediction = _predict_ridge(state, split_features["train"])
        val_prediction = _predict_ridge(state, split_features["val"])
        train_metrics = _metrics(targets["train"], train_prediction)
        val_metrics = _metrics(targets["val"], val_prediction)
        candidates.append((
            val_metrics["SROCC"],
            val_metrics["PLCC"],
            -val_metrics["RMSE"],
            -float(alpha),
            float(alpha),
            train_metrics,
            val_metrics,
        ))
    best = max(candidates, key=lambda item: item[:4])
    alpha, train_metrics, val_metrics = best[4], best[5], best[6]

    combined_features = np.concatenate(
        [split_features["train"], split_features["val"]], axis=0
    )
    combined_target = np.concatenate([targets["train"], targets["val"]], axis=0)
    final_state = _fit_ridge(combined_features, combined_target, alpha)
    test_prediction = _predict_ridge(final_state, split_features["test"])
    test_metrics = _metrics(targets["test"], test_prediction)
    return alpha, {"train": train_metrics, "val": val_metrics, "test": test_metrics}


def _result_row(
    checkpoint_name: str,
    checkpoint_path: str,
    metadata: Dict,
    kind: str,
    feature_name: str,
    dimension: int,
    alpha: float | str,
    split_seed: int,
    split_metrics: Dict[str, Dict[str, float]],
) -> Dict:
    row = {
        "checkpoint": checkpoint_name,
        "checkpoint_path": checkpoint_path,
        **metadata,
        "kind": kind,
        "feature": feature_name,
        "dimension": dimension,
        "alpha": alpha,
        "split_seed": split_seed,
    }
    for split in ("train", "val", "test"):
        for metric, value in split_metrics[split].items():
            row[f"{split}_{metric}"] = value
    return row


def _write_csv(path: str, rows: list[Dict], overwrite: bool) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite or choose another path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    alphas = _parse_alphas(args.alphas)
    device = _resolve_device(args.device)
    downstream_cfg = load_yaml(args.downstream_config)
    split_seed = int(
        args.split_seed
        if args.split_seed is not None
        else downstream_cfg.get("split_seed", downstream_cfg.get("seed", 42))
    )
    set_seed(split_seed)
    dataset, indices = _build_dataset_and_indices(downstream_cfg, split_seed)
    indices = _truncate_indices(indices, args.max_samples_per_split)
    loaders = _make_loaders(
        dataset,
        indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    print(
        f"[INFO] device={device}, split_seed={split_seed}, "
        + ", ".join(f"{name}={len(values)}" for name, values in indices.items())
    )

    all_rows = []
    for checkpoint_name, checkpoint_path in checkpoints:
        print(f"\n[INFO] Loading {checkpoint_name}: {checkpoint_path}")
        model, pretrain_cfg, metadata = _load_visual_projector(checkpoint_path, device)
        print(
            f"[INFO] checkpoint_epoch={metadata['checkpoint_epoch']}, "
            f"loaded_keys={metadata['loaded_keys']}"
        )
        features, zero_shot, targets = _extract_features(
            model, loaders, pretrain_cfg, device
        )

        checkpoint_rows = []
        shared_feature_names = set(features["train"])
        if not all(set(features[split]) == shared_feature_names for split in features):
            raise RuntimeError("Feature names differ across dataset splits.")
        preferred_order = (
            "fusion",
            "projector",
            "fusion_projector",
            "mean_residual_coordinates",
            "component_anchor_coordinates",
        )
        ordered_names = [name for name in preferred_order if name in shared_feature_names]
        ordered_names.extend(sorted(shared_feature_names - set(ordered_names)))
        for feature_name in ordered_names:
            split_feature = {
                split: features[split][feature_name]
                for split in ("train", "val", "test")
            }
            alpha, metrics = _select_and_evaluate_ridge(
                split_feature, targets, alphas
            )
            row = _result_row(
                checkpoint_name,
                checkpoint_path,
                metadata,
                "ridge",
                feature_name,
                int(split_feature["train"].shape[1]),
                alpha,
                split_seed,
                metrics,
            )
            checkpoint_rows.append(row)
            print(
                f"[RIDGE] {feature_name:30s} dim={row['dimension']:3d} "
                f"alpha={alpha:g} val={row['val_SROCC']:.5f} "
                f"test={row['test_SROCC']:.5f} PLCC={row['test_PLCC']:.5f} "
                f"RMSE={row['test_RMSE']:.5f}"
            )

        zero_names = set(zero_shot["train"])
        for score_name in sorted(zero_names):
            metrics = {
                split: _metrics(targets[split], zero_shot[split][score_name])
                for split in ("train", "val", "test")
            }
            row = _result_row(
                checkpoint_name,
                checkpoint_path,
                metadata,
                "zero_shot",
                score_name,
                1,
                "",
                split_seed,
                metrics,
            )
            checkpoint_rows.append(row)
            print(
                f"[ZERO ] {score_name:30s} "
                f"val={row['val_SROCC']:.5f} test={row['test_SROCC']:.5f}"
            )

        ridge_by_name = {
            row["feature"]: row
            for row in checkpoint_rows
            if row["kind"] == "ridge"
        }
        if "fusion" in ridge_by_name and "projector" in ridge_by_name:
            delta = (
                float(ridge_by_name["projector"]["test_SROCC"])
                - float(ridge_by_name["fusion"]["test_SROCC"])
            )
            action = (
                "projector_candidate"
                if delta >= 0.01
                else "no_extra_projector_signal"
            )
            for row in checkpoint_rows:
                row["projector_minus_fusion_test_SROCC"] = delta
                row["decision"] = action
            print(
                f"[DECISION] projector-fusion test SROCC={delta:+.5f}: {action}"
            )
        all_rows.extend(checkpoint_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(args.output_csv, all_rows, overwrite=bool(args.overwrite))
    print(f"\n[INFO] Saved {len(all_rows)} diagnostic rows to {args.output_csv}")


if __name__ == "__main__":
    main()
