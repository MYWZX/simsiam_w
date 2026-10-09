#!/usr/bin/env python3
"""Diagnose where RichHF local relative relations are represented.

The script compares the disposable SimSiam projector space with the persistent
alignment space used by downstream IQA.  It is evaluation-only: checkpoints and
datasets are never modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.mixed_pretrain_dataset import RichHF18KPretrainDataset  # noqa: E402
from models.two_stage_models import CrossModalIQAJointPretrain  # noqa: E402
from train_cm_pretrain import (  # noqa: E402
    _map_to_patch_weights,
    _parse_richhf_prompt_phrases,
    _symmetric_simsiam_similarity,
)
from utils.train_utils import load_yaml, set_seed  # noqa: E402


DEFAULT_CONFIG = "configs/richhf18k_cm_pretrain_local_feedback_relative_rank_balanced_lr.yaml"
DEFAULT_CHECKPOINTS = (
    "balanced=outputs/richhf18k_cm_pretrain_local_feedback_balanced_lr/best_pretrain.pth",
    "relative_joint=outputs/richhf18k_cm_pretrain_local_feedback_relative_rank_balanced_lr/best_joint_pretrain.pth",
    "relative_align=outputs/richhf18k_cm_pretrain_local_feedback_relative_rank_balanced_lr/best_align_pretrain.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RichHF local relation gaps in disposable SimSiam and "
            "persistent downstream alignment spaces."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Checkpoint to evaluate; repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--output",
        default="outputs/local_relation_gap_diagnosis.csv",
        help="Summary CSV path.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit. Omit it for the complete validation split.",
    )
    parser.add_argument("--map-margin", type=float, default=None)
    parser.add_argument("--word-margin", type=float, default=None)
    parser.add_argument("--map-active-threshold", type=float, default=None)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA autocast.",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _parse_checkpoints(values: Sequence[str]) -> List[Tuple[str, Path]]:
    checkpoints: List[Tuple[str, Path]] = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must use LABEL=PATH syntax: {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in labels:
            raise ValueError(f"Checkpoint label must be non-empty and unique: {label!r}")
        path = _resolve(raw_path.strip())
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        labels.add(label)
        checkpoints.append((label, path))
    return checkpoints


def _build_validation_loader(
    cfg: Dict, batch_size: int | None, num_workers: int | None
) -> DataLoader:
    dcfg = cfg["dataset"]
    if str(dcfg.get("type", "")).lower() != "mixed_pretrain":
        raise ValueError("The diagnostic expects dataset.type=mixed_pretrain.")

    richhf_items = [
        item
        for item in dcfg.get("datasets", [])
        if str(item.get("type", "")).lower() in {"richhf18k", "richhf-18k", "richhf"}
    ]
    if len(richhf_items) != 1:
        raise ValueError(f"Expected exactly one RichHF dataset entry, found {len(richhf_items)}.")

    item = copy.deepcopy(richhf_items[0])
    item.update(
        {
            "input_size": dcfg.get("input_size", item.get("input_size", 224)),
            "split": item.get("val_split", "val"),
            "use_local_feedback": True,
            "supervision_view": cfg.get("train", {}).get(
                "human_supervision_view", item.get("supervision_view", "augmented")
            ),
        }
    )
    dataset = RichHF18KPretrainDataset(item, train=False)
    lcfg = cfg.get("loader", {})
    workers = int(lcfg.get("num_workers", 4) if num_workers is None else num_workers)
    size = int(lcfg.get("batch_size", 8) if batch_size is None else batch_size)
    return DataLoader(
        dataset,
        batch_size=size,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(lcfg.get("pin_memory", True)),
        drop_last=False,
    )


def _load_checkpoint(path: Path) -> Dict:
    # Explicit weights_only=False is needed because these trusted local
    # checkpoints also contain their YAML configuration.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_config_without_downloads(base_cfg: Dict, checkpoint: Dict) -> Dict:
    cfg = copy.deepcopy(checkpoint.get("cfg", base_cfg))
    # Every encoder parameter is restored below.  Constructing without
    # pretrained weights prevents an unnecessary network/cache dependency.
    cfg["model"]["global_encoder"]["pretrained"] = False
    cfg["model"]["local_encoder"]["pretrained"] = False
    cfg["model"].setdefault("text_encoder", {})["clip_pretrained"] = None
    return cfg


def _load_model(base_cfg: Dict, path: Path, device: torch.device):
    checkpoint = _load_checkpoint(path)
    model_cfg = _model_config_without_downloads(base_cfg, checkpoint)
    model = CrossModalIQAJointPretrain(model_cfg)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        preview_missing = ", ".join(missing[:8])
        preview_unexpected = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"Checkpoint/model mismatch for {path}: missing={len(missing)} "
            f"[{preview_missing}], unexpected={len(unexpected)} [{preview_unexpected}]"
        )
    epoch = int(checkpoint.get("epoch", -1))
    selection_metric = str(checkpoint.get("selection_metric", ""))
    selection_value = checkpoint.get("selection_value", "")
    del state, checkpoint
    gc.collect()
    model.to(device).eval()
    return model, epoch, selection_metric, selection_value


def _pool_regions(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(features).clamp(0.0, 1.0)
    pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.normalize(pooled, dim=-1)


def _persistent_text_features(
    model: CrossModalIQAJointPretrain,
    texts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Run text through the persistent path used by patch alignment."""
    encoded = model.iqa.backbone.text_encoder.encode_cached(texts, device)
    fused = model.iqa.text_to_fused(encoded)
    aligned = model.iqa.align_text_adapter(fused)
    return F.normalize(model.iqa.align_patch_txt_proj(aligned), dim=-1)


class RelationAccumulator:
    def __init__(self, margin: float):
        self.margin = float(margin)
        self.total = 0
        self.positive: List[np.ndarray] = []
        self.negative: List[np.ndarray] = []
        self.weights: List[np.ndarray] = []

    def add(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        valid: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        positive = positive.detach().float().cpu().view(-1)
        negative = negative.detach().float().cpu().view(-1)
        valid = valid.detach().float().cpu().view(-1) > 0
        if not (positive.numel() == negative.numel() == valid.numel()):
            raise ValueError("Similarity and validity tensors have inconsistent lengths.")
        self.total += valid.numel()
        if not valid.any():
            return
        if weight is None:
            selected_weight = torch.ones(int(valid.sum()), dtype=torch.float32)
        else:
            selected_weight = weight.detach().float().cpu().view(-1)[valid].clamp_min(0.0)
        self.positive.append(positive[valid].numpy())
        self.negative.append(negative[valid].numpy())
        self.weights.append(selected_weight.numpy())

    def summary(self) -> Dict[str, float | int]:
        if not self.positive:
            return {
                "n_total": self.total,
                "n_valid": 0,
                "valid_ratio": 0.0,
                "margin": self.margin,
            }
        positive = np.concatenate(self.positive).astype(np.float64, copy=False)
        negative = np.concatenate(self.negative).astype(np.float64, copy=False)
        weights = np.concatenate(self.weights).astype(np.float64, copy=False)
        gap = positive - negative
        violation = gap < self.margin
        weight_sum = float(weights.sum())
        weighted_gap = float(np.dot(gap, weights) / weight_sum) if weight_sum > 0 else math.nan
        weighted_violation = (
            float(np.dot(violation.astype(np.float64), weights) / weight_sum)
            if weight_sum > 0
            else math.nan
        )
        percentiles = np.percentile(gap, [10, 25, 50, 75, 90])
        return {
            "n_total": self.total,
            "n_valid": int(gap.size),
            "valid_ratio": float(gap.size / max(1, self.total)),
            "margin": self.margin,
            "positive_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "gap_mean": float(gap.mean()),
            "gap_std": float(gap.std()),
            "gap_median": float(np.median(gap)),
            "gap_p10": float(percentiles[0]),
            "gap_p25": float(percentiles[1]),
            "gap_p50": float(percentiles[2]),
            "gap_p75": float(percentiles[3]),
            "gap_p90": float(percentiles[4]),
            "violation_rate": float(violation.mean()),
            "weighted_gap_mean": weighted_gap,
            "weighted_violation_rate": weighted_violation,
        }


def _new_accumulators(map_margin: float, word_margin: float):
    return {
        ("map", "projector"): RelationAccumulator(map_margin),
        ("map", "persistent"): RelationAccumulator(map_margin),
        ("word", "projector"): RelationAccumulator(word_margin),
        ("word", "persistent"): RelationAccumulator(word_margin),
    }


@torch.inference_mode()
def diagnose_checkpoint(
    model: CrossModalIQAJointPretrain,
    loader: Iterable[Dict],
    device: torch.device,
    map_margin: float,
    word_margin: float,
    active_threshold: float,
    amp_enabled: bool,
    max_batches: int | None,
):
    accumulators = _new_accumulators(map_margin, word_margin)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = batch["image1"].to(device, non_blocking=True)
        spatial_map = batch["misalignment_map1"].to(device, non_blocking=True)
        map_valid = batch["misalignment_map_valid"].to(device, non_blocking=True).float()
        answer_confidence = batch["answer_confidence"].to(device, non_blocking=True).float()
        prompts = list(batch["prompt"])
        answers = list(batch["answer"])

        with autocast(device_type=device.type, enabled=amp_enabled):
            score = model.iqa.forward_single(image, prompts=prompts)
            patch_features = score["align_patch_features"]
            prompt_features = score["align_text_feature"]
            misaligned_weight = _map_to_patch_weights(spatial_map, patch_features.shape[1])
            aligned_weight = 1.0 - misaligned_weight
            region_valid = (
                map_valid
                * (aligned_weight.sum(dim=1) > 1e-4).float()
                * (misaligned_weight.amax(dim=1) > active_threshold).float()
            )

            # Persistent region features: exactly the p_img/p_txt space used
            # to form downstream patch attention and evidence.
            aligned_persistent = _pool_regions(patch_features, aligned_weight)
            misaligned_persistent = _pool_regions(patch_features, misaligned_weight)
            persistent_map_positive = F.cosine_similarity(
                aligned_persistent, prompt_features, dim=-1
            )
            persistent_map_negative = F.cosine_similarity(
                misaligned_persistent, prompt_features, dim=-1
            )

            # Disposable SimSiam projector/predictor relation used by training.
            projected_prompt = model._project_text(prompt_features)
            projected_aligned = model.project_patch_regions(patch_features, aligned_weight)
            projected_misaligned = model.project_patch_regions(patch_features, misaligned_weight)
            projected_map_positive = _symmetric_simsiam_similarity(
                projected_aligned["p"],
                projected_prompt["z_proj"],
                projected_prompt["p"],
                projected_aligned["z_proj"],
            )
            projected_map_negative = _symmetric_simsiam_similarity(
                projected_misaligned["p"],
                projected_prompt["z_proj"],
                projected_prompt["p"],
                projected_misaligned["z_proj"],
            )

            correct, wrong, correct_valid, wrong_valid = _parse_richhf_prompt_phrases(
                prompts, batch["prompt_word_labels"], device
            )
            word_valid = correct_valid * wrong_valid

            projected_correct = model.project_aux_texts(correct, device)
            projected_wrong = model.project_aux_texts(wrong, device)
            answer_encoded = model.iqa.backbone.text_encoder.encode_cached(answers, device)
            answer_fused = model.iqa.text_to_fused(answer_encoded)
            projected_answer = model._project_text(answer_fused)
            projected_word_positive = _symmetric_simsiam_similarity(
                projected_correct["p"],
                projected_answer["z_proj"],
                projected_answer["p"],
                projected_correct["z_proj"],
            )
            projected_word_negative = _symmetric_simsiam_similarity(
                projected_wrong["p"],
                projected_answer["z_proj"],
                projected_answer["p"],
                projected_wrong["z_proj"],
            )

            # Apply the same retained downstream text path to answer/correct/
            # wrong phrases, then compare them in align_patch_txt_proj space.
            persistent_answer = _persistent_text_features(model, answers, device)
            persistent_correct = _persistent_text_features(model, correct, device)
            persistent_wrong = _persistent_text_features(model, wrong, device)
            persistent_word_positive = F.cosine_similarity(
                persistent_answer, persistent_correct, dim=-1
            )
            persistent_word_negative = F.cosine_similarity(
                persistent_answer, persistent_wrong, dim=-1
            )

        accumulators[("map", "projector")].add(
            projected_map_positive, projected_map_negative, region_valid
        )
        accumulators[("map", "persistent")].add(
            persistent_map_positive, persistent_map_negative, region_valid
        )
        accumulators[("word", "projector")].add(
            projected_word_positive,
            projected_word_negative,
            word_valid,
            answer_confidence,
        )
        accumulators[("word", "persistent")].add(
            persistent_word_positive,
            persistent_word_negative,
            word_valid,
            answer_confidence,
        )

        if (batch_index + 1) % 20 == 0:
            print(f"  processed {batch_index + 1} batches", flush=True)
    return accumulators


def _write_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "checkpoint",
        "checkpoint_path",
        "epoch",
        "selection_metric",
        "selection_value",
        "relation",
        "space",
        "n_total",
        "n_valid",
        "valid_ratio",
        "margin",
        "positive_mean",
        "negative_mean",
        "gap_mean",
        "gap_std",
        "gap_median",
        "gap_p10",
        "gap_p25",
        "gap_p50",
        "gap_p75",
        "gap_p90",
        "violation_rate",
        "weighted_gap_mean",
        "weighted_violation_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config_path = _resolve(args.config)
    cfg = load_yaml(str(config_path))
    loader = _build_validation_loader(cfg, args.batch_size, args.num_workers)
    checkpoints = _parse_checkpoints(args.checkpoint or DEFAULT_CHECKPOINTS)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu.")

    tcfg = cfg.get("train", {})
    map_margin = float(
        args.map_margin
        if args.map_margin is not None
        else tcfg.get("img_prompt_local_rank_margin", tcfg.get("img_prompt_misaligned_max_similarity", 0.10))
    )
    word_margin = float(
        args.word_margin
        if args.word_margin is not None
        else tcfg.get("prompt_answer_word_rank_margin", tcfg.get("prompt_answer_misaligned_max_similarity", 0.10))
    )
    active_threshold = float(
        args.map_active_threshold
        if args.map_active_threshold is not None
        else tcfg.get("misalignment_map_active_threshold", 0.05)
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    print(
        f"[INFO] val_samples={len(loader.dataset)}, device={device}, amp={amp_enabled}, "
        f"map_margin={map_margin}, word_margin={word_margin}, "
        f"map_active_threshold={active_threshold}"
    )
    rows: List[Dict] = []
    for label, checkpoint_path in checkpoints:
        print(f"[INFO] Loading {label}: {checkpoint_path}", flush=True)
        model, epoch, selection_metric, selection_value = _load_model(
            cfg, checkpoint_path, device
        )
        print(f"[INFO] Diagnosing {label} (epoch={epoch})", flush=True)
        accumulators = diagnose_checkpoint(
            model=model,
            loader=loader,
            device=device,
            map_margin=map_margin,
            word_margin=word_margin,
            active_threshold=active_threshold,
            amp_enabled=amp_enabled,
            max_batches=args.max_batches,
        )
        for (relation, space), accumulator in accumulators.items():
            summary = accumulator.summary()
            row = {
                "checkpoint": label,
                "checkpoint_path": str(checkpoint_path),
                "epoch": epoch,
                "selection_metric": selection_metric,
                "selection_value": selection_value,
                "relation": relation,
                "space": space,
                **summary,
            }
            rows.append(row)
            print(
                f"  {relation:4s}/{space:10s}: valid={summary['n_valid']}/{summary['n_total']} "
                f"gap={_format_value(summary.get('gap_mean'))} "
                f"violation={_format_value(summary.get('violation_rate'))}"
            )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = _resolve(args.output)
    _write_rows(output_path, rows)
    print(f"[DONE] Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
