#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Fuse multiple weak alignment signals into an adaptive pseudo-align score.

The fusion is confidence-aware:
  alpha = softmax((log(prior) + logit(confidence)) / temperature)
  score = sum_i alpha_i * source_score_i

Missing or failed sources get a low confidence and do not dominate the result.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: Path, encoding: str = "auto") -> pd.DataFrame:
    if encoding and str(encoding).lower() not in {"", "auto"}:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[CSV] loaded {path} with encoding={enc}")
            return df
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_err}")


def _score(df: pd.DataFrame, score_col: str, status_col: str | None, default: float) -> Tuple[np.ndarray, np.ndarray]:
    if score_col not in df.columns:
        return np.full(len(df), default, dtype=np.float64), np.zeros(len(df), dtype=bool)
    score = pd.to_numeric(df[score_col], errors="coerce").astype("float64")
    valid = score.notna().to_numpy()
    if status_col and status_col in df.columns:
        ok = df[status_col].astype(str).str.strip().str.lower().isin(["", "ok"]).to_numpy()
        valid = valid & ok
    values = score.fillna(default).clip(0.0, 1.0).to_numpy(dtype=np.float64).copy()
    values[~valid] = default
    return values, valid


def _confidence(
    df: pd.DataFrame,
    conf_col: str | None,
    fallback: np.ndarray,
    valid: np.ndarray,
    default_invalid: float,
) -> np.ndarray:
    if conf_col and conf_col in df.columns:
        conf = pd.to_numeric(df[conf_col], errors="coerce").astype("float64").to_numpy()
        conf = np.where(np.isfinite(conf), conf, fallback)
    else:
        conf = fallback
    conf = np.clip(conf, 0.0, 1.0)
    return np.where(valid, conf, default_invalid)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.nanmax(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1e-8, None)


def _cap_and_renormalize(alpha: np.ndarray, caps: np.ndarray, steps: int = 4) -> np.ndarray:
    """Apply per-source maximum alpha caps and keep each row summing to 1."""
    capped = alpha.copy()
    caps = np.clip(caps.astype(np.float64), 1e-6, 1.0)
    if np.all(caps >= 1.0):
        return capped

    for _ in range(max(1, steps)):
        over = capped > caps[None, :]
        if not over.any():
            break
        capped = np.minimum(capped, caps[None, :])
        fixed_mass = (capped * over).sum(axis=1, keepdims=True)
        free = ~over
        free_mass = (capped * free).sum(axis=1, keepdims=True)
        remaining = np.clip(1.0 - fixed_mass, 0.0, 1.0)
        scale = np.divide(
            remaining,
            np.clip(free_mass, 1e-8, None),
            out=np.ones_like(remaining),
            where=free_mass > 1e-8,
        )
        capped = np.where(free, capped * scale, capped)

    row_sum = np.clip(capped.sum(axis=1, keepdims=True), 1e-8, None)
    return capped / row_sum


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def _parse_source(cfg: Dict, name: str, default_score: float, default_conf: float) -> Dict:
    sources = cfg.get("sources", {})
    scfg = sources.get(name, {})
    return {
        "score_col": scfg.get("score_column", ""),
        "status_col": scfg.get("status_column", ""),
        "confidence_col": scfg.get("confidence_column", ""),
        "default_score": float(scfg.get("default_score", default_score)),
        "default_confidence": float(scfg.get("default_confidence", default_conf)),
        "confidence_scale": float(scfg.get("confidence_scale", 1.0)),
        "confidence_power": float(scfg.get("confidence_power", 1.0)),
        "max_alpha": float(scfg.get("max_alpha", 1.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build adaptive pseudo-align score for AIGIQA-20K.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("adaptive_pseudo_align", full_cfg)

    input_csv = Path(cfg["input_csv"])
    output_csv = Path(cfg["output_csv"])
    input_encoding = str(cfg.get("input_encoding", "auto"))
    output_encoding = str(cfg.get("output_encoding", "utf-8-sig"))
    temperature = float(cfg.get("temperature", 0.7))
    min_valid_sources = int(cfg.get("min_valid_sources", 2))
    fallback_source = str(cfg.get("fallback_source", "clip")).lower()
    prior_weights = cfg.get("prior_weights", {"clip": 0.35, "vqa": 0.30, "pa": 0.20, "obj": 0.15})

    out_col = cfg.get("score_column", "adaptive_pseudo_align_score")
    status_col = cfg.get("status_column", "adaptive_pseudo_align_status")
    valid_count_col = cfg.get("valid_count_column", "adaptive_source_valid_count")
    reliability_col = cfg.get("reliability_column", "adaptive_reliability_score")

    df = read_csv_robust(input_csv, input_encoding)

    source_names = ["clip", "vqa", "pa", "obj"]
    parsed = {
        "clip": _parse_source(cfg, "clip", default_score=0.5, default_conf=0.05),
        "vqa": _parse_source(cfg, "vqa", default_score=0.5, default_conf=0.05),
        "pa": _parse_source(cfg, "pa", default_score=0.5, default_conf=0.05),
        "obj": _parse_source(cfg, "obj", default_score=0.5, default_conf=0.05),
    }

    scores = []
    confs = []
    valids = []
    for name in source_names:
        scfg = parsed[name]
        s, valid = _score(df, scfg["score_col"], scfg["status_col"], scfg["default_score"])
        # Without a dedicated confidence column, high-confidence positive matches
        # are those the source itself scores clearly.
        fallback_conf = np.maximum(s, np.abs(s - 0.5) * 2.0)
        c = _confidence(df, scfg["confidence_col"], fallback_conf, valid, scfg["default_confidence"])
        c = np.clip((c ** max(scfg["confidence_power"], 1e-6)) * scfg["confidence_scale"], 0.0, 1.0)
        scores.append(s)
        confs.append(c)
        valids.append(valid)

    score_mat = np.stack(scores, axis=1)
    conf_mat = np.stack(confs, axis=1)
    valid_mat = np.stack(valids, axis=1)
    valid_count = valid_mat.sum(axis=1)

    prior = np.asarray([float(prior_weights.get(name, 1.0)) for name in source_names], dtype=np.float64)
    prior = prior / np.clip(prior.sum(), 1e-8, None)
    logits = np.log(np.clip(prior, 1e-8, None))[None, :] + _logit(conf_mat)
    alpha = _softmax(logits / max(temperature, 1e-6), axis=1)
    caps = np.asarray([parsed[name]["max_alpha"] for name in source_names], dtype=np.float64)
    alpha = _cap_and_renormalize(alpha, caps)
    fused = np.sum(alpha * score_mat, axis=1)

    fallback_idx = source_names.index(fallback_source) if fallback_source in source_names else 0
    fused = np.where(valid_count >= min_valid_sources, fused, score_mat[:, fallback_idx])
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    reliability = np.sum(alpha * conf_mat, axis=1).clip(0.0, 1.0).astype(np.float32)

    df[out_col] = fused
    df[status_col] = np.where(valid_count >= min_valid_sources, "ok", f"fallback_{fallback_source}")
    df[valid_count_col] = valid_count.astype(np.int32)
    df[reliability_col] = reliability
    for i, name in enumerate(source_names):
        df[f"adaptive_alpha_{name}"] = alpha[:, i].astype(np.float32)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=output_encoding)

    print("Done.")
    print(f"rows={len(df)}, output_csv={output_csv}")
    print(f"{out_col}: min={float(df[out_col].min()):.4f}, mean={float(df[out_col].mean()):.4f}, max={float(df[out_col].max()):.4f}")
    print(f"{status_col}: {df[status_col].value_counts(dropna=False).to_dict()}")
    alpha_means = {name: float(df[f'adaptive_alpha_{name}'].mean()) for name in source_names}
    print(f"alpha_mean={alpha_means}")


if __name__ == "__main__":
    main()
