#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: Path, encoding: str = "auto") -> pd.DataFrame:
    if encoding and str(encoding).lower() not in {"", "auto"}:
        return pd.read_csv(path, encoding=encoding)
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[CSV] loaded {path} with encoding={enc}")
            return df
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_err}")


def _valid_score(df: pd.DataFrame, score_col: str, status_col: str | None, default: float = 0.0) -> pd.Series:
    if score_col not in df.columns:
        raise KeyError(f"Missing score column: {score_col}")
    score = pd.to_numeric(df[score_col], errors="coerce").astype("float64")
    if status_col and status_col in df.columns:
        ok = df[status_col].astype(str).str.strip().str.lower().isin(["", "ok"])
        score = score.where(ok, np.nan)
    return score.fillna(default).clip(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Build 20K semantic confidence from image-prompt, image-answer, and prompt-answer scores.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("semantic_confidence", full_cfg)

    input_csv = Path(cfg["input_csv"])
    output_csv = Path(cfg["output_csv"])
    input_encoding = str(cfg.get("input_encoding", "auto"))
    output_encoding = str(cfg.get("output_encoding", "utf-8-sig"))

    ip_col = cfg.get("image_prompt_score_column", "pseudo_align_score")
    ip_status_col = cfg.get("image_prompt_status_column", "pseudo_align_status")
    ia_col = cfg.get("image_answer_score_column", "image_answer_score")
    ia_status_col = cfg.get("image_answer_status_column", "image_answer_status")
    pa_col = cfg.get("prompt_answer_score_column", "prompt_answer_score")
    pa_status_col = cfg.get("prompt_answer_status_column", "prompt_answer_status")

    out_col = cfg.get("score_column", "semantic_confidence_score")
    out_status_col = cfg.get("status_column", "semantic_confidence_status")
    out_level_col = cfg.get("level_column", "semantic_confidence_level")
    mode = str(cfg.get("mode", "min")).lower()
    high_threshold = float(cfg.get("high_threshold", 0.65))
    mid_threshold = float(cfg.get("mid_threshold", 0.35))

    df = read_csv_robust(input_csv, encoding=input_encoding)
    ip = _valid_score(df, ip_col, ip_status_col)
    ia = _valid_score(df, ia_col, ia_status_col)
    pa = _valid_score(df, pa_col, pa_status_col)

    if mode == "min":
        semantic = np.minimum.reduce([ip.to_numpy(), ia.to_numpy(), pa.to_numpy()])
    elif mode == "mean":
        semantic = np.stack([ip.to_numpy(), ia.to_numpy(), pa.to_numpy()], axis=0).mean(axis=0)
    elif mode == "weighted":
        weights: List[float] = [float(x) for x in cfg.get("weights", [0.4, 0.3, 0.3])]
        if len(weights) != 3:
            raise ValueError("weights must contain three values for IP, IA, PA.")
        w = np.asarray(weights, dtype=np.float64)
        w = w / max(w.sum(), 1e-8)
        semantic = w[0] * ip.to_numpy() + w[1] * ia.to_numpy() + w[2] * pa.to_numpy()
    else:
        raise ValueError("mode must be one of: min, mean, weighted")

    semantic = np.clip(semantic, 0.0, 1.0).astype(np.float32)
    df[out_col] = semantic
    df[out_status_col] = "ok"
    df[out_level_col] = "low"
    df.loc[df[out_col] >= mid_threshold, out_level_col] = "mid"
    df.loc[df[out_col] >= high_threshold, out_level_col] = "high"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=output_encoding)

    print("Done.")
    print(f"rows={len(df)}, output_csv={output_csv}")
    print(f"{out_col}: min={float(df[out_col].min()):.4f}, mean={float(df[out_col].mean()):.4f}, max={float(df[out_col].max()):.4f}")
    print(f"{out_level_col}: {df[out_level_col].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
