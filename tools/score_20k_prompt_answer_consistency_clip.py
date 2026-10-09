#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate pseudo prompt-answer consistency scores for AIGIQA-20K using CLIP text embeddings.

Input CSV example:
    data/AIGIQA-20K/info_merge_with_answer_pseudo_align.csv
Required columns:
    prompt, answer
Output CSV:
    keeps all original columns and appends:
        prompt_answer_raw     : raw CLIP text-text cosine similarity
        prompt_answer_score   : normalized score in [0, 1]
        prompt_answer_status  : ok / empty_prompt / empty_answer / bad_answer_status / error
        prompt_answer_error   : error message

This script is weak supervision, not human alignment MOS. It is used to decide
which prompt-answer pairs are reliable enough for positive SimSiam pretraining.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

try:
    import torch
    import open_clip
except ImportError as e:
    torch = None
    open_clip = None
    _OPEN_CLIP_IMPORT_ERROR = e
else:
    _OPEN_CLIP_IMPORT_ERROR = None


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: Path, encoding: str = "auto") -> pd.DataFrame:
    if encoding and str(encoding).lower() not in ["auto", ""]:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[CSV] loaded {path} with encoding={enc}")
            return df
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with encodings {encodings}. Last error: {last_err}")


def normalize_scores(raw: pd.Series, mode: str = "percentile", q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    raw_numeric = pd.to_numeric(raw, errors="coerce").astype("float64")
    valid_mask = raw_numeric.notna().to_numpy()
    out = np.full(len(raw_numeric), np.nan, dtype=np.float32)
    if not valid_mask.any():
        return pd.Series(out, index=raw.index, dtype="float32")

    valid_values = raw_numeric.to_numpy(dtype=np.float64)[valid_mask]
    if mode == "cos01":
        values = ((valid_values + 1.0) / 2.0).clip(0.0, 1.0)
    elif mode == "minmax":
        lo, hi = float(np.nanmin(valid_values)), float(np.nanmax(valid_values))
        values = np.full_like(valid_values, 0.5) if abs(hi - lo) < 1e-8 else ((valid_values - lo) / (hi - lo)).clip(0.0, 1.0)
    elif mode == "percentile":
        lo, hi = float(np.nanquantile(valid_values, q_low)), float(np.nanquantile(valid_values, q_high))
        values = np.full_like(valid_values, 0.5) if abs(hi - lo) < 1e-8 else ((valid_values - lo) / (hi - lo)).clip(0.0, 1.0)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    out[valid_mask] = values.astype(np.float32)
    return pd.Series(out, index=raw.index, dtype="float32")


def build_model(cfg: Dict):
    if torch is None or open_clip is None:
        raise ImportError("torch and open_clip are required. Install with: pip install torch open_clip_torch") from _OPEN_CLIP_IMPORT_ERROR
    model_name = cfg.get("clip_model_name", "ViT-B-16")
    pretrained = cfg.get("clip_pretrained", "openai")
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"[Model] open_clip model={model_name}, pretrained={pretrained}, device={device}")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, device


def encode_text_pair_batch(model, tokenizer, device: torch.device, prompts: List[str], answers: List[str]) -> np.ndarray:
    with torch.no_grad():
        prompt_tokens = tokenizer(prompts).to(device, non_blocking=True)
        answer_tokens = tokenizer(answers).to(device, non_blocking=True)

        prompt_features = model.encode_text(prompt_tokens)
        answer_features = model.encode_text(answer_tokens)
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        answer_features = answer_features / answer_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        scores = (prompt_features * answer_features).sum(dim=-1)
        return scores.detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N missing rows.")
    parser.add_argument("--dry-run", action="store_true", help="Only check text fields; do not score.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute rows even if existing score is ok.")
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("prompt_answer_consistency_score", full_cfg)

    input_csv = Path(cfg["input_csv"])
    output_csv = Path(cfg["output_csv"])
    prompt_col = cfg.get("prompt_column", "prompt")
    answer_col = cfg.get("answer_column", "answer")
    answer_status_col = cfg.get("answer_status_column", "answer_status")
    only_ok_answer = bool(cfg.get("only_ok_answer", True))

    raw_col = cfg.get("raw_score_column", "prompt_answer_raw")
    score_col = cfg.get("score_column", "prompt_answer_score")
    status_col = cfg.get("status_column", "prompt_answer_status")
    error_col = cfg.get("error_column", "prompt_answer_error")

    batch_size = int(cfg.get("batch_size", 256))
    save_every = int(cfg.get("save_every", 1000))
    normalize_mode = cfg.get("normalize_mode", "percentile")
    q_low = float(cfg.get("normalize_q_low", 0.05))
    q_high = float(cfg.get("normalize_q_high", 0.95))
    input_encoding = str(cfg.get("input_encoding", "auto"))
    output_encoding = str(cfg.get("output_encoding", "utf-8-sig"))

    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    if output_csv.exists():
        print(f"[Resume] load output_csv={output_csv}")
        df = read_csv_robust(output_csv, encoding=input_encoding)
    else:
        print(f"[Start] load input_csv={input_csv}")
        df = read_csv_robust(input_csv, encoding=input_encoding)

    for col in [raw_col, score_col, status_col, error_col]:
        if col not in df.columns:
            df[col] = np.nan if col in [raw_col, score_col] else ""
    for col in [status_col, error_col]:
        df[col] = df[col].astype("object").where(pd.notna(df[col]), "")
    for col in [raw_col, score_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [prompt_col, answer_col]:
        if col not in df.columns:
            raise KeyError(f"CSV missing required column: {col}. Existing columns: {list(df.columns)}")

    todo_indices = []
    for idx, row in df.iterrows():
        if (not args.overwrite) and str(row.get(status_col, "")) == "ok" and pd.notna(row.get(raw_col, np.nan)):
            continue
        todo_indices.append(idx)
        if args.limit is not None and len(todo_indices) >= args.limit:
            break

    print(f"[Data] rows={len(df)}, todo={len(todo_indices)}, input={input_csv}")
    if args.dry_run:
        for idx in todo_indices[: min(len(todo_indices), 20)]:
            print(f"[DRY] idx={idx} prompt={str(df.at[idx, prompt_col])[:90]} | answer={str(df.at[idx, answer_col])[:90]}")
        print("Dry-run done. No score is written.")
        return

    model, tokenizer, device = build_model(cfg)
    pending_indices: List[int] = []
    pending_prompts: List[str] = []
    pending_answers: List[str] = []
    processed = 0
    skipped = 0
    errors = 0

    def flush_batch():
        nonlocal processed, errors, pending_indices, pending_prompts, pending_answers
        if not pending_indices:
            return
        try:
            scores = encode_text_pair_batch(model, tokenizer, device, pending_prompts, pending_answers)
            for i, score in zip(pending_indices, scores):
                df.at[i, raw_col] = float(score)
                df.at[i, status_col] = "ok"
                df.at[i, error_col] = ""
                processed += 1
        except Exception as e:
            msg = repr(e)
            for i in pending_indices:
                df.at[i, status_col] = "error"
                df.at[i, error_col] = msg
                errors += 1
        finally:
            pending_indices = []
            pending_prompts = []
            pending_answers = []

    pbar = tqdm(todo_indices, desc="Scoring P-A", dynamic_ncols=True)
    for n, idx in enumerate(pbar, start=1):
        prompt = "" if pd.isna(df.at[idx, prompt_col]) else str(df.at[idx, prompt_col]).strip()
        answer = "" if pd.isna(df.at[idx, answer_col]) else str(df.at[idx, answer_col]).strip()
        if not prompt:
            df.at[idx, status_col] = "empty_prompt"
            df.at[idx, error_col] = "empty prompt"
            skipped += 1
            continue
        if not answer:
            df.at[idx, status_col] = "empty_answer"
            df.at[idx, error_col] = "empty answer"
            skipped += 1
            continue
        if only_ok_answer and answer_status_col in df.columns and str(df.at[idx, answer_status_col]).strip().lower() != "ok":
            df.at[idx, status_col] = "bad_answer_status"
            df.at[idx, error_col] = f"{answer_status_col} is not ok"
            skipped += 1
            continue

        pending_indices.append(idx)
        pending_prompts.append(prompt)
        pending_answers.append(answer)

        if len(pending_indices) >= batch_size:
            flush_batch()
        if n % save_every == 0:
            flush_batch()
            df[score_col] = normalize_scores(df[raw_col], normalize_mode, q_low, q_high)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding=output_encoding)
            pbar.set_postfix(processed=processed, skipped=skipped, errors=errors)

    flush_batch()
    df[score_col] = normalize_scores(df[raw_col], normalize_mode, q_low, q_high)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=output_encoding)

    ok_count = int((df[status_col] == "ok").sum())
    valid_scores = pd.to_numeric(df.loc[df[status_col] == "ok", score_col], errors="coerce").dropna()
    print("Done.")
    print(f"processed_now={processed}, ok_total={ok_count}, skipped={skipped}, errors={errors}")
    print(f"output_csv={output_csv}")
    if len(valid_scores) > 0:
        print(f"{score_col}: min={valid_scores.min():.4f}, mean={valid_scores.mean():.4f}, max={valid_scores.max():.4f}")


if __name__ == "__main__":
    main()
