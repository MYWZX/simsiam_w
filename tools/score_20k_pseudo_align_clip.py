#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate pseudo image-prompt alignment scores for AIGIQA-20K using CLIPScore.

Input CSV example:
    data/AIGIQA-20K/info_merge_with_answer_local.csv
Required columns:
    name, prompt
Output CSV:
    keeps all original columns and appends:
        pseudo_align_raw     : raw CLIP cosine similarity
        pseudo_align_score   : normalized score in [0, 1]
        pseudo_align_status  : ok / missing_image / empty_prompt / error
        pseudo_align_error   : error message

This script does NOT use answer. It scores image-prompt consistency only.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import yaml

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required. Install with: pip install open_clip_torch"
    ) from e


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: Path, encoding: str = "auto") -> pd.DataFrame:
    """Read CSV robustly.

    Some generated CSV files may be saved as GBK/ANSI/GB18030 by Excel or pandas
    on Windows, so plain UTF-8 reading can fail. This function first honors an
    explicit encoding, otherwise tries common encodings in order.
    """
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
            continue
    raise UnicodeDecodeError(
        last_err.encoding, last_err.object, last_err.start, last_err.end,
        f"Failed to read CSV with encodings {encodings}. Last error: {last_err.reason}"
    )


def find_image_path(image_root: Path, image_name: str, search_subdirs: List[str]) -> Optional[Path]:
    """Find image by exact name in root and configured subdirs."""
    candidates = [image_root / image_name]
    candidates += [image_root / sub / image_name for sub in search_subdirs]

    # If CSV name lacks extension, try common extensions.
    stem = Path(image_name).stem
    if Path(image_name).suffix == "":
        candidates += [image_root / f"{stem}{ext}" for ext in IMAGE_EXTS]
        for sub in search_subdirs:
            candidates += [image_root / sub / f"{stem}{ext}" for ext in IMAGE_EXTS]

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def safe_open_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def normalize_scores(raw: pd.Series, mode: str = "percentile", q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """Normalize raw cosine scores to [0, 1].

    This implementation avoids assigning float64 arrays into a float32 pandas
    Series, which can raise LossySetitemError / TypeError in newer pandas
    versions. It computes with numpy and returns a fresh float32 Series.
    """
    raw_numeric = pd.to_numeric(raw, errors="coerce").astype("float64")
    valid_mask = raw_numeric.notna().to_numpy()

    out = np.full(len(raw_numeric), np.nan, dtype=np.float32)
    if not valid_mask.any():
        return pd.Series(out, index=raw.index, dtype="float32")

    valid_values = raw_numeric.to_numpy(dtype=np.float64)[valid_mask]

    if mode == "cos01":
        values = ((valid_values + 1.0) / 2.0).clip(0.0, 1.0)
        out[valid_mask] = values.astype(np.float32)
        return pd.Series(out, index=raw.index, dtype="float32")

    if mode == "minmax":
        lo, hi = float(np.nanmin(valid_values)), float(np.nanmax(valid_values))
    elif mode == "percentile":
        lo, hi = float(np.nanquantile(valid_values, q_low)), float(np.nanquantile(valid_values, q_high))
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    if abs(hi - lo) < 1e-8:
        values = np.full_like(valid_values, 0.5, dtype=np.float64)
    else:
        values = ((valid_values - lo) / (hi - lo)).clip(0.0, 1.0)

    out[valid_mask] = values.astype(np.float32)
    return pd.Series(out, index=raw.index, dtype="float32")


@torch.no_grad()
def encode_batch(
    model,
    tokenizer,
    preprocess,
    device: torch.device,
    image_paths: List[Path],
    prompts: List[str],
) -> np.ndarray:
    images = []
    for p in image_paths:
        images.append(preprocess(safe_open_image(p)))
    image_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)

    text_tensor = tokenizer(prompts).to(device, non_blocking=True)

    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tensor)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    scores = (image_features * text_features).sum(dim=-1)
    return scores.detach().float().cpu().numpy()


def build_model(cfg: Dict):
    model_name = cfg.get("clip_model_name", "ViT-B-16")
    pretrained = cfg.get("clip_pretrained", "openai")
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    print(f"[Model] open_clip model={model_name}, pretrained={pretrained}, device={device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, preprocess, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N missing rows.")
    parser.add_argument("--dry-run", action="store_true", help="Only check paths/prompts; do not score.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute rows even if existing score is ok.")
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("pseudo_align_score", full_cfg)

    input_csv = Path(cfg["input_csv"])
    output_csv = Path(cfg["output_csv"])
    image_root = Path(cfg["image_root"])
    search_subdirs = cfg.get("search_subdirs", ["train", "val", "test"])

    image_col = cfg.get("image_name_column", "name")
    prompt_col = cfg.get("prompt_column", "prompt")
    raw_col = cfg.get("raw_score_column", "pseudo_align_raw")
    score_col = cfg.get("score_column", "pseudo_align_score")
    status_col = cfg.get("status_column", "pseudo_align_status")
    error_col = cfg.get("error_column", "pseudo_align_error")

    batch_size = int(cfg.get("batch_size", 64))
    save_every = int(cfg.get("save_every", 500))
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

    required_cols = [image_col, prompt_col]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"CSV missing required column: {col}. Existing columns: {list(df.columns)}")

    # Select rows to process.
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
            image_name = str(df.at[idx, image_col])
            prompt = str(df.at[idx, prompt_col]) if pd.notna(df.at[idx, prompt_col]) else ""
            img_path = find_image_path(image_root, image_name, search_subdirs)
            print(f"[DRY] idx={idx} name={image_name} path={img_path} prompt={prompt[:120]}")
        print("Dry-run done. No score is written.")
        return

    model, tokenizer, preprocess, device = build_model(cfg)

    pending_indices: List[int] = []
    pending_paths: List[Path] = []
    pending_prompts: List[str] = []

    processed = 0
    missing = 0
    empty_prompt = 0
    errors = 0

    def flush_batch():
        nonlocal processed, errors, pending_indices, pending_paths, pending_prompts
        if not pending_indices:
            return
        try:
            scores = encode_batch(model, tokenizer, preprocess, device, pending_paths, pending_prompts)
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
            pending_paths = []
            pending_prompts = []

    pbar = tqdm(todo_indices, desc="Scoring", dynamic_ncols=True)
    for n, idx in enumerate(pbar, start=1):
        image_name = str(df.at[idx, image_col])
        prompt = df.at[idx, prompt_col]
        prompt = "" if pd.isna(prompt) else str(prompt).strip()

        if not prompt:
            df.at[idx, status_col] = "empty_prompt"
            df.at[idx, error_col] = "empty prompt"
            empty_prompt += 1
            continue

        img_path = find_image_path(image_root, image_name, search_subdirs)
        if img_path is None:
            df.at[idx, status_col] = "missing_image"
            df.at[idx, error_col] = f"image not found: {image_name}"
            missing += 1
            continue

        pending_indices.append(idx)
        pending_paths.append(img_path)
        pending_prompts.append(prompt)

        if len(pending_indices) >= batch_size:
            flush_batch()

        if n % save_every == 0:
            flush_batch()
            # Normalize existing raw values and save checkpoint.
            df[score_col] = normalize_scores(df[raw_col], normalize_mode, q_low, q_high)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding=output_encoding)
            pbar.set_postfix(processed=processed, missing=missing, empty=empty_prompt, errors=errors)

    flush_batch()
    df[score_col] = normalize_scores(df[raw_col], normalize_mode, q_low, q_high)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=output_encoding)

    ok_count = int((df[status_col] == "ok").sum())
    print("Done.")
    print(f"processed_now={processed}, ok_total={ok_count}, missing={missing}, empty_prompt={empty_prompt}, errors={errors}")
    print(f"output_csv={output_csv}")
    valid_scores = pd.to_numeric(df.loc[df[status_col] == "ok", score_col], errors="coerce").dropna()
    if len(valid_scores) > 0:
        print(
            f"{score_col}: min={valid_scores.min():.4f}, "
            f"mean={valid_scores.mean():.4f}, max={valid_scores.max():.4f}"
        )


if __name__ == "__main__":
    main()
