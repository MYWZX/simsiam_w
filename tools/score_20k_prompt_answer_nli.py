#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Score AIGIQA-20K prompt-answer consistency with an NLI model.

The generated answer is treated as a textual description of the image. A high
entailment score means the answer supports the prompt requirements, while a high
contradiction score means the answer conflicts with them. The output score is a
weak reliability signal for prompt-answer SimSiam, not a human alignment label.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


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


def _find_label_indices(id2label: Dict[int, str]) -> Tuple[int, int, int | None]:
    labels = {int(i): str(v).lower() for i, v in id2label.items()}
    entail = next((i for i, v in labels.items() if "entail" in v), None)
    contra = next((i for i, v in labels.items() if "contrad" in v), None)
    neutral = next((i for i, v in labels.items() if "neutral" in v), None)
    if entail is not None and contra is not None:
        return entail, contra, neutral
    # Common MNLI order for several DeBERTa/RoBERTa checkpoints.
    return 2, 0, 1 if len(labels) >= 3 else None


def build_model(cfg: Dict):
    if torch is None:
        raise ImportError(
            "torch and transformers are required. Install with: pip install torch transformers"
        ) from _IMPORT_ERROR
    model_name = cfg.get("model_name", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    token = cfg.get("hf_token", None) or os.environ.get("HF_TOKEN", None)
    local_files_only = bool(cfg.get("local_files_only", False))
    if bool(cfg.get("disable_xet", True)):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"[Model] NLI model={model_name}, device={device}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=token,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    entail_idx, contra_idx, neutral_idx = _find_label_indices(model.config.id2label)
    print(
        f"[Labels] entailment={entail_idx}, contradiction={contra_idx}, "
        f"neutral={neutral_idx}, id2label={model.config.id2label}"
    )
    return model, tokenizer, device, entail_idx, contra_idx, neutral_idx


def score_batch(
    model,
    tokenizer,
    device,
    prompts: List[str],
    answers: List[str],
    entail_idx: int,
    contra_idx: int,
    neutral_idx: int | None,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        encoded = tokenizer(
            prompts,
            answers,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        entail = probs[:, entail_idx]
        contra = probs[:, contra_idx]
        neutral = probs[:, neutral_idx] if neutral_idx is not None else torch.zeros_like(entail)
        score = (0.5 + 0.5 * (entail - contra)).clamp(0.0, 1.0)
        confidence = torch.maximum(torch.maximum(entail, contra), neutral).clamp(0.0, 1.0)
        return (
            score.detach().cpu().float().numpy(),
            confidence.detach().cpu().float().numpy(),
            entail.detach().cpu().float().numpy(),
            contra.detach().cpu().float().numpy(),
            neutral.detach().cpu().float().numpy(),
        )


def main():
    parser = argparse.ArgumentParser(description="Score 20K prompt-answer consistency with an NLI model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("prompt_answer_nli_score", full_cfg)

    input_csv = Path(cfg["input_csv"])
    output_csv = Path(cfg["output_csv"])
    input_encoding = str(cfg.get("input_encoding", "auto"))
    output_encoding = str(cfg.get("output_encoding", "utf-8-sig"))
    prompt_col = cfg.get("prompt_column", "prompt")
    answer_col = cfg.get("answer_column", "answer")
    answer_status_col = cfg.get("answer_status_column", "answer_status")
    only_ok_answer = bool(cfg.get("only_ok_answer", True))

    score_col = cfg.get("score_column", "prompt_answer_nli_score")
    confidence_col = cfg.get("confidence_column", "prompt_answer_nli_confidence")
    entail_col = cfg.get("entailment_column", "prompt_answer_nli_entailment")
    contra_col = cfg.get("contradiction_column", "prompt_answer_nli_contradiction")
    neutral_col = cfg.get("neutral_column", "prompt_answer_nli_neutral")
    status_col = cfg.get("status_column", "prompt_answer_nli_status")
    error_col = cfg.get("error_column", "prompt_answer_nli_error")

    batch_size = int(cfg.get("batch_size", 32))
    save_every = int(cfg.get("save_every", 1000))
    max_length = int(cfg.get("max_length", 256))

    if output_csv.exists():
        print(f"[Resume] load output_csv={output_csv}")
        df = read_csv_robust(output_csv, input_encoding)
    else:
        print(f"[Start] load input_csv={input_csv}")
        df = read_csv_robust(input_csv, input_encoding)

    for col in [score_col, confidence_col, entail_col, contra_col, neutral_col]:
        if col not in df.columns:
            df[col] = np.nan
    for col in [status_col, error_col]:
        if col not in df.columns:
            df[col] = ""
    for col in [status_col, error_col]:
        df[col] = df[col].astype("object").where(pd.notna(df[col]), "")
    for col in [score_col, confidence_col, entail_col, contra_col, neutral_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [prompt_col, answer_col]:
        if col not in df.columns:
            raise KeyError(f"CSV missing required column: {col}")

    todo_indices = []
    for idx, row in df.iterrows():
        if (not args.overwrite) and str(row.get(status_col, "")).strip().lower() == "ok" and pd.notna(row.get(score_col, np.nan)):
            continue
        todo_indices.append(idx)
        if args.limit is not None and len(todo_indices) >= args.limit:
            break
    print(f"[Data] rows={len(df)}, todo={len(todo_indices)}")

    if args.dry_run:
        for idx in todo_indices[:20]:
            print(f"[DRY] idx={idx} prompt={str(df.at[idx, prompt_col])[:100]} | answer={str(df.at[idx, answer_col])[:100]}")
        return

    model, tokenizer, device, entail_idx, contra_idx, neutral_idx = build_model(cfg)

    pending_indices: List[int] = []
    pending_prompts: List[str] = []
    pending_answers: List[str] = []
    processed = skipped = errors = 0

    def flush_batch():
        nonlocal processed, errors, pending_indices, pending_prompts, pending_answers
        if not pending_indices:
            return
        try:
            scores, confs, entails, contras, neutrals = score_batch(
                model, tokenizer, device, pending_prompts, pending_answers,
                entail_idx, contra_idx, neutral_idx, max_length,
            )
            for i, s, c, e, co, n in zip(pending_indices, scores, confs, entails, contras, neutrals):
                df.at[i, score_col] = float(s)
                df.at[i, confidence_col] = float(c)
                df.at[i, entail_col] = float(e)
                df.at[i, contra_col] = float(co)
                df.at[i, neutral_col] = float(n)
                df.at[i, status_col] = "ok"
                df.at[i, error_col] = ""
                processed += 1
        except Exception as exc:
            msg = repr(exc)
            for i in pending_indices:
                df.at[i, status_col] = "error"
                df.at[i, error_col] = msg
                errors += 1
        finally:
            pending_indices = []
            pending_prompts = []
            pending_answers = []

    for n, idx in enumerate(tqdm(todo_indices, desc="Scoring NLI", dynamic_ncols=True), start=1):
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
        if only_ok_answer and answer_status_col in df.columns:
            if str(df.at[idx, answer_status_col]).strip().lower() != "ok":
                df.at[idx, status_col] = "bad_answer_status"
                df.at[idx, error_col] = f"{answer_status_col}={df.at[idx, answer_status_col]}"
                skipped += 1
                continue

        pending_indices.append(idx)
        pending_prompts.append(prompt)
        pending_answers.append(answer)
        if len(pending_indices) >= batch_size:
            flush_batch()
        if n % save_every == 0:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding=output_encoding)

    flush_batch()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=output_encoding)
    ok = df[status_col].astype(str).str.lower().eq("ok")
    print("Done.")
    print(f"processed={processed}, skipped={skipped}, errors={errors}, output_csv={output_csv}")
    if ok.any():
        vals = pd.to_numeric(df.loc[ok, score_col], errors="coerce")
        print(f"{score_col}: min={vals.min():.4f}, mean={vals.mean():.4f}, max={vals.max():.4f}")


if __name__ == "__main__":
    main()
