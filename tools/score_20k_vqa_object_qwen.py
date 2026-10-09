#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Score AIGIQA-20K image-prompt consistency with a VLM.

The default backend is Qwen2.5-VL. It returns three weak-supervision scores:
  - vqa_consistency_score: whether the image satisfies the prompt.
  - object_attribute_score: whether entities/attributes/counts/relations match.
  - vlm_answer_confidence: whether the generated answer faithfully describes the image.

The script writes CSV columns only; it does not change any training code.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml

try:
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLMModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLMModel
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None
except ImportError as e:
    torch = None
    AutoProcessor = None
    VLMModel = None
    process_vision_info = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


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


def find_image_path(image_root: Path, image_name: str, search_subdirs: List[str]) -> Optional[Path]:
    candidates = [image_root / image_name]
    candidates += [image_root / sub / image_name for sub in search_subdirs]
    stem = Path(image_name).stem
    if Path(image_name).suffix == "":
        candidates += [image_root / f"{stem}{ext}" for ext in IMAGE_EXTS]
        for sub in search_subdirs:
            candidates += [image_root / sub / f"{stem}{ext}" for ext in IMAGE_EXTS]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def clamp01(value, default: float = 0.5) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return float(np.clip(v, 0.0, 1.0))


def parse_json_object(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")
    return json.loads(match.group(0))


def build_instruction(prompt: str, answer: str) -> str:
    return (
        "You are evaluating image-text alignment for an AI-generated image. "
        "Use the image as the source of truth. Compare it with the prompt and "
        "the generated answer. Return JSON only, with numeric scores in [0, 1].\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Generated answer:\n{answer}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "image_only_answer": "one concise image-only caption",\n'
        '  "vqa_consistency_score": number,\n'
        '  "object_attribute_score": number,\n'
        '  "answer_confidence": number,\n'
        '  "main_mismatch": "short reason"\n'
        "}\n\n"
        "Scoring rules:\n"
        "- vqa_consistency_score: how well the image satisfies the whole prompt.\n"
        "- object_attribute_score: objects, attributes, count, spatial relation, and style match.\n"
        "- answer_confidence: the answer describes visible image content and avoids hallucinations.\n"
        "- image_only_answer: describe only visible content; do not copy unsupported prompt details.\n"
        "Be strict about wrong objects, colors, counts, text, and spatial relations."
    )


def build_model(cfg: Dict):
    if torch is None or AutoProcessor is None or VLMModel is None:
        raise ImportError(
            "Qwen VLM scoring needs torch, transformers, accelerate, and qwen-vl-utils. "
            "Install with: pip install torch transformers accelerate qwen-vl-utils"
        ) from _IMPORT_ERROR
    model_name = cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    torch_dtype = cfg.get("torch_dtype", "auto")
    device_map = cfg.get("device_map", "auto")
    token = cfg.get("hf_token", None) or os.environ.get("HF_TOKEN", None)
    local_files_only = bool(cfg.get("local_files_only", False))
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    if bool(cfg.get("disable_xet", True)):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"[Model] VLM model={model_name}, dtype={torch_dtype}, device_map={device_map}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        token=token,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = VLMModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=token,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, processor


def score_one(model, processor, image_path: Path, prompt: str, answer: str, max_new_tokens: int) -> Dict:
    with torch.no_grad():
        instruction = build_instruction(prompt, answer)
        if process_vision_info is None:
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": instruction}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    parsed = parse_json_object(output_text)
    return {
        "raw": output_text,
        "image_only_answer": str(parsed.get("image_only_answer", ""))[:1000],
        "vqa_consistency_score": clamp01(parsed.get("vqa_consistency_score"), 0.5),
        "object_attribute_score": clamp01(parsed.get("object_attribute_score"), 0.5),
        "answer_confidence": clamp01(parsed.get("answer_confidence"), 0.5),
        "main_mismatch": str(parsed.get("main_mismatch", ""))[:500],
    }


def main():
    parser = argparse.ArgumentParser(description="Score 20K VQA/object consistency with Qwen2.5-VL.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("vqa_object_score", full_cfg)

    input_csv = Path(cfg["input_csv"])
    output_csv = Path(cfg["output_csv"])
    image_root = Path(cfg["image_root"])
    search_subdirs = cfg.get("search_subdirs", ["train", "val", "test"])
    input_encoding = str(cfg.get("input_encoding", "auto"))
    output_encoding = str(cfg.get("output_encoding", "utf-8-sig"))

    image_col = cfg.get("image_name_column", "name")
    prompt_col = cfg.get("prompt_column", "prompt")
    answer_col = cfg.get("answer_column", "answer")
    answer_status_col = cfg.get("answer_status_column", "answer_status")
    only_ok_answer = bool(cfg.get("only_ok_answer", True))

    vqa_col = cfg.get("vqa_score_column", "vqa_consistency_score")
    obj_col = cfg.get("object_score_column", "object_attribute_score")
    answer_conf_col = cfg.get("answer_confidence_column", "vlm_answer_confidence")
    image_only_answer_col = cfg.get("image_only_answer_column", "vlm_image_answer")
    mismatch_col = cfg.get("mismatch_column", "vqa_object_mismatch")
    raw_col = cfg.get("raw_column", "vqa_object_raw")
    status_col = cfg.get("status_column", "vqa_object_status")
    error_col = cfg.get("error_column", "vqa_object_error")

    max_new_tokens = int(cfg.get("max_new_tokens", 160))
    save_every = int(cfg.get("save_every", 100))

    if output_csv.exists():
        print(f"[Resume] load output_csv={output_csv}")
        df = read_csv_robust(output_csv, input_encoding)
    else:
        print(f"[Start] load input_csv={input_csv}")
        df = read_csv_robust(input_csv, input_encoding)

    for col in [vqa_col, obj_col, answer_conf_col]:
        if col not in df.columns:
            df[col] = np.nan
    for col in [mismatch_col, raw_col, status_col, error_col]:
        if col not in df.columns:
            df[col] = ""
    if image_only_answer_col not in df.columns:
        df[image_only_answer_col] = ""
    for col in [mismatch_col, raw_col, status_col, error_col, image_only_answer_col]:
        df[col] = df[col].astype("object").where(pd.notna(df[col]), "")
    for col in [vqa_col, obj_col, answer_conf_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [image_col, prompt_col, answer_col]:
        if col not in df.columns:
            raise KeyError(f"CSV missing required column: {col}")

    todo_indices = []
    for idx, row in df.iterrows():
        if (not args.overwrite) and str(row.get(status_col, "")).strip().lower() == "ok" and pd.notna(row.get(vqa_col, np.nan)):
            continue
        todo_indices.append(idx)
        if args.limit is not None and len(todo_indices) >= args.limit:
            break
    print(f"[Data] rows={len(df)}, todo={len(todo_indices)}")

    if args.dry_run:
        for idx in todo_indices[:20]:
            image_path = find_image_path(image_root, str(df.at[idx, image_col]), search_subdirs)
            print(f"[DRY] idx={idx} image={image_path} prompt={str(df.at[idx, prompt_col])[:100]}")
        return

    model, processor = build_model(cfg)
    processed = skipped = errors = 0

    for n, idx in enumerate(tqdm(todo_indices, desc="Scoring VQA/object", dynamic_ncols=True), start=1):
        prompt = "" if pd.isna(df.at[idx, prompt_col]) else str(df.at[idx, prompt_col]).strip()
        answer = "" if pd.isna(df.at[idx, answer_col]) else str(df.at[idx, answer_col]).strip()
        image_name = "" if pd.isna(df.at[idx, image_col]) else str(df.at[idx, image_col]).strip()
        image_path = find_image_path(image_root, image_name, search_subdirs)
        if image_path is None:
            df.at[idx, status_col] = "missing_image"
            df.at[idx, error_col] = f"missing image: {image_name}"
            skipped += 1
            continue
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
        try:
            scored = score_one(model, processor, image_path, prompt, answer, max_new_tokens)
            df.at[idx, vqa_col] = scored["vqa_consistency_score"]
            df.at[idx, obj_col] = scored["object_attribute_score"]
            df.at[idx, answer_conf_col] = scored["answer_confidence"]
            df.at[idx, image_only_answer_col] = scored["image_only_answer"]
            df.at[idx, mismatch_col] = scored["main_mismatch"]
            df.at[idx, raw_col] = scored["raw"]
            df.at[idx, status_col] = "ok"
            df.at[idx, error_col] = ""
            processed += 1
        except Exception as exc:
            df.at[idx, status_col] = "error"
            df.at[idx, error_col] = repr(exc)
            errors += 1
        if n % save_every == 0:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding=output_encoding)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=output_encoding)
    print("Done.")
    print(f"processed={processed}, skipped={skipped}, errors={errors}, output_csv={output_csv}")


if __name__ == "__main__":
    main()
