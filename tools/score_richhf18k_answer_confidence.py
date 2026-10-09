import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score RichHF-18K generated answers with CLIP image-text similarity."
    )
    parser.add_argument(
        "--config",
        default="configs/richhf18k_answer_confidence.yaml",
        help="Path to the scoring config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score at most N pending rows per split for a smoke test. Calibration is skipped.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("answer_confidence", cfg)


def atomic_save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def is_valid_answer(row):
    answer = row.get("answer", "")
    status = str(row.get("answer_status", "ok")).strip().lower()
    return status == "ok" and isinstance(answer, str) and bool(answer.strip())


def resolve_image_path(processed_dir, image_path):
    path = Path(str(image_path))
    if not path.is_absolute():
        path = processed_dir / path
    return path


def initialize_columns(df):
    numeric_defaults = {
        "answer_clip_raw": np.nan,
        "answer_confidence": np.nan,
        "answer_weight": 0.0,
    }
    text_defaults = {
        "answer_clip_status": "pending",
        "answer_confidence_level": "unscored",
        "answer_confidence_error": "",
        "answer_clip_model": "",
    }
    for column, default in numeric_defaults.items():
        if column in df.columns:
            # Pandas may infer partially filled checkpoint columns as Arrow
            # strings. Force writable NumPy floats before resuming scoring.
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
        else:
            df[column] = np.full(len(df), default, dtype=np.float64)
    for column, default in text_defaults.items():
        if column in df.columns:
            df[column] = df[column].fillna("").astype(object)
        else:
            df[column] = pd.Series([default] * len(df), dtype=object)
    return df


def load_split(input_path, output_path, overwrite):
    source_path = input_path
    if output_path.exists() and not overwrite:
        source_path = output_path
    if not source_path.exists():
        raise FileNotFoundError(f"CSV not found: {source_path}")
    df = pd.read_csv(source_path, keep_default_na=False)
    if "image_path" not in df.columns or "answer" not in df.columns:
        raise KeyError(f"{source_path} must contain image_path and answer columns")
    return initialize_columns(df)


@torch.inference_mode()
def encode_batch(model, preprocess, tokenizer, image_paths, answers, device):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(preprocess(image.convert("RGB")))
    image_tensor = torch.stack(images).to(device, non_blocking=True)
    text_tensor = tokenizer(answers).to(device, non_blocking=True)

    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        similarities = (image_features * text_features).sum(dim=-1)
    return similarities.cpu().numpy()


def score_split(
    df,
    split,
    output_path,
    processed_dir,
    model,
    preprocess,
    tokenizer,
    model_tag,
    device,
    batch_size,
    save_every,
    limit,
):
    eligible = df.apply(is_valid_answer, axis=1)
    invalid = ~eligible
    df.loc[invalid, "answer_clip_status"] = "invalid_answer"
    df.loc[invalid, "answer_confidence_level"] = "invalid"
    df.loc[invalid, "answer_weight"] = 0.0

    pending = df.index[eligible & (df["answer_clip_status"].astype(str) != "ok")].tolist()
    if limit is not None:
        pending = pending[:limit]
    print(f"[{split}] rows={len(df)}, eligible={int(eligible.sum())}, pending={len(pending)}")

    completed_since_save = 0
    progress = tqdm(range(0, len(pending), batch_size), desc=f"CLIP {split}")
    for start in progress:
        batch_indices = pending[start : start + batch_size]
        valid_indices = []
        image_paths = []
        answers = []

        for index in batch_indices:
            image_path = resolve_image_path(processed_dir, df.at[index, "image_path"])
            if not image_path.exists():
                df.at[index, "answer_clip_status"] = "error"
                df.at[index, "answer_confidence_error"] = f"image not found: {image_path}"
                continue
            valid_indices.append(index)
            image_paths.append(image_path)
            answers.append(str(df.at[index, "answer"]).strip())

        if valid_indices:
            try:
                scores = encode_batch(
                    model, preprocess, tokenizer, image_paths, answers, device
                )
                for index, score in zip(valid_indices, scores):
                    df.at[index, "answer_clip_raw"] = float(score)
                    df.at[index, "answer_clip_status"] = "ok"
                    df.at[index, "answer_confidence_error"] = ""
                    df.at[index, "answer_clip_model"] = model_tag
            except Exception as exc:
                batch_error = f"{type(exc).__name__}: {exc}"
                for index, image_path, answer in zip(
                    valid_indices, image_paths, answers
                ):
                    try:
                        score = encode_batch(
                            model,
                            preprocess,
                            tokenizer,
                            [image_path],
                            [answer],
                            device,
                        )[0]
                        df.at[index, "answer_clip_raw"] = float(score)
                        df.at[index, "answer_clip_status"] = "ok"
                        df.at[index, "answer_confidence_error"] = ""
                        df.at[index, "answer_clip_model"] = model_tag
                    except Exception as item_exc:
                        item_error = f"{type(item_exc).__name__}: {item_exc}"
                        df.at[index, "answer_clip_status"] = "error"
                        df.at[index, "answer_confidence_error"] = (
                            f"batch={batch_error}; item={item_error}"
                        )

        completed_since_save += len(batch_indices)
        if completed_since_save >= save_every:
            atomic_save_csv(df, output_path)
            completed_since_save = 0

    atomic_save_csv(df, output_path)
    return df


def fit_train_calibration(train_df, q_low, q_high):
    valid = train_df.loc[
        train_df["answer_clip_status"].astype(str) == "ok", "answer_clip_raw"
    ]
    scores = pd.to_numeric(valid, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(scores) == 0:
        raise RuntimeError("No valid training CLIP scores are available for calibration")
    lo = float(np.quantile(scores, q_low))
    hi = float(np.quantile(scores, q_high))
    if hi <= lo:
        raise RuntimeError(f"Invalid train calibration range: lo={lo}, hi={hi}")
    return lo, hi, len(scores)


def validate_scoring_coverage(frames, minimum_success_rate):
    for split, df in frames.items():
        eligible = df.apply(is_valid_answer, axis=1)
        ok = eligible & (df["answer_clip_status"].astype(str) == "ok")
        eligible_count = int(eligible.sum())
        ok_count = int(ok.sum())
        success_rate = ok_count / max(eligible_count, 1)
        print(
            f"[{split}] scoring coverage={ok_count}/{eligible_count} "
            f"({success_rate:.2%})"
        )
        if success_rate < minimum_success_rate:
            failures = df.loc[
                eligible & ~ok,
                ["image_path", "answer_clip_status", "answer_confidence_error"],
            ].head(3)
            raise RuntimeError(
                f"{split} CLIP scoring coverage {success_rate:.2%} is below "
                f"the required {minimum_success_rate:.2%}. Example failures:\n"
                f"{failures.to_string(index=False)}"
            )


def apply_calibration(df, lo, hi, medium_threshold, high_threshold):
    raw = pd.to_numeric(df["answer_clip_raw"], errors="coerce")
    ok = (df["answer_clip_status"].astype(str) == "ok") & raw.notna()
    confidence = ((raw - lo) / (hi - lo)).clip(0.0, 1.0)

    df.loc[:, "answer_confidence"] = np.nan
    df.loc[:, "answer_weight"] = 0.0
    df.loc[ok, "answer_confidence"] = confidence.loc[ok]
    df.loc[ok, "answer_weight"] = confidence.loc[ok]
    df.loc[ok, "answer_confidence_level"] = "low"
    df.loc[ok & (confidence >= medium_threshold), "answer_confidence_level"] = "medium"
    df.loc[ok & (confidence >= high_threshold), "answer_confidence_level"] = "high"
    df.loc[~ok, "answer_confidence_level"] = "invalid"
    return df


def print_summary(split, df):
    ok = df["answer_clip_status"].astype(str) == "ok"
    confidence = pd.to_numeric(df.loc[ok, "answer_confidence"], errors="coerce").dropna()
    levels = df["answer_confidence_level"].value_counts(dropna=False).to_dict()
    if len(confidence):
        print(
            f"[{split}] confidence mean={confidence.mean():.4f}, "
            f"median={confidence.median():.4f}, levels={levels}"
        )
    else:
        print(f"[{split}] no valid confidence scores, levels={levels}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["processed_dir"]).resolve()
    splits = list(cfg.get("splits", ["train", "val", "test"]))
    input_pattern = cfg.get("input_pattern", "{split}_with_answer.csv")
    output_pattern = cfg.get(
        "output_pattern", "{split}_with_answer_confidence.csv"
    )

    paths = {}
    for split in splits:
        input_path = processed_dir / input_pattern.format(split=split)
        output_path = processed_dir / output_pattern.format(split=split)
        paths[split] = (input_path, output_path)
        print(f"[{split}] input={input_path}")
        print(f"[{split}] output={output_path}")
        if not input_path.exists() and not output_path.exists():
            raise FileNotFoundError(f"Neither input nor resumable output exists for {split}")

    if args.dry_run:
        print("Dry run complete. No model was loaded and no files were written.")
        return

    import open_clip

    device_name = cfg.get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    model_name = cfg.get("clip_model_name", "ViT-B-16")
    pretrained = cfg.get("clip_pretrained", "openai")
    model_tag = f"open_clip:{model_name}:{pretrained}"
    print(f"Loading {model_tag} on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    frames = {}
    for split in splits:
        input_path, output_path = paths[split]
        df = load_split(input_path, output_path, args.overwrite)
        frames[split] = score_split(
            df=df,
            split=split,
            output_path=output_path,
            processed_dir=processed_dir,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            model_tag=model_tag,
            device=device,
            batch_size=int(cfg.get("batch_size", 64)),
            save_every=int(cfg.get("save_every", 500)),
            limit=args.limit,
        )

    if args.limit is not None:
        print(
            "Smoke-test scoring finished. Calibration was intentionally skipped; "
            "rerun without --limit to finish and calibrate all splits."
        )
        return

    if "train" not in frames:
        raise KeyError("The train split is required to fit answer confidence calibration")
    validate_scoring_coverage(
        frames, float(cfg.get("minimum_success_rate", 0.98))
    )
    q_low = float(cfg.get("train_quantile_low", 0.05))
    q_high = float(cfg.get("train_quantile_high", 0.95))
    medium_threshold = float(cfg.get("medium_threshold", 0.30))
    high_threshold = float(cfg.get("high_threshold", 0.65))
    lo, hi, train_count = fit_train_calibration(frames["train"], q_low, q_high)
    print(
        f"Train-only calibration: q{q_low:.2f}={lo:.6f}, "
        f"q{q_high:.2f}={hi:.6f}, n={train_count}"
    )

    for split, df in frames.items():
        calibrated = apply_calibration(
            df, lo, hi, medium_threshold, high_threshold
        )
        frames[split] = calibrated
        atomic_save_csv(calibrated, paths[split][1])
        print_summary(split, calibrated)

    calibration = {
        "method": "train_quantile_minmax",
        "clip_model": model_tag,
        "train_quantile_low": q_low,
        "train_quantile_high": q_high,
        "train_raw_low": lo,
        "train_raw_high": hi,
        "train_valid_count": train_count,
        "medium_threshold": medium_threshold,
        "high_threshold": high_threshold,
        "note": "Validation and test scores use the unchanged train calibration.",
    }
    calibration_path = processed_dir / cfg.get(
        "calibration_file", "answer_confidence_calibration.json"
    )
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=True, indent=2)
    print(f"Saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
