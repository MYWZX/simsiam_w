from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_SPLIT_SIZES = {
    "train": 15810,
    "validation": 995,
    "test": 955,
}

OUTPUT_SPLIT_NAMES = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

CSV_COLUMNS = [
    "name",
    "image_path",
    "prompt",
    "answer",
    "answer_status",
    "answer_confidence",
    "artifact_score",
    "aesthetics_score",
    "misalignment_score",
    "overall_score",
    "quality_score",
    "align_score",
    "misalignment_map_path",
    "prompt_misalignment_label",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export downloaded RichHF-18K parquet shards to images and CSV files."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/RichHF-18K/raw/data",
        help="Directory containing train/validation/test parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/RichHF-18K/processed",
        help="Destination for extracted images and metadata CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite images that already exist.",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Export only misalignment maps; keep existing CSV/Answer files untouched.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Do not fail when a split size differs from the official release.",
    )
    return parser.parse_args()


def _as_float(value: Any, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], got {score}")
    return score


def _json_value(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False)


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).copy()
        if value.get("path"):
            return Image.open(value["path"]).copy()
    if isinstance(value, (str, Path)):
        return Image.open(value).copy()
    raise TypeError(f"Unsupported RichHF image value: {type(value)!r}")


def _find_shards(raw_dir: Path, split: str) -> list[str]:
    shards = sorted(raw_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No {split} parquet shards found under {raw_dir.resolve()}"
        )
    return [str(path) for path in shards]


def export_split(
    raw_dir: Path,
    output_dir: Path,
    source_split: str,
    overwrite: bool,
    maps_only: bool = False,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: "
            "pip install datasets pyarrow pillow"
        ) from exc

    output_split = OUTPUT_SPLIT_NAMES[source_split]
    image_dir = output_dir / output_split / "images"
    misalignment_map_dir = output_dir / output_split / "misalignment_maps"
    image_dir.mkdir(parents=True, exist_ok=True)
    misalignment_map_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_split}.csv"

    dataset = load_dataset(
        "parquet",
        data_files={source_split: _find_shards(raw_dir, source_split)},
        split=source_split,
        streaming=True,
    )

    count = 0
    if maps_only:
        for sample in dataset:
            required = {"filename", "misalignment_map"}
            missing = required - set(sample)
            if missing:
                raise RuntimeError(
                    f"RichHF parquet sample is missing fields: {sorted(missing)}"
                )
            name = Path(str(sample["filename"])).name
            if not name:
                raise RuntimeError(f"Invalid RichHF filename: {sample['filename']!r}")
            map_path = misalignment_map_dir / name
            if overwrite or not map_path.exists():
                heatmap = _decode_image(sample["misalignment_map"]).convert("L")
                heatmap.save(map_path, format="PNG")
            count += 1
            if count % 1000 == 0:
                print(f"[{source_split}] exported {count} misalignment maps")
        return count

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for sample in dataset:
            required = {
                "filename",
                "caption",
                "image",
                "artifact_score",
                "aesthetics_score",
                "misalignment_score",
                "misalignment_map",
                "overall_score",
                "prompt_misalignment_label",
            }
            missing = required - set(sample)
            if missing:
                raise RuntimeError(
                    f"RichHF parquet sample is missing fields: {sorted(missing)}"
                )

            name = Path(str(sample["filename"])).name
            if not name:
                raise RuntimeError(f"Invalid RichHF filename: {sample['filename']!r}")
            image_path = image_dir / name
            if overwrite or not image_path.exists():
                image = _decode_image(sample["image"]).convert("RGB")
                image.save(image_path, format="PNG")

            misalignment_map_path = misalignment_map_dir / name
            if overwrite or not misalignment_map_path.exists():
                misalignment_map = _decode_image(sample["misalignment_map"]).convert("L")
                misalignment_map.save(misalignment_map_path, format="PNG")

            artifact = _as_float(sample["artifact_score"], "artifact_score")
            aesthetics = _as_float(sample["aesthetics_score"], "aesthetics_score")
            alignment = _as_float(
                sample["misalignment_score"], "misalignment_score"
            )
            overall = _as_float(sample["overall_score"], "overall_score")
            prompt = str(sample["caption"]).strip()
            if not prompt:
                raise RuntimeError(f"Empty prompt for RichHF sample {name}")

            relative_image_path = image_path.relative_to(output_dir).as_posix()
            writer.writerow(
                {
                    "name": name,
                    "image_path": relative_image_path,
                    "prompt": prompt,
                    "answer": "",
                    "answer_status": "pending",
                    "answer_confidence": "",
                    "artifact_score": artifact,
                    "aesthetics_score": aesthetics,
                    "misalignment_score": alignment,
                    "overall_score": overall,
                    # RichHF states that all scores are higher-is-better.
                    "quality_score": artifact,
                    "align_score": alignment,
                    "misalignment_map_path": misalignment_map_path.relative_to(output_dir).as_posix(),
                    "prompt_misalignment_label": _json_value(
                        sample["prompt_misalignment_label"]
                    ),
                    "split": output_split,
                }
            )
            count += 1
            if count % 1000 == 0:
                print(f"[{source_split}] exported {count} samples")

    return count


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"RichHF raw directory not found: {raw_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for source_split in ("train", "validation", "test"):
        counts[source_split] = export_split(
            raw_dir=raw_dir,
            output_dir=output_dir,
            source_split=source_split,
            overwrite=bool(args.overwrite),
            maps_only=bool(args.maps_only),
        )

    mismatches = {
        split: (counts[split], expected)
        for split, expected in EXPECTED_SPLIT_SIZES.items()
        if counts[split] != expected
    }
    mode = "misalignment-map export" if args.maps_only else "full export"
    print(f"RichHF-18K {mode} complete:")
    for split, count in counts.items():
        print(f"  {split}: {count}")
    print(f"  total: {sum(counts.values())}")
    print(f"  output: {output_dir.resolve()}")

    if mismatches and not args.allow_count_mismatch:
        details = ", ".join(
            f"{split}={actual} (expected {expected})"
            for split, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(f"RichHF split-size verification failed: {details}")


if __name__ == "__main__":
    main()
