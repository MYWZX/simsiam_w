from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROMPT = (
    "Describe only the visible content of this image in one concise, factual "
    "English sentence. Mention the main objects, their counts when clear, "
    "important visible attributes, actions, and spatial relationships. Do not "
    "guess hidden details, creators, or intentions. Do not discuss image quality, "
    "do not refer to an input prompt, and do not add a preamble."
)

RESULT_COLUMNS = [
    "row_index",
    "name",
    "answer",
    "answer_status",
    "answer_confidence",
    "answer_error",
    "answer_model",
    "answer_prompt_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate image-only Qwen answers for processed RichHF-18K CSV files. "
            "Each worker writes its own shard, so the script is safe for multi-GPU use."
        )
    )
    parser.add_argument(
        "--processed-dir",
        default="data/RichHF-18K/processed",
        help="Directory containing train.csv, val.csv, test.csv and extracted images.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id or local Qwen2.5-VL model path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never contact Hugging Face; require all model files to exist locally.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes NF4 quantization for GPUs with limited memory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device such as cuda:0. Defaults to cuda:LOCAL_RANK.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Worker shard index. Defaults to RANK from torchrun, otherwise 0.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total workers. Defaults to WORLD_SIZE from torchrun, otherwise 1.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Persist the current answer shard after this many new samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate at most this many pending samples per split and worker.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate successful rows already present in this worker shard.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing answer shards without loading Qwen.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow merging when some rows or shard files are still missing.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Image-only captioning instruction. The dataset Prompt is never supplied.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _resolve_parallel_args(args: argparse.Namespace) -> tuple[int, int, int]:
    shard_id = args.shard_id
    if shard_id is None:
        shard_id = int(os.environ.get("RANK", "0"))
    num_shards = args.num_shards
    if num_shards is None:
        num_shards = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(shard_id)))
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id={shard_id} is outside [0, {num_shards})")
    return shard_id, num_shards, local_rank


def _shard_path(
    processed_dir: Path,
    split: str,
    shard_id: int,
    num_shards: int,
) -> Path:
    return (
        processed_dir
        / "answer_shards"
        / f"{split}.shard-{shard_id:05d}-of-{num_shards:05d}.csv"
    )


def _load_existing_results(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    _, rows = _read_csv(path)
    results: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            row_index = int(row["row_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid row_index in answer shard {path}") from exc
        results[row_index] = row
    return results


def _save_results(path: Path, results: dict[int, dict[str, str]]) -> None:
    ordered = [results[index] for index in sorted(results)]
    _write_csv_atomic(path, RESULT_COLUMNS, ordered)


def _clean_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    prefixes = (
        "The image shows ",
        "This image shows ",
        "The image depicts ",
        "This image depicts ",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            if text:
                text = text[0].upper() + text[1:]
            break
    return text


def _batches(items: list[tuple[int, dict[str, str]]], size: int):
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


class QwenAnswerGenerator:
    def __init__(self, args: argparse.Namespace, local_rank: int):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies are missing. Install them with: "
                "pip install -U transformers accelerate qwen-vl-utils safetensors"
            ) from exc

        if args.device is not None:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype_map[args.dtype],
            "attn_implementation": args.attn_implementation,
            "low_cpu_mem_usage": True,
            "local_files_only": bool(args.local_files_only),
        }
        if args.load_in_4bit:
            if device.type != "cuda":
                raise RuntimeError("--load-in-4bit requires a CUDA device")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "4-bit loading requires transformers BitsAndBytesConfig"
                ) from exc
            compute_dtype = dtype_map[args.dtype]
            if compute_dtype == "auto" or compute_dtype == torch.float32:
                compute_dtype = torch.bfloat16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["device_map"] = {"": device.index or 0}
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                **model_kwargs,
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                **model_kwargs,
            ).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            local_files_only=bool(args.local_files_only),
        )
        self.process_vision_info = process_vision_info
        self.torch = torch
        self.device = device
        self.max_new_tokens = int(args.max_new_tokens)
        self.prompt = args.prompt

    def generate(self, image_paths: list[Path]) -> list[str]:
        messages = []
        for image_path in image_paths:
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path.resolve().as_uri(),
                            },
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]
            )

        texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [_clean_answer(answer) for answer in answers]


def generate_split(
    args: argparse.Namespace,
    generator: QwenAnswerGenerator,
    processed_dir: Path,
    split: str,
    shard_id: int,
    num_shards: int,
) -> None:
    input_path = processed_dir / f"{split}.csv"
    _, rows = _read_csv(input_path)
    shard_path = _shard_path(processed_dir, split, shard_id, num_shards)
    results = _load_existing_results(shard_path)

    assigned = [
        (index, row)
        for index, row in enumerate(rows)
        if index % num_shards == shard_id
    ]
    pending = []
    for index, row in assigned:
        existing = results.get(index)
        if (
            not args.overwrite
            and existing is not None
            and existing.get("answer_status", "").lower() == "ok"
            and existing.get("answer", "").strip()
        ):
            continue
        pending.append((index, row))
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]

    print(
        f"[{split}] worker={shard_id}/{num_shards} assigned={len(assigned)} "
        f"existing={len(results)} pending={len(pending)}"
    )
    newly_processed = 0
    last_saved_count = 0
    for batch in _batches(pending, args.batch_size):
        valid_items: list[tuple[int, dict[str, str], Path]] = []
        for index, row in batch:
            image_value = str(row.get("image_path", "")).strip()
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = processed_dir / image_path
            if not image_path.exists():
                results[index] = {
                    "row_index": str(index),
                    "name": row.get("name", ""),
                    "answer": "",
                    "answer_status": "error",
                    "answer_confidence": "",
                    "answer_error": f"Image not found: {image_path}",
                    "answer_model": args.model,
                    "answer_prompt_version": "richhf_image_only_v1",
                }
                newly_processed += 1
                continue
            valid_items.append((index, row, image_path))

        if valid_items:
            try:
                answers = generator.generate([item[2] for item in valid_items])
                if len(answers) != len(valid_items):
                    raise RuntimeError(
                        f"Qwen returned {len(answers)} answers for {len(valid_items)} images"
                    )
                if any(not answer for answer in answers):
                    raise RuntimeError("Qwen returned one or more empty answers")
                for (index, row, _), answer in zip(valid_items, answers):
                    results[index] = {
                        "row_index": str(index),
                        "name": row.get("name", ""),
                        "answer": answer,
                        "answer_status": "ok",
                        # Confidence is filled by a later independent filtering pass.
                        "answer_confidence": "",
                        "answer_error": "",
                        "answer_model": args.model,
                        "answer_prompt_version": "richhf_image_only_v1",
                    }
                    newly_processed += 1
            except Exception as exc:
                if generator.device.type == "cuda":
                    generator.torch.cuda.empty_cache()
                error = f"{type(exc).__name__}: {exc}"
                for index, row, _ in valid_items:
                    results[index] = {
                        "row_index": str(index),
                        "name": row.get("name", ""),
                        "answer": "",
                        "answer_status": "error",
                        "answer_confidence": "",
                        "answer_error": error[:1000],
                        "answer_model": args.model,
                        "answer_prompt_version": "richhf_image_only_v1",
                    }
                    newly_processed += 1

        if newly_processed - last_saved_count >= max(1, args.save_every):
            _save_results(shard_path, results)
            last_saved_count = newly_processed
            print(
                f"[{split}] worker={shard_id}/{num_shards} "
                f"saved={len(results)} new={newly_processed}"
            )

    _save_results(shard_path, results)
    ok_count = sum(
        row.get("answer_status", "").lower() == "ok" for row in results.values()
    )
    error_count = sum(
        row.get("answer_status", "").lower() == "error"
        for row in results.values()
    )
    print(
        f"[{split}] worker={shard_id}/{num_shards} complete: "
        f"ok={ok_count}, error={error_count}, shard={shard_path}"
    )


def merge_split(
    args: argparse.Namespace,
    processed_dir: Path,
    split: str,
    num_shards: int,
) -> None:
    input_path = processed_dir / f"{split}.csv"
    fieldnames, rows = _read_csv(input_path)
    merged_results: dict[int, dict[str, str]] = {}
    missing_shards = []
    for shard_id in range(num_shards):
        shard_path = _shard_path(processed_dir, split, shard_id, num_shards)
        if not shard_path.exists():
            missing_shards.append(str(shard_path))
            continue
        for index, result in _load_existing_results(shard_path).items():
            if index in merged_results:
                raise RuntimeError(f"Duplicate RichHF answer for {split} row {index}")
            merged_results[index] = result

    missing_rows = [index for index in range(len(rows)) if index not in merged_results]
    if (missing_shards or missing_rows) and not args.allow_incomplete:
        raise RuntimeError(
            f"Cannot merge incomplete {split} answers: "
            f"missing_shards={len(missing_shards)}, missing_rows={len(missing_rows)}. "
            "Finish generation or pass --allow-incomplete for inspection only."
        )

    added_columns = [
        "answer",
        "answer_status",
        "answer_confidence",
        "answer_error",
        "answer_model",
        "answer_prompt_version",
    ]
    output_fields = list(fieldnames)
    for column in added_columns:
        if column not in output_fields:
            output_fields.append(column)

    for index, row in enumerate(rows):
        result = merged_results.get(index)
        if result is None:
            row["answer_status"] = row.get("answer_status", "pending") or "pending"
            continue
        for column in added_columns:
            row[column] = result.get(column, "")

    output_path = processed_dir / f"{split}_with_answer.csv"
    _write_csv_atomic(output_path, output_fields, rows)
    ok_count = sum(row.get("answer_status", "").lower() == "ok" for row in rows)
    error_count = sum(
        row.get("answer_status", "").lower() == "error" for row in rows
    )
    print(
        f"[{split}] merged rows={len(rows)}, ok={ok_count}, error={error_count}, "
        f"pending={len(rows) - ok_count - error_count}, output={output_path}"
    )


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed RichHF directory not found: {processed_dir.resolve()}"
        )
    shard_id, num_shards, local_rank = _resolve_parallel_args(args)

    if args.merge_only:
        if shard_id != 0:
            print(f"Worker {shard_id} skips merge; only worker 0 merges answer shards.")
            return
        for split in args.splits:
            merge_split(args, processed_dir, split, num_shards)
        return

    generator = QwenAnswerGenerator(args, local_rank=local_rank)
    for split in args.splits:
        generate_split(
            args=args,
            generator=generator,
            processed_dir=processed_dir,
            split=split,
            shard_id=shard_id,
            num_shards=num_shards,
        )


if __name__ == "__main__":
    main()
