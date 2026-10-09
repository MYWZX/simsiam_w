"""Read-only sensitivity probe using synthetic subject/action Prompt changes.

Negatives are candidates, not human-verified image facts. Develop on validation
data; reserve the test split for a final, one-time check.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import re
import statistics
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import build_dataloaders  # noqa: E402
from models.two_stage_models import CrossModalIQAFullModel  # noqa: E402

PAIRS = {
    "subject": (
        ("cat", "dog"), ("dog", "cat"), ("man", "woman"),
        ("woman", "man"), ("boy", "girl"), ("girl", "boy"),
        ("car", "boat"), ("boat", "car"), ("bird", "fish"),
        ("fish", "bird"), ("horse", "cow"), ("cow", "horse"),
        ("elephant", "giraffe"), ("giraffe", "elephant"),
    ),
    "action": (
        ("eating", "cooking"), ("cooking", "eating"),
        ("walking", "running"), ("running", "walking"),
        ("sitting", "standing"), ("standing", "sitting"),
        ("swimming", "flying"), ("flying", "swimming"),
    ),
}


def _word(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def make_negative(prompt: str, kind: str) -> tuple[str, str] | None:
    """Return one grammatical-form-preserving replacement, or reject ambiguity."""
    found = [(src, dst) for src, dst in PAIRS[kind] if _word(src).search(prompt)]
    if len(found) != 1:
        return None
    source, target = found[0]
    if _word(target).search(prompt):
        return None

    def keep_case(match: re.Match[str]) -> str:
        original = match.group()
        return target.upper() if original.isupper() else target.title() if original.istitle() else target

    changed = _word(source).sub(keep_case, prompt)
    return (changed, f"{source}->{target}") if changed != prompt else None


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Use --checkpoint LABEL=PATH.")
    label, raw = value.split("=", 1)
    path = Path(raw.strip())
    if not path.is_absolute():
        path = ROOT / path
    if not label.strip() or not path.is_file():
        raise ValueError(f"Invalid checkpoint: {value}")
    return label.strip(), path


def _snapshot(path: Path) -> dict:
    snapshot = path.parent / "config_snapshot.yaml"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing checkpoint config snapshot: {snapshot}")
    return yaml.safe_load(snapshot.read_text(encoding="utf-8"))


def _make_loader(cfg: dict, split: str, batch_size: int):
    cfg = copy.deepcopy(cfg)
    if cfg["dataset"].get("type") not in {"agiqa3k_simsiam", "aigciqa2023_iqa"}:
        raise ValueError("Only 3K and 2023 alignment datasets are supported.")
    root = Path(cfg["dataset"]["root_dir"])
    if not root.is_absolute():
        cfg["dataset"]["root_dir"] = str((ROOT / root).resolve())
    cfg["dataset"]["use_perturb"] = False
    cfg["loader"].update(batch_size=batch_size, num_workers=0, pin_memory=False)
    _, val, test = build_dataloaders(cfg)
    return val if split == "val" else test


def _load_model(path: Path, device: torch.device) -> tuple[CrossModalIQAFullModel, int]:
    # Local user-owned project checkpoints only: state and effective cfg are required.
    # Optimizer state makes these checkpoints large; map storage lazily.
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = copy.deepcopy(ckpt["cfg"])
    cfg["model"]["global_encoder"]["pretrained"] = False
    cfg["model"]["local_encoder"]["pretrained"] = False
    cfg["model"]["text_encoder"]["clip_pretrained"] = None
    model = CrossModalIQAFullModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    epoch = int(ckpt.get("epoch", -1))
    del ckpt
    if model.quality_branch_use_text_gate:
        raise ValueError("Probe requires quality_branch_use_text_gate=false.")
    return model.to(device).eval(), epoch


@torch.inference_mode()
def score_batch(model, images, positives, negatives, local_crops=None):
    """Reuse image features; only the Prompt changes between comparisons."""
    out = model.forward_single(images, prompts=positives, local_crops=local_crops)
    scores = {"positive": out["s_align"].view(-1).float().cpu().tolist()}
    for kind, prompts in negatives.items():
        text = model.backbone.encode_text(prompts, images.device)
        text_fused = model.text_to_fused(text)
        changed = model._align_from_global_and_patch(
            out["z"], out["patch_tokens"], text_fused,
            patch_hw=out.get("patch_hw"),
        )
        scores[kind] = changed["s_align"].view(-1).float().cpu().tolist()
    return scores


def summarize(rows: list[dict], eligible_count: int) -> list[dict]:
    result = []
    for label in sorted({row["checkpoint"] for row in rows}):
        for kind in PAIRS:
            gaps = [float(row["gap"]) for row in rows
                    if row["checkpoint"] == label and row["kind"] == kind]
            result.append({
                "checkpoint": label, "kind": kind,
                "eligible_images": eligible_count, "candidate_pairs": len(gaps),
                "coverage": len(gaps) / eligible_count if eligible_count else 0.0,
                "positive_gap_rate": sum(gap > 0 for gap in gaps) / len(gaps) if gaps else "",
                "mean_gap": statistics.mean(gaps) if gaps else "",
                "median_gap": statistics.median(gaps) if gaps else "",
            })
    return result


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--min-align-score", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only; omit for the full split.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 <= args.min_align_score <= 1:
        parser.error("--min-align-score must be in [0, 1].")
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Batch size and maximum batches must be positive.")

    checkpoints = [parse_checkpoint(value) for value in args.checkpoint]
    if len({label for label, _ in checkpoints}) != len(checkpoints):
        parser.error("Checkpoint labels must be unique.")
    prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    pair_file = prefix.with_name(prefix.name + ".pairs.csv")
    summary_file = prefix.with_name(prefix.name + ".summary.csv")
    if pair_file.exists() or summary_file.exists():
        raise FileExistsError("Refusing to overwrite diagnostic output.")
    if args.split == "test":
        print("[WARN] Do not tune or choose a model using this test-split probe.")

    snapshots = [(label, path, _snapshot(path)) for label, path in checkpoints]
    reference = snapshots[0][2]
    for label, _, cfg in snapshots[1:]:
        if cfg["dataset"] != reference["dataset"] or cfg.get("split_seed") != reference.get("split_seed"):
            raise ValueError(f"{label} has a different dataset config or split seed.")
    loader = _make_loader(reference, args.split, args.batch_size)
    device = torch.device(args.device)
    rows: list[dict] = []
    reference_cases: set[tuple[str, str]] | None = None
    eligible_count = 0
    for label, checkpoint, _ in snapshots:
        model, epoch = _load_model(checkpoint, device)
        current_cases: set[tuple[str, str]] = set()
        current_eligible = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            human = batch["align_score"].view(-1).tolist()
            selected = [i for i, value in enumerate(human) if value >= args.min_align_score]
            current_eligible += len(selected)
            if not selected:
                continue
            positives = [batch["prompt"][i] for i in selected]
            paths = [batch["path"][i] for i in selected]
            mutations = {kind: [make_negative(prompt, kind) for prompt in positives]
                         for kind in PAIRS}
            negatives = {kind: [item[0] if item else prompt
                                for item, prompt in zip(mutations[kind], positives)]
                         for kind in PAIRS}
            images = batch["image"][selected].to(device)
            crops = batch.get("local_crops")
            crops = crops[selected].to(device) if crops is not None else None
            scores = score_batch(model, images, positives, negatives, crops)
            for position, (path, prompt) in enumerate(zip(paths, positives)):
                for kind in PAIRS:
                    item = mutations[kind][position]
                    if item is None:
                        continue
                    current_cases.add((path, kind))
                    rows.append({
                        "checkpoint": label, "checkpoint_epoch": epoch,
                        "split": args.split, "path": path, "kind": kind,
                        "replacement": item[1],
                        "human_align": human[selected[position]],
                        "positive_prompt": prompt, "negative_prompt": item[0],
                        "positive_score": scores["positive"][position],
                        "negative_score": scores[kind][position],
                        "gap": scores["positive"][position] - scores[kind][position],
                    })
        if reference_cases is None:
            reference_cases, eligible_count = current_cases, current_eligible
        elif current_cases != reference_cases or current_eligible != eligible_count:
            raise RuntimeError("Eligible images changed across checkpoints.")
        print(f"{label}: {len(current_cases)} candidate pairs / {current_eligible} eligible images.")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if not rows:
        raise RuntimeError("No eligible pairs; check threshold and vocabulary.")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_file, rows, [
        "checkpoint", "checkpoint_epoch", "split", "path", "kind", "replacement",
        "human_align", "positive_prompt", "negative_prompt", "positive_score",
        "negative_score", "gap",
    ])
    summary = summarize(rows, eligible_count)
    _write_csv(summary_file, summary, [
        "checkpoint", "kind", "eligible_images", "candidate_pairs", "coverage",
        "positive_gap_rate", "mean_gap", "median_gap",
    ])
    for row in summary:
        print(row)
    print(f"Saved {pair_file} and {summary_file}. Manually inspect candidate validity.")


if __name__ == "__main__":
    main()
