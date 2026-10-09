"""Print or explicitly execute matched RQ1/RQ3 experiments; no GPU selection."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"1k": "agiqa1k", "3k": "agiqa3k", "2023": "aigciqa2023"}
PRETRAIN_CONFIG = ROOT / "configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml"


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    output_dir: Path
    checkpoint: Path | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pretrain", "downstream", "adapter"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="1k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-seed", type=int, default=42,
                        help="RQ1 checkpoint seed, independently of downstream training seed.")
    parser.add_argument("--fixed-epochs", type=int, default=17,
                        help="Same epoch budget in both RQ1 groups; original scheduler horizon stays fixed.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rq_diagnostics_v1"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/richhf18k_cm_pretrain_decoupled_quality_align/best_quality_pretrain.pth"),
                        help="One fixed checkpoint for the four RQ3 conditions.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run the printed commands. Omission is a write-free dry run.")
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def build_plan(args) -> list[Job]:
    root = resolve_project_path(args.output_root)
    jobs = []
    if args.stage == "pretrain":
        if args.fixed_epochs < 1:
            raise ValueError("--fixed-epochs must be positive.")
        for group in ("full", "no_regular_simsiam"):
            name = f"rq1_pretrain_{group}_seed{args.seed}"
            output_dir = root / name
            command = [
                sys.executable, str(ROOT / "train_cm_pretrain.py"),
                "--config", str(PRETRAIN_CONFIG), "--seed", str(args.seed),
                "--fixed-epochs", str(args.fixed_epochs),
                "--output-dir", str(output_dir),
                "--results-csv", str(root / "all_results_rq1_pretrain.csv"),
            ]
            if group == "no_regular_simsiam":
                command.append("--disable-regular-simsiam")
            jobs.append(Job(name, command, output_dir))
        return jobs

    dataset = DATASETS[args.dataset]
    config = ROOT / f"configs/{dataset}_cm_iqa_decoupled_quality_align.yaml"
    if args.stage == "downstream":
        cases = [
            (group, root / f"rq1_pretrain_{group}_seed{args.pretrain_seed}" / "best_quality_pretrain.pth", [])
            for group in ("full", "no_regular_simsiam")
        ]
        rq = "rq1"
    else:
        checkpoint = resolve_project_path(args.checkpoint)
        cases = [
            (f"{mode}_{initialization}", checkpoint,
             ["--quality-adapter-mode", mode, "--quality-adapter-init", initialization])
            for mode in ("replace", "parallel_score")
            for initialization in ("pretrained", "random")
        ]
        rq = "rq3"
    for case, checkpoint, flags in cases:
        name = f"{dataset}_{rq}_{case}_seed{args.seed}"
        output_dir = root / name
        command = [
            sys.executable, str(ROOT / "train_cm_iqa.py"),
            "--config", str(config), "--seed", str(args.seed),
            "--output-dir", str(output_dir),
            "--results-csv", str(root / f"all_results_{rq}_{dataset}.csv"),
            "--pretrained-checkpoint", str(checkpoint),
            *flags,
        ]
        jobs.append(Job(name, command, output_dir, checkpoint))
    return jobs


def execute_plan(jobs: list[Job]) -> None:
    # Validate the entire batch before starting its first expensive experiment.
    for job in jobs:
        config = Path(job.command[job.command.index("--config") + 1])
        if not config.is_file():
            raise FileNotFoundError(f"Config not found: {config}")
        if "--fixed-epochs" in job.command:
            # Keep dry-run dependency-free; YAML is needed only to validate
            # an actual training batch before reserving any output directory.
            import yaml

            cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
            horizon = int(cfg.get("train", {}).get("epochs", 40))
            budget = int(job.command[job.command.index("--fixed-epochs") + 1])
            if not 1 <= budget <= horizon:
                raise ValueError(f"Fixed training epochs must be in [1, {horizon}], got {budget}.")
        if job.checkpoint is not None and not job.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {job.checkpoint}")
        if job.output_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse output directory: {job.output_dir}. "
                "Use a new --output-root; no automatic overwrite or retry."
            )
    for job in jobs:
        # Atomic reservation also prevents concurrent invocations sharing a run.
        job.output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Starting {job.name}", flush=True)
        subprocess.run(job.command, cwd=ROOT, check=True)


def main(argv=None):
    args = parse_args(argv)
    jobs = build_plan(args)
    print("RQ1: four ordinary loss terms only; human/local/anchor supervision remains.")
    print("RQ3: quality scoring interfaces only; no synthetic Final MOS.")
    for job in jobs:
        print(shlex.join(job.command))
    if args.execute:
        execute_plan(jobs)
    else:
        print("Dry run: no files written and no training started. Add --execute to run.")


if __name__ == "__main__":
    main()
