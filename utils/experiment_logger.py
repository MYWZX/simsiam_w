# 日志
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def make_run_id(stage: str, cfg: Dict[str, Any]) -> str:
    """Create a readable unique run id for one experiment."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _json_dumps({
        "stage": stage,
        "output_dir": cfg.get("output_dir", ""),
        "seed": cfg.get("seed", ""),
        "dataset": cfg.get("dataset", {}).get("type", ""),
        "train": cfg.get("train", {}),
    })
    short_hash = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    return f"{stage}_{now}_{short_hash}"


def resolve_results_csv(cfg: Dict[str, Any]) -> str:
    """Resolve global CSV path. All runs append to this one file."""
    return str(cfg.get("results_csv", "./outputs/all_results.csv"))


def dump_config_snapshot(cfg: Dict[str, Any], output_dir: str, config_path: Optional[str] = None) -> str:
    """Save the effective config after command-line overrides.

    Copying the source YAML loses overrides such as ``--seed`` and
    ``--output-dir``. Serializing ``cfg`` makes the snapshot reproducible and
    keeps repeated-seed experiments from all appearing to use the YAML seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, "config_snapshot.yaml")
    snapshot = dict(cfg)
    if config_path:
        snapshot["source_config"] = str(config_path)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)
    return dst


class CSVExperimentLogger:
    """Append experiment records to one global CSV.

    Each row can represent run_start / epoch / best_val / best_test / final.
    The training scripts already call this class with event/epoch/metrics.
    """

    HEADER = [
        "time",
        "run_id",
        "stage",
        "event",
        "epoch",
        "best_epoch",
        "is_best",
        "output_dir",
        "config_path",
        "metrics_json",
        "config_json",
    ]

    def __init__(
        self,
        csv_path: str,
        run_id: str,
        stage: str,
        cfg: Dict[str, Any],
        output_dir: str,
        config_path: str = "",
    ) -> None:
        self.csv_path = csv_path
        self.run_id = run_id
        self.stage = stage
        self.cfg = cfg
        self.output_dir = output_dir
        self.config_path = config_path

        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        if need_header:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log(
        self,
        event: str,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        best_epoch: int = -1,
        is_best: bool = False,
        include_config: bool = False,
    ) -> None:
        metrics = metrics or {}
        config_obj = self.cfg if include_config or event == "run_start" else {}
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.run_id,
            self.stage,
            event,
            int(epoch) if epoch is not None else "",
            int(best_epoch) if best_epoch is not None else "",
            int(bool(is_best)),
            self.output_dir,
            self.config_path,
            _json_dumps(metrics),
            _json_dumps(config_obj),
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# Backward-compatible simple logger name, in case old scripts import ExperimentLogger.
class ExperimentLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        if need_header:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "run_id", "stage", "event", "epoch", "best_epoch", "metrics", "config"])

    def log(self, run_id, stage, event, epoch, best_epoch, metrics, config):
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                run_id,
                stage,
                event,
                epoch,
                best_epoch,
                _json_dumps(metrics),
                _json_dumps(config),
            ])
