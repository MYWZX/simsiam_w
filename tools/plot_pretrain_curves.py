#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretraining loss curves from CSVExperimentLogger output.")
    parser.add_argument("--csv", type=str, default="./outputs/all_results_20k.csv", help="Global results CSV.")
    parser.add_argument("--output-dir", type=str, required=True, help="Filter rows by this experiment output_dir.")
    parser.add_argument("--save", type=str, default="", help="Output png path. Default: <output-dir>/pretrain_curves.png")
    parser.add_argument("--run-id", type=str, default="", help="Optional exact run_id filter.")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Plot all matching epoch rows. By default only the latest run is plotted.",
    )
    return parser.parse_args()


def _load_metrics(metrics_json: str) -> Dict:
    try:
        obj = json.loads(metrics_json)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _select_latest_run(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest run even when a fixed run_id/output_dir is reused."""
    rows = rows.sort_index()
    run_start_indices = rows.index[rows["event"] == "run_start"].tolist()
    if run_start_indices:
        return rows.loc[run_start_indices[-1] :]

    epoch_rows = rows[rows["event"] == "epoch"]
    if epoch_rows.empty:
        return rows
    epochs = pd.to_numeric(epoch_rows["epoch"], errors="coerce")
    restart_positions = [i for i in range(1, len(epochs)) if epochs.iloc[i] <= epochs.iloc[i - 1]]
    if restart_positions:
        return rows.loc[epoch_rows.index[restart_positions[-1]] :]
    return rows


def _extract_epoch_metrics(
    csv_path: Path,
    output_dir: str,
    run_id: str = "",
    latest_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "event" not in df.columns or "metrics_json" not in df.columns:
        raise RuntimeError(f"Unexpected CSV format: {csv_path}")

    output_dir_norm = str(Path(output_dir).as_posix()).rstrip("/")
    logged_output_norm = df.get("output_dir", "").astype(str).map(lambda x: str(Path(x).as_posix()).rstrip("/"))
    matched = df[logged_output_norm == output_dir_norm].copy()
    if run_id:
        matched = matched[matched["run_id"].astype(str) == run_id]
    if matched.empty:
        # Fall back to suffix matching because Linux/Windows path spelling can differ.
        suffix = output_dir_norm.split("/")[-1]
        matched = df[df.get("output_dir", "").astype(str).str.contains(suffix, regex=False)].copy()
        if run_id:
            matched = matched[matched["run_id"].astype(str) == run_id]
    if latest_only and not matched.empty:
        matched = _select_latest_run(matched)
    rows = matched[matched["event"] == "epoch"].copy()
    if rows.empty:
        candidates = (
            df[df["event"] == "epoch"].get("output_dir", pd.Series(dtype=str))
            .astype(str)
            .drop_duplicates()
            .tail(12)
            .tolist()
        )
        msg = f"No epoch rows found for output_dir={output_dir} in {csv_path}"
        if candidates:
            msg += "\nAvailable recent output_dir values:\n  " + "\n  ".join(candidates)
        raise RuntimeError(msg)

    records: List[Dict] = []
    for _, row in rows.iterrows():
        rec = {
            "epoch": int(row["epoch"]),
            "is_best": bool(int(row.get("is_best", 0))),
            "run_id": str(row.get("run_id", "")),
        }
        rec.update(_load_metrics(str(row["metrics_json"])))
        records.append(rec)
    out = pd.DataFrame(records).sort_values("epoch").reset_index(drop=True)
    return out


def _curve_groups():
    return [
        ("Total Loss", ["train_loss", "val_loss"]),
        ("Image Relations", ["train_img", "val_img", "train_img_answer", "val_img_answer"]),
        ("Text-conditioned Relations", ["train_img_prompt", "val_img_prompt", "train_prompt_answer", "val_prompt_answer"]),
        ("Quality Supervision", ["train_quality_reg", "val_quality_reg", "train_quality_rank", "val_quality_rank", "train_quality_srocc", "val_quality_srocc"]),
        ("Align / Final Supervision", ["train_align_reg", "val_align_reg", "train_align_rank", "val_align_rank", "train_align_srocc", "val_align_srocc", "train_final_reg", "val_final_reg", "train_final_rank", "val_final_rank", "train_final_srocc", "val_final_srocc"]),
        ("Validation Proxies", ["val_quality_proxy_SROCC", "val_align_proxy_SROCC", "val_final_proxy_SROCC", "val_joint_SROCC", "val_quality_proxy_PLCC", "val_align_proxy_PLCC", "val_final_proxy_PLCC", "val_joint_PLCC", "val_proxy_SROCC", "val_proxy_PLCC"]),
    ]


def _plot_curves_matplotlib(metrics: pd.DataFrame, save_path: Path) -> Path:
    import matplotlib.pyplot as plt

    groups = _curve_groups()

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), dpi=160)
    axes = axes.reshape(-1)
    x = metrics["epoch"]

    for ax, (title, keys) in zip(axes, groups):
        plotted = False
        for key in keys:
            if key in metrics.columns and metrics[key].notna().any():
                ax.plot(x, metrics[key], marker="o", linewidth=1.4, markersize=2.8, label=key)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if title == "Validation Proxies":
            ylabel = "correlation"
        else:
            ylabel = "loss"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        best_rows = metrics[metrics.get("is_best", False) == True]
        if not best_rows.empty:
            best_epoch = int(best_rows.iloc[-1]["epoch"])
            ax.axvline(best_epoch, color="0.35", linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_curves_svg(metrics: pd.DataFrame, save_path: Path) -> Path:
    """Dependency-free SVG fallback for training servers without matplotlib."""
    width, height = 1400, 1100
    panel_w, panel_h = 660, 320
    lefts = [60, 740]
    tops = [55, 405, 755]
    pad_left, pad_right, pad_top, pad_bottom = 72, 20, 48, 52
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    groups = _curve_groups()
    epochs = pd.to_numeric(metrics["epoch"], errors="coerce").astype(float)
    x_min, x_max = float(epochs.min()), float(epochs.max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    best_rows = metrics[metrics["is_best"] == True]
    best_epoch = float(best_rows.iloc[-1]["epoch"]) if not best_rows.empty else None

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="700" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="600">Pretraining Curves</text>',
    ]

    for panel_idx, (title, keys) in enumerate(groups):
        panel_x = lefts[panel_idx % 2]
        panel_y = tops[panel_idx // 2]
        plot_x0 = panel_x + pad_left
        plot_x1 = panel_x + panel_w - pad_right
        plot_y0 = panel_y + pad_top
        plot_y1 = panel_y + panel_h - pad_bottom
        present = [k for k in keys if k in metrics.columns and pd.to_numeric(metrics[k], errors="coerce").notna().any()]
        values = []
        for key in present:
            values.extend(pd.to_numeric(metrics[key], errors="coerce").dropna().astype(float).tolist())
        if values:
            y_min, y_max = min(values), max(values)
            span = y_max - y_min
            pad = span * 0.08 if span > 0 else max(abs(y_min) * 0.08, 1e-6)
            y_min, y_max = y_min - pad, y_max + pad
        else:
            y_min, y_max = 0.0, 1.0

        def sx(value: float) -> float:
            return plot_x0 + (value - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)

        def sy(value: float) -> float:
            return plot_y1 - (value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)

        svg.extend([
            f'<text x="{panel_x + panel_w / 2:.1f}" y="{panel_y + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="600">{escape(title)}</text>',
            f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_x1 - plot_x0}" height="{plot_y1 - plot_y0}" fill="none" stroke="#9ca3af" stroke-width="1"/>',
        ])
        for tick in range(5):
            frac = tick / 4
            ty = plot_y1 - frac * (plot_y1 - plot_y0)
            value = y_min + frac * (y_max - y_min)
            svg.append(f'<line x1="{plot_x0}" y1="{ty:.1f}" x2="{plot_x1}" y2="{ty:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            svg.append(f'<text x="{plot_x0 - 8}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{value:.4g}</text>')
        for tick in range(5):
            frac = tick / 4
            tx = plot_x0 + frac * (plot_x1 - plot_x0)
            value = x_min + frac * (x_max - x_min)
            svg.append(f'<text x="{tx:.1f}" y="{plot_y1 + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{value:.0f}</text>')
        svg.append(f'<text x="{(plot_x0 + plot_x1) / 2:.1f}" y="{panel_y + panel_h - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">epoch</text>')

        if best_epoch is not None:
            bx = sx(best_epoch)
            svg.append(f'<line x1="{bx:.1f}" y1="{plot_y0}" x2="{bx:.1f}" y2="{plot_y1}" stroke="#4b5563" stroke-width="1" stroke-dasharray="5 4"/>')
            svg.append(f'<text x="{min(bx + 5, plot_x1 - 72):.1f}" y="{plot_y0 + 13}" font-family="Arial, sans-serif" font-size="11" fill="#374151">best {best_epoch:.0f}</text>')

        for series_idx, key in enumerate(present):
            series = pd.to_numeric(metrics[key], errors="coerce")
            points = []
            for epoch, value in zip(epochs, series):
                if math.isfinite(float(epoch)) and pd.notna(value) and math.isfinite(float(value)):
                    points.append(f"{sx(float(epoch)):.1f},{sy(float(value)):.1f}")
            if points:
                color = colors[series_idx % len(colors)]
                svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
                legend_x = plot_x0 + (series_idx % 3) * 185
                legend_y = panel_y + panel_h - 28 + (series_idx // 3) * 15
                svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 16}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
                svg.append(f'<text x="{legend_x + 21}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="10" fill="#111827">{escape(key)}</text>')
        if not present:
            svg.append(f'<text x="{(plot_x0 + plot_x1) / 2:.1f}" y="{(plot_y0 + plot_y1) / 2:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">no data</text>')

    svg.append("</svg>")
    save_path = save_path.with_suffix(".svg")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(svg), encoding="utf-8")
    return save_path


def _plot_curves(metrics: pd.DataFrame, save_path: Path) -> Path:
    try:
        return _plot_curves_matplotlib(metrics, save_path)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("[WARN] matplotlib is unavailable; saving an SVG curve instead.")
        return _plot_curves_svg(metrics, save_path)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    save_path = Path(args.save) if args.save else output_dir / "pretrain_curves.png"

    metrics = _extract_epoch_metrics(
        csv_path,
        args.output_dir,
        run_id=args.run_id,
        latest_only=not args.all_runs,
    )
    actual_save_path = _plot_curves(metrics, save_path)
    metrics_csv = actual_save_path.with_name(f"{actual_save_path.stem}_data.csv")
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"epochs={len(metrics)}, first={int(metrics['epoch'].min())}, last={int(metrics['epoch'].max())}")
    print(f"run_id={metrics.iloc[-1].get('run_id', '')}")
    print(f"saved={actual_save_path}")
    print(f"data={metrics_csv}")


if __name__ == "__main__":
    main()
