#!/usr/bin/env python3
"""Generate two ablation grouped bar charts with error bars.

Chart 1:
    Datasets on x-axis: GSE115978, GSE123139, GSE153935, GSE166555
    Bars per dataset: full, no-GAT, no_C-DVAE (mean average_accuracy with std error bars)

Chart 2:
    Datasets on x-axis: lung_full, sapiens_full, Zhengsorted, Overall(7 datasets)
    Bars per dataset: full, no-GAT, no_C-DVAE (mean average_accuracy with std error bars)
    The Overall(7 datasets) bar for each stage is computed from the 7 dataset-level means.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES: List[str] = ["full", "no_C-DVAE", "no_GAT"]
STAGE_LABELS: Dict[str, str] = {
    "full": "full",
    "no_GAT": "no-GAT",
    "no_C-DVAE": "no-C-DVAE",
}

FIRST_FOUR = ["GSE115978", "GSE123139", "GSE153935", "GSE166555"]
LAST_THREE = ["lung_full", "sapiens_full", "Zhengsorted"]
ALL_SEVEN = FIRST_FOUR + LAST_THREE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "sclightgat_exp_results" / "ablation" / "ablation_metrics_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "sclightgat_exp_results" / "ablation"


def _stage_stats(df: pd.DataFrame, dataset: str, stage: str) -> tuple[float, float]:
    part = df[(df["dataset"] == dataset) & (df["stage"] == stage)]
    vals = pd.to_numeric(part["average_accuracy"], errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return np.nan, 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0


def _overall_stage_stats(df: pd.DataFrame, stage: str) -> tuple[float, float]:
    # Compute overall from 7 dataset-level means for this stage.
    means = []
    for ds in ALL_SEVEN:
        mean_val, _ = _stage_stats(df, ds, stage)
        if not np.isnan(mean_val):
            means.append(mean_val)
    if not means:
        return np.nan, 0.0
    arr = np.asarray(means, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _plot_grouped_bars(
    x_labels: List[str],
    means_by_stage: Dict[str, List[float]],
    errs_by_stage: Dict[str, List[float]],
    save_path: Path,
    errorbar_indices: List[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(x_labels), dtype=float)
    group_gap = 0.28
    x = x * (1.0 + group_gap)
    width = 0.22
    offsets = [-width, 0.0, width]
    colors = {
        "full": "#a3e9e1",
        "no_C-DVAE": "#91cef3",
        "no_GAT": "#e9aa9a",
    }

    for stage, off in zip(STAGES, offsets):
        bar_x = x + off
        ax.bar(
            bar_x,
            means_by_stage[stage],
            width=width,
            label=STAGE_LABELS[stage],
            color=colors[stage],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
        )

        # Draw error bars manually so we can suppress them for specific groups (e.g., overall).
        idxs = errorbar_indices if errorbar_indices is not None else list(range(len(x_labels)))
        if idxs:
            idxs_arr = np.asarray(idxs, dtype=int)
            y_vals = np.asarray(means_by_stage[stage], dtype=float)[idxs_arr]
            err_vals = np.asarray(errs_by_stage[stage], dtype=float)[idxs_arr]
            valid = np.isfinite(y_vals) & np.isfinite(err_vals) & (err_vals > 0)
            if np.any(valid):
                ax.errorbar(
                    bar_x[idxs_arr][valid],
                    y_vals[valid],
                    yerr=err_vals[valid],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=4,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    legend = ax.legend(
        title="Stage",
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation bar charts with error bars")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to ablation_metrics_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"dataset", "stage", "average_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    chart1_labels = FIRST_FOUR
    means1 = {stage: [] for stage in STAGES}
    errs1 = {stage: [] for stage in STAGES}
    for ds in chart1_labels:
        for stage in STAGES:
            m, s = _stage_stats(df, ds, stage)
            means1[stage].append(m)
            errs1[stage].append(s)

    chart2_labels = LAST_THREE + ["Overall"]
    means2 = {stage: [] for stage in STAGES}
    errs2 = {stage: [] for stage in STAGES}

    for ds in LAST_THREE:
        for stage in STAGES:
            m, s = _stage_stats(df, ds, stage)
            means2[stage].append(m)
            errs2[stage].append(s)

    for stage in STAGES:
        m_all, s_all = _overall_stage_stats(df, stage)
        means2[stage].append(m_all)
        errs2[stage].append(s_all)

    out1 = args.out_dir / "ablation_bar_chart_1.png"
    out2 = args.out_dir / "ablation_bar_chart_2.png"

    _plot_grouped_bars(
        x_labels=chart1_labels,
        means_by_stage=means1,
        errs_by_stage=errs1,
        save_path=out1,
        errorbar_indices=list(range(len(chart1_labels))),
    )
    _plot_grouped_bars(
        x_labels=chart2_labels,
        means_by_stage=means2,
        errs_by_stage=errs2,
        save_path=out2,
        errorbar_indices=list(range(len(LAST_THREE))),
    )

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
