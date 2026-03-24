#!/usr/bin/env python3
"""Plot benchmark grouped bar charts from CSV summaries.

This script is designed for the "fully completed" case where benchmark outputs
have been aggregated into CSV files (for example: metrics_summary.csv).

Requirements implemented:
- Methods in legend: scLightGAT, Celltypist, CHETAH, scGPT, Seurat, SingleR
- Two charts with the same dataset grouping logic as the ablation plot
- Missing/incomplete data is skipped safely (bar becomes empty)
- Subtype/CAF is ignored by only using 7 target datasets
- Primary source is benchmark/results, fallback is results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS: List[str] = ["scLightGAT", "Celltypist", "CHETAH", "scGPT", "Seurat", "SingleR"]

METHOD_LABELS: Dict[str, str] = {
    "scLightGAT": "scLightGAT",
    "Celltypist": "Celltypist",
    "CHETAH": "CHETAH",
    "scGPT": "scGPT",
    "Seurat": "Seurat",
    "SingleR": "SingleR",
}

FIRST_FOUR = ["GSE115978", "GSE123139", "GSE153935", "GSE166555"]
LAST_THREE = ["lung_full", "sapiens_full", "Zhengsorted"]
ALL_SEVEN = FIRST_FOUR + LAST_THREE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "sclightgat_exp_results" / "benchmark_comparison"

# Benchmark is the primary source. Non-scLightGAT methods can fall back to results/.
METHOD_ROOTS: Dict[str, List[Path]] = {
    "scLightGAT": [PROJECT_ROOT / "sclightgat_exp_results"],
    "Celltypist": [PROJECT_ROOT / "benchmark" / "results" / "Celltypist", PROJECT_ROOT / "results" / "Celltypist"],
    "CHETAH": [PROJECT_ROOT / "benchmark" / "results" / "CHETAH", PROJECT_ROOT / "results" / "CHETAH"],
    "scGPT": [PROJECT_ROOT / "benchmark" / "results" / "scGPT", PROJECT_ROOT / "results" / "scGPT"],
    "Seurat": [PROJECT_ROOT / "benchmark" / "results" / "Seurat", PROJECT_ROOT / "results" / "Seurat"],
    "SingleR": [PROJECT_ROOT / "benchmark" / "results" / "SingleR", PROJECT_ROOT / "results" / "SingleR"],
}

METHOD_COLORS: Dict[str, str] = {
    "scLightGAT": "#8dd3c7",
    "Celltypist": "#80b1d3",
    "CHETAH": "#fdb462",
    "scGPT": "#b3de69",
    "Seurat": "#fccde5",
    "SingleR": "#fb8072",
}


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def _extract_metric_column(df: pd.DataFrame) -> str | None:
    # Keep compatibility with benchmark CSV schemas.
    for col in ["average_accuracy", "mean_accuracy", "accuracy", "weighted_accuracy"]:
        if col in df.columns:
            return col
    return None


def _read_values_from_csv(csv_path: Path, dataset: str) -> List[float]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    metric_col = _extract_metric_column(df)
    if metric_col is None:
        return []

    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == dataset]

    vals = _safe_numeric(df[metric_col])
    return vals.tolist()


def _candidate_csv_paths(root: Path, dataset: str) -> List[Path]:
    # Priority order: dataset metrics_summary -> dataset any csv -> root metrics_summary.
    candidates: List[Path] = []
    ds_dir = root / dataset

    ds_metrics = ds_dir / "metrics_summary.csv"
    if ds_metrics.exists():
        candidates.append(ds_metrics)

    if ds_dir.exists():
        for p in sorted(ds_dir.glob("*.csv")):
            if p not in candidates:
                candidates.append(p)

    root_metrics = root / "metrics_summary.csv"
    if root_metrics.exists() and root_metrics not in candidates:
        candidates.append(root_metrics)

    return candidates


def _collect_values_from_csvs(method: str, dataset: str) -> np.ndarray:
    roots = METHOD_ROOTS.get(method, [])
    for root in roots:
        if not root.exists():
            continue

        for csv_path in _candidate_csv_paths(root, dataset):
            vals = _read_values_from_csv(csv_path, dataset)
            if vals:
                return np.asarray(vals, dtype=float)

    return np.asarray([], dtype=float)


def _method_dataset_stats(method: str, dataset: str) -> tuple[float, float]:
    vals = _collect_values_from_csvs(method, dataset)
    if vals.size == 0:
        return np.nan, 0.0
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return mean_val, std_val


def _overall_method_stats(dataset_means: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(dataset_means, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, 0.0
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean_val, std_val


def _plot_grouped_bars(
    x_labels: List[str],
    means_by_method: Dict[str, List[float]],
    errs_by_method: Dict[str, List[float]],
    save_path: Path,
    errorbar_indices: Iterable[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    n_groups = len(x_labels)
    n_methods = len(METHODS)

    x = np.arange(n_groups, dtype=float)
    group_gap = 0.34
    x = x * (1.0 + group_gap)

    width = 0.12
    center = (n_methods - 1) / 2.0

    if errorbar_indices is None:
        err_idxs = np.arange(n_groups, dtype=int)
    else:
        err_idxs = np.asarray(list(errorbar_indices), dtype=int)

    for i, method in enumerate(METHODS):
        offset = (i - center) * width
        bar_x = x + offset

        y = np.asarray(means_by_method[method], dtype=float)
        e = np.asarray(errs_by_method[method], dtype=float)

        ax.bar(
            bar_x,
            y,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
        )

        if err_idxs.size > 0:
            valid_idx = err_idxs[(err_idxs >= 0) & (err_idxs < n_groups)]
            if valid_idx.size > 0:
                y_sub = y[valid_idx]
                e_sub = e[valid_idx]
                valid = np.isfinite(y_sub) & np.isfinite(e_sub) & (e_sub > 0)
                if np.any(valid):
                    ax.errorbar(
                        bar_x[valid_idx][valid],
                        y_sub[valid],
                        yerr=e_sub[valid],
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.0,
                        capsize=3.5,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend = ax.legend(
        title="Method",
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout(rect=[0, 0, 0.86, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global METHOD_ROOTS
    parser = argparse.ArgumentParser(description="Plot benchmark method comparison from CSV summaries")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Root directory containing per-method result folders (e.g. result/)",
    )
    parser.add_argument(
        "--sclightgat_dir",
        type=Path,
        default=None,
        help="Root directory for scLightGAT results (e.g. /path/to/sclightgat_exp_results)",
    )
    args = parser.parse_args()

    if args.results_dir:
        for m in METHODS:
            if m == "scLightGAT":
                continue
            METHOD_ROOTS[m] = [args.results_dir / m] + METHOD_ROOTS.get(m, [])
    if args.sclightgat_dir:
        METHOD_ROOTS["scLightGAT"] = [args.sclightgat_dir] + METHOD_ROOTS.get("scLightGAT", [])

    chart1_labels = FIRST_FOUR
    means1 = {method: [] for method in METHODS}
    errs1 = {method: [] for method in METHODS}

    for ds in chart1_labels:
        for method in METHODS:
            m, s = _method_dataset_stats(method, ds)
            means1[method].append(m)
            errs1[method].append(s)

    chart2_labels = LAST_THREE + ["Overall"]
    means2 = {method: [] for method in METHODS}
    errs2 = {method: [] for method in METHODS}

    for method in METHODS:
        dataset_means: List[float] = []

        for ds in LAST_THREE:
            m, s = _method_dataset_stats(method, ds)
            means2[method].append(m)
            errs2[method].append(s)

        for ds in ALL_SEVEN:
            m_all_ds, _ = _method_dataset_stats(method, ds)
            dataset_means.append(m_all_ds)

        m_all, s_all = _overall_method_stats(dataset_means)
        means2[method].append(m_all)
        errs2[method].append(s_all)

    out1 = args.out_dir / "benchmark_methods_bar_chart_1_csv.png"
    out2 = args.out_dir / "benchmark_methods_bar_chart_2_csv.png"

    _plot_grouped_bars(
        x_labels=chart1_labels,
        means_by_method=means1,
        errs_by_method=errs1,
        save_path=out1,
        errorbar_indices=range(len(chart1_labels)),
    )

    # Keep "Overall" without error bars, same convention as your original plot.
    _plot_grouped_bars(
        x_labels=chart2_labels,
        means_by_method=means2,
        errs_by_method=errs2,
        save_path=out2,
        errorbar_indices=range(len(LAST_THREE)),
    )

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
