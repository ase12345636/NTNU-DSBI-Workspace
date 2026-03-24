#!/usr/bin/env python3
"""Plot subtype benchmark bar charts from CSV summaries.

Outputs:
1) One chart per broad type (including CAF) with subtype names on x-axis.
   - For CAF: use text inside parentheses as x labels when available.
   - For CAF: append an extra x label "Overall".
2) One summary chart with x-axis as broad types + final "Overall".

Design notes:
- This is the "future complete data" script: it relies on CSV sources.
- Missing files/methods/subtypes are skipped safely.
- Source policy is strict CSV-only:
    - scLightGAT: sclightgat_exp_results
    - other methods: benchmark/results only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS: List[str] = ["scLightGAT", "Celltypist", "CHETAH", "scGPT", "Seurat", "SingleR"]
BROAD_TYPES: List[str] = ["B cells", "CD4+T cells", "CD8+T cells", "DC", "Plasma cells", "CAF"]
NON_CAF_BROAD_TYPES: List[str] = [b for b in BROAD_TYPES if b != "CAF"]

CANONICAL_SUBTYPE_ORDER: Dict[str, List[str]] = {
    "B cells": [
        "Follicular B cells",
        "Germinal B cells",
        "MALT B cells",
        "Memory B cells",
        "Naive B cells",
    ],
    "CD4+T cells": [
        "CD4+Tfh/Th cells",
        "CD4+exhausted T cells",
        "CD4+memory T cells",
        "CD4+naive T cells",
        "CD4+reg T cells",
    ],
    "CD8+T cells": [
        "CD8+MAIT T cells",
        "CD8+Naive T cells",
        "CD8+exhausted T cells",
        "CD8+memory T cells",
    ],
    "DC": ["DC", "cDC"],
    "Plasma cells": ["IgA+ Plasma", "IgG+ Plasma", "Plasma cells", "Plasmablasts"],
}

METHOD_COLORS: Dict[str, str] = {
    "scLightGAT": "#8dd3c7",
    "Celltypist": "#80b1d3",
    "CHETAH": "#fdb462",
    "scGPT": "#b3de69",
    "Seurat": "#fccde5",
    "SingleR": "#fb8072",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "sclightgat_exp_results" / "benchmark_comparison" / "subtype"

METHOD_ROOTS: Dict[str, List[Path]] = {
    "scLightGAT": [PROJECT_ROOT / "sclightgat_exp_results"],
    "Celltypist": [PROJECT_ROOT / "benchmark" / "results" / "Celltypist"],
    "CHETAH": [PROJECT_ROOT / "benchmark" / "results" / "CHETAH"],
    "scGPT": [PROJECT_ROOT / "benchmark" / "results" / "scGPT"],
    "Seurat": [PROJECT_ROOT / "benchmark" / "results" / "Seurat"],
    "SingleR": [PROJECT_ROOT / "benchmark" / "results" / "SingleR"],
}


def _safe_name(name: str) -> str:
    return name.replace("+", "plus").replace(" ", "_")


def _extract_paren_short_name(label: str) -> str:
    m = re.search(r"\(([^()]+)\)", label)
    if m:
        return m.group(1).strip()
    return label.strip()


def _arr_stats(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, 0.0
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean_val, std_val


def _read_csv_safe(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def _find_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _load_subtype_detailed_df(method: str) -> pd.DataFrame:
    roots = METHOD_ROOTS.get(method, [])
    candidates_rel = [
        Path("Subtype") / "hierarchical" / "subtype_detailed_summary.csv",
        Path("Subtype") / "subtype_detailed_summary.csv",
        Path("subtype_detailed_summary.csv"),
    ]

    for root in roots:
        csv_path = _find_first_existing(root / rel for rel in candidates_rel)
        if csv_path is not None:
            df = _read_csv_safe(csv_path)
            required = {"broad_type", "subtype", "accuracy"}
            if required.issubset(df.columns):
                return df

    return pd.DataFrame(columns=["run_number", "broad_type", "subtype", "accuracy", "support", "timestamp"])


def _load_caf_detailed_df(method: str) -> pd.DataFrame:
    roots = METHOD_ROOTS.get(method, [])
    candidates_rel = [
        Path("caf.mode") / "majortype_detailed_summary.csv",
        Path("caf.mode") / "celltype_detailed_summary.csv",
        Path("caf.mode") / "CAF" / "majortype_detailed_summary.csv",
        Path("caf.mode") / "CAF" / "celltype_detailed_summary.csv",
    ]

    for root in roots:
        csv_path = _find_first_existing(root / rel for rel in candidates_rel)
        if csv_path is not None:
            df = _read_csv_safe(csv_path)
            if {"celltype", "accuracy"}.issubset(df.columns):
                return df

    return pd.DataFrame(columns=["run_number", "celltype", "accuracy", "support", "timestamp"])


def _load_caf_summary_df(method: str) -> pd.DataFrame:
    roots = METHOD_ROOTS.get(method, [])
    candidates_rel = [
        Path("caf.mode") / "celltype_metrics_summary.csv",
        Path("caf.mode") / "CAF" / "celltype_metrics_summary.csv",
        Path("caf.mode") / "CAF" / "metrics_summary.csv",
        Path("caf.mode") / "metrics_summary.csv",
    ]

    for root in roots:
        csv_path = _find_first_existing(root / rel for rel in candidates_rel)
        if csv_path is not None:
            return _read_csv_safe(csv_path)

    return pd.DataFrame()


def _subtype_values_from_csv(method: str, broad_type: str, subtype: str) -> np.ndarray:
    df = _load_subtype_detailed_df(method)
    if df.empty:
        return np.asarray([], dtype=float)

    part = df[(df["broad_type"].astype(str) == broad_type) & (df["subtype"].astype(str) == subtype)]
    vals = pd.to_numeric(part["accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
    return vals


def _broad_overall_values_from_csv(method: str, broad_type: str) -> np.ndarray:
    if broad_type == "CAF":
        df = _load_caf_summary_df(method)
        if df.empty:
            return np.asarray([], dtype=float)

        metric_col = None
        for col in ["average_accuracy", "mean_accuracy", "accuracy", "weighted_accuracy"]:
            if col in df.columns:
                metric_col = col
                break
        if metric_col is None:
            return np.asarray([], dtype=float)

        vals = pd.to_numeric(df[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
        return vals

    df = _load_subtype_detailed_df(method)
    if df.empty:
        return np.asarray([], dtype=float)
    part = df[df["broad_type"].astype(str) == broad_type]
    if part.empty:
        return np.asarray([], dtype=float)
    # Compute weighted overall accuracy per run from individual subtypes
    per_run: list[float] = []
    for _, run_df in part.groupby("run_number"):
        accs = pd.to_numeric(run_df["accuracy"], errors="coerce")
        sups = pd.to_numeric(run_df["support"], errors="coerce")
        valid = accs.notna() & sups.notna() & (sups > 0)
        if valid.any():
            per_run.append(float((accs[valid] * sups[valid]).sum() / sups[valid].sum()))
    return np.asarray(per_run, dtype=float)


def _caf_subtype_values_from_csv(method: str, subtype_display: str) -> np.ndarray:
    detailed_df = _load_caf_detailed_df(method)
    if not detailed_df.empty and {"celltype", "accuracy"}.issubset(detailed_df.columns):
        labels = detailed_df["celltype"].astype(str).tolist()
        mapped = [_extract_paren_short_name(x) for x in labels]
        part = detailed_df[np.asarray(mapped, dtype=object) == subtype_display]
        vals = pd.to_numeric(part["accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size > 0:
            return vals

    summary_df = _load_caf_summary_df(method)
    if {"celltype", "mean_accuracy"}.issubset(summary_df.columns):
        temp = summary_df.copy()
        temp["mapped"] = temp["celltype"].astype(str).map(_extract_paren_short_name)
        part = temp[temp["mapped"] == subtype_display]
        vals = pd.to_numeric(part["mean_accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
        return vals

    return np.asarray([], dtype=float)


def _collect_non_caf_subtype_labels() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {b: [] for b in NON_CAF_BROAD_TYPES}

    for broad in NON_CAF_BROAD_TYPES:
        available: set[str] = set()
        for method in METHODS:
            df = _load_subtype_detailed_df(method)
            if df.empty:
                continue
            part = df[df["broad_type"].astype(str) == broad]
            if part.empty:
                continue
            available.update(part["subtype"].astype(str).tolist())

        canonical = CANONICAL_SUBTYPE_ORDER.get(broad, [])
        labels = [s for s in canonical if s in available]
        out[broad] = labels

    return out


def _collect_caf_labels() -> List[str]:
    labels: List[str] = []
    for method in METHODS:
        detailed_df = _load_caf_detailed_df(method)
        if {"celltype"}.issubset(detailed_df.columns):
            for raw in detailed_df["celltype"].astype(str).tolist():
                short = _extract_paren_short_name(raw)
                if short not in labels:
                    labels.append(short)

        summary_df = _load_caf_summary_df(method)
        if {"celltype"}.issubset(summary_df.columns):
            for raw in summary_df["celltype"].astype(str).tolist():
                short = _extract_paren_short_name(raw)
                if short not in labels:
                    labels.append(short)

    labels = [x for x in labels if x.lower() != "overall"]
    return labels


def _plot_grouped_bars(
    x_labels: List[str],
    means_by_method: Dict[str, List[float]],
    errs_by_method: Dict[str, List[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    errorbar_indices: Iterable[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(11, 1.2 * len(x_labels) + 6), 6.8))

    n_groups = len(x_labels)
    n_methods = len(METHODS)

    x = np.arange(n_groups, dtype=float)
    group_gap = 0.30
    x = x * (1.0 + group_gap)

    width = 0.11
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
            label=method,
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

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
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
    parser = argparse.ArgumentParser(description="Plot subtype benchmark comparison from CSV")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
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

    non_caf_labels = _collect_non_caf_subtype_labels()
    caf_labels = _collect_caf_labels()

    broad_overall_stats: Dict[str, Dict[str, Tuple[float, float]]] = {
        m: {} for m in METHODS
    }

    for broad in BROAD_TYPES:
        if broad == "CAF":
            x_labels = caf_labels + ["Overall"]
        else:
            x_labels = non_caf_labels.get(broad, [])

        if not x_labels:
            continue

        means_by_method = {m: [] for m in METHODS}
        errs_by_method = {m: [] for m in METHODS}

        for method in METHODS:
            if broad == "CAF":
                for subtype_label in caf_labels:
                    vals = _caf_subtype_values_from_csv(method, subtype_label)
                    mean_v, std_v = _arr_stats(vals)
                    means_by_method[method].append(mean_v)
                    errs_by_method[method].append(std_v)

                overall_vals = _broad_overall_values_from_csv(method, "CAF")
                mean_v, std_v = _arr_stats(overall_vals)
                means_by_method[method].append(mean_v)
                errs_by_method[method].append(std_v)
                broad_overall_stats[method]["CAF"] = (mean_v, std_v)
            else:
                for subtype_label in x_labels:
                    vals = _subtype_values_from_csv(method, broad, subtype_label)
                    mean_v, std_v = _arr_stats(vals)
                    means_by_method[method].append(mean_v)
                    errs_by_method[method].append(std_v)

                overall_vals = _broad_overall_values_from_csv(method, broad)
                mean_v, std_v = _arr_stats(overall_vals)
                broad_overall_stats[method][broad] = (mean_v, std_v)

        title = f"{broad}"
        out_path = args.out_dir / f"subtype_{_safe_name(broad)}_methods_bar_chart_csv.png"

        err_indices = range(len(x_labels) - 1) if broad == "CAF" else range(len(x_labels))
        _plot_grouped_bars(
            x_labels=x_labels,
            means_by_method=means_by_method,
            errs_by_method=errs_by_method,
            title=title,
            ylabel="Accuracy",
            save_path=out_path,
            errorbar_indices=err_indices,
        )
        print(f"Saved: {out_path}")

    # Broad summary excludes CAF; CAF remains in its dedicated chart.
    summary_x = NON_CAF_BROAD_TYPES + ["Overall"]
    summary_means = {m: [] for m in METHODS}
    summary_errs = {m: [] for m in METHODS}

    for method in METHODS:
        broad_vals: List[float] = []
        for broad in NON_CAF_BROAD_TYPES:
            mv, sv = broad_overall_stats[method].get(broad, (np.nan, 0.0))
            summary_means[method].append(mv)
            summary_errs[method].append(sv)
            broad_vals.append(mv)

        overall_mv, overall_sv = _arr_stats(broad_vals)
        summary_means[method].append(overall_mv)
        summary_errs[method].append(overall_sv)

    summary_out = args.out_dir / "subtype_broadtype_summary_methods_bar_chart_csv.png"
    _plot_grouped_bars(
        x_labels=summary_x,
        means_by_method=summary_means,
        errs_by_method=summary_errs,
        title="Subtype Broad-Type Summary by Method (CSV)",
        ylabel="Accuracy",
        save_path=summary_out,
        errorbar_indices=range(len(NON_CAF_BROAD_TYPES)),
    )
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
