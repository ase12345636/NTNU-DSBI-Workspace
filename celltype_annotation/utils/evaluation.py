"""Accuracy report parsing, metrics aggregation, and subtype analysis.

Matches batch_effect/utils/evaluation.py in style: all metric-related
logic lives here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# ============================================================
# Accuracy Report Parsing
# ============================================================

def parse_accuracy_report(report_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Extract weighted and average accuracy from a text report."""
    weighted, average = None, None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Weighted Accuracy:"):
            try: weighted = float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError): pass
        elif line.startswith("Average Accuracy:"):
            try: average = float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError): pass
        elif line.startswith(("Strict Accuracy:", "Overall Accuracy:")) and weighted is None:
            m = re.search(r":\s*([0-9]*\.?[0-9]+)", line)
            if m:
                weighted = float(m.group(1))
                if average is None:
                    average = weighted
    return weighted, average


_PER_CLASS_RE = re.compile(
    r"^\s*(.+?):\s*([0-9]*\.?[0-9]+)\s*\([0-9]*\.?[0-9]+%\),\s*support=([0-9]+)\s*$"
)


def parse_per_class_accuracy(report_path: Path) -> List[Dict[str, object]]:
    rows, in_section = [], False
    for line in report_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Per-Class Accuracy"):
            in_section = True; continue
        if not in_section:
            continue
        if not line or line.startswith("==="):
            if rows: break
            continue
        m = _PER_CLASS_RE.match(line)
        if m:
            rows.append({"celltype": m.group(1).strip(),
                         "accuracy": float(m.group(2)),
                         "support": int(m.group(3))})
    return rows


# ============================================================
# Metrics Aggregation
# ============================================================

def _infer_run_number(tag: str, fallback: int) -> int:
    lower = tag.lower()
    if lower.startswith("run"):
        digits = "".join(ch for ch in lower[3:] if ch.isdigit())
        if digits: return int(digits)
    if lower.isdigit():
        return int(lower)
    return fallback


def _iter_run_dirs(results_root, run_tags, exclude_datasets=None, include_datasets=None):
    """Yield (dataset_dir, idx, run_dir) tuples for matching dataset/run pairs."""
    if not results_root.exists():
        return
    excluded = {n.lower() for n in (exclude_datasets or [])}
    included = None if include_datasets is None else {n.lower() for n in include_datasets}

    for dd in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if included is not None and dd.name.lower() not in included:
            continue
        if dd.name.lower() in excluded:
            continue
        if run_tags is None:
            runs = sorted(p for p in dd.iterdir() if p.is_dir())
        else:
            runs = [dd / t for t in run_tags if (dd / t).is_dir()]
        for idx, rd in enumerate(runs, 1):
            yield dd, idx, rd


def collect_rows(results_root, run_tags=None, exclude_datasets=None, include_datasets=None):
    rows = []
    for dd, idx, rd in _iter_run_dirs(results_root, run_tags, exclude_datasets, include_datasets):
        rp = rd / "accuracy_report.txt"
        w, a = parse_accuracy_report(rp)
        rows.append({"dataset": dd.name, "run_number": _infer_run_number(rd.name, idx),
                      "weighted_accuracy": w, "average_accuracy": a,
                      "timestamp": rd.name, "report_path": str(rp) if rp.exists() else ""})
    return rows


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows found."
    parts = []
    for m in ["weighted_accuracy", "average_accuracy"]:
        v = pd.to_numeric(df[m], errors="coerce").dropna()
        if v.empty:
            parts.append(f"{m}: no valid values")
        else:
            parts.append(f"{m}: mean={v.mean():.4f}, std={v.std():.4f}, "
                         f"min={v.min():.4f}, max={v.max():.4f}")
    return "\n".join(parts)


def write_celltype_summary_csv(results_root, run_tags=None,
                               exclude_datasets=None, include_datasets=None):
    """Per-class accuracy summary across runs (detailed + aggregated CSVs)."""
    rows = []
    for dd, idx, rd in _iter_run_dirs(results_root, run_tags, exclude_datasets, include_datasets):
        rp = rd / "accuracy_report.txt"
        rn = _infer_run_number(rd.name, idx)
        for cr in parse_per_class_accuracy(rp):
            rows.append({"run_number": rn, "celltype": str(cr["celltype"]),
                         "accuracy": cr["accuracy"], "support": int(cr["support"]),
                         "timestamp": rd.name})

    results_root.mkdir(parents=True, exist_ok=True)
    detail_path = results_root / "majortype_detailed_summary.csv"
    summary_path = results_root / "celltype_metrics_summary.csv"

    if not rows:
        pd.DataFrame(columns=["run_number","celltype","accuracy","support","timestamp"]).to_csv(detail_path, index=False)
        pd.DataFrame(columns=["celltype","mean_accuracy","std_accuracy","min_accuracy","max_accuracy","total_support"]).to_csv(summary_path, index=False)
        return detail_path, summary_path

    df = pd.DataFrame(rows)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").round(4)
    df["support"] = pd.to_numeric(df["support"], errors="coerce").fillna(0).astype(int)
    df.sort_values(["run_number", "celltype"]).to_csv(detail_path, index=False)

    summary = (df.groupby("celltype", as_index=False)
               .agg(mean_accuracy=("accuracy","mean"), std_accuracy=("accuracy","std"),
                    min_accuracy=("accuracy","min"), max_accuracy=("accuracy","max"),
                    total_support=("support","sum"))
               .sort_values("celltype"))
    for c in ["mean_accuracy","std_accuracy","min_accuracy","max_accuracy"]:
        summary[c] = summary[c].round(4)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path


# ============================================================
# Subtype Accuracy Reports
# ============================================================

DEFAULT_BROAD_TO_SUBTYPES = {
    "B cells": ["Follicular B cells","Germinal B cells","MALT B cells","Memory B cells","Naive B cells"],
    "CD4+T cells": ["CD4+Tfh/Th cells","CD4+exhausted T cells","CD4+memory T cells","CD4+naive T cells","CD4+reg T cells"],
    "CD8+T cells": ["CD8+MAIT T cells","CD8+Naive T cells","CD8+exhausted T cells","CD8+memory T cells"],
    "DC": ["DC", "cDC"],
    "Plasma cells": ["IgA+ Plasma","IgG+ Plasma","Plasma cells","Plasmablasts"],
}


def write_subtype_accuracy_reports(adata, pred_col, output_dir, tool_name,
                                   explicit_gt_col="Celltype_subtraining",
                                   mapping=None):
    """Per-broad-type subtype accuracy reports + detailed CSV."""
    gt_col = explicit_gt_col if explicit_gt_col and explicit_gt_col in adata.obs.columns else None
    if gt_col is None or pred_col not in adata.obs.columns:
        return None

    mapping = mapping or DEFAULT_BROAD_TO_SUBTYPES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_all = adata.obs[gt_col].astype(str)
    y_pred_all = adata.obs[pred_col].astype(str)
    detail_rows, sections, broad_acc = [], [], {}

    for broad_type, subtypes in mapping.items():
        subtype_set = set(subtypes)
        mask = y_true_all.isin(subtype_set)
        if mask.sum() == 0:
            continue

        gt = y_true_all[mask].to_numpy()
        pred_mapped = np.array([p if p in subtype_set else "Other" for p in y_pred_all[mask].to_numpy()])
        n = len(gt)
        acc = float(np.mean(pred_mapped == gt))
        broad_acc[broad_type] = acc

        for st in subtypes:
            sm = gt == st
            if sm.sum() == 0: continue
            detail_rows.append({"broad_type": broad_type, "subtype": st,
                                "accuracy": round(float(np.mean(pred_mapped[sm] == st)), 4),
                                "support": int(sm.sum())})

        safe = broad_type.replace("+","plus").replace(" ","_")
        within = pred_mapped != "Other"
        within_acc = float(np.mean(pred_mapped[within] == gt[within])) if within.any() else 0.0
        header = (f"{broad_type}  -  Acc: {acc:.4f} ({acc*100:.2f}%), "
                  f"Within: {within_acc:.4f}, Cells: {n}")
        lines = ["="*55, header, "="*55]
        labels = [s for s in subtypes if (gt==s).any() or (pred_mapped==s).any()]
        if (pred_mapped=="Other").any(): labels.append("Other")
        lines.append(classification_report(gt, pred_mapped, labels=labels, digits=2, zero_division=0))
        sections.append("\n".join(lines))
        Path(output_dir / f"accuracy_report_{safe}.txt").write_text("\n".join(lines)+"\n")

    # Combined report
    all_sub = set()
    for v in mapping.values(): all_sub.update(v)
    pooled = y_true_all.isin(all_sub)
    overall = float(np.mean(y_pred_all[pooled].to_numpy() == y_true_all[pooled].to_numpy())) if pooled.any() else None

    header = ["="*60, "Subtype Experiment - Accuracy Report", "="*60, ""]
    if overall is not None:
        header.append(f"Overall Subtype Accuracy: {overall:.4f} ({overall*100:.2f}%)")
    header.append("\nPer-broad-type:")
    for bt in mapping:
        if bt in broad_acc:
            header.append(f"  {bt}: {broad_acc[bt]:.4f}")
    header.append("")
    Path(output_dir / "accuracy_report.txt").write_text("\n".join(header + sections)+"\n")

    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(output_dir / "subtype_detailed_accuracy.csv", index=False)
    return pd.DataFrame(detail_rows) if detail_rows else None


def collect_subtype_summary_csv(results_root, dataset_name="Subtype"):
    """Merge per-run subtype CSVs into summary."""
    dd = Path(results_root) / dataset_name
    if not dd.exists():
        return

    rows = []
    for idx, rd in enumerate(sorted(p for p in dd.iterdir() if p.is_dir()), 1):
        csv = rd / "subtype_detailed_accuracy.csv"
        if not csv.exists(): continue
        df = pd.read_csv(csv)
        if df.empty: continue
        rn = _infer_run_number(rd.name, idx)
        for _, r in df.iterrows():
            rows.append({"run_number": rn, "broad_type": str(r.get("broad_type","")),
                         "subtype": str(r.get("subtype","")),
                         "accuracy": pd.to_numeric(r.get("accuracy"), errors="coerce"),
                         "support": int(pd.to_numeric(r.get("support"), errors="coerce") or 0),
                         "timestamp": rd.name})

    detailed_csv = dd / "subtype_detailed_summary.csv"
    if not rows:
        pd.DataFrame(columns=["run_number","broad_type","subtype","accuracy","support","timestamp"]).to_csv(detailed_csv, index=False)
        return

    det = pd.DataFrame(rows).sort_values(["run_number","broad_type","subtype"]).reset_index(drop=True)
    det["accuracy"] = pd.to_numeric(det["accuracy"], errors="coerce").round(4)
    det.to_csv(detailed_csv, index=False)

    metric_key = {"B cells": "b_cells_avg_acc", "CD4+T cells": "cd4t_avg_acc",
                  "CD8+T cells": "cd8t_avg_acc", "DC": "dc_avg_acc", "Plasma cells": "plasma_avg_acc"}
    summary_rows = []
    for rn in sorted(det["run_number"].dropna().unique()):
        rdf = det[det["run_number"] == rn]
        row = {"dataset": dataset_name, "run_number": int(rn)}
        for bt, key in metric_key.items():
            vals = pd.to_numeric(rdf[rdf["broad_type"]==bt]["accuracy"], errors="coerce").dropna()
            row[key] = round(float(vals.mean()), 4) if len(vals) else np.nan
        summary_rows.append(row)

    cols = ["dataset","run_number"] + list(metric_key.values())
    pd.DataFrame(summary_rows)[cols].to_csv(dd / "subtype_metrics_summary.csv", index=False)
    print(f"[subtype-metrics] {detailed_csv}")
