"""Unified pipeline for cell type annotation benchmarks.

Each tool defines train_fn / predict_fn, then calls
run_annotation_pipeline() — same pattern as batch_effect/utils/pipeline.py.

    train_fn(train_path, model_dir, label_col, seed, args) -> model_handle
    predict_fn(test_path, model_handle, pred_col, seed, args) -> AnnData
"""

import shutil
import subprocess
import random
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy.io import mmwrite
from scipy.sparse import issparse, csc_matrix

from utils.label_utils import (
    generate_accuracy_report,
    get_broad_type,
    get_ground_truth_column,
    standardize_labels_series,
)

BROAD_TYPES_FOR_SUBTYPE = [
    "CD4+T cells", "CD8+T cells", "B cells", "Plasma cells", "DC",
]


# ============================================================
# I/O Helpers
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def load_adata(path):
    return ad.read_h5ad(Path(path))


# ============================================================
# Label Helpers
# ============================================================

def resolve_label_col(adata, preferred=None):
    if preferred and preferred in adata.obs.columns:
        return preferred
    for col in ["Celltype_training", "Ground Truth", "Manual_celltype", "final_celltype"]:
        if col in adata.obs.columns:
            return col
    auto = get_ground_truth_column(adata)
    if auto is not None:
        return auto
    raise ValueError("Unable to infer label column from adata.obs")


# ============================================================
# Seed & Run Utilities
# ============================================================

def seed_python_numpy(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate_run_seeds(run_time, base_seed=None):
    if base_seed is not None:
        return [base_seed + i for i in range(run_time)]
    rng = random.Random(time.time_ns())
    seeds = set()
    while len(seeds) < run_time:
        seeds.add(rng.randrange(1, 2**31 - 1))
    return sorted(seeds)


def build_seed_plan(run_time, base_seed=None):
    major = generate_run_seeds(run_time, base_seed)
    sub_base = None if base_seed is None else base_seed + run_time
    sub = generate_run_seeds(run_time, sub_base)
    return major, sub


# ============================================================
# R Interop (Python ↔ R data exchange)
# ============================================================

def export_adata_for_r(adata, output_dir):
    """Export adata as mtx + csv for native R loading (no reticulate)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = adata.X
    if not issparse(X):
        X = csc_matrix(X)
    mmwrite(str(output_dir / "matrix.mtx"), X.T)
    pd.DataFrame({"barcode": adata.obs_names}).to_csv(output_dir / "barcodes.csv", index=False)
    pd.DataFrame({"feature": adata.var_names}).to_csv(output_dir / "features.csv", index=False)
    adata.obs.to_csv(output_dir / "metadata.csv")


def import_predictions_from_r(adata, csv_path, pred_col):
    """Read predictions CSV from R and attach to adata.obs."""
    df = pd.read_csv(csv_path)
    preds = pd.Series(df["prediction"].values, index=df["barcode"].astype(str).values)
    adata.obs[pred_col] = preds.reindex(adata.obs_names.astype(str)).values
    return adata


def train_r_tool(train_path, model_dir, label_col, seed, r_script,
                 extra_args=None):
    """Generic R-based training: export → Rscript → cleanup → return .rds."""
    ensure_dir(model_dir)
    r_exchange = model_dir / "r_exchange"
    export_adata_for_r(load_adata(train_path), r_exchange)
    cmd = ["Rscript", str(r_script), "--task", "train_reference",
           "--input_dir", str(r_exchange), "--save_path", str(model_dir),
           "--label_col", label_col, "--seed", str(seed)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)
    shutil.rmtree(r_exchange)
    return model_dir / "reference.rds"


def predict_r_tool(test_path, rds_path, pred_col, seed, r_script,
                   extra_args=None):
    """Generic R-based prediction: export → Rscript → read predictions → cleanup."""
    adata = load_adata(test_path)
    tmp = rds_path.parent / "_predict_tmp"
    ensure_dir(tmp)
    r_exchange = tmp / "r_exchange"
    export_adata_for_r(adata, r_exchange)
    cmd = ["Rscript", str(r_script), "--task", "predict",
           "--input_dir", str(r_exchange), "--reference_rds", str(rds_path),
           "--save_path", str(tmp), "--seed", str(seed)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)
    import_predictions_from_r(adata, tmp / "predictions.csv", pred_col)
    shutil.rmtree(tmp)
    return adata


# ============================================================
# Argument Parser
# ============================================================

def create_argument_parser(tool_name):
    parser = argparse.ArgumentParser(description=f"{tool_name} cell type annotation benchmark")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument("--run_time", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--caf_mode", "--caf", action="store_true")
    return parser


def _resolve_paths(args, tool_name):
    dp = args.data_path
    data = {
        "train":         dp / "Integrated_training" / "train.h5ad",
        "test_dir":      dp / "Independent_testing",
        "subtype_train": dp / "Subtype" / "train.h5ad",
        "subtype_test":  dp / "Subtype" / "test.h5ad",
        "caf_train":     dp / "caf.data" / "caf_train.h5ad",
        "caf_test":      dp / "caf.data" / "caf_test.h5ad",
    }
    benchmark_dir = Path(__file__).resolve().parent.parent
    if args.save_path:
        results_root = args.save_path / tool_name
    else:
        results_root = benchmark_dir / "results" / tool_name
    model_root = benchmark_dir / tool_name / "models"
    return data, results_root, model_root


# ============================================================
# Post-Prediction Helpers
# ============================================================

def save_predictions(adata, pred_col, tool_name, test_name, output_dir,
                     gt_col=None, caf_mode=False):
    from utils.function import write_umap_figures, write_confusion_matrix

    ensure_dir(output_dir)
    adata.obs["scLightGAT_pred"] = adata.obs[pred_col].astype(str)
    adata.write_h5ad(output_dir / "adata_with_predictions.h5ad")

    write_umap_figures(adata, pred_col, output_dir, tool_name,
                       explicit_gt_col=gt_col, use_major_type_filter=not caf_mode)

    report = _build_accuracy_report(adata, pred_col, test_name,
                                    explicit_gt_col=gt_col, caf_mode=caf_mode)
    if report is not None:
        (output_dir / "accuracy_report.txt").write_text(report, encoding="utf-8")

    write_confusion_matrix(adata, pred_col, output_dir,
                           explicit_gt_col=gt_col,
                           title=f"{tool_name} Prediction Confusion Matrix",
                           use_major_type=not caf_mode)
    print(f"[predict] {test_name} -> {output_dir}")


def _build_accuracy_report(adata, pred_col, dataset_name,
                           explicit_gt_col=None, caf_mode=False):
    gt_col = explicit_gt_col or get_ground_truth_column(adata)
    if gt_col is None or gt_col not in adata.obs.columns:
        return None
    y_true = standardize_labels_series(adata.obs[gt_col].astype(str))
    y_pred = standardize_labels_series(adata.obs[pred_col].astype(str))
    if not caf_mode:
        return generate_accuracy_report(y_true, y_pred, dataset_name=dataset_name)
    return generate_accuracy_report(
        y_true, y_pred, dataset_name=f"{dataset_name} (CAF strict)",
        allowed_celltypes_only=False, myeloid_group=False,
        fine_grained_tolerance=False, epithelial_tolerance=False,
    )


def _write_metrics(results_root, run_tags, exclude=None, include=None):
    from utils.evaluation import collect_rows, summarize

    rows = collect_rows(results_root, run_tags,
                        exclude_datasets=exclude, include_datasets=include)
    df = pd.DataFrame(rows)
    ensure_dir(results_root)
    csv = results_root / "metrics_summary.csv"
    df.to_csv(csv, index=False)
    if not df.empty:
        for ds, ds_df in df.groupby("dataset"):
            p = results_root / ds / "metrics_summary.csv"
            ensure_dir(p.parent)
            ds_df.to_csv(p, index=False)
    print(f"[metrics] {csv}")
    print(summarize(df))


# ============================================================
# Subtype Orchestration
# ============================================================

def default_train_subtype(train_fn, subtype_train_path, model_dir, seed, args):
    adata = load_adata(subtype_train_path)
    subtype_col = "Celltype_subtraining"

    broad_col = ("Celltype_training"
                 if "Celltype_training" in adata.obs.columns else None)
    if broad_col is None:
        adata.obs["_derived_broad"] = adata.obs[subtype_col].astype(str).map(get_broad_type)
        broad_col = "_derived_broad"

    ensure_dir(model_dir)
    broad_handle = train_fn(subtype_train_path, model_dir / "broad",
                            broad_col, seed, args)

    subtype_handles = {}
    for bt in BROAD_TYPES_FOR_SUBTYPE:
        mask = adata.obs[broad_col].astype(str) == bt
        subset = adata[mask].copy()
        if subset.n_obs == 0 or subset.obs[subtype_col].nunique() < 2:
            continue
        safe = bt.replace(" ", "_").replace("+", "plus")
        tmp = model_dir / f"temp_{safe}.h5ad"
        subset.write_h5ad(tmp)
        sub_handle = train_fn(tmp, model_dir / f"subtype_{safe}",
                              subtype_col, seed, args)
        subtype_handles[bt] = sub_handle
        tmp.unlink(missing_ok=True)
        print(f"[subtype-train] {bt}: done")

    return broad_handle, subtype_handles


def save_subtype_results(out, pred_col, bp, sp, test_path, output_dir, tool_name):
    """Shared post-prediction save for subtype experiments."""
    from utils.function import write_umap_figures, write_confusion_matrix, write_subtype_figures
    from utils.evaluation import write_subtype_accuracy_reports

    ensure_dir(output_dir)
    out.obs["scLightGAT_pred"] = out.obs[pred_col].astype(str)
    out.write_h5ad(output_dir / "adata_with_predictions.h5ad")

    write_umap_figures(out, pred_col, output_dir, tool_name, use_major_type_filter=False)

    broad_gt = "Celltype_training" if "Celltype_training" in out.obs.columns else None
    sub_gt = "Celltype_subtraining" if "Celltype_subtraining" in out.obs.columns else None

    if sub_gt:
        write_subtype_figures(out, pred_col, output_dir, tool_name, explicit_gt_col=sub_gt)
        write_subtype_accuracy_reports(out, pred_col, output_dir, tool_name, explicit_gt_col=sub_gt)

    parts = []
    if broad_gt:
        parts.append(generate_accuracy_report(
            standardize_labels_series(out.obs[broad_gt].astype(str)),
            standardize_labels_series(out.obs[bp].astype(str)),
            dataset_name=f"{test_path.stem} (Broad)"))
    if sub_gt:
        parts.append(generate_accuracy_report(
            standardize_labels_series(out.obs[sub_gt].astype(str)),
            standardize_labels_series(out.obs[pred_col].astype(str)),
            dataset_name=f"{test_path.stem} (Final Subtype)"))
    if parts:
        (output_dir / "accuracy_report.txt").write_text("\n\n".join(parts), encoding="utf-8")

    write_confusion_matrix(out, pred_col, output_dir,
                           explicit_gt_col=sub_gt or broad_gt,
                           title=f"{tool_name} Prediction Confusion Matrix",
                           use_major_type=False)
    print(f"[subtype-predict] {test_path.name} -> {output_dir}")


def default_subtype_predict(predict_fn, test_path, broad_handle,
                            subtype_handles, output_dir, tool_name,
                            pred_col, seed, args):
    prefix = pred_col.replace("_pred", "")
    bp = f"{prefix}_broad_pred"
    sp = f"{prefix}_subtype_pred"

    adata = predict_fn(test_path, broad_handle, bp, seed, args)
    out = adata.copy()
    out.obs[sp] = out.obs[bp].astype(str)
    out.obs[pred_col] = out.obs[bp].astype(str)

    ensure_dir(output_dir)
    for bt, sub_h in subtype_handles.items():
        mask = out.obs[bp].astype(str) == bt
        if mask.sum() == 0:
            continue
        subset = out[mask].copy()
        safe = bt.replace(" ", "_").replace("+", "plus")
        tmp = output_dir / f"_tmp_{safe}.h5ad"
        subset.write_h5ad(tmp)

        a_sub = predict_fn(tmp, sub_h, sp, seed, args)
        preds = a_sub.obs[sp].astype(str)
        broad_check = preds.map(get_broad_type)
        accepted = preds.where(broad_check == bt, bt)

        out.obs.loc[subset.obs_names, sp] = preds.values
        out.obs.loc[subset.obs_names, pred_col] = accepted.values
        tmp.unlink(missing_ok=True)

    save_subtype_results(out, pred_col, bp, sp, test_path, output_dir, tool_name)


# ============================================================
# Main Pipeline
# ============================================================

def run_annotation_pipeline(tool_name, pred_col, train_fn, predict_fn, args,
                            train_subtype_fn=None, subtype_predict_fn=None):
    from utils.evaluation import collect_subtype_summary_csv, write_celltype_summary_csv

    data, results_root, model_root = _resolve_paths(args, tool_name)
    major_seeds, subtype_seeds = build_seed_plan(args.run_time, base_seed=args.base_seed)

    if args.caf_mode:
        print(f"\n{'='*60}\n[{tool_name}] CAF Mode\n{'='*60}")
        caf_root = results_root / "caf.mode"
        caf_models = model_root / "caf_mode" / "major_type"
        tags = []
        for i, seed in enumerate(major_seeds, 1):
            seed_python_numpy(seed)
            tag = str(i)
            tags.append(tag)
            print(f"\nCAF Run {i}/{args.run_time} (seed={seed})")
            h = train_fn(data["caf_train"], caf_models / tag,
                         "Celltype_training", seed, args)
            a = predict_fn(data["caf_test"], h, pred_col, seed, args)
            save_predictions(a, pred_col, tool_name, data["caf_test"].stem,
                             caf_root / "CAF" / tag, caf_mode=True)
        _write_metrics(caf_root, tags, include=["CAF"])
        write_celltype_summary_csv(caf_root, run_tags=tags, include_datasets=["CAF"])
        return

    # ---- Major Type ----
    print(f"\n{'='*60}\n[{tool_name}] Major Type Experiment\n{'='*60}")
    tags = []
    for i, seed in enumerate(major_seeds, 1):
        seed_python_numpy(seed)
        tag = str(i)
        tags.append(tag)
        mdir = model_root / "major_type" / tag
        print(f"\nMajor Run {i}/{args.run_time} (seed={seed})")
        h = train_fn(data["train"], mdir, "Celltype_training", seed, args)
        for tp in sorted(data["test_dir"].glob("*.h5ad")):
            a = predict_fn(tp, h, pred_col, seed, args)
            save_predictions(a, pred_col, tool_name, tp.stem,
                             results_root / tp.stem / tag)
    _write_metrics(results_root, tags, exclude=["Subtype"])

    # ---- Subtype ----
    print(f"\n{'='*60}\n[{tool_name}] Subtype Experiment\n{'='*60}")
    stags = []
    for i, seed in enumerate(subtype_seeds, 1):
        seed_python_numpy(seed)
        tag = str(i)
        stags.append(tag)
        mdir = model_root / "subtype" / tag
        print(f"\nSubtype Run {i}/{args.run_time} (seed={seed})")

        if train_subtype_fn:
            bh, sh = train_subtype_fn(data["subtype_train"], mdir, seed, args)
        else:
            bh, sh = default_train_subtype(
                train_fn, data["subtype_train"], mdir, seed, args)

        odir = results_root / "Subtype" / tag
        _subtype_predict = subtype_predict_fn or default_subtype_predict
        _subtype_predict(predict_fn, data["subtype_test"], bh, sh,
                         odir, tool_name, pred_col, seed, args)

    collect_subtype_summary_csv(results_root, dataset_name="Subtype")
    print(f"\n{'='*60}\n[{tool_name}] All experiments completed!\n{'='*60}")
