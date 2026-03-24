"""Celltypist cell type annotation benchmark."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import celltypist
import numpy as np
from scipy import sparse as sp

from utils.label_utils import get_ground_truth_column, standardize_labels_series
from utils.pipeline import (
    create_argument_parser, run_annotation_pipeline,
    ensure_dir, load_adata, resolve_label_col, seed_python_numpy,
)


# ============================================================
# Helpers
# ============================================================

def _normalize_log1p_10k(adata):
    norm = adata.copy()
    x = norm.X
    if sp.issparse(x):
        x = x.tocsr()
        rs = np.asarray(x.sum(axis=1)).reshape(-1)
        rs[rs == 0] = 1.0
        x = x.multiply(10000.0 / rs[:, None]).tocsr()
        x.data = np.log1p(x.data)
        norm.X = x
    else:
        x = np.asarray(x, dtype=np.float32)
        rs = x.sum(axis=1); rs[rs == 0] = 1.0
        norm.X = np.log1p(x * (10000.0 / rs)[:, None])
    return norm


def _annotate(adata, model_path):
    return celltypist.annotate(
        filename=adata, model=str(model_path),
        majority_voting=False, mode="best match", transpose_input=False,
    )


# ============================================================
# Train / Predict
# ============================================================

def train_celltypist(train_path, model_dir, label_col, seed, args):
    seed_python_numpy(seed)
    adata = load_adata(train_path)
    resolved = resolve_label_col(adata, label_col)
    model = celltypist.train(
        X=adata, labels=resolved, transpose_input=False,
        check_expression=False, with_mean=True, n_jobs=-1, use_SGD=True,
    )
    ensure_dir(model_dir)
    model_path = model_dir / "model.pkl"
    model.write(str(model_path))
    return model_path


def predict_celltypist(test_path, model_handle, pred_col, seed, args):
    adata = load_adata(test_path)
    try:
        result = _annotate(adata, model_handle)
    except ValueError as e:
        if "Invalid expression matrix" not in str(e): raise
        result = _annotate(_normalize_log1p_10k(adata), model_handle)

    labels_df = result.predicted_labels.copy()
    labels_df.index = labels_df.index.astype(str)
    out = adata.copy()
    out.obs.index = out.obs.index.astype(str)
    pred = labels_df["predicted_labels"].reindex(out.obs.index)
    out.obs[pred_col] = standardize_labels_series(pred.astype(str))

    gt_col = get_ground_truth_column(out)
    if gt_col and gt_col in out.obs.columns:
        out.obs[gt_col] = standardize_labels_series(out.obs[gt_col].astype(str))
    return out


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = create_argument_parser("Celltypist")
    args = parser.parse_args()
    run_annotation_pipeline(
        "Celltypist", "celltypist_pred",
        train_celltypist, predict_celltypist, args,
    )
