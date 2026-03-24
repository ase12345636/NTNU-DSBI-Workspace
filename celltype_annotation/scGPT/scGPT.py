"""scGPT cell type annotation benchmark.

Reference-mapping: embed cells (pretrained scGPT or trainset PCA),
then kNN label propagation from reference to query.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from utils.label_utils import get_broad_type

from utils.pipeline import (
    create_argument_parser, run_annotation_pipeline,
    save_subtype_results, load_adata, ensure_dir,
    resolve_label_col, seed_python_numpy,
    BROAD_TYPES_FOR_SUBTYPE,
)

DEFAULT_K = 15
DEFAULT_PCA_COMPONENTS = 64


# ============================================================
# Embedding
# ============================================================

def _to_dense(adata):
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _embed_scgpt(adata, model_dir, batch_size=64):
    import scgpt as scg
    data = adata.copy()
    gene_col = next((c for c in ("feature_name","gene_name","index") if c in data.var.columns), None)
    if gene_col is None:
        data.var["index"] = data.var.index.astype(str)
        gene_col = "index"
    embedded = scg.tasks.embed_data(data, model_dir, gene_col=gene_col,
                                     batch_size=batch_size, return_new_adata=True)
    return np.asarray(embedded.X, dtype=np.float32)


def _fit_pca(adata, n_components, seed):
    x = _to_dense(adata)
    nc = min(max(1, n_components), x.shape[0], x.shape[1])
    pca = PCA(n_components=nc, random_state=seed)
    return pca.fit_transform(x).astype(np.float32), pca, adata.var_names.astype(str).to_numpy()


def _project_pca(adata, pca, train_genes):
    query_genes = adata.var_names.astype(str).to_numpy()
    gene_idx = {g: i for i, g in enumerate(query_genes)}
    pairs = [(j, gene_idx[g]) for j, g in enumerate(train_genes) if g in gene_idx]
    if not pairs:
        raise ValueError("No overlapping genes for PCA projection.")
    x_raw = _to_dense(adata)
    x = np.zeros((x_raw.shape[0], len(train_genes)), dtype=np.float32)
    ti, qi = zip(*pairs)
    x[:, list(ti)] = x_raw[:, list(qi)]
    return pca.transform(x).astype(np.float32)


def _build_backend(adata, args):
    if args.embedding_mode == "pretrained":
        return _embed_scgpt(adata, args.scgpt_model_dir), {"mode": "pretrained", "model_dir": args.scgpt_model_dir}
    emb, pca, genes = _fit_pca(adata, args.pca_components, seed=0)
    return emb, {"mode": "trainset", "pca": pca, "gene_order": genes}


def _embed_query(adata, backend):
    if backend["mode"] == "pretrained":
        return _embed_scgpt(adata, backend["model_dir"])
    return _project_pca(adata, backend["pca"], backend["gene_order"])


# ============================================================
# kNN Prediction
# ============================================================

def _knn_predict(ref_emb, ref_labels, query_emb, k, seed):
    k_eff = max(1, min(k, ref_emb.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(ref_emb)
    dist, idx = nn.kneighbors(query_emb)
    labels_np = ref_labels.astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    preds = []
    for i in range(query_emb.shape[0]):
        nl = labels_np[idx[i]]
        uniq, counts = np.unique(nl, return_counts=True)
        cands = uniq[counts == counts.max()]
        preds.append(str(cands[int(rng.integers(0, len(cands)))]) if len(cands) > 1
                     else str(cands[0]))
    return pd.Series(preds)


# ============================================================
# Train / Predict
# ============================================================

def train_scgpt(train_path, model_dir, label_col, seed, args):
    adata = load_adata(train_path)
    resolved = resolve_label_col(adata, label_col)
    emb, backend = _build_backend(adata, args)
    return {"embeddings": emb, "labels": adata.obs[resolved].astype(str).copy(),
            "backend": backend}


def predict_scgpt(test_path, model_handle, pred_col, seed, args):
    adata = load_adata(test_path)
    query_emb = _embed_query(adata, model_handle["backend"])
    pred = _knn_predict(model_handle["embeddings"], model_handle["labels"],
                        query_emb, args.k, seed)
    out = adata.copy()
    out.obs[pred_col] = pred.values
    return out


# ============================================================
# Custom Subtype (embed-once-slice for efficiency)
# ============================================================

def train_subtype_scgpt(subtype_train_path, model_dir, seed, args):
    adata = load_adata(subtype_train_path)
    subtype_col = "Celltype_subtraining"
    broad_col = ("Celltype_training" if "Celltype_training" in adata.obs.columns else None)
    if broad_col is None:
        adata.obs["_derived_broad"] = adata.obs[subtype_col].astype(str).map(get_broad_type)
        broad_col = "_derived_broad"

    emb, backend = _build_backend(adata, args)
    broad_handle = {"embeddings": emb, "labels": adata.obs[broad_col].astype(str).copy(),
                    "backend": backend}

    subtype_handles = {}
    for bt in BROAD_TYPES_FOR_SUBTYPE:
        mask = adata.obs[broad_col].astype(str) == bt
        sub = adata[mask]
        if mask.sum() == 0 or sub.obs[subtype_col].astype(str).nunique() < 2:
            continue
        subtype_handles[bt] = {"embeddings": emb[np.asarray(mask)],
                               "labels": sub.obs[subtype_col].astype(str).copy(),
                               "backend": backend}
    return broad_handle, subtype_handles


def subtype_predict_scgpt(predict_fn, test_path, broad_handle,
                          subtype_handles, output_dir, tool_name,
                          pred_col, seed, args):
    """Embed query ONCE, kNN per type — reuse save_subtype_results."""
    prefix = pred_col.replace("_pred", "")
    bp, sp = f"{prefix}_broad_pred", f"{prefix}_subtype_pred"

    adata = load_adata(test_path)
    out = adata.copy()
    query_emb = _embed_query(out, broad_handle["backend"])

    # Broad prediction
    broad_pred = _knn_predict(broad_handle["embeddings"], broad_handle["labels"],
                              query_emb, args.k, seed)
    out.obs[bp] = broad_pred.values
    out.obs[sp] = out.obs[bp].astype(str)
    out.obs[pred_col] = out.obs[bp].astype(str)

    # Per-broad-type subtype refinement
    for bt, sub_h in subtype_handles.items():
        mask = out.obs[bp].astype(str) == bt
        if mask.sum() == 0:
            continue
        sub_pred = _knn_predict(sub_h["embeddings"], sub_h["labels"],
                                query_emb[np.asarray(mask)], args.k, seed)
        broad_check = sub_pred.map(get_broad_type)
        accepted = sub_pred.where(broad_check == bt, bt)
        cells = out.obs_names[np.asarray(mask)]
        out.obs.loc[cells, sp] = sub_pred.values
        out.obs.loc[cells, pred_col] = accepted.values

    save_subtype_results(out, pred_col, bp, sp, test_path, output_dir, tool_name)


# ============================================================
# Auto-discover scGPT checkpoint
# ============================================================

def _find_scgpt_model_dir():
    env = os.environ.get("SCGPT_MODEL_DIR", "").strip()
    if env and (Path(env) / "best_model.pt").exists():
        return Path(env)
    base = Path(__file__).resolve().parent.parent / "scGPT" / "models"
    for name in ("scGPT_human", "whole-human", "whole_human"):
        if (base / name / "best_model.pt").exists():
            return base / name
    return None


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = create_argument_parser("scGPT")
    parser.add_argument("--embedding_mode", choices=["trainset","pretrained"], default="trainset")
    parser.add_argument("--pca_components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--scgpt_model_dir", type=Path, default=None)
    args = parser.parse_args()

    if args.embedding_mode == "pretrained" and args.scgpt_model_dir is None:
        args.scgpt_model_dir = _find_scgpt_model_dir()
        if args.scgpt_model_dir is None:
            raise FileNotFoundError("scGPT model dir not found.")

    run_annotation_pipeline(
        "scGPT", "scgpt_pred",
        train_scgpt, predict_scgpt, args,
        train_subtype_fn=train_subtype_scgpt,
        subtype_predict_fn=subtype_predict_scgpt,
    )
