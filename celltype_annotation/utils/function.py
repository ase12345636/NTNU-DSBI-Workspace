"""Visualization functions for annotation benchmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from utils.label_utils import (
    ALLOWED_CELLTYPES,
    collapse_major_type_label,
    collapse_major_type_labels,
    get_aligned_color_palette,
    get_ground_truth_column,
    standardize_labels_series,
)


# ============================================================
# UMAP Figures
# ============================================================

def _ensure_umap(vis):
    """Compute UMAP if not already present."""
    if "X_umap" in vis.obsm:
        return vis
    use_rep = "X_scVI" if "X_scVI" in vis.obsm else ("X_pca" if "X_pca" in vis.obsm else None)
    sc.pp.neighbors(vis, use_rep=use_rep)
    sc.tl.umap(vis)
    return vis


def write_umap_figures(adata, pred_col, output_dir, tool_name,
                       explicit_gt_col=None, use_major_type_filter=True):
    """Write paired GT vs Prediction UMAP + standalone prediction UMAP."""
    output_dir = Path(output_dir)
    gt_col = explicit_gt_col or get_ground_truth_column(adata)
    vis = _ensure_umap(adata.copy())

    plot_gt, plot_pred, plot_title = gt_col, pred_col, tool_name

    if use_major_type_filter and gt_col and gt_col in vis.obs.columns:
        mg = collapse_major_type_labels(vis.obs[gt_col].astype(str))
        mp = collapse_major_type_labels(vis.obs[pred_col].astype(str))
        valid = mg.notna() & mp.notna()
        if valid.sum() > 0:
            vis = vis[valid].copy()
            vis.obs["_major_gt"] = mg[valid].astype(str)
            vis.obs["_major_pred"] = mp[valid].astype(str)
            plot_gt, plot_pred, plot_title = "_major_gt", "_major_pred", f"{tool_name} (Major Type)"

    # Paired GT vs Prediction
    if plot_gt and plot_gt in vis.obs.columns:
        gt_pal, pred_pal = get_aligned_color_palette(vis.obs[plot_gt], vis.obs[plot_pred])
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        sc.pl.umap(vis, color=plot_gt, ax=axes[0], show=False, title="Ground Truth",
                    frameon=True, legend_loc=None, size=50, palette=gt_pal)
        sc.pl.umap(vis, color=plot_pred, ax=axes[1], show=False, title=plot_title,
                    frameon=True, legend_loc=None, size=50, palette=pred_pal)
        labels = sorted(set(gt_pal) | set(pred_pal))
        cmap = {**gt_pal, **pred_pal}
        fig.legend([Line2D([0],[0],marker="o",linestyle="",color=cmap[l],markersize=6) for l in labels],
                   labels, loc="center left", bbox_to_anchor=(0.93,0.5), frameon=False, fontsize=10)
        plt.tight_layout(rect=[0,0,0.94,1])
        plt.savefig(output_dir / "umap_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Standalone prediction
    cmap20 = plt.get_cmap("tab20")
    uniq = sorted({str(x) for x in vis.obs[plot_pred]})
    pal = {l: cmap20(i%20) for i, l in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(vis, color=plot_pred, ax=ax, show=False, title=plot_title,
                frameon=True, legend_loc=None, size=50, palette=pal)
    fig.legend([Line2D([0],[0],marker="o",linestyle="",color=pal[l],markersize=6) for l in sorted(pal)],
               sorted(pal), loc="center left", bbox_to_anchor=(0.92,0.5), frameon=False, fontsize=10)
    plt.tight_layout(rect=[0,0,0.90,1])
    plt.savefig(output_dir / "umap_scLightGAT.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Confusion Matrix
# ============================================================

def write_confusion_matrix(adata, pred_col, output_dir, explicit_gt_col=None,
                           title="Prediction Confusion Matrix",
                           filename="sclightgat_test_confusion_matrix.png",
                           use_major_type=True):
    gt_col = explicit_gt_col or get_ground_truth_column(adata)
    if gt_col is None:
        return

    yt = standardize_labels_series(adata.obs[gt_col].astype(str))
    yp = standardize_labels_series(adata.obs[pred_col].astype(str))

    if use_major_type:
        yt, yp = collapse_major_type_labels(yt), collapse_major_type_labels(yp)
        valid = yt.notna() & yp.notna()
        yt, yp = yt[valid].astype(str), yp[valid].astype(str)
        labels = sorted({collapse_major_type_label(x) for x in ALLOWED_CELLTYPES} - {None})
    else:
        valid = yt.notna() & yp.notna()
        yt, yp = yt[valid].astype(str), yp[valid].astype(str)
        labels = sorted(set(yt) | set(yp))

    if len(yt) == 0 or not labels:
        return

    cm = confusion_matrix(yt, yp, labels=labels)
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, vmin=0, vmax=cm.max() if cm.size else 1,
                cbar_kws={"label": "Count"}, annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel("Prediction", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Subtype Figures (per-broad-type UMAP + confusion)
# ============================================================

def write_subtype_figures(adata, pred_col, output_dir, tool_name,
                          explicit_gt_col="Celltype_subtraining",
                          broad_to_subtypes=None):
    """Per-broad-type UMAP comparison + confusion matrix figures."""
    gt_col = explicit_gt_col if explicit_gt_col in adata.obs.columns else None
    if gt_col is None:
        return

    vis = _ensure_umap(adata.copy())
    output_dir = Path(output_dir)

    from utils.evaluation import DEFAULT_BROAD_TO_SUBTYPES
    mapping = broad_to_subtypes or DEFAULT_BROAD_TO_SUBTYPES
    y_true_all = vis.obs[gt_col].astype(str)

    for broad_type, subtypes in mapping.items():
        mask = y_true_all.isin(set(subtypes))
        if mask.sum() == 0:
            continue
        sub = vis[mask].copy()
        safe = broad_type.replace("+","plus").replace(" ","_")

        # Per-broad UMAP
        gt_pal, pred_pal = get_aligned_color_palette(sub.obs[gt_col].astype(str),
                                                      sub.obs[pred_col].astype(str))
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        sc.pl.umap(sub, color=gt_col, ax=axes[0], show=False,
                    title=f"{broad_type} Ground Truth", frameon=True,
                    legend_loc=None, size=50, palette=gt_pal)
        sc.pl.umap(sub, color=pred_col, ax=axes[1], show=False,
                    title=f"{tool_name} ({broad_type})", frameon=True,
                    legend_loc=None, size=50, palette=pred_pal)
        labels_leg = sorted(set(gt_pal) | set(pred_pal))
        cmap = {**gt_pal, **pred_pal}
        fig.legend([Line2D([0],[0],marker="o",linestyle="",color=cmap[l],markersize=6) for l in labels_leg],
                   labels_leg, loc="center left", bbox_to_anchor=(0.93,0.5), frameon=False, fontsize=10)
        plt.tight_layout(rect=[0,0,0.94,1])
        plt.savefig(output_dir / f"subtype_umap_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Per-broad confusion
        yt = sub.obs[gt_col].astype(str)
        yp = sub.obs[pred_col].astype(str)
        valid_pred = yp.isin(set(subtypes))
        if valid_pred.sum() == 0:
            continue
        yt_v, yp_v = yt[valid_pred], yp[valid_pred]
        cm_labels = sorted(set(yt_v) | set(yp_v))
        cm = confusion_matrix(yt_v, yp_v, labels=cm_labels)
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0

        fig, ax = plt.subplots(figsize=(max(7, len(cm_labels)*1.8), max(6, len(cm_labels)*1.6)))
        sns.heatmap(cm/row_sums, annot=cm, fmt="g", cmap="Blues", xticklabels=cm_labels,
                    yticklabels=cm_labels, vmin=0, vmax=1, cbar_kws={"label":"Proportion"}, ax=ax)
        ax.set_xlabel("Predicted Subtype", fontsize=11, fontweight="bold")
        ax.set_ylabel("Ground Truth Subtype", fontsize=11, fontweight="bold")
        ax.set_title(f"{broad_type} - Subtype Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"subtype_confusion_matrix_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
