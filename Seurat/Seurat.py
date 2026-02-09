import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Run Seurat v5 batch correction with Python evaluation')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--batch_key', type=str, default='batch')
parser.add_argument('--celltype_key', type=str, default='celltype')
parser.add_argument('--run_times', type=int, default=1)
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
r_script_path = os.path.join(script_dir, "Seurat.R")

# Load and preprocess data ONCE (outside the loop)
print(f"Loading data from {args.dataset_path}")
adata_raw = sc.read_h5ad(args.dataset_path)

print("Preprocessing data (filtering and normalization)...")
# Basic filtering (same as other methods for fairness)
sc.pp.filter_cells(adata_raw, min_genes=200)
sc.pp.filter_genes(adata_raw, min_cells=3)

# Calculate QC metrics and filter
adata_raw.var["mt"] = adata_raw.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata_raw, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
adata_raw = adata_raw[adata_raw.obs["pct_counts_mt"] < 5.0].copy()

print(f"Preprocessed dataset: {adata_raw.shape[0]} cells, {adata_raw.shape[1]} genes")

# Normalize and log-transform (Seurat will use this as input for HVG selection)
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)

# Export preprocessed data ONCE using Parquet (much faster than CSV)
print("Exporting preprocessed data to CSV...")
temp_data_dir = os.path.join(args.save_path, "Seurat/temp_data/")
os.makedirs(temp_data_dir, exist_ok=True)

# Transpose: R expects genes x cells
counts_matrix = adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X
counts_df = pd.DataFrame(
    counts_matrix.T,
    index=adata_raw.var_names,
    columns=adata_raw.obs_names
)
counts_df.to_csv(os.path.join(temp_data_dir, "counts.csv"))

metadata_df = adata_raw.obs[[args.batch_key, args.celltype_key]].copy()
metadata_df.to_csv(os.path.join(temp_data_dir, "metadata.csv"))

print("Data export complete!\n")

for run_id in range(1, args.run_times + 1):
    print(f"\n{'='*60}")
    print(f"Run {run_id}/{args.run_times}")
    print(f"{'='*60}\n")
    
    seed = int(time.time() * 1000000) % (2**31)
    np.random.seed(seed)
    
    run_out = os.path.join(args.save_path, f"Seurat/{run_id}/")
    os.makedirs(run_out, exist_ok=True)

    # Call R script
    cmd = [
        "Rscript",
        r_script_path,
        "--data_dir", temp_data_dir,
        "--save_path", run_out,
        "--batch_key", args.batch_key,
        "--celltype_key", args.celltype_key,
        "--seed", str(seed),
        "--run_id", str(run_id)
    ]
    
    print(f"\n{'='*60}")
    print("Running Seurat batch correction...")
    print(f"{'='*60}\n")
    
    subprocess.run(cmd, check=True)
    
    # Read runtime from R script
    runtime_file = os.path.join(run_out, "runtime.txt")
    with open(runtime_file, 'r') as f:
        runtime = float(f.read().strip())
    os.remove(runtime_file)  # Clean up

    # Python evaluation
    from utils.evaluation import evaluate_embedding_scib

    temp_h5ad = os.path.join(run_out, "temp_integrated.h5ad")
    adata = sc.read_h5ad(temp_h5ad)
    
    # R saved PCA in .X, move it to .obsm['X_pca'] for consistency
    adata.obsm['X_pca'] = adata.X.copy()
    
    # Compute neighbors for clustering and UMAP
    sc.pp.neighbors(adata, use_rep='X_pca', random_state=seed)
    sc.tl.umap(adata)
    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
    umap_df['batch'] = adata.obs[args.batch_key].values
    umap_df['celltype'] = adata.obs[args.celltype_key].values
    umap_df.to_csv(os.path.join(run_out, "umap_coordinates.csv"), index=False)
    
    # Find best Leiden resolution automatically
    from utils.function import find_best_leiden_resolution
    leiden_results = find_best_leiden_resolution(
        adata, celltype_key=args.celltype_key, seed=seed,
        res_min=0.1, res_max=2.0, res_step=0.1,
        save_path=os.path.join(run_out, "resolution_search.csv")
    )
    best_res = leiden_results['best_resolution']

    # Plot UMAPs
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(
        adata, 
        color=[args.batch_key], 
        save=None,
        show=False,
        frameon=False,
        title='',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(f"{run_out}/Seurat_umap_batch.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(
        adata, 
        color=[args.celltype_key], 
        save=None,
        show=False,
        frameon=False,
        title='',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(f"{run_out}/Seurat_umap_celltype.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(
        adata, 
        color=['leiden'], 
        save=None,
        show=False,
        frameon=False,
        title='',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(f"{run_out}/Seurat_umap_leiden.png", dpi=300, bbox_inches='tight')
    plt.close()

    metrics = evaluate_embedding_scib(
        adata,
        embed_key="X_pca",
        batch_key=args.batch_key,
        celltype_key=args.celltype_key,
        leiden_resolution=best_res,
        random_state=seed
    )
    metrics['runtime_seconds'] = runtime
    metrics['best_leiden_resolution'] = best_res

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(run_out, "Seurat_metrics.csv"), index=False)

    print(f"\n[Seed {seed}] Seurat metrics (with best resolution {best_res:.1f}):")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  NMI={metrics['NMI']:.4f}, ARI={metrics['ARI']:.4f}")
    print(f"  ASW_bio={metrics['ASW_bio']:.4f}, ASW_batch={metrics['ASW_batch']:.4f}")
    print(f"  AVG_bio={metrics['AVG_bio']:.4f}, AVG_batch={metrics['AVG_batch']:.4f}")
    print(f"Results saved to {run_out}")

    # Clean up temp files
    os.remove(temp_h5ad)

print(f"\nAll runs completed. Results saved to {args.save_path}")

# Clean up shared temp data
import shutil
shutil.rmtree(temp_data_dir)
print("Cleaned up temporary data files.")
