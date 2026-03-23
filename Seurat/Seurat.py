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

from utils.dataset import preprocess_adata
from utils.function import MemoryTracker

parser = argparse.ArgumentParser(description='Run Seurat v5 batch correction with Python evaluation')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--batch_key', type=str, default='batch')
parser.add_argument('--celltype_key', type=str, default='celltype')
parser.add_argument('--run_times', type=int, default=1)
parser.add_argument('--compute_oc', action='store_true', help='Compute OverCorrection score (slow)')
parser.add_argument('--ATAC', action='store_true', help='ATAC mode: skip preprocess_adata and use input h5ad directly')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
r_script_path = os.path.join(script_dir, "Seurat.R")

# Load and preprocess data ONCE (outside the loop)
print(f"Loading data from {args.dataset_path}")
adata_raw = sc.read_h5ad(args.dataset_path)

# Use shared preprocessing function for fairness (same as other methods)
if args.compute_oc or args.ATAC:
    adata = adata_raw.copy()
else:
    adata, _, _, _ = preprocess_adata(adata_raw.copy(), args.celltype_key, args.batch_key)
print(f"Preprocessed dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
# Ensure unique cell names
adata.obs_names_make_unique()

# Compute raw (unintegrated) PCA for OC baseline later
sc.pp.pca(adata, n_comps=50)
raw_pca_df = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)

# Export preprocessed data as h5ad (avoids CSV duplicate row.names issues)
temp_data_dir = os.path.join(args.save_path, "Seurat/temp_data/")
os.makedirs(temp_data_dir, exist_ok=True)
temp_input_h5ad = os.path.join(temp_data_dir, "preprocessed.h5ad")
adata.write(temp_input_h5ad)
print(f"Exported preprocessed data to {temp_input_h5ad}\n")

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
        "--input_h5ad", temp_input_h5ad,
        "--save_path", run_out,
        "--batch_key", args.batch_key,
        "--celltype_key", args.celltype_key,
        "--seed", str(seed),
        "--run_id", str(run_id)
    ]
    
    print(f"\n{'='*60}")
    print("Running Seurat batch correction...")
    print(f"{'='*60}\n")
    
    # Start timing and memory tracking
    # track_children=True to monitor R subprocess memory
    tracker = MemoryTracker(track_children=True)
    tracker.start()
    start_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    mem_metrics = tracker.stop()
    runtime = end_time - start_time

    # Python evaluation
    from utils.evaluation import evaluate_embedding_scib

    temp_h5ad = os.path.join(run_out, "temp_integrated.h5ad")
    adata = sc.read_h5ad(temp_h5ad)
    
    # R saved integrated PCA in .X → store as X_seurat
    adata.obsm['X_seurat'] = adata.X.copy()
    
    # Align raw (unintegrated) PCA by cell name for OC baseline
    adata.obsm['X_pca'] = raw_pca_df.loc[adata.obs_names].values
    
    # Compute neighbors for clustering and UMAP
    sc.pp.neighbors(adata, use_rep='X_seurat', random_state=seed)
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
        embed_key="X_seurat",
        batch_key=args.batch_key,
        celltype_key=args.celltype_key,
        leiden_resolution=best_res,
        random_state=seed,
        compute_oc=args.compute_oc
    )
    metrics['runtime_seconds'] = runtime
    metrics['best_leiden_resolution'] = best_res
    metrics['peak_ram_mb'] = mem_metrics['peak_ram_mb']
    metrics['peak_vram_mb'] = mem_metrics['peak_vram_mb']

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(run_out, "Seurat_metrics.csv"), index=False)

    print(f"\n[Seed {seed}] Seurat metrics (with best resolution {best_res:.1f}):")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  RAM: {mem_metrics['peak_ram_mb']:.2f} MB, VRAM: {mem_metrics['peak_vram_mb']:.2f} MB")
    print(f"  NMI={metrics['NMI']:.4f}, ARI={metrics['ARI']:.4f}")
    print(f"  ASW_bio={metrics['ASW_bio']:.4f}, ASW_batch={metrics['ASW_batch']:.4f}")
    print(f"  AVG_bio={metrics['AVG_bio']:.4f}, AVG_batch={metrics['AVG_batch']:.4f}")
    print(f"Results saved to {run_out}")

    # Clean up temp files
    os.remove(temp_h5ad)

print(f"\nAll runs completed. Results saved to {args.save_path}")

# Clean up shared temp data
os.remove(temp_input_h5ad)
import shutil
shutil.rmtree(temp_data_dir, ignore_errors=True)
print("Cleaned up temporary data files.")
