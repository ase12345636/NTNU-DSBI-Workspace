import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import preprocess_adata
from utils.evaluation import evaluate_embedding_scib
from utils.function import find_best_leiden_resolution, MemoryTracker

parser = argparse.ArgumentParser(description='Run Raw (PCA) batch correction')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to input h5ad file')
parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
parser.add_argument('--batch_key', type=str, default='batch', help='Batch column name')
parser.add_argument('--celltype_key', type=str, default='celltype', help='Cell type column name')
parser.add_argument('--run_times', type=int, default=1, help='Number of times to run the batch correction')
parser.add_argument('--compute_oc', action='store_true', help='Compute OverCorrection score (slow)')
parser.add_argument('--ATAC', action='store_true', help='ATAC mode: skip preprocess_adata and use input h5ad directly')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
print(f"Loading data from {args.dataset_path}")
adata_raw = sc.read_h5ad(args.dataset_path)

for run_id in range(1, args.run_times + 1):
    print(f"\n{'='*60}")
    print(f"Run {run_id}/{args.run_times}")
    print(f"{'='*60}\n")

    seed = int(time.time() * 1000000) % (2**31)
    np.random.seed(seed)

    run_out_path = os.path.join(args.save_path, f"raw/{run_id}/")
    os.makedirs(run_out_path, exist_ok=True)

    if args.compute_oc or args.ATAC:
        adata = adata_raw.copy()
    else:
        adata, _, _, _ = preprocess_adata(adata_raw.copy(), args.celltype_key, args.batch_key)
    print(f"Preprocessed dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
    # Start timing and memory tracking
    tracker = MemoryTracker()
    tracker.start()
    start_time = time.time()
    sc.tl.pca(adata, random_state=seed)
    end_time = time.time()
    mem_metrics = tracker.stop()
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata)
    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
    umap_df['batch'] = adata.obs[args.batch_key].values
    umap_df['celltype'] = adata.obs[args.celltype_key].values
    umap_df.to_csv(os.path.join(run_out_path, "umap_coordinates.csv"), index=False)

    # Find best Leiden resolution
    leiden_results = find_best_leiden_resolution(
        adata, 
        celltype_key=args.celltype_key,
        seed=seed,
        res_min=0.1,
        res_max=2.0,
        res_step=0.1,
        save_path=os.path.join(run_out_path, "resolution_search.csv")
    )

    best_resolution = leiden_results['best_resolution']

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
    plt.savefig(f"{run_out_path}/raw_umap_batch.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{run_out_path}/raw_umap_celltype.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{run_out_path}/raw_umap_leiden.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Evaluate
    metrics_raw = evaluate_embedding_scib(
                                        adata, 
                                        embed_key="X_pca", 
                                        batch_key=args.batch_key, 
                                        celltype_key=args.celltype_key, 
                                        leiden_resolution=best_resolution,
                                        random_state=seed,
                                        compute_oc=args.compute_oc
                                        )
    
    # End timing and add to metrics
    runtime = end_time - start_time
    metrics_raw['runtime_seconds'] = runtime
    metrics_raw['best_leiden_resolution'] = best_resolution
    metrics_raw['peak_ram_mb'] = mem_metrics['peak_ram_mb']
    metrics_raw['peak_vram_mb'] = mem_metrics['peak_vram_mb']
    metrics_df = pd.DataFrame([metrics_raw])
    metrics_df.to_csv(os.path.join(run_out_path, "raw_metrics.csv"), index=False)

    print(f"\n[Seed {seed}] Raw metrics (with best resolution {best_resolution:.1f}):")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  RAM: {mem_metrics['peak_ram_mb']:.2f} MB, VRAM: {mem_metrics['peak_vram_mb']:.2f} MB")
    print(f"  NMI={metrics_raw['NMI']:.4f}, ARI={metrics_raw['ARI']:.4f}")
    print(f"  ASW_bio={metrics_raw['ASW_bio']:.4f}, ASW_batch={metrics_raw['ASW_batch']:.4f}")
    print(f"  AVG_bio={metrics_raw['AVG_bio']:.4f}, AVG_batch={metrics_raw['AVG_batch']:.4f}")
    print(f"Results saved to {run_out_path}")

print(f"\n{'='*60}")
print(f"All runs completed!")
print(f"Results saved to {args.save_path}")
print(f"{'='*60}")
