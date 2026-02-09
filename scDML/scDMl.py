import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scDML import scDMLModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import preprocess_adata
from utils.evaluation import evaluate_embedding_scib
from utils.function import find_best_leiden_resolution

parser = argparse.ArgumentParser(description='Run scDML batch correction')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to input h5ad file')
parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
parser.add_argument('--batch_key', type=str, default='batch', help='Batch column name')
parser.add_argument('--celltype_key', type=str, default='celltype', help='Cell type column name')
parser.add_argument('--run_times', type=int, default=1, help='Number of times to run the batch correction')
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

    run_out_path = os.path.join(args.save_path, f"scDML/{run_id}/")
    os.makedirs(run_out_path, exist_ok=True)

    adata, X, y, b = preprocess_adata(adata_raw.copy(), args.celltype_key, args.batch_key)
    print(f"Preprocessed dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")

    print(f"\n{'='*60}")
    print("Running scDML batch correction...")
    print(f"{'='*60}\n")

    # Start timing
    start_time = time.time()
    # scDML integration
    model = scDMLModel()
    # adata = model.preprocess(adata, batch_key=args.batch_key)
    model.integrate(adata, 
                    batch_key=args.batch_key, 
                    ncluster_list=[15], 
                    merge_rule="rule2", 
                    expect_num_cluster=15, 
                    mode="unsupervised",
                    seed=seed)
    end_time = time.time()
    # UMAP
    sc.pp.neighbors(adata, use_rep='X_emb', random_state=seed)
    sc.tl.umap(adata)
    
    # Save UMAP coordinates
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
    plt.savefig(f"{run_out_path}/scDML_umap_batch.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{run_out_path}/scDML_umap_celltype.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{run_out_path}/scDML_umap_leiden.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Evaluate
    metrics_scDML = evaluate_embedding_scib(
                                        adata, 
                                        embed_key="X_emb", 
                                        batch_key=args.batch_key, 
                                        celltype_key=args.celltype_key, 
                                        leiden_resolution=best_resolution,
                                        random_state=seed
                                        )
    
    # Add runtime to metrics
    runtime = end_time - start_time
    metrics_scDML['runtime_seconds'] = runtime
    metrics_scDML['best_leiden_resolution'] = best_resolution
    metrics_df = pd.DataFrame([metrics_scDML])
    metrics_df.to_csv(os.path.join(run_out_path, "scDML_metrics.csv"), index=False)

    print(f"\n[Seed {seed}] scDML metrics (with best resolution {best_resolution:.1f}):")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  NMI={metrics_scDML['NMI']:.4f}, ARI={metrics_scDML['ARI']:.4f}")
    print(f"  ASW_bio={metrics_scDML['ASW_bio']:.4f}, ASW_batch={metrics_scDML['ASW_batch']:.4f}")
    print(f"  AVG_bio={metrics_scDML['AVG_bio']:.4f}, AVG_batch={metrics_scDML['AVG_batch']:.4f}")
    
    # Save integrated adata with trajectory-ready information
    adata.write_h5ad(os.path.join(run_out_path, "adata_scDML_integrated.h5ad"))
    print(f"  Integrated adata saved: {run_out_path}/adata_scDML_integrated.h5ad")
    
    print(f"Results saved to {run_out_path}")

print(f"\n{'='*60}")
print(f"All runs completed!")
print(f"Results saved to {args.save_path}")
print(f"{'='*60}")
