import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scvi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import preprocess_adata
from utils.evaluation import evaluate_embedding_scib
from utils.function import find_best_leiden_resolution

parser = argparse.ArgumentParser(description='Run scVi batch correction')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to input h5ad file')
parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
parser.add_argument('--batch_key', type=str, default='batch', help='Batch column name')
parser.add_argument('--celltype_key', type=str, default='celltype', help='Cell type column name')
parser.add_argument('--run_times', type=int, default=1, help='Number of times to run the batch correction')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
run_out_path = os.path.join(args.save_path, f"scVi/")
print(f"Loading data from {args.dataset_path}")
adata_raw = sc.read_h5ad(args.dataset_path)

for run_id in range(1, args.run_times + 1):
    print(f"\n{'='*60}")
    print(f"Run {run_id}/{args.run_times}")
    print(f"{'='*60}\n")

    seed = int(time.time() * 1000000) % (2**31)
    np.random.seed(seed)

    run_out_path = os.path.join(run_out_path, f"{run_id}/")
    os.makedirs(run_out_path, exist_ok=True)

    adata, X, y, b = preprocess_adata(adata_raw.copy(), args.celltype_key, args.batch_key)
    print(f"Preprocessed dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")

    sc.tl.pca(adata, random_state=seed)

    print(f"\n{'='*60}")
    print("Running scVi batch correction...")
    print(f"{'='*60}\n")

    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, batch_key = args.batch_key)
    model = scvi.model.SCVI(adata)
    model.train()
    adata.obsm["X_scVI"] = model.get_latent_representation()

    # UMAP
    sc.pp.neighbors(adata, use_rep='X_scVI', random_state=seed)
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
    sc.pl.umap(
        adata, 
        color=[args.batch_key], 
        save=None,
        show=False
    )
    plt.savefig(f"{run_out_path}/scVi_umap_batch.png", dpi=300, bbox_inches='tight')
    plt.close()

    sc.pl.umap(
        adata, 
        color=[args.celltype_key], 
        save=None,
        show=False
    )
    plt.savefig(f"{run_out_path}/scVi_umap_celltype.png", dpi=300, bbox_inches='tight')
    plt.close()

    sc.pl.umap(
        adata, 
        color=['leiden'], 
        save=None,
        show=False
    )
    plt.savefig(f"{run_out_path}/scVi_umap_leiden.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Evaluate
    metrics_scVi = evaluate_embedding_scib(
                                        adata, 
                                        embed_key="X_scVI", 
                                        batch_key=args.batch_key, 
                                        celltype_key=args.celltype_key, 
                                        leiden_resolution=best_resolution,
                                        random_state=seed
                                        )
    metrics_scVi['best_leiden_resolution'] = best_resolution
    metrics_df = pd.DataFrame([metrics_scVi])
    metrics_df.to_csv(os.path.join(run_out_path, "scVi_metrics.csv"), index=False)

    print(f"\n[Seed {seed}] scVi metrics (with best resolution {best_resolution:.1f}):")
    print(f"  NMI={metrics_scVi['NMI']:.4f}, ARI={metrics_scVi['ARI']:.4f}")
    print(f"  ASW_bio={metrics_scVi['ASW_bio']:.4f}, ASW_batch={metrics_scVi['ASW_batch']:.4f}")
    print(f"  AVG_bio={metrics_scVi['AVG_bio']:.4f}, AVG_batch={metrics_scVi['AVG_batch']:.4f}")
    print(f"Results saved to {run_out_path}")

print(f"\n{'='*60}")
print(f"All runs completed!")
print(f"Results saved to {args.save_path}")
print(f"{'='*60}")
