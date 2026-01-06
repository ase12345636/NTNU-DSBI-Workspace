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
from utils.function import find_best_leiden_resolution

parser = argparse.ArgumentParser(description='Run Raw (PCA) batch correction')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to input h5ad file')
parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
parser.add_argument('--batch_key', type=str, default='batch', help='Batch column name')
parser.add_argument('--celltype_key', type=str, default='celltype', help='Cell type column name')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
print(f"Loading data from {args.dataset_path}")
adata_raw = sc.read_h5ad(args.dataset_path)

seed = int(time.time() * 1000000) % (2**31)
np.random.seed(seed)

run_out_path = os.path.join(args.save_path + f"raw/")
os.makedirs(run_out_path, exist_ok=True)

adata, X, y, b = preprocess_adata(adata_raw.copy(), args.celltype_key, args.batch_key)
print(f"Preprocessed dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")

sc.tl.pca(adata)
sc.pp.neighbors(adata, random_state=seed)
sc.tl.umap(adata)

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

sc.pl.umap(
    adata, 
    color=[args.batch_key], 
    save=None,
    show=False
)
plt.savefig(f"{run_out_path}/raw_umap_batch.png", dpi=300, bbox_inches='tight')
plt.close()

sc.pl.umap(
    adata, 
    color=[args.celltype_key], 
    save=None,
    show=False
)
plt.savefig(f"{run_out_path}/raw_umap_celltype.png", dpi=300, bbox_inches='tight')
plt.close()

sc.pl.umap(
    adata, 
    color=['leiden'], 
    save=None,
    show=False
)
plt.savefig(f"{run_out_path}/raw_umap_leiden.png", dpi=300, bbox_inches='tight')
plt.close()

metrics_raw = evaluate_embedding_scib(
                                    adata, 
                                    embed_key="X_pca", 
                                    batch_key=args.batch_key, 
                                    celltype_key=args.celltype_key, 
                                    leiden_resolution=best_resolution,
                                    random_state=seed
                                    )
metrics_raw['best_leiden_resolution'] = best_resolution
metrics_df = pd.DataFrame([metrics_raw])
metrics_df.to_csv(os.path.join(run_out_path, "raw_metrics.csv"), index=False)

print(f"\n[Seed {seed}] Raw metrics (with best resolution {best_resolution:.1f}):")
print(f"  NMI={metrics_raw['NMI']:.4f}, ARI={metrics_raw['ARI']:.4f}")
print(f"  ASW_bio={metrics_raw['ASW_bio']:.4f}, ASW_batch={metrics_raw['ASW_batch']:.4f}")
print(f"  AVG_bio={metrics_raw['AVG_bio']:.4f}, AVG_batch={metrics_raw['AVG_batch']:.4f}")
print(f"Results saved to {run_out_path}")

print(f"\n{'='*60}")
print(f"All runs completed!")
print(f"Results saved to {args.save_path}")
print(f"{'='*60}")
