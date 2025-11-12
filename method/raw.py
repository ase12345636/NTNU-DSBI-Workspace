import os
import numpy as np
import scanpy as sc
from utils.plot import plot

def raw(adata, out_path, batch, celltype):
    out_path = out_path + "raw"

    sc.tl.pca(adata)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(f"{out_path}_emb.npy", adata.obsm['X_pca'])

    plot(adata, "X_pca", batch, celltype, out_path)
    return adata
