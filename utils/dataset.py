import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import scanpy as sc


def preprocess_adata(adata, label_col, batch_col, 
                     min_genes=200, min_cells=3, 
                     mt_threshold=5.0, target_sum=1e4,
                     n_top_genes=2000,
                     min_mean=0.0125, max_mean=3, min_disp=0.5):
    ad_bc = adata.copy()
    
    # Filter cells and genes
    sc.pp.filter_cells(ad_bc, min_genes=min_genes)
    sc.pp.filter_genes(ad_bc, min_cells=min_cells)
    
    # Calculate QC metrics
    ad_bc.var["mt"] = ad_bc.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ad_bc, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    ad_bc = ad_bc[ad_bc.obs["pct_counts_mt"] < mt_threshold].copy()
    
    # Normalize and log-transform
    sc.pp.normalize_total(ad_bc, target_sum=target_sum)
    sc.pp.log1p(ad_bc)
    
    # Select highly variable genes
    if ad_bc.n_vars < n_top_genes:
        sc.pp.highly_variable_genes(
            ad_bc, 
            min_mean=min_mean, 
            max_mean=max_mean, 
            min_disp=min_disp
        )
    else:
        sc.pp.highly_variable_genes(ad_bc, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_col)
    
    ad_bc = ad_bc[:, ad_bc.var["highly_variable"]].copy()
    
    # Extract data
    X = ad_bc.X.tocsr() if sp.issparse(ad_bc.X) else sp.csr_matrix(ad_bc.X)
    y = ad_bc.obs[label_col].astype(str).to_numpy()
    b = ad_bc.obs[batch_col].astype(str).to_numpy()
    
    print(f"Filtered shape: {X.shape}")
    
    return ad_bc, X, y, b


class SupBatchDataset(Dataset):
    """Dataset for Supervised Contrastive Learning with batch indices."""
    def __init__(self, X, y, b):
        self.X = X
        self.y = np.asarray(y)
        self.b = np.asarray(b)
        self.is_sparse = sp.issparse(X)
        self.n = X.shape[0]
        # Map original labels/batches to integer indices
        self.classes_, self.yi = np.unique(self.y, return_inverse=True)
        self.batches_, self.bi = np.unique(self.b, return_inverse=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        xi = self.X[i]
        if self.is_sparse:
            xi = xi.toarray().ravel()
        else:
            xi = np.asarray(xi).ravel()
        return (
            torch.from_numpy(xi.astype(np.float32, copy=False)),
            torch.tensor(self.yi[i], dtype=torch.long),
            torch.tensor(self.bi[i], dtype=torch.long)
        )
