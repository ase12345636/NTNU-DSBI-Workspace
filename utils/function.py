import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import normalized_mutual_info_score


def find_best_leiden_resolution(adata, celltype_key, seed=42, 
                                 res_min=0.1, res_max=2.0, res_step=0.1,
                                 save_path=None):
    """
    Test Leiden clustering across a range of resolutions and select the one with highest NMI.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with neighbors already computed
    celltype_key : str
        Column name in adata.obs containing cell type labels
    seed : int, optional
        Random seed for reproducibility (default: 42)
    res_min : float, optional
        Minimum resolution to test (default: 0.1)
    res_max : float, optional
        Maximum resolution to test (default: 2.0)
    res_step : float, optional
        Step size for resolution increment (default: 0.1)
    save_path : str, optional
        Path to save resolution search results CSV. If None, results are not saved.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'best_resolution': Best Leiden resolution
        - 'best_nmi': NMI score at best resolution
        - 'all_resolutions': Array of all tested resolutions
        - 'all_nmi_scores': Array of all NMI scores
        - 'resolution_df': DataFrame with resolution search results
    """
    
    print(f"\n{'='*60}")
    print(f"Testing Leiden clustering resolutions from {res_min} to {res_max}...")
    print(f"{'='*60}\n")
    
    resolutions = np.arange(res_min, res_max + res_step/2, res_step)
    nmi_scores = []
    celltype_labels = adata.obs[celltype_key].astype(str).to_numpy()
    
    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=seed, 
                     key_added=f"leiden_res_{res:.1f}")
        cluster_labels = adata.obs[f"leiden_res_{res:.1f}"].astype(str).to_numpy()
        nmi = normalized_mutual_info_score(celltype_labels, cluster_labels)
        nmi_scores.append(nmi)
        print(f"Resolution {res:.1f}: NMI = {nmi:.4f}")
    
    # Find best resolution
    best_idx = np.argmax(nmi_scores)
    best_resolution = resolutions[best_idx]
    best_nmi = nmi_scores[best_idx]
    
    print(f"\n{'='*60}")
    print(f"Best resolution: {best_resolution:.1f} with NMI = {best_nmi:.4f}")
    print(f"{'='*60}\n")
    
    # Create results dataframe
    resolution_df = pd.DataFrame({
        'resolution': resolutions,
        'NMI': nmi_scores
    })
    
    # Save if path provided
    if save_path is not None:
        resolution_df.to_csv(save_path, index=False)
        print(f"Resolution search results saved to {save_path}")
    
    # Set the best resolution as the default leiden clustering
    adata.obs['leiden'] = adata.obs[f"leiden_res_{best_resolution:.1f}"]
    
    return {
        'best_resolution': best_resolution,
        'best_nmi': best_nmi,
        'all_resolutions': resolutions,
        'all_nmi_scores': np.array(nmi_scores),
        'resolution_df': resolution_df
    }