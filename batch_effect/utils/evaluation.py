import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples
)
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from harmonypy import compute_lisi


def compute_overcorrection_from_adata(
    adata,
    embed_key="X_pca",
    celltype_col="celltype",
    n_neighbors=100,
    n_pools=100,
    n_samples_per_pool=100,
    seed=124,
    baseline=None
):
    """
    Compute Over-correction Score using pooled sampling method.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing obs and embeddings
    embed_key : str
        Key for embedding in adata.obsm
    celltype_col : str
        obs column containing cell types
    n_neighbors : int
        Number of neighbors to consider
    n_pools : int
        Number of pooling iterations
    n_samples_per_pool : int
        Number of samples per pool
    seed : int
        Random seed for reproducibility
    baseline : float or None
        If provided, returns (current_score - baseline)
    
    Returns
    -------
    float
        Over-correction score (or normalized score if baseline provided)
    """
    
    # 1. Extract Embedding
    if embed_key in adata.obsm.keys():
        Z = adata.obsm[embed_key]
    elif embed_key == "X":
        Z = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    else:
        raise ValueError(f"Embedding '{embed_key}' not found in adata.obsm")

    # 2. Prepare label data
    celltype_series = pd.Series(adata.obs[celltype_col]).reset_index(drop=True)
    celltype_values = celltype_series.values
    celltype_dict = celltype_series.value_counts().to_dict()
    
    n_cells = Z.shape[0]
    n_neighbors = min(n_neighbors, n_cells - 1)
    
    # 3. Compute neighbor matrix (exclude self)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=-1)
    nne.fit(Z)
    kmatrix = nne.kneighbors_graph(Z) - sp_sparse.identity(n_cells)
    
    # 4. Pooling random sampling
    np.random.seed(seed)
    total_score = 0
    
    for t in range(n_pools):
        indices = np.random.choice(np.arange(n_cells), size=min(n_samples_per_pool, n_cells), replace=False)
        pool_purity = []
        
        for i in indices:
            neighbor_indices = kmatrix[i].nonzero()[1]
            # Capping: limit neighbors based on cell type count
            max_compare = min(celltype_dict[celltype_values[i]], n_neighbors)
            
            # Calculate proportion of same-type neighbors
            if len(neighbor_indices) > 0 and max_compare > 0:
                is_same_type = (celltype_values[neighbor_indices[:max_compare]] == celltype_values[i])
                pool_purity.append(np.mean(is_same_type))
            else:
                pool_purity.append(1.0)  # No neighbors = assume pure
            
        total_score += np.mean(pool_purity)

    # 5. Calculate final raw OC (1 - average purity)
    raw_oc_mean = 1 - (total_score / float(n_pools))

    # 6. Return normalized if baseline provided
    if baseline is not None:
        oc_norm = float(raw_oc_mean - baseline)
        print(f"[OC] baseline={baseline:.4f}, integrated={raw_oc_mean:.4f}, normalized={oc_norm:.4f}")
        return oc_norm
    
    print(f"[OC] raw OC = {raw_oc_mean:.4f}")
    return float(raw_oc_mean)


def calculate_batch_kl_py(embedding, batch_labels, n_cells=100, n_neighbors=100, replicates=200):
    """計算 Batch KL Divergence (原始散度值，越低越好)"""
    np.random.seed(42)
    batch_series = pd.Series(batch_labels)
    batch_counts = batch_series.value_counts(normalize=True).sort_index()
    p_population = batch_counts.values
    unique_batches = batch_counts.index.tolist()
    
    k = min(5 * len(unique_batches), n_neighbors)
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1).fit(embedding)
    
    kl_replicates = []
    for _ in range(replicates):
        boot_idx = np.random.choice(embedding.shape[0], size=n_cells, replace=False)
        _, indices = nn.kneighbors(embedding[boot_idx])
        kl_samples = [entropy(batch_series.iloc[idx].value_counts(normalize=True).reindex(unique_batches, fill_value=0).values + 1e-10, 
                             p_population + 1e-10, base=2) for idx in indices]
        kl_replicates.append(np.mean(kl_samples))
    return np.mean(kl_replicates)


def calculate_lisi(embedding, adata_obs, batch_key, celltype_key):
    """計算原始 iLISI (有效批次數)"""
    lisi_res = compute_lisi(embedding, adata_obs, [batch_key, celltype_key])
    lisi_raw = lisi_res[:, 0]
    ilisi_std = np.median(lisi_raw)
    
    return ilisi_std


def evaluate_embedding_scib(ad_tmp, embed_key="X_supcon", batch_key="batch", celltype_key="celltype", leiden_resolution=0.1, random_state=42, compute_oc=False):
    """
    重新整合的評估管線，包含 scIB 與 scBCN 論文關鍵指標。
    
    Parameters
    ----------
    compute_oc : bool
        Whether to compute OverCorrection score (default False, as it's slow)
    """
    results = {}
    embedding = ad_tmp.obsm[embed_key]
    if hasattr(embedding, "toarray"): 
        embedding = embedding.toarray()
    
    y = ad_tmp.obs[celltype_key].astype(str).to_numpy()
    b = ad_tmp.obs[batch_key].astype(str).to_numpy()

    # -------- Biological conservation (生物保留) --------
    # 計算聚類標籤
    sc.pp.neighbors(ad_tmp, n_neighbors=15, use_rep=embed_key)
    sc.tl.leiden(ad_tmp, resolution=leiden_resolution, random_state=random_state, key_added="eval_leiden")
    cluster_labels = ad_tmp.obs["eval_leiden"].astype(str).to_numpy()

    results["NMI"] = normalized_mutual_info_score(y, cluster_labels)
    results["ARI"] = adjusted_rand_score(y, cluster_labels)
    
    # ASW_bio (Cell type Silhouette)
    sil_samples = silhouette_samples(embedding, y)
    asw_bio_raw = pd.DataFrame({"ct": y, "sil": sil_samples}).groupby("ct")["sil"].mean().mean()
    results["ASW_bio"] = abs(asw_bio_raw)
    # AVG_bio 
    results["AVG_bio"] = np.mean([results["NMI"], results["ARI"], results["ASW_bio"]])

    # -------- Batch correction (批次校正) --------
    # ASW_batch (越高代表混合越好)
    sil_batch = silhouette_samples(embedding, b)
    asw_batch_raw = pd.DataFrame({"b": b, "sil": sil_batch}).groupby("b")["sil"].mean().mean()
    results["ASW_batch"] = 1 - abs(asw_batch_raw)
    
    # iLISI (原始值) 與 iLISI_N (歸一化得分)
    ilisi_raw = calculate_lisi(embedding, ad_tmp.obs, batch_key, celltype_key)
    results["iLISI"] = ilisi_raw
    
    # Batch KL (原始散度值)
    results["BatchKL"] = calculate_batch_kl_py(embedding, b)

    # Graph connectivity
    graph_conn = 0
    unique_types = ad_tmp.obs[celltype_key].unique()
    for l in unique_types:
        idx = np.where(ad_tmp.obs[celltype_key] == l)[0]
        if len(idx) > 1:
            sub_conn = ad_tmp.obsp["connectivities"][np.ix_(idx, idx)].max(axis=1).mean()
            graph_conn += sub_conn
    results["GraphConn_celltype"] = graph_conn / len(unique_types)

    results["AVG_batch"] = np.mean([results["ASW_batch"], results["GraphConn_celltype"]])

    # -------- Over-correction (過度校正) --------
    if compute_oc:
        if "X_pca" not in ad_tmp.obsm: 
            sc.pp.pca(ad_tmp)
        # 取得原始 PCA 的 OC 作為 baseline
        oc_raw = compute_overcorrection_from_adata(ad_tmp, embed_key="X_pca", celltype_col=celltype_key)
        # 計算當前 Embedding 的 OC 並與 baseline 相減
        results["OverCorrection"] = compute_overcorrection_from_adata(
            ad_tmp, embed_key=embed_key, celltype_col=celltype_key, baseline=oc_raw
        )
    else:
        results["OverCorrection"] = np.nan

    return results
