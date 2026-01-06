import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples
)
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from harmonypy import compute_lisi


def celltype_asw(embedding, celltype_labels, metric="euclidean"):
    """CellType ASW (Büttner et al., 2019). Higher is better."""
    if issparse(embedding):
        embedding = embedding.A
    sil = silhouette_samples(embedding, celltype_labels, metric=metric)
    df = pd.DataFrame({"celltype": celltype_labels, "sil": sil})
    avg_sil = df.groupby("celltype")["sil"].mean().mean()
    return abs(avg_sil)


def batch_asw(embedding, batch_labels, metric="euclidean"):
    """Batch ASW (Büttner et al., 2019). Closer to 0 is better (lower separation)."""
    if issparse(embedding):
        embedding = embedding.A
    sil = silhouette_samples(embedding, batch_labels, metric=metric)
    df = pd.DataFrame({"batch": batch_labels, "sil": sil})
    avg_sil = df.groupby("batch")["sil"].mean().mean()
    return abs(avg_sil)


def batch_entropy_mixing(embedding, batch_labels, k=50):
    """Batch Entropy Mixing Score (Xiong et al., 2022). Higher (closer to 1) is better."""
    if issparse(embedding):
        embedding = embedding.A

    nn = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    idx = nn.kneighbors(return_distance=False)[:, 1:]

    entropies = []
    B = len(np.unique(batch_labels))
    eps = 1e-10

    for i in range(len(embedding)):
        neighbors = batch_labels[idx[i]]
        p = pd.Series(neighbors).value_counts(normalize=True)
        H_i = -(p * np.log(p + eps)).sum()
        entropies.append(H_i)

    H_mean = np.nanmean(entropies)
    H_norm = H_mean / np.log(B)

    return H_norm


def compute_overcorrection_from_adata(
    adata,
    embed_key="X_pca",
    celltype_col="celltype",
    batch_col="batch",
    k=15,
    baseline=None,
):
    """
    Compute normalized over-correction score directly from AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing obs and embeddings
    embed_key : str
        Key for embedding, supports:
        - adata.obsm["X_pca"]
        - adata.obsm["X_supcon"]
        - adata.obsm["X_perturbed"]
    celltype_col : str
        obs column containing cell types
    batch_col : str
        obs column containing batch labels (not used in score)
    k : int
        Number of neighbors
    baseline : float or None
        Raw over-correction score from *before integration*

    Returns
    -------
    float
        If baseline is None → return raw OC score  
        If baseline is given → return normalized OC score
    """

    # --- 1. extract embedding ---
    if embed_key in adata.obsm.keys():
        Z = adata.obsm[embed_key]
    elif embed_key == "X":
        Z = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    else:
        raise ValueError(f"Embedding '{embed_key}' not found in adata.obsm")

    # --- 2. extract labels ---
    celltypes = np.array(adata.obs[celltype_col])
    batches = np.array(adata.obs[batch_col])

    # --- 3. compute neighbors ---
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(Z)
    _, idx = nbrs.kneighbors(Z)
    idx = idx[:, 1:]  # remove self

    # --- 4. OC score ---
    same_type_counts = np.array([
        np.sum(celltypes[idx[i]] == celltypes[i]) for i in range(len(celltypes))
    ])
    oc = 1 - same_type_counts / k
    oc_mean = float(np.mean(oc))

    # --- 5. return normalized or raw ---
    if baseline is not None:
        oc_norm = oc_mean - baseline
        print(f"[OC] baseline={baseline:.4f}, integrated={oc_mean:.4f}, normalized={oc_norm:.4f}")
        return oc_norm

    print(f"[OC] raw OC = {oc_mean:.4f}")
    return oc_mean


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


def evaluate_embedding_scib(ad_tmp, embed_key="X_supcon", batch_key="batch", celltype_key="celltype", leiden_resolution=0.1, random_state=42):
    """
    重新整合的評估管線，包含 scIB 與 scBCN 論文關鍵指標。
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
    if "X_pca" not in ad_tmp.obsm: 
        sc.pp.pca(ad_tmp)
    # 取得原始 PCA 的 OC 作為 baseline
    oc_raw = compute_overcorrection_from_adata(ad_tmp, embed_key="X_pca", celltype_col=celltype_key)
    # 計算當前 Embedding 的 OC 並與 baseline 相減
    results["OverCorrection"] = compute_overcorrection_from_adata(
        ad_tmp, embed_key=embed_key, celltype_col=celltype_key, baseline=oc_raw
    )

    return results
