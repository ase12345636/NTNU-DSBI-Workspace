import os
import glob
import numpy as np
import scanpy as sc
import pandas as pd
from utils import metrics, preprocess


def evaluate_from_embeddings(
	data: str,
	out_path: str,
	batch: str,
	celltype: str,
	pattern: str = "*_emb.npy",
):
	os.makedirs(out_path, exist_ok=True)

	emb_files = sorted(glob.glob(os.path.join(out_path, pattern)))

	combined_df = pd.DataFrame()

	adata = preprocess.filter_cells_type1(data)
	
	for emb_path in emb_files:

		method_name = os.path.basename(emb_path).replace("_emb.npy", "")

		adata.obs['celltype'] = adata.obs[celltype]
		adata.obs['BATCH'] = adata.obs[batch]

		X_emb = np.load(emb_path)
		adata.obsm['X_emb'] = X_emb

		n_celltype = int(adata.obs[celltype].nunique())

		_, res_df = metrics.evaluate_dataset(adata, method=method_name, n_celltype=n_celltype)

		if combined_df.empty:
			combined_df = res_df
		else:
			combined_df = combined_df.join(res_df, how='outer')

	metrics_csv = os.path.join(out_path, "metrics_from_embeddings.csv")
	combined_df.to_csv(metrics_csv)
	return combined_df, metrics_csv


if __name__ == "__main__":
	for i in range (5):
		data = "/Group16T/raw_data/scCobra/Immune_ALL_human.h5ad"
		out_path = f"/Group16T/common/ccuc/scCobra/result/immune/{i}/"
		batch = 'batch'
		celltype = 'final_annotation'
		df, csv_path = evaluate_from_embeddings(data, out_path, batch, celltype)

		data = "/Group16T/raw_data/scCobra/Lung_atlas_public.h5ad"
		out_path = f"/Group16T/common/ccuc/scCobra/result/lung/{i}/"
		batch = 'batch'
		celltype = 'cell_type'
		df, csv_path = evaluate_from_embeddings(data, out_path, batch, celltype)

		data = "/Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad"
		out_path = f"/Group16T/common/ccuc/scCobra/result/pancreas/{i}/"
		batch = 'tech'
		celltype = 'celltype'
		df, csv_path = evaluate_from_embeddings(data, out_path, batch, celltype)
	
	

