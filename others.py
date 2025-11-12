from method.raw import raw
from method.Harmony import Harmony
from method.scVi import scVi
from utils.preprocess import filter_cells_type1
from utils import metrics
import os
import pandas as pd

data = "/Group16T/raw_data/scCobra/Immune_ALL_human.h5ad"
out_path = "/Group16T/common/ccuc/scCobra/result/immune/"
batch = 'batch'
celltype = 'final_annotation'
adata = filter_cells_type1(data)

os.makedirs(out_path, exist_ok=True)
adata_copy = adata.copy()
raw(adata_copy, out_path, batch, celltype)
adata_copy = adata.copy()
Harmony(adata_copy, out_path, batch, celltype)
adata_copy = adata.copy()
scVi(adata_copy, out_path, batch, celltype)

data = "/Group16T/raw_data/scCobra/Lung_atlas_public.h5ad"
out_path = "/Group16T/common/ccuc/scCobra/result/lung/"
batch = 'batch'
celltype = 'cell_type'
adata = filter_cells_type1(data)

os.makedirs(out_path, exist_ok=True)
adata_copy = adata.copy()
raw(adata_copy, out_path, batch, celltype)
adata_copy = adata.copy()
Harmony(adata_copy, out_path, batch, celltype)
adata_copy = adata.copy()
scVi(adata_copy, out_path, batch, celltype)




data = "/Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad"
out_path = "/Group16T/common/ccuc/scCobra/result/pancreas/"
batch = 'tech'
celltype = 'celltype'
adata = filter_cells_type1(data)

adata_copy = adata.copy()
raw(adata_copy, out_path, batch, celltype)
adata_copy = adata.copy()
Harmony(adata_copy, out_path, batch, celltype)
adata_copy = adata.copy()
scVi(adata_copy, out_path, batch, celltype)
