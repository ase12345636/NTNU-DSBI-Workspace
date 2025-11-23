from method.scdml import scdml
import scanpy as sc
import os
import pandas as pd
import time, random, numpy as np, os
import torch
from utils.preprocess import filter_cells_type1


for i in range (5):
    random.seed(int(time.time()))
    torch.cuda.manual_seed_all(int(time.time()))
    
    data = "/Group16T/raw_data/scCobra/Immune_ALL_human.h5ad"
    out_path = f"/Group16T/common/ccuc/scCobra/result_org/immune/{i}/"
    batch = 'batch'
    celltype = 'final_annotation'
    adata = filter_cells_type1(data)
    scdml(adata, out_path, batch, celltype)

    data = "/Group16T/raw_data/scCobra/Lung_atlas_public.h5ad"
    out_path = f"/Group16T/common/ccuc/scCobra/result_org/lung/{i}/"
    batch = 'batch'
    celltype = 'cell_type'
    adata = filter_cells_type1(data)
    scdml(adata, out_path, batch, celltype)


    data = "/Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad"
    out_path = f"/Group16T/common/ccuc/scCobra/result_org/pancreas/{i}/"
    batch = 'tech'
    celltype = 'celltype'
    adata = filter_cells_type1(data)
    scdml(adata, out_path, batch, celltype)