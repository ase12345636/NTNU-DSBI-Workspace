#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(argparse)
  library(Matrix)
  library(future)
})

# Set multiprocessing
options(future.globals.maxSize = 32 * 1024^3)
plan("multicore", workers = 8)

parser <- ArgumentParser(description='Run Seurat v5 batch correction')
parser$add_argument('--data_dir', type='character', required=TRUE)
parser$add_argument('--save_path', type='character', required=TRUE)
parser$add_argument('--batch_key', type='character', default='batch')
parser$add_argument('--celltype_key', type='character', default='celltype')
parser$add_argument('--seed', type='integer', default=42)
parser$add_argument('--run_id', type='integer', default=1)
args <- parser$parse_args()

set.seed(args$seed)
dir.create(args$save_path, showWarnings = FALSE, recursive = TRUE)

cat("Loading counts and metadata from CSV...\n")
counts <- as.matrix(read.csv(file.path(args$data_dir, "counts.csv"), row.names=1, check.names=FALSE))
metadata <- read.csv(file.path(args$data_dir, "metadata.csv"), row.names=1)

cat(sprintf("Dataset: %d cells, %d genes\n", ncol(counts), nrow(counts)))

# Create Seurat object
seurat_obj <- CreateSeuratObject(
  counts = counts,
  meta.data = metadata
)

# Set data
seurat_obj <- SetAssayData(seurat_obj, layer="data", new.data=counts)

# HVG, scale, PCA
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method="vst", nfeatures=2000, verbose=FALSE)
seurat_obj <- ScaleData(seurat_obj, verbose=FALSE)
seurat_obj <- RunPCA(seurat_obj, npcs=50, verbose=FALSE, seed.use=args$seed)

# Split by batch and prepare integration
seurat_list <- SplitObject(seurat_obj, split.by=args$batch_key)
seurat_list <- lapply(seurat_list, function(x) {
  x <- FindVariableFeatures(x, selection.method="vst", nfeatures=2000, verbose=FALSE)
})

features <- SelectIntegrationFeatures(seurat_list, nfeatures=2000)

seurat_list <- lapply(seurat_list, function(x) {
  x <- ScaleData(x, features=features, verbose=FALSE)
  x <- RunPCA(x, features=features, npcs=50, verbose=FALSE, seed.use=args$seed)
})

start_time <- Sys.time()
anchors <- FindIntegrationAnchors(seurat_list, anchor.features=features, reduction="rpca", dims=1:50, verbose=FALSE)
seurat_integrated <- IntegrateData(anchors, dims=1:50, verbose=FALSE)

seurat_integrated <- ScaleData(seurat_integrated, verbose=FALSE)
seurat_integrated <- RunPCA(seurat_integrated, npcs=50, verbose=FALSE, seed.use=args$seed)
end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units="secs"))

cat(sprintf("Seurat integration completed in %.2f seconds\n", runtime))
# Save h5ad for Python evaluation
library(reticulate)
use_python("/Group16T/common/ccuc/miniconda3/envs/sctools/bin/python", required=TRUE)
sc <- import("scanpy")
np <- import("numpy")
pd <- import("pandas")

adata_integrated <- sc$AnnData(X=np$array(Embeddings(seurat_integrated,"pca")),
                               obs=pd$DataFrame(seurat_integrated@meta.data))
temp_h5ad <- file.path(args$save_path,"temp_integrated.h5ad")
adata_integrated$write(temp_h5ad)

cat("Seurat integration complete. Handing off to Python for evaluation and visualization.\n")
