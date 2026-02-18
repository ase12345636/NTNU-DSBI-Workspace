#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(argparse)
  library(Matrix)
  library(future)
  library(reticulate)
})

# Set multiprocessing
options(future.globals.maxSize = Inf)
plan("multicore", workers = 8)

# Use the sctools conda environment python
use_python("/Group16T/common/ccuc/miniconda3/envs/sctools/bin/python", required=TRUE)

parser <- ArgumentParser(description='Run Seurat v5 batch correction')
parser$add_argument('--input_h5ad', type='character', required=TRUE)
parser$add_argument('--save_path', type='character', required=TRUE)
parser$add_argument('--batch_key', type='character', default='batch')
parser$add_argument('--celltype_key', type='character', default='celltype')
parser$add_argument('--seed', type='integer', default=42)
parser$add_argument('--run_id', type='integer', default=1)
args <- parser$parse_args()

set.seed(args$seed)
dir.create(args$save_path, showWarnings = FALSE, recursive = TRUE)

# Read h5ad via reticulate
cat("Loading preprocessed data from h5ad...\n")
sc <- import("scanpy")
np <- import("numpy")
pd <- import("pandas")

adata <- sc$read_h5ad(args$input_h5ad)

# Extract counts matrix (genes x cells) and metadata
# reticulate auto-converts scipy sparse to R dgCMatrix (S4 sparse)
X <- adata$X
if (is(X, "sparseMatrix")) {
  counts <- t(as.matrix(X))   # cells x genes -> genes x cells
} else {
  counts <- t(as.matrix(X))
}
rownames(counts) <- as.character(adata$var_names$tolist())
colnames(counts) <- as.character(adata$obs_names$tolist())

metadata <- as.data.frame(adata$obs)

cat(sprintf("Dataset: %d cells, %d genes\n", ncol(counts), nrow(counts)))

# Create Seurat object
seurat_obj <- CreateSeuratObject(
  counts = counts,
  meta.data = metadata
)

# Set data layer (preprocessed log-normalized values)
seurat_obj <- SetAssayData(seurat_obj, layer="data", new.data=counts)

# Data is already HVG-selected from Python preprocessing (same as other methods)
# Use all genes as features (they are already HVG)
all_features <- rownames(seurat_obj)

# Set variable features explicitly on the main object (required before split)
VariableFeatures(seurat_obj) <- all_features

# Scale and PCA
seurat_obj <- ScaleData(seurat_obj, verbose=FALSE)
seurat_obj <- RunPCA(seurat_obj, npcs=50, verbose=FALSE, seed.use=args$seed)

# Split by batch and prepare integration
seurat_list <- SplitObject(seurat_obj, split.by=args$batch_key)

# Use all features (already HVG from Python)
features <- all_features

seurat_list <- lapply(seurat_list, function(x) {
  VariableFeatures(x) <- features
  x <- ScaleData(x, features=features, verbose=FALSE)
  x <- RunPCA(x, features=features, npcs=50, verbose=FALSE, seed.use=args$seed)
  return(x)
})

anchors <- FindIntegrationAnchors(seurat_list, anchor.features=features, reduction="rpca", dims=1:50, verbose=FALSE)
seurat_integrated <- IntegrateData(anchors, dims=1:50, verbose=FALSE)

seurat_integrated <- ScaleData(seurat_integrated, verbose=FALSE)
seurat_integrated <- RunPCA(seurat_integrated, npcs=50, verbose=FALSE, seed.use=args$seed)

# Save h5ad for Python evaluation
adata_integrated <- sc$AnnData(X=np$array(Embeddings(seurat_integrated,"pca")),
                               obs=pd$DataFrame(seurat_integrated@meta.data))
temp_h5ad <- file.path(args$save_path,"temp_integrated.h5ad")
adata_integrated$write(temp_h5ad)

cat("Seurat integration complete. Handing off to Python for evaluation and visualization.\n")
