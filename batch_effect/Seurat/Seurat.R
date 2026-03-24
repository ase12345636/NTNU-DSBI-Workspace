#!/usr/bin/env Rscript
# Seurat v5 RPCA batch correction
# Called by Seurat.py — reads mtx + csv (native R), integrates, exports embedding CSV.
# No reticulate / Python dependency required.

# ============================================================
# 1. Setup
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(argparse)
  library(Matrix)
  library(future)
})

options(future.globals.maxSize = Inf)
plan("multicore", workers = 8)

# ============================================================
# 2. Parse Arguments
# ============================================================

parser <- ArgumentParser(description='Run Seurat v5 batch correction')
parser$add_argument('--input_dir', type='character', required=TRUE,
                    help='Directory containing matrix.mtx, barcodes.csv, features.csv, metadata.csv')
parser$add_argument('--batch_key', type='character', default='batch')
parser$add_argument('--seed', type='integer', default=42)
args <- parser$parse_args()

set.seed(args$seed)

# ============================================================
# 3. Load Data (native R formats)
# ============================================================

cat("Loading data from mtx + csv...\n")

counts <- readMM(file.path(args$input_dir, "matrix.mtx"))
counts <- as(counts, "dgCMatrix")
barcodes <- read.csv(file.path(args$input_dir, "barcodes.csv"))
features <- read.csv(file.path(args$input_dir, "features.csv"))
metadata <- read.csv(file.path(args$input_dir, "metadata.csv"), row.names = 1)

rownames(counts) <- features$feature
colnames(counts) <- barcodes$barcode

cat(sprintf("Dataset: %d cells, %d genes\n", ncol(counts), nrow(counts)))

# ============================================================
# 4. Create Seurat Object & Integration (timed)
# ============================================================

timer_start <- proc.time()

seurat_obj <- CreateSeuratObject(counts = counts, meta.data = metadata)
seurat_obj[["RNA"]]$data <- counts

all_features <- rownames(seurat_obj)
VariableFeatures(seurat_obj) <- all_features

seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
seurat_obj <- RunPCA(seurat_obj, npcs = 50, verbose = FALSE, seed.use = args$seed)

# ============================================================
# 5. RPCA Integration
# ============================================================

seurat_list <- SplitObject(seurat_obj, split.by = args$batch_key)

seurat_list <- lapply(seurat_list, function(x) {
  VariableFeatures(x) <- all_features
  x <- ScaleData(x, features = all_features, verbose = FALSE)
  x <- RunPCA(x, features = all_features, npcs = 50, verbose = FALSE, seed.use = args$seed)
  return(x)
})

anchors <- FindIntegrationAnchors(
  seurat_list, anchor.features = all_features,
  reduction = "rpca", dims = 1:50, verbose = FALSE
)
seurat_integrated <- IntegrateData(anchors, dims = 1:50, verbose = FALSE)

seurat_integrated <- ScaleData(seurat_integrated, verbose = FALSE)
seurat_integrated <- RunPCA(seurat_integrated, npcs = 50, verbose = FALSE, seed.use = args$seed)

timer_end <- proc.time()
algo_seconds <- (timer_end - timer_start)["elapsed"]
cat(sprintf("Algorithm runtime: %.4f seconds\n", algo_seconds))
write.csv(data.frame(runtime_seconds = algo_seconds),
          file.path(args$input_dir, "timing.csv"), row.names = FALSE)

# ============================================================
# 6. Export Embedding as CSV
# ============================================================

embeddings <- Embeddings(seurat_integrated, "pca")
write.csv(embeddings, file.path(args$input_dir, "embedding.csv"))

cat("Seurat integration complete.\n")
