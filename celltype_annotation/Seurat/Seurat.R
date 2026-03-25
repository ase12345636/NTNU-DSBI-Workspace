#!/usr/bin/env Rscript
# Seurat reference-based cell type annotation (label transfer).
# Called by Seurat.py — reads mtx + csv (native R), no reticulate needed.
# Tasks:
#   - train_reference: Build Seurat reference object, save as .rds
#   - predict: Load reference .rds, FindTransferAnchors + TransferData, write predictions.csv

# ============================================================
# 1. Setup
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(argparse)
  library(Matrix)
})

# ============================================================
# 2. Parse Arguments
# ============================================================

parser <- ArgumentParser(description = "Seurat label transfer for cell type annotation")
parser$add_argument("--task", type = "character", required = TRUE,
                    help = "train_reference or predict")
parser$add_argument("--input_dir", type = "character", required = TRUE,
                    help = "Directory with matrix.mtx, barcodes.csv, features.csv, metadata.csv")
parser$add_argument("--reference_rds", type = "character",
                    help = "Reference .rds file (for predict task)")
parser$add_argument("--save_path", type = "character", required = TRUE,
                    help = "Save directory")
parser$add_argument("--label_col", type = "character", default = "Celltype_training",
                    help = "Label column in metadata")
parser$add_argument("--seed", type = "integer", default = 42,
                    help = "Random seed")
args <- parser$parse_args()

set.seed(args$seed)
dir.create(args$save_path, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 3. Native R I/O helpers
# ============================================================

load_from_exchange <- function(input_dir) {
  counts <- readMM(file.path(input_dir, "matrix.mtx"))
  counts <- as(counts, "dgCMatrix")
  barcodes <- read.csv(file.path(input_dir, "barcodes.csv"))
  features <- read.csv(file.path(input_dir, "features.csv"))
  metadata <- read.csv(file.path(input_dir, "metadata.csv"), row.names = 1)

  rownames(counts) <- features$feature
  colnames(counts) <- barcodes$barcode

  list(counts = counts, metadata = metadata, barcodes = barcodes$barcode)
}

# ============================================================
# 4. Tasks
# ============================================================

if (args$task == "train_reference") {
  cat("Loading training data from", args$input_dir, "\n")
  data <- load_from_exchange(args$input_dir)

  cat(sprintf("[train_reference] Creating Seurat object: %d cells, %d genes\n",
              ncol(data$counts), nrow(data$counts)))

  reference <- CreateSeuratObject(counts = data$counts, meta.data = data$metadata)
  reference <- FindVariableFeatures(reference, verbose = FALSE)
  reference <- ScaleData(reference, verbose = FALSE)
  reference <- RunPCA(reference, npcs = 30, verbose = FALSE, seed.use = args$seed)
  reference <- RunUMAP(reference, dims = 1:30, umap.method = "uwot",
                       metric = "cosine", verbose = FALSE, seed.use = args$seed)

  reference_path <- file.path(args$save_path, "reference.rds")
  saveRDS(reference, reference_path)
  cat("[train_reference] Saved to", reference_path, "\n")

} else if (args$task == "predict") {
  if (is.null(args$reference_rds)) {
    stop("--reference_rds is required for predict task")
  }

  cat("Loading reference from", args$reference_rds, "\n")
  reference <- readRDS(args$reference_rds)

  cat("Loading query data from", args$input_dir, "\n")
  data <- load_from_exchange(args$input_dir)

  cat(sprintf("[predict] Creating query Seurat object: %d cells, %d genes\n",
              ncol(data$counts), nrow(data$counts)))

  query <- CreateSeuratObject(counts = data$counts, meta.data = data$metadata)

  # Process query
  common_features <- intersect(rownames(query), rownames(reference))
  cat(sprintf("[predict] Common features: %d / %d query genes\n",
              length(common_features), nrow(query)))

  query <- ScaleData(query, features = common_features, verbose = FALSE)
  query <- RunPCA(query, features = common_features, npcs = 30, verbose = FALSE,
                  seed.use = args$seed)

  if (!(args$label_col %in% colnames(reference@meta.data))) {
    stop(paste0("Label column '", args$label_col, "' not found in reference metadata."))
  }

  cat("[predict] Finding anchors...\n")
  anchors <- FindTransferAnchors(
    reference = reference, query = query,
    dims = 1:30, reference.reduction = "pca", verbose = FALSE
  )

  cat("[predict] Transferring labels...\n")
  ref_labels <- reference@meta.data[[args$label_col]]
  predictions_df <- TransferData(
    anchorset = anchors, refdata = ref_labels,
    dims = 1:30, verbose = FALSE
  )
  query <- AddMetaData(query, metadata = predictions_df)

  # Extract predictions
  pred_candidates <- c("predicted.id", "predicted.celltype", "predicted_celltype")
  pred_col <- pred_candidates[pred_candidates %in% colnames(query@meta.data)][1]
  if (is.na(pred_col) || length(pred_col) == 0) {
    stop("Could not find predicted label column after TransferData")
  }
  predictions <- as.character(query@meta.data[[pred_col]])

  # Write predictions as CSV
  pred_df <- data.frame(
    barcode = data$barcodes,
    prediction = predictions,
    stringsAsFactors = FALSE
  )
  pred_path <- file.path(args$save_path, "predictions.csv")
  write.csv(pred_df, pred_path, row.names = FALSE)
  cat("[predict] Predictions saved to", pred_path, "\n")

} else {
  stop(paste("Unknown task:", args$task))
}
