#!/usr/bin/env Rscript
# SingleR reference-based cell type annotation.
# Called by SingleR.py — reads mtx + csv (native R), no reticulate needed.
# Tasks:
#   - train_reference: Build SingleCellExperiment reference, save as .rds
#   - predict: Load reference .rds, classify query cells, write predictions.csv

# ============================================================
# 1. Setup
# ============================================================

suppressPackageStartupMessages({
  library(argparse)
  library(Matrix)
  library(SingleCellExperiment)
  library(S4Vectors)
  library(SingleR)
  library(SummarizedExperiment)
})

# ============================================================
# 2. Parse Arguments
# ============================================================

parser <- ArgumentParser(description = "SingleR cell type annotation")
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

make_sce <- function(data, label_col = NULL, assay_name = "counts") {
  assays_list <- list()
  assays_list[[assay_name]] <- data$counts
  sce <- SingleCellExperiment(
    assays = assays_list,
    colData = DataFrame(data$metadata)
  )
  if (!is.null(label_col)) {
    if (!(label_col %in% colnames(colData(sce)))) {
      stop(paste0("Label column '", label_col, "' not found in metadata."))
    }
    sce$celltypes <- as.character(colData(sce)[[label_col]])
  }
  sce
}

# ============================================================
# 4. Tasks
# ============================================================

if (args$task == "train_reference") {
  cat("Loading training data from", args$input_dir, "\n")
  data <- load_from_exchange(args$input_dir)
  reference <- make_sce(data, label_col = args$label_col)

  cat(sprintf("[train_reference] Reference: %d cells, %d genes\n",
              ncol(reference), nrow(reference)))

  reference_obj <- list(ref = reference, label_col = args$label_col)

  reference_path <- file.path(args$save_path, "reference.rds")
  saveRDS(reference_obj, reference_path)
  cat("[train_reference] Saved to", reference_path, "\n")

} else if (args$task == "predict") {
  if (is.null(args$reference_rds)) {
    stop("--reference_rds is required for predict task")
  }

  cat("Loading reference from", args$reference_rds, "\n")
  reference_obj <- readRDS(args$reference_rds)
  reference <- reference_obj$ref
  label_col <- reference_obj$label_col

  if (!(label_col %in% colnames(colData(reference)))) {
    stop(paste0("Label column '", label_col, "' is missing in reference."))
  }

  cat("Loading query data from", args$input_dir, "\n")
  data <- load_from_exchange(args$input_dir)
  query <- make_sce(data, label_col = NULL)

  common_genes <- intersect(rownames(reference), rownames(query))
  if (length(common_genes) == 0) {
    stop("No overlapping genes between reference and query.")
  }

  reference_common <- reference[common_genes, ]
  query_common <- query[common_genes, ]

  cat(sprintf("[predict] Query: %d cells, %d genes (%d overlapping)\n",
              ncol(query_common), nrow(query_common), length(common_genes)))

  de_method <- if (requireNamespace("scran", quietly = TRUE)) "wilcox" else "classic"
  if (de_method == "classic") {
    cat("[predict][warn] scran not found; falling back to de.method='classic'\n")
  }

  cat("[predict] Running SingleR classifier...\n")
  pred <- SingleR(
    test = query_common,
    ref = reference_common,
    labels = as.character(colData(reference_common)[[label_col]]),
    de.method = de_method,
    assay.type.test = "counts",
    assay.type.ref = "counts"
  )

  labels <- as.character(pred$labels)
  labels[is.na(labels)] <- "Unknown"

  # Write predictions as CSV
  pred_df <- data.frame(
    barcode = data$barcodes,
    prediction = labels,
    stringsAsFactors = FALSE
  )
  pred_path <- file.path(args$save_path, "predictions.csv")
  write.csv(pred_df, pred_path, row.names = FALSE)
  cat("[predict] Predictions saved to", pred_path, "\n")

} else {
  stop(paste("Unknown task:", args$task))
}
