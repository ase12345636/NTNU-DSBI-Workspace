"""Seurat cell type annotation benchmark."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline import (
    create_argument_parser, run_annotation_pipeline,
    train_r_tool, predict_r_tool,
)

R_SCRIPT = Path(__file__).parent / "Seurat.R"


def train_seurat(train_path, model_dir, label_col, seed, args):
    rds = train_r_tool(train_path, model_dir, label_col, seed, R_SCRIPT)
    return (rds, label_col)


def predict_seurat(test_path, model_handle, pred_col, seed, args):
    rds_path, label_col = model_handle
    return predict_r_tool(test_path, rds_path, pred_col, seed, R_SCRIPT,
                          extra_args=["--label_col", label_col])


if __name__ == "__main__":
    parser = create_argument_parser("Seurat")
    args = parser.parse_args()
    run_annotation_pipeline(
        "Seurat", "seurat_pred",
        train_seurat, predict_seurat, args,
    )
