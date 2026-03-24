"""SingleR cell type annotation benchmark."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline import (
    create_argument_parser, run_annotation_pipeline,
    train_r_tool, predict_r_tool,
)

R_SCRIPT = Path(__file__).parent / "SingleR.R"


def train_singler(train_path, model_dir, label_col, seed, args):
    return train_r_tool(train_path, model_dir, label_col, seed, R_SCRIPT)


def predict_singler(test_path, model_handle, pred_col, seed, args):
    return predict_r_tool(test_path, model_handle, pred_col, seed, R_SCRIPT)


if __name__ == "__main__":
    parser = create_argument_parser("SingleR")
    args = parser.parse_args()
    run_annotation_pipeline(
        "SingleR", "singler_pred",
        train_singler, predict_singler, args,
    )
