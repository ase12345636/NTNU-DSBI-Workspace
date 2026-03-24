import os
import sys
import scanpy as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def integrate_raw(adata, seed, run_out_path, args):
    """Raw baseline: only compute PCA (no batch correction)."""
    sc.tl.pca(adata, random_state=seed)
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('raw')
    args = parser.parse_args()
    run_integration_pipeline("raw", integrate_raw, "X_pca", args, run_pca=False)

