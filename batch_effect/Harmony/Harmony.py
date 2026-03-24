import os
import sys
from harmony import harmonize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def integrate_harmony(adata, seed, run_out_path, args):
    """Run Harmony batch correction on PCA embeddings."""
    adata.obsm['X_pca_harmony'] = harmonize(
        adata.obsm['X_pca'], adata.obs,
        batch_key=args.batch_key, random_state=seed, use_gpu=True
    )
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('harmony')
    args = parser.parse_args()
    run_integration_pipeline("harmony", integrate_harmony, "X_pca_harmony", args)

