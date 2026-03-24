import os
import sys
from scCobra import scCobra

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def integrate_sccobra(adata, seed, run_out_path, args):
    """Run scCobra batch correction."""
    adata = scCobra(
        adata, batch_name=args.batch_key, processed=True,
        outdir=run_out_path, show=False, gpu=0,
        ignore_umap=True, seed=seed
    )
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scCobra')
    args = parser.parse_args()
    run_integration_pipeline("scCobra", integrate_sccobra, "latent", args)

