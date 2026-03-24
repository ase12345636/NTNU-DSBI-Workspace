import os
import sys
import scvi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def integrate_scvi(adata, seed, run_out_path, args):
    """Run scVI batch correction."""
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, batch_key=args.batch_key)
    model = scvi.model.SCVI(adata)
    model.train()
    adata.obsm["X_scVI"] = model.get_latent_representation()
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scVi')
    args = parser.parse_args()
    run_integration_pipeline("scVi", integrate_scvi, "X_scVI", args)

