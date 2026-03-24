import os
import sys
from scDML import scDMLModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def integrate_scdml(adata, seed, run_out_path, args):
    """Run scDML batch correction."""
    model = scDMLModel()
    model.integrate(
        adata, batch_key=args.batch_key, ncluster_list=[15],
        merge_rule="rule2", expect_num_cluster=15,
        mode="unsupervised", seed=seed
    )
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scDML')
    args = parser.parse_args()
    run_integration_pipeline("scDML", integrate_scdml, "X_emb", args)

