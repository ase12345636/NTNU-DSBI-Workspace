import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: compute metrics CSVs if missing by reusing evaluate_from_embeddings
try:
    from metric import evaluate_from_embeddings
    CAN_COMPUTE = True
except Exception:
    evaluate_from_embeddings = None
    CAN_COMPUTE = False


def find_datasets(root_dir: str):
    datasets = []
    if not os.path.isdir(root_dir):
        return datasets
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            datasets.append(name)
    return datasets


def find_seeds(dataset_dir: str):
    seeds = []
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path) and name.isdigit():
            seeds.append(name)
    return seeds


def read_metrics_csvs(dataset_dir: str, seeds: list):
    """Read all metrics_from_embeddings.csv under given seeds.
    Returns a dict seed -> DataFrame (index: metric, columns: methods)
    """
    per_seed = {}
    for seed in seeds:
        csv_path = os.path.join(dataset_dir, seed, "metrics_from_embeddings.csv")
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0)
                per_seed[seed] = df
            except Exception as e:
                print(f"Warn: failed reading {csv_path}: {e}")
        else:
            # silently skip missing CSVs; may be computed by caller
            continue
    return per_seed


def aggregate_metrics(per_seed: dict):
    """Aggregate into long-form DataFrame with columns:
    dataset, metric, method, seed, value
    """
    rows = []
    for seed, df in per_seed.items():
        for metric in df.index:
            for method in df.columns:
                val = df.loc[metric, method]
                if pd.isna(val):
                    continue
                rows.append({
                    'metric': metric,
                    'method': method,
                    'seed': seed,
                    'value': float(val),
                })
    return pd.DataFrame(rows)


def plot_metric_bar(dataset: str, metric: str, df_long: pd.DataFrame, save_dir: str):
    df_m = df_long[df_long['metric'] == metric]
    if df_m.empty:
        return None
    stats = df_m.groupby('method')['value'].agg(['mean', 'std']).reindex(sorted(df_m['method'].unique()))

    plt.figure(figsize=(6, 4))
    x = np.arange(len(stats.index))
    y = stats['mean'].values
    yerr = stats['std'].values
    plt.bar(x, y, yerr=yerr, capsize=5, color="#69b3a2")
    plt.xticks(x, stats.index, rotation=0)
    plt.ylabel(metric)
    plt.title(f"{dataset}: {metric} (mean ± std)")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset}_{metric.replace('/', '-')}_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'result'))
    datasets = find_datasets(root)
    if not datasets:
        print(f"No datasets found under: {root}")
        return

    print(f"Datasets: {datasets}")
    for ds in datasets:
        ds_dir = os.path.join(root, ds)
        seeds = find_seeds(ds_dir)

        # Compute metrics if missing and we can compute
        if CAN_COMPUTE:
            data_cfg = {
                'immune': {
                    'data': "/Group16T/raw_data/scCobra/Immune_ALL_human.h5ad",
                    'batch': 'batch',
                    'celltype': 'final_annotation',
                },
                'lung': {
                    'data': "/Group16T/raw_data/scCobra/Lung_atlas_public.h5ad",
                    'batch': 'batch',
                    'celltype': 'cell_type',
                },
                'pancreas': {
                    'data': "/Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad",
                    'batch': 'tech',
                    'celltype': 'celltype',
                }
            }
            cfg = data_cfg.get(ds)
            if cfg is not None:
                for seed in seeds:
                    out_dir = os.path.join(ds_dir, seed)
                    csv_path = os.path.join(out_dir, "metrics_from_embeddings.csv")
                    if not os.path.isfile(csv_path):
                        try:
                            print(f"Computing metrics for {ds}/{seed} → {csv_path}")
                            evaluate_from_embeddings(
                                data=cfg['data'],
                                out_path=out_dir,
                                batch=cfg['batch'],
                                celltype=cfg['celltype'],
                            )
                        except Exception as e:
                            print(f"Warn: failed computing metrics for {ds}/{seed}: {e}")

        per_seed = read_metrics_csvs(ds_dir, seeds)
        if not per_seed:
            print(f"No CSVs found under {ds_dir}; skipping.")
            continue
        df_long = aggregate_metrics(per_seed)
        metrics = sorted(df_long['metric'].unique())
        save_dir = os.path.join(ds_dir, 'plots')
        outputs = []
        for m in metrics:
            out = plot_metric_bar(ds, m, df_long, save_dir)
            if out:
                outputs.append(out)
        if outputs:
            print(f"Saved {len(outputs)} plots to {save_dir}")


if __name__ == '__main__':
    main()
