import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import os
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Set style
sc.settings.set_figure_params(dpi=300, facecolor='white', frameon=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Define paths
BASE_PATH = "/Group16T/common/ccuc/Workspace/result"
DATASETS = ["immune", "lung", "pancreas"]
TOOLS = ["raw", "scanorama", "scCobra", "scDML", "scVi", "Seurat", "harmony", "scLCY"]
OUTPUT_DIR = "/Group16T/common/ccuc/Workspace/analysis_output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for each dataset
for dataset in DATASETS:
    os.makedirs(os.path.join(OUTPUT_DIR, dataset), exist_ok=True)

# ============================================================
# Task 1: Generate UMAP visualizations (batch vs celltype)
# ============================================================
print("=" * 60)
print("Task 1: Generating UMAP visualizations")
print("=" * 60)

# Collect UMAP data for each dataset
for dataset in DATASETS:
    print(f"\n{dataset.upper()} Dataset:")
    
    # Collect data from all tools
    umap_data_dict = {}
    adata_dict = {}
    
    for tool in TOOLS:
        tool_path = os.path.join(BASE_PATH, dataset, tool)
        
        if not os.path.exists(tool_path):
            print(f"  Skipping {tool} (not found)")
            continue
        
        # All tools use round 1
        data_path = os.path.join(tool_path, "1")
        if not os.path.isdir(data_path):
            print(f"  Skipping {tool} (round 1 not found)")
            continue
        
        # Read UMAP coordinates
        umap_file = os.path.join(data_path, "umap_coordinates.csv")
        if os.path.exists(umap_file):
            try:
                df = pd.read_csv(umap_file)
                
                # Create AnnData object for scanpy plotting
                adata = sc.AnnData(X=np.zeros((len(df), 10)))  # Dummy X
                adata.obsm['X_umap'] = df[['UMAP1', 'UMAP2']].values
                adata.obs['batch'] = df['batch'].values
                adata.obs['celltype'] = df['celltype'].values
                
                umap_data_dict[tool] = df
                adata_dict[tool] = adata
                print(f"  ✓ Loaded {tool}: {len(df)} cells")
            except Exception as e:
                print(f"  ✗ Error loading {tool}: {e}")
    
    if not adata_dict:
        print(f"  No data found for {dataset}")
        continue
    
    # Get all unique batches and celltypes for unified legend
    all_batches = set()
    all_celltypes = set()
    for adata in adata_dict.values():
        all_batches.update(adata.obs['batch'].unique())
        all_celltypes.update(adata.obs['celltype'].unique())
    all_batches = sorted(list(all_batches))
    all_celltypes = sorted(list(all_celltypes))
    
    # ===== Figure 1: Batch coloring =====
    n_tools = len(adata_dict)
    n_cols = 4  # Fixed: 4 tools per row
    n_rows = 2  # Fixed: 2 rows
    
    # Sort tools with raw first
    tool_order = ['raw'] + [t for t in sorted(adata_dict.keys()) if t != 'raw']
    
    # Create figure with extra column for legend
    fig = plt.figure(figsize=(5*n_cols + 1.0, 4*n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.15],
                         hspace=0.1, wspace=0.2, top=0.9, bottom=0.05, left=0.05, right=0.99)
    fig.suptitle(f"{dataset.upper()} - UMAP colored by Batch", 
                fontsize=16, fontweight='bold')
    
    # Plot each tool
    batch_colors = None
    for idx, tool in enumerate(tool_order):
        if tool not in adata_dict:
            continue
        adata = adata_dict[tool]
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Use scanpy's plotting without legend
        sc.pl.umap(adata, color='batch', ax=ax, show=False, 
                  frameon=True, title=tool.upper(), s=3, legend_loc='none')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Store colors from first plot
        if batch_colors is None and hasattr(adata, 'uns') and 'batch_colors' in adata.uns:
            batch_colors = adata.uns['batch_colors']
    
    # Create unified legend on the right
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis('off')
    
    # Get colors from scanpy's default palette
    if batch_colors is None:
        from matplotlib import cm
        batch_palette = sc.pl.palettes.default_102
        batch_colors = [batch_palette[i % len(batch_palette)] for i in range(len(all_batches))]
    
    # Use circular markers (Line2D) instead of square patches
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=batch_colors[i] if i < len(batch_colors) else 'gray',
                             markersize=8, label=batch, markeredgewidth=0) 
                      for i, batch in enumerate(all_batches)]
    legend_ax.legend(handles=legend_elements, title="Batch", 
                    loc='center left', fontsize=9, frameon=False,
                    title_fontsize=11)
    
    output_file = os.path.join(OUTPUT_DIR, dataset, f"{dataset}_batch_umap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved batch UMAP to {output_file}")
    plt.close()
    
    # ===== Figure 2: Celltype coloring =====
    fig = plt.figure(figsize=(5*n_cols + 1.0, 4*n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.15],
                         hspace=0.1, wspace=0.2, top=0.9, bottom=0.05, left=0.05, right=0.99)
    fig.suptitle(f"{dataset.upper()} - UMAP colored by Cell Type", 
                fontsize=16, fontweight='bold')
    
    # Plot each tool with raw first
    celltype_colors = None
    for idx, tool in enumerate(tool_order):
        if tool not in adata_dict:
            continue
        adata = adata_dict[tool]
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Use scanpy's plotting without legend
        sc.pl.umap(adata, color='celltype', ax=ax, show=False, 
                  frameon=True, title=tool.upper(), s=3, legend_loc='none')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Store colors from first plot
        if celltype_colors is None and hasattr(adata, 'uns') and 'celltype_colors' in adata.uns:
            celltype_colors = adata.uns['celltype_colors']
    
    # Create unified legend on the right
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis('off')
    
    # Get colors from scanpy's default palette
    if celltype_colors is None:
        celltype_palette = sc.pl.palettes.default_102
        celltype_colors = [celltype_palette[i % len(celltype_palette)] for i in range(len(all_celltypes))]
    
    # Use circular markers (Line2D) instead of square patches
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=celltype_colors[i] if i < len(celltype_colors) else 'gray',
                             markersize=8, label=celltype, markeredgewidth=0) 
                      for i, celltype in enumerate(all_celltypes)]
    legend_ax.legend(handles=legend_elements, title="Cell Type", 
                    loc='center left', fontsize=8, frameon=False,
                    title_fontsize=11)
    
    output_file = os.path.join(OUTPUT_DIR, dataset, f"{dataset}_celltype_umap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved celltype UMAP to {output_file}")
    plt.close()

# ============================================================
# Task 2: Generate metrics bar plots (for all tools except raw)
# ============================================================
print("\n" + "=" * 60)
print("Task 2: Generating metrics bar plots")
print("=" * 60)

# Collect all metrics by dataset
metrics_by_dataset = {}

for dataset in DATASETS:
    metrics_by_dataset[dataset] = {}
    
    for tool in TOOLS:
        # Skip raw tool for metrics
        if tool == "raw":
            continue
            
        tool_path = os.path.join(BASE_PATH, dataset, tool)
        
        if not os.path.exists(tool_path):
            continue
        
        print(f"Collecting metrics for {dataset}/{tool}... ", end="")
        
        metrics_list = []
        
        # Collect metrics from all rounds
        for round_num in [1, 2, 3, 4, 5]:
            round_path = os.path.join(tool_path, str(round_num))
            metrics_file = os.path.join(round_path, f"{tool}_metrics.csv")
            
            if os.path.exists(metrics_file):
                try:
                    metrics = pd.read_csv(metrics_file)
                    metrics_list.append(metrics)
                except Exception as e:
                    print(f"Warning: Failed to read {metrics_file}: {e}")
        
        if metrics_list:
            metrics_by_dataset[dataset][tool] = metrics_list
            print(f"✓ Found {len(metrics_list)} rounds")
        else:
            print("✗ No metrics found")

# Generate plots for each dataset and each metric
for dataset in DATASETS:
    if dataset not in metrics_by_dataset or not metrics_by_dataset[dataset]:
        print(f"\nNo metrics data for {dataset}")
        continue
    
    print(f"\n{dataset.upper()} - Generating metric plots...")
    
    # Get all metric names for this dataset
    all_metric_names = set()
    for tool, metrics_list in metrics_by_dataset[dataset].items():
        if metrics_list:
            metric_names = [col for col in metrics_list[0].columns 
                           if col != 'best_leiden_resolution']
            all_metric_names.update(metric_names)
    
    # Create metrics subdirectory for this dataset
    metrics_dir = os.path.join(OUTPUT_DIR, dataset, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    for metric_name in sorted(all_metric_names):
        print(f"  Plotting {metric_name}... ", end="")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_data = []
        labels = []
        colors_list = plt.cm.tab10(np.linspace(0, 1, len(metrics_by_dataset[dataset])))
        
        for (tool, metrics_list), color in zip(sorted(metrics_by_dataset[dataset].items()), colors_list):
            # Extract metric values from all rounds
            values = []
            for metrics in metrics_list:
                if metric_name in metrics.columns:
                    values.append(metrics[metric_name].values[0])
            
            if values:
                values = np.array(values)
                avg = np.mean(values)
                std = np.std(values)
                
                plot_data.append({
                    'tool': tool,
                    'avg': avg,
                    'std': std,
                    'values': values,
                    'color': color
                })
                labels.append(tool.upper())
        
        if plot_data:
            # Create bar plot
            num_tools = len(plot_data)
            # Very thin bars that stick together
            bar_width = 0.15  # Thin bars
            x_pos = np.arange(num_tools) * 0.15  # Bars stick together
            avgs = [d['avg'] for d in plot_data]
            stds = [d['std'] for d in plot_data]
            colors = [d['color'] for d in plot_data]
            
            # Set y-axis limits to not start from 0 (add some padding)
            all_values = []
            for d in plot_data:
                all_values.extend(d['values'])
            y_min = min(all_values) * 0.95
            y_max = max(all_values) * 1.05
            
            bars = ax.bar(x_pos, avgs, yerr=stds, capsize=3, color=colors, 
                         alpha=1.0, edgecolor='black', linewidth=1.2, width=bar_width)
            
            # Overlay individual points
            for i, d in enumerate(plot_data):
                values = d['values']
                # Limit jitter to stay within the bar width
                x_jitter = np.random.normal(x_pos[i], 0.015, size=len(values))
                # Clip to ensure points don't overflow into adjacent bars
                x_jitter = np.clip(x_jitter, x_pos[i] - bar_width/2, x_pos[i] + bar_width/2)
                ax.scatter(x_jitter, values, color='darkred', s=50, alpha=0.6, zorder=3)
            
            ax.set_xlabel("Tool", fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f"{dataset.upper()} - {metric_name}", 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            output_file = os.path.join(metrics_dir, f"{metric_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {output_file}")
            plt.close()
        else:
            print("✗ No valid data")

# ============================================================
# Task 3: Generate Runtime Comparison
# ============================================================
print("\n" + "=" * 60)
print("Task 3: Generating Runtime Comparison")
print("=" * 60)

# Collect runtime data by dataset
runtime_by_dataset = {}

for dataset in DATASETS:
    runtime_by_dataset[dataset] = {}
    
    for tool in TOOLS:
        # Skip raw tool for runtime
        if tool == "raw":
            continue
            
        tool_path = os.path.join(BASE_PATH, dataset, tool)
        
        if not os.path.exists(tool_path):
            continue
        
        print(f"Collecting runtime for {dataset}/{tool}... ", end="")
        
        runtime_list = []
        
        # Collect runtime from all rounds
        for round_num in [1, 2, 3, 4, 5]:
            round_path = os.path.join(tool_path, str(round_num))
            metrics_file = os.path.join(round_path, f"{tool}_metrics.csv")
            
            if os.path.exists(metrics_file):
                try:
                    metrics = pd.read_csv(metrics_file)
                    if 'runtime_seconds' in metrics.columns:
                        runtime_list.append(metrics['runtime_seconds'].values[0])
                except Exception as e:
                    print(f"Warning: Failed to read {metrics_file}: {e}")
        
        if runtime_list:
            runtime_by_dataset[dataset][tool] = runtime_list
            print(f"✓ Found {len(runtime_list)} rounds")
        else:
            print("✗ No runtime data found")

# Generate overall runtime comparison (all datasets)
print(f"\nGenerating overall runtime comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Runtime Comparison Across Datasets', fontsize=16, fontweight='bold')

for idx, dataset in enumerate(DATASETS):
    ax = axes[idx]
    
    if dataset not in runtime_by_dataset or not runtime_by_dataset[dataset]:
        ax.text(0.5, 0.5, f"No data for {dataset}", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{dataset.upper()}", fontsize=14, fontweight='bold')
        continue
    
    plot_data = []
    labels = []
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(runtime_by_dataset[dataset])))
    
    for (tool, runtime_list), color in zip(sorted(runtime_by_dataset[dataset].items()), colors_list):
        runtime_array = np.array(runtime_list)
        avg_runtime = np.mean(runtime_array)
        std_runtime = np.std(runtime_array)
        
        plot_data.append({
            'tool': tool,
            'avg': avg_runtime,
            'std': std_runtime,
            'values': runtime_array,
            'color': color
        })
        labels.append(tool.upper())
    
    if plot_data:
        # Create bar plot
        num_tools = len(plot_data)
        bar_width = 0.15
        x_pos = np.arange(num_tools) * 0.15
        avgs = [d['avg'] for d in plot_data]
        stds = [d['std'] for d in plot_data]
        colors = [d['color'] for d in plot_data]
        
        bars = ax.bar(x_pos, avgs, yerr=stds, capsize=3, color=colors, 
                     alpha=1.0, edgecolor='black', linewidth=1.2, width=bar_width)
        
        # Overlay individual points
        for i, d in enumerate(plot_data):
            values = d['values']
            x_jitter = np.random.normal(x_pos[i], 0.015, size=len(values))
            x_jitter = np.clip(x_jitter, x_pos[i] - bar_width/2, x_pos[i] + bar_width/2)
            ax.scatter(x_jitter, values, color='darkred', s=50, alpha=0.6, zorder=3)
        
        ax.set_xlabel("Tool", fontsize=11, fontweight='bold')
        ax.set_ylabel("Runtime (seconds)", fontsize=11, fontweight='bold')
        ax.set_title(f"{dataset.upper()}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for i, (x, y) in enumerate(zip(x_pos, avgs)):
            if y < 60:
                ax.text(x, y, f'{y:.1f}s', ha='center', va='bottom', fontsize=8)
            elif y < 3600:
                ax.text(x, y, f'{y/60:.1f}m', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(x, y, f'{y/3600:.1f}h', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, "runtime_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved overall runtime comparison to {output_file}")
plt.close()

# Generate detailed runtime table
print(f"\nGenerating runtime summary table...")

runtime_summary = []
for dataset in DATASETS:
    if dataset not in runtime_by_dataset or not runtime_by_dataset[dataset]:
        continue
    
    for tool, runtime_list in sorted(runtime_by_dataset[dataset].items()):
        runtime_array = np.array(runtime_list)
        runtime_summary.append({
            'Dataset': dataset,
            'Tool': tool,
            'Avg Runtime (s)': f"{np.mean(runtime_array):.2f}",
            'Std Runtime (s)': f"{np.std(runtime_array):.2f}",
            'Min Runtime (s)': f"{np.min(runtime_array):.2f}",
            'Max Runtime (s)': f"{np.max(runtime_array):.2f}",
            'Runs': len(runtime_list)
        })

if runtime_summary:
    runtime_df = pd.DataFrame(runtime_summary)
    output_csv = os.path.join(OUTPUT_DIR, "runtime_summary.csv")
    runtime_df.to_csv(output_csv, index=False)
    print(f"✓ Saved runtime summary table to {output_csv}")
    print("\nRuntime Summary:")
    print(runtime_df.to_string(index=False))

print("\n" + "=" * 60)
print("Analysis complete!")
print(f"Output files saved to: {OUTPUT_DIR}")
print("=" * 60)
