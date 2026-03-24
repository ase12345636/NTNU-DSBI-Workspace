bash run_benchmark.sh \
    --dataset_path /Group16T/common/ccuc/scLightGAT/scLightGAT_data \
    --save_path /Group16T/common/ccuc/Workspace/celltype_annotation/result \
    --run_times 5

python3 analyze/plot_benchmark_methods_bar_charts_csv.py \
    --results_dir /Group16T/common/ccuc/Workspace/celltype_annotation/result \
    --sclightgat_dir /Group16T/common/ccuc/scLightGAT/sclightgat_exp_results \
    --out_dir /Group16T/common/ccuc/Workspace/celltype_annotation/result/figures

python3 analyze/plot_subtype_methods_bar_charts_csv.py \
    --results_dir /Group16T/common/ccuc/Workspace/celltype_annotation/result \
    --sclightgat_dir /Group16T/common/ccuc/scLightGAT/sclightgat_exp_results \
    --out_dir /Group16T/common/ccuc/Workspace/celltype_annotation/result/figures/subtype