source /Group16T/common/ccuc/miniconda3/etc/profile.d/conda.sh
conda activate sctools

n_runs=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_path)
            dataset_path="$2"
            shift 2
            ;;
        --save_path)
            save_path="$2"
            shift 2
            ;;
        --run_times)
            n_runs="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -z "$dataset_path" ]; then
    echo "Error: --dataset_path is required."
    exit 1
fi

if [ -z "$save_path" ]; then
    echo "Error: --save_path is required."
    exit 1
fi

echo "Dataset: $dataset_path"
echo "Save path: $save_path"
echo "Run times: $n_runs"

echo "Running Celltypist"
python3 Celltypist/Celltypist.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs"
echo "Running Celltypist (CAF mode)"
python3 Celltypist/Celltypist.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs" --caf_mode

echo "Running CHETAH"
python3 CHETAH/CHETAH.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs"
echo "Running CHETAH (CAF mode)"
python3 CHETAH/CHETAH.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs" --caf_mode

echo "Running Seurat"
python3 Seurat/Seurat.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs"
echo "Running Seurat (CAF mode)"
python3 Seurat/Seurat.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs" --caf_mode

conda activate scGPT
echo "Running scGPT"
python3 scGPT/scGPT.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs"
echo "Running scGPT (CAF mode)"
python3 scGPT/scGPT.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs" --caf_mode

conda activate SingleR
echo "Running SingleR"
python3 SingleR/SingleR.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs"
echo "Running SingleR (CAF mode)"
python3 SingleR/SingleR.py \
    --data_path "$dataset_path" \
    --save_path "$save_path" \
    --run_time "$n_runs" --caf_mode
