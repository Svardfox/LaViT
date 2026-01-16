export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration - modify these paths as needed
CHECKPOINT="path_to_checkpoint"
DATA_ROOT="path_to_data"
OUTPUT_FILE="path_to_output_file"

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found at $CHECKPOINT"
    CHECKPOINT=$(ls -d path_to_checkpoints/* 2>/dev/null | tail -n 1)
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found. Please train first."
        exit 1
    fi
    echo "Using latest found: $CHECKPOINT"
fi

echo "Evaluating MMVP..."
echo "Model: $CHECKPOINT"
echo "Output: $OUTPUT_FILE"
python evaluation/utils/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "mmvp" \
    --data_root "$DATA_ROOT" \
    --output_file "$OUTPUT_FILE" \
    --max_samples 10 \
    --force_lvr
echo "Done."