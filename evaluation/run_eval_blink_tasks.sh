export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Get project root (LaViT_github directory)
# Script is in evaluation/, so go up one level
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration - modify these paths as needed
CHECKPOINT="path_to_checkpoint"
DATA_ROOT="path_to_data"
OUTPUT_DIR="path_to_output_file"
TASKS=("Relative_Reflectance" "Relative_Depth" "Spatial_Relation")

if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Evaluating BLINK Tasks"
echo "=========================================="
echo "Model: $CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Tasks: ${TASKS[@]}"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Running Task: $TASK"
    echo "--------------------------------------------------"
    OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_lavit_3b.jsonl"
    if python evaluation/utils/run_eval.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "blink" \
        --data_root "$DATA_ROOT" \
        --task_name "$TASK" \
        --output_file "$OUTPUT_FILE"; then
        if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
            echo "✓ Task $TASK completed"
            echo "Results:"
            python evaluation/utils/answer_checker.py \
                --results_file "$OUTPUT_FILE" \
                --show_examples 0 2>&1 | grep -E "(Total Samples|Correct|Wrong|Accuracy|Failed Extraction)"
        else
            echo "✗ Task $TASK failed: No output file generated"
        fi
    else
        echo "✗ Task $TASK failed"
    fi
done

echo ""
echo "=========================================="
echo "All Tasks Done."
echo "=========================================="