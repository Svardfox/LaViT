export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
CHECKPOINT="$HOME/autodl-tmp/my_qwen_model/Qwen2.5-VL-3B-Instruct"
DATA_ROOT="/root/autodl-tmp/ViLR/data/BLINK"
OUTPUT_DIR="/root/autodl-tmp/ViLR/training/eval_results/blink"
TASKS=("IQ_Test" "Relative_Reflectance" "Spatial_Relation")
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
    OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_qwen25_3b.jsonl"
    python ../utils/run_eval.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "blink" \
        --data_root "$DATA_ROOT" \
        --task_name "$TASK" \
        --output_file "$OUTPUT_FILE"
    if [ $? -eq 0 ]; then
        echo "✓ Task $TASK completed"
        python ../utils/answer_checker.py \
            --results_file "$OUTPUT_FILE" \
            --show_examples 0 2>&1 | grep -E "(总样本数|正确数|准确率)"
    else
        echo "✗ Task $TASK failed"
    fi
done
echo ""
echo "=========================================="
echo "All Tasks Done."
echo "=========================================="