export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
CHECKPOINT="/root/autodl-tmp/ViLR/training/checkpoints/debug_run/checkpoint-1250"
DATA_ROOT="/root/autodl-tmp/ViLR/data/MMVP"
OUTPUT_FILE="/root/autodl-tmp/ViLR/training/eval_results/mmvp_results.jsonl"
if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found at $CHECKPOINT"
    CHECKPOINT=$(ls -d /root/autodl-tmp/ViLR/training/checkpoints/debug_run/checkpoint* | tail -n 1)
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found. Please train first."
        exit 1
    fi
    echo "Using latest found: $CHECKPOINT"
fi
echo "Evaluating MMVP..."
echo "Model: $CHECKPOINT"
echo "Output: $OUTPUT_FILE"
python ../utils/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "mmvp" \
    --data_root "$DATA_ROOT" \
    --output_file "$OUTPUT_FILE" \
    --max_samples 10 \
    --force_lvr
echo "Done."