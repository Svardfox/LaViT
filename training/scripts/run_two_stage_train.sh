set -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
DATASET_ROOT=${DATASET_ROOT:-"$PROJECT_ROOT/../LaViT-15k"}
if [ ! -d "$DATASET_ROOT" ]; then
    echo "LaViT-15k dataset not found at: $DATASET_ROOT"
    echo "Set DATASET_ROOT to the LaViT-15k directory."
    exit 1
fi
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
export DATA_JSON=${DATA_JSON:-"$DATASET_ROOT/data/metadata/lavit_15k_for_training.json"}
export IMAGE_ROOT=${IMAGE_ROOT:-"$DATASET_ROOT/data/images"}
export VTOP_DIR=${VTOP_DIR:-"$DATASET_ROOT/data/features/vtop"}
export ATTN_DIR=${ATTN_DIR:-"$DATASET_ROOT/data/features/trajectories"}
STAGE1_OUTPUT=${STAGE1_OUTPUT:-"$PROJECT_ROOT/checkpoints/two_stage/stage1_bottleneck"}
STAGE2_OUTPUT=${STAGE2_OUTPUT:-"$PROJECT_ROOT/checkpoints/two_stage/stage2_joint"}
STAGE1_STEPS=400
STAGE2_STEPS=600
if [ ! -d "$MODEL_PATH" ]; then
    echo "Local model not found at $MODEL_PATH, using HF ID Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
fi
echo "============================================"
echo "Stage 1: Bottleneck Training (${STAGE1_STEPS} steps)"
echo "============================================"
python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --v_top_dir $VTOP_DIR \
    --attention_dir $ATTN_DIR \
    --output_dir $STAGE1_OUTPUT \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps $STAGE1_STEPS \
    --loss_scale_vtop 0.3 \
    --loss_scale_traj 0.3 \
    --training_stage 1 \
    --bottleneck_block_prompt True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.06 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --run_name "vilr-two-stage-s1-bottleneck" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples 8765 \
    --do_train
if [ $? -ne 0 ]; then
    echo "Stage 1 failed! Exiting..."
    exit 1
fi
echo ""
echo "============================================"
echo "Stage 2: Joint Training (${STAGE2_STEPS} steps)"
echo "Loading checkpoint from Stage 1..."
echo "============================================"
LATEST_CKPT=$(ls -td ${STAGE1_OUTPUT}/checkpoint-* 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found in Stage 1 output! Using final model..."
    LATEST_CKPT=$STAGE1_OUTPUT
fi
echo "Resuming from: $LATEST_CKPT"
python src/train.py \
    --model_name_or_path $LATEST_CKPT \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --v_top_dir $VTOP_DIR \
    --attention_dir $ATTN_DIR \
    --output_dir $STAGE2_OUTPUT \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps $STAGE2_STEPS \
    --loss_scale_vtop 0.3 \
    --loss_scale_traj 0.3 \
    --training_stage 2 \
    --bottleneck_block_prompt True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --run_name "vilr-two-stage-s2-joint" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples 8765 \
    --do_train
echo ""
echo "============================================"
echo "Two-Stage Training Complete!"
echo "Stage 1 checkpoint: $STAGE1_OUTPUT"
echo "Stage 2 (final) checkpoint: $STAGE2_OUTPUT"
echo "============================================"