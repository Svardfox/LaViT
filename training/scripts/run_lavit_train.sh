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
export DATA_JSON=${DATA_JSON:-"$DATASET_ROOT/data/metadata/lavit_15k_for_training.json"}
export IMAGE_ROOT=${IMAGE_ROOT:-"$DATASET_ROOT/data/images"}
export VTOP_DIR=${VTOP_DIR:-"$DATASET_ROOT/data/features/vtop"}
export ATTN_DIR=${ATTN_DIR:-"$DATASET_ROOT/data/features/trajectories"}
TRAINING_STAGE=0
BOTTLENECK_BLOCK_PROMPT=True
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/checkpoints/lavit"}
NUM_LAVIT_TOKENS=${NUM_LAVIT_TOKENS:-4}
MAX_STEPS=${MAX_STEPS:-1000}
MAX_SAMPLES=${MAX_SAMPLES:-14567}
python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --v_top_dir $VTOP_DIR \
    --attention_dir $ATTN_DIR \
    --output_dir $OUTPUT_DIR \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps $MAX_STEPS \
    --loss_scale_vtop 0.3 \
    --loss_scale_traj 0.3 \
    --training_stage $TRAINING_STAGE \
    --bottleneck_block_prompt $BOTTLENECK_BLOCK_PROMPT \
    --use_trajectory_supervision $USE_TRAJECTORY_SUPERVISION \
    --num_lavit_tokens $NUM_LAVIT_TOKENS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --run_name "lavit-single-stage-4lavit-1000steps" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples $MAX_SAMPLES \
    --do_train