# LaViT: Latent Visual Thought

This repository contains the official implementation of LaViT.

## Repository Structure

- `data_construction/`: Scripts for extracting Trajectories and V_top features from Visual-CoT and other datasets.
- `training/`: Core training code for LaViT, including dataset implementation and model definition.
- `evaluation/`: Evaluation scripts for MMVP, Blink, and other benchmarks.
- `src/`: Common utilities shared across components.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## Usage

### Building the LaViT Dataset

LaViT training uses **image–question–answer** triplets together with **V\_top features** and **trajectory attention distributions**. The high-level pipeline is:

- **Step 1: Build Visual-CoT trajectories from raw multimodal data**
  - Use the scripts under `data_construction/viscot/` (e.g. `extract_viscot_trajectories.py`, `extract_viscot_vtop.py`).


- **Step 2: Extract attention trajectories with a teacher model**
  - Use `data_construction/extraction/teacher_traj_extractor_attention.py`, which:
    - Calls a strong vision-language backbone (e.g. Qwen2.5-VL);
    - Uses an LVR/LaViT-style prompt to run step-wise reasoning;
    - Saves per-step attention over image patches to JSON (with fields like `p_t`) for trajectory supervision.

- **Step 3: Merge everything into an enriched JSON for training**
  - Use `training/src/preprocess_enrich_json.py` to combine raw trajectories, V\_top tensors and attention JSON into a single enriched JSON:
    - For each sample, it writes:
      - `image_relative_path`
      - `ground_truth_enriched`
      - `v_top_path_abs`
      - `attention_path_abs`
    - The script produces a file like `trajectories_vtop_enriched_all_final.json`.

- **Step 4: Check paths and configs**
  - Make sure the following paths match your environment (edit them in the scripts if needed):
    - `DATA_JSON`: path to the enriched JSON
    - `IMAGE_ROOT`: image root directory
    - `VTOP_DIR`: directory with V\_top tensors
    - `ATTN_DIR`: directory with attention JSON/tensors

> Note: the data construction code is research-oriented and may require manual adjustment. Please read the scripts and adapt paths to your setup.

### One-Stage LaViT Training (Original)

The one-stage setting uses `training_stage=0` and jointly optimizes **CE loss + V\_top loss + trajectory loss**. This is the simplest configuration to reproduce.

- **Script**: `training/scripts/run_lavit_train.sh`
- **Key environment variables (edit at the top of the script)**:
  - `DATA_JSON`: enriched JSON
  - `IMAGE_ROOT`: image directory
  - `VTOP_DIR`, `ATTN_DIR`: directories for V\_top and attention features
  - `MODEL_PATH`: pretrained backbone (local path or HF Hub ID)
- **Default config**:
  - `TRAINING_STAGE=0`
  - `NUM_LAVIT_TOKENS` (actually the number of `<lvr*>` special tokens) = 4
  - `loss_scale_vtop=0.3`, `loss_scale_traj=0.3`
  - `max_steps=1000`

Example:

```bash
cd training
bash scripts/run_lavit_train.sh
```

You can increase training length or change loss weights by editing `max_steps`, `loss_scale_vtop`, `loss_scale_traj`, etc. in the script.

### Two-Stage LaViT Training (Bottleneck + Joint)

The two-stage setting switches `training_stage` between a **hard bottleneck phase (vision-only supervision)** and a **joint phase (vision + text)**:

- **Script**: `training/scripts/run_two_stage_train.sh`
- **Stage 1: Bottleneck**
  - `training_stage=1`
  - CE loss disabled; only V\_top + trajectory losses are active;
  - A hard attention bottleneck ensures answer tokens cannot directly attend to image tokens and must go through `<lvr>` tokens.
- **Stage 2: Joint**
  - `training_stage=2`
  - Restore standard attention; jointly optimize text + vision losses;
  - Initialize from the Stage 1 checkpoint.

Important script variables:

- Data paths (`DATA_JSON`, `IMAGE_ROOT`, `VTOP_DIR`, `ATTN_DIR`) are similar to one-stage training.
- `STAGE1_OUTPUT`, `STAGE2_OUTPUT`: output dirs for the two stages.
- `STAGE1_STEPS`, `STAGE2_STEPS`: training steps per stage.

Example:

```bash
cd training
bash scripts/run_two_stage_train.sh
```

After training you will obtain:

- Stage 1 checkpoint: `STAGE1_OUTPUT`
- Stage 2 (final) checkpoint: `STAGE2_OUTPUT` (recommended for downstream evaluation)

### Naive Text-Only SFT (Baseline)

For ablations, this repo provides a **pure text SFT baseline** that does **not** use `<lvr>` tokens, V\_top supervision, or trajectory supervision.

- **Script**: `training/scripts/run_text_only_sft.sh`
- **Characteristics**:
  - No `<lvr*>` tokens are inserted;
  - No V\_top / attention features are loaded;
  - Only standard language modeling (CE) loss is used.
- **Key environment variables**:
  - `MODEL_PATH`: pretrained Qwen2.5-VL base
  - `DATA_JSON`: you can reuse the same enriched JSON (the script only uses text and image paths)
  - `IMAGE_ROOT`, `OUTPUT_DIR`: image root and output directory.

Example:

```bash
cd training
bash scripts/run_text_only_sft.sh
```

Internally this script calls `training/src/train_text_only.py`, so you can directly compare LaViT vs. text-only SFT under the same data.

### Evaluation

Evaluation code lives in `evaluation/utils/` and has two entry points:

- Base model: `run_eval_base.py` (evaluates the original Qwen2.5-VL model)
- LaViT model: `run_eval.py` (evaluates LaViT checkpoints with `<lvr*>` tokens)

#### Evaluating LaViT

Entry: `evaluation/utils/run_eval.py`. Key arguments:

- `--checkpoint`: LaViT checkpoint path (e.g. Stage 2 output of two-stage training)
- `--dataset`: dataset name, e.g.:
  - `mmvp` (MMVP benchmark)
  - `blink` (BLINK; requires `--task_name` such as `Counting`, `Jigsaw`)
  - `vsp`, `mmstar`, `vstar`, etc.
- `--data_root`: dataset root directory or parquet/jsonl file path
- `--output_file`: JSONL path for saving raw model outputs
- Optional:
  - `--task_name`: sub-task name for multi-task datasets like BLINK
  - `--max_samples`: limit the number of evaluated samples
  - `--force_lvr`: append `<lvr>` tokens to the prompt, to test whether the model uses the bottleneck tokens
  - `--mask_lvr`: mask `<lvr>` tokens in the attention mask, to test how much the model relies on them

Example (MMVP):

```bash
cd evaluation/utils
python run_eval.py \
  --checkpoint /path/to/lavit_checkpoint \
  --dataset mmvp \
  --data_root /path/to/MMVP \
  --output_file /path/to/save/mmvp_lavit_results.jsonl \
  --max_samples 100
```

#### Evaluating the Base Qwen2.5-VL Model

Entry: `evaluation/utils/run_eval_base.py`, which evaluates the original backbone with the same adapter interface.

Key arguments:

- `--model_path`: base model path or HF ID (e.g. `Qwen/Qwen2.5-VL-3B-Instruct`)
- `--dataset`, `--data_root`, `--output_file`, `--task_name`, `--max_samples`: same semantics as in `run_eval.py`.

Example:

```bash
cd training/src
python evaluation/run_eval_base.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset mmvp \
  --data_root /path/to/MMVP \
  --output_file /path/to/save/mmvp_base_qwen_results.jsonl
```

Both evaluation scripts write JSONL files with:

- `id`, `prompt`, `ground_truth`, `model_output`
- `lvr_count`: number of `<lvr>` / `<lvr1>` / `<lvr2>` … tokens in the output (always 0 for the base model)

You can then compute accuracy and analyze `<lvr>` usage based on these logs.

## License
[License Info]
