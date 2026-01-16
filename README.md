# LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2601.10129-b31b1b.svg)](https://arxiv.org/abs/2601.10129)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow.svg)](https://huggingface.co/Svard/LaViT-3B)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

This repository contains the official implementation of **LaViT**, a method for training vision-language models through visual thought trajectory supervision. LaViT extracts visual reasoning processes from large teacher models and uses them as supervision signals to train smaller, more efficient student models.


## 🚀 Quick Start

### 1. Environment Setup

```bash
cd LaViT
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 2. Dependencies

Main dependencies include:
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face model library
- `accelerate` - Distributed training acceleration
- `Pillow` - Image processing
- `pandas`, `numpy` - Data processing
- `wandb` - Experiment tracking (optional)

## 📊 Data Construction

LaViT training requires three types of data:

1. **Image-Question-Answer triplets**: Basic multimodal data
2. **V_top features**: Top-layer visual representations from teacher model
3. **Attention trajectory distributions**: Attention distributions during stepwise reasoning

### Preparing Visual-CoT Dataset

Before extracting trajectories, you need to download and prepare the **Visual-CoT dataset** from Hugging Face:

- **Dataset**: [Visual-CoT](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- **Download**: Use the Hugging Face datasets library or download directly from the repository
- **Required files**:
  - `viscot_363k.json` or `viscot_mixed_2m.json`: Main data file containing image-question-answer triplets
  - Image files: All images referenced in the JSON file

**Dataset structure**: The Visual-CoT dataset contains samples with the following format:
```json
{
  "image": ["relative/path/to/image.jpg"],
  "conversations": [
    {"from": "human", "value": "Question text"},
    {"from": "gpt", "value": "<answer>Answer text</answer>"}
  ],
  "question_id": "unique_id",
  "dataset": "dataset_name",
  "split": "train"
}
```

### Extracting Visual-CoT Trajectories

Use `data_construction/scripts/extract_viscot_trajectories.py` to extract trajectories from the Visual-CoT dataset:

```bash
cd LaViT/data_construction/scripts
python extract_viscot_trajectories.py \
    --data_file /path/to/viscot_363k.json \
    --base_image_dir /path/to/viscot/images \
    --output_dir /path/to/output \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --method attention  # or gradient
```


### Extracting V_top Features

Use `data_construction/extraction/v_top_layer_extractor.py` to extract top-layer visual features.

### Building Training Data

Merge extracted features into the JSON format required for training, containing:
- `image_relative_path`: Relative path to image
- `ground_truth_enriched`: Enriched answer text
- `v_top_path_abs`: Absolute path to V_top features
- `attention_path_abs`: Absolute path to attention trajectories

## 🎓 Training

### Single-Stage Training (Recommended to Start)

Single-stage training jointly optimizes CE loss, V_top loss, and trajectory loss:

```bash
cd LaViT/training
bash scripts/run_lavit_train.sh
```

### Two-Stage Training

Two-stage training consists of a bottleneck stage and a joint stage:

**Stage 1: Bottleneck Training**
- Optimize only visual losses (V_top + trajectory)
- Force answer tokens to access visual information through `<lvr>` tokens
- `training_stage=1`

**Stage 2: Joint Training**
- Restore standard attention mechanism
- Jointly optimize text and visual losses
- `training_stage=2`

```bash
cd LaViT/training
bash scripts/run_two_stage_train.sh
```


## 📈 Evaluation

### Evaluating LaViT Models

Use `evaluation/utils/run_eval.py` to evaluate trained models:

```bash
cd LaViT/evaluation/utils
python run_eval.py \
    --checkpoint /path/to/lavit_checkpoint \
    --dataset mmvp \
    --data_root /path/to/MMVP \
    --output_file /path/to/results.jsonl
```

Supported datasets:
- `mmvp`: MMVP benchmark
- `blink`: BLINK benchmark (requires `--task_name` parameter, e.g., `IQ_Test`, `Relative_Reflectance`, `Spatial_Relation`)
- `vsp`, `mmstar`, `vstar`, etc.

### Evaluating Base Models

Evaluate the original Qwen2.5-VL model as a baseline:

```bash
cd LaViT/evaluation/utils
python run_eval_base.py \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset mmvp \
    --data_root /path/to/MMVP \
    --output_file /path/to/base_results.jsonl
```

### Batch Evaluation Examples

**Evaluating BLINK Tasks**:

```bash
cd LaViT/evaluation/blink
bash run_eval_blink_tasks.sh
```

**Evaluating MMVP**:

```bash
cd LaViT/evaluation/mmvp
bash run_eval_mmvp.sh
```

### Evaluation Options

- `--force_lvr`: Force append `<lvr>` tokens to the prompt (test if model uses bottleneck tokens)
- `--mask_lvr`: Mask `<lvr>` tokens in attention mask (test model's reliance on them)
- `--max_samples`: Limit number of evaluation samples (for debugging)


## 📄 License

This project is licensed under the Apache-2.0 License. 

## 🙏 Acknowledgments

This project is implemented based on the Qwen2.5-VL model. Thanks to the support of related open-source projects:

- [LVR (Latent Visual Reasoning)](https://github.com/VincentLeebang/lvr) - Official codebase for the paper "Latent Visual Reasoning"
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL) - Multimodal large language model series developed by Qwen team, Alibaba Cloud

## 📋 TODO

- [x] Release LaViT-3B model weights
- [ ] Upload complete training dataset LaViT-15k

## 📖 Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@misc{wu2026lavitaligninglatentvisual,
      title={LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning}, 
      author={Linquan Wu and Tianxiang Jiang and Yifei Dong and Haoyu Yang and Fengji Zhang and Shichaang Meng and Ai Xuan and Linqi Song and Jacky Keung},
      year={2026},
      eprint={2601.10129},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.10129}, 
}
```

## 🔗 Related Links

- **Paper**: [arXiv:2601.10129](https://arxiv.org/abs/2601.10129)
- **HuggingFace Model**: [Svard/LaViT-3B](https://huggingface.co/Svard/LaViT-3B)
- **Base Model**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
