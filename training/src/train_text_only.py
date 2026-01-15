import os
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Qwen2VLProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from transformers import Trainer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_only_dataset import TextOnlyDataset, TextOnlyCollator
logger = logging.getLogger(__name__)
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    freeze_vision: bool = field(default=True, metadata={"help": "Whether to freeze vision encoder"})
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze LLM"})
@dataclass
class DataArguments:
    data_json: str = field(metadata={"help": "Path to enriched data json"})
    image_root: str = field(metadata={"help": "Path to image root directory"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "For debugging, truncate dataset"})
    min_pixels: Optional[int] = field(default=256*28*28, metadata={"help": "Minimum number of pixels for image"})
    max_pixels: Optional[int] = field(default=1280*28*28, metadata={"help": "Maximum number of pixels for image"})
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info("=" * 60)
    logger.info("Text-Only SFT Training (Ablation Study)")
    logger.info("No <lvr> tokens, no v_top, no trajectory supervision")
    logger.info("=" * 60)
    set_seed(training_args.seed)
    try:
        processor = Qwen2VLProcessor.from_pretrained(
            model_args.model_name_or_path, 
            min_pixels=data_args.min_pixels, 
            max_pixels=data_args.max_pixels
        )
    except Exception as e:
        logger.warning(f"Failed to load processor from {model_args.model_name_or_path}: {e}")
        raise e
    dataset = TextOnlyDataset(
        data_json_path=data_args.data_json,
        image_root=data_args.image_root,
        processor=processor,
        max_samples=data_args.max_samples
    )
    logger.info(f"Loading standard Qwen2.5-VL model (no ViLR wrapper)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path
    )
    if model_args.freeze_vision:
        for param in model.model.visual.parameters():
            param.requires_grad = False
        logger.info("Froze Vision Encoder")
    if model_args.freeze_llm:
        logger.info("Freezing LLM (text model) parameters...")
        for name, param in model.model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
        logger.info("Froze LLM parameters")
    collator = TextOnlyCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer
    )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
if __name__ == "__main__":
    main()