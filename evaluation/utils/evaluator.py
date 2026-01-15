import torch
from transformers import Qwen2VLProcessor
from PIL import Image
import sys
import os
# Add training/src to path to import modeling_lavit
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
training_src_path = os.path.join(project_root, "training", "src")
sys.path.insert(0, training_src_path)
from modeling_lavit import LaViTQwen2VL, LaViTConfig
class LaViTEvaluator:
    def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None
        self._load_model()
    def _load_model(self):
        print(f"Loading model from {self.checkpoint_path}...")
        try:
            config = LaViTConfig.from_pretrained(self.checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load LaViTConfig from {self.checkpoint_path}, using default. Error: {e}")
            config = LaViTConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        if not os.path.exists(os.path.join(self.checkpoint_path, "preprocessor_config.json")):
             print("preprocessor_config.json not found in checkpoint. Loading from base model.")
             base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
             self.processor = Qwen2VLProcessor.from_pretrained(base_model_path)
             from transformers import AutoTokenizer
             try:
                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
                self.processor.tokenizer = tokenizer
             except Exception as e:
                 print(f"Warning: Could not load tokenizer from checkpoint: {e}. Using base tokenizer.")
        else:
            self.processor = Qwen2VLProcessor.from_pretrained(self.checkpoint_path)
        if "<lvr>" not in self.processor.tokenizer.get_vocab():
            print("Warning: <lvr> token not found in tokenizer vocabulary. Adding it.")
            self.processor.tokenizer.add_tokens(["<lvr>"], special_tokens=True)
        self.model = LaViTQwen2VL.from_pretrained(
            self.checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        print("Model loaded successfully.")
    def generate(self, image_path, question, max_new_tokens=512, force_lvr=False, mask_lvr=False):
        try:
            if isinstance(image_path, list):
                images = []
                for img in image_path:
                    if isinstance(img, str):
                        images.append(Image.open(img).convert("RGB"))
                    else:
                        images.append(img.convert("RGB"))
            elif isinstance(image_path, str):
                images = [Image.open(image_path).convert("RGB")]
            else:
                images = [image_path.convert("RGB")]
        except Exception as e:
            print(f"Error loading image(s) {image_path}: {e}")
            return None
        question = question.replace("<image>", "").strip()
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        if force_lvr:
            lvr_id = self.processor.tokenizer.convert_tokens_to_ids("<lvr>")
            if lvr_id is None:
                print("Error: <lvr> token not found in tokenizer. Cannot force lvr.")
            else:
                lvr_tokens = torch.tensor([[lvr_id] * 4], device=self.device, dtype=inputs.input_ids.dtype)
                inputs.input_ids = torch.cat([inputs.input_ids, lvr_tokens], dim=1)
                inputs.attention_mask = torch.cat([inputs.attention_mask, torch.ones((1, 4), device=self.device, dtype=inputs.attention_mask.dtype)], dim=1)
        if mask_lvr:
            lvr_id = self.processor.tokenizer.convert_tokens_to_ids("<lvr>")
            if lvr_id is not None:
                lvr_mask = (inputs.input_ids == lvr_id)
                inputs.attention_mask[lvr_mask] = 0
                if lvr_mask.any():
                    print(f"DEBUG: Masked {lvr_mask.sum().item()} <lvr> tokens in input.")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return output_text