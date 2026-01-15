import torch
from transformers import Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
class BaseQwenEvaluator:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    def _load_model(self):
        print(f"Loading base Qwen2.5-VL model from {self.model_path}...")
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        print("Model loaded successfully.")
    def generate(self, image_path, question, max_new_tokens=512, force_lvr=False):
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