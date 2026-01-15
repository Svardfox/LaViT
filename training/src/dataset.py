import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import logging
import math
logger = logging.getLogger(__name__)
class LaViTDataset(Dataset):
    def __init__(
        self, 
        data_json_path, 
        image_root, 
        v_top_dir,
        attention_dir,
        processor,
        max_samples=None,
        lavit_token="<lavit>",
        num_lavit_tokens=4
    ):
        self.image_root = image_root
        self.v_top_dir = v_top_dir
        self.attention_dir = attention_dir
        self.processor = processor
        self.num_lavit_tokens = num_lavit_tokens
        self.dataset_root = None
        if self.image_root:
            abs_image_root = os.path.abspath(self.image_root)
            marker = os.path.join(os.sep, "data", "images")
            idx = abs_image_root.rfind(marker)
            if idx != -1:
                self.dataset_root = abs_image_root[:idx]
        self.lavit_tokens = [f"<lvr{i+1}>" for i in range(num_lavit_tokens)]
        logger.info(f"Loading dataset from {data_json_path}")
        with open(data_json_path, 'r') as f:
            full_data = json.load(f)
            self.data = full_data['results']
        if max_samples:
            self.data = self.data[:max_samples]
        logger.info(f"Loaded {len(self.data)} samples")
        self.lavit_token_ids = []
        for token in self.lavit_tokens:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.processor.tokenizer.unk_token_id:
                logger.warning(f"LaViT token {token} not found in tokenizer! Ensure it's added.")
            self.lavit_token_ids.append(token_id)
    def _find_best_grid(self, N, original_w, original_h):
        target_ratio = original_h / original_w
        best_h, best_w = 1, N
        best_diff = float('inf')
        for h in range(1, int(N**0.5) + 2):
            if N % h == 0:
                w = N // h
                ratio1 = h / w
                diff1 = abs(ratio1 - target_ratio)
                if diff1 < best_diff:
                    best_diff = diff1
                    best_h, best_w = h, w
                ratio2 = w / h
                diff2 = abs(ratio2 - target_ratio)
                if diff2 < best_diff:
                    best_diff = diff2
                    best_h, best_w = w, h
        return best_h, best_w
    def _resize_feature(self, tensor, current_N, target_h, target_w, original_w, original_h):
        if current_N == target_h * target_w:
            return tensor
        h, w = self._find_best_grid(current_N, original_w, original_h)
        is_1d = (tensor.dim() == 1)
        if is_1d:
            x = tensor.view(1, 1, h, w)
        else:
            D = tensor.shape[1]
            x = tensor.view(h, w, D).permute(2, 0, 1).unsqueeze(0)
        x_new = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if is_1d:
            return x_new.flatten()
        else:
            return x_new.squeeze(0).permute(1, 2, 0).reshape(-1, x_new.shape[1])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        ground_truth = item.get('ground_truth_enriched', '') 
        image_path = os.path.join(self.image_root, item['image_relative_path'])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise e
        lavit_sequence = "".join(self.lavit_tokens)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": lavit_sequence + " " + ground_truth 
            }
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor (
            text=[text_input],
            images=[image],
            padding=False, 
            return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        pixel_values = inputs.pixel_values 
        image_grid_thw = inputs.image_grid_thw[0]
        labels = input_ids.clone()
        lavit_indices = []
        for token_id in self.lavit_token_ids:
            indices = (input_ids == token_id).nonzero(as_tuple=True)[0]
            lavit_indices.extend(indices.tolist())
        lavit_indices = torch.tensor(sorted(lavit_indices), dtype=torch.long)
        if len(lavit_indices) != self.num_lavit_tokens:
            if len(lavit_indices) < self.num_lavit_tokens:
               raise ValueError(f"Numbered LaViT token count mismatch! Expected {self.num_lavit_tokens}, found {len(lavit_indices)}")
        lavit_idx = lavit_indices[-1]
        lavit_start_idx = lavit_indices[0]
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_start_indices = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
        vision_end_indices = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
        vision_start_idx = vision_start_indices[0] if len(vision_start_indices) > 0 else torch.tensor(0)
        vision_end_idx = vision_end_indices[0] if len(vision_end_indices) > 0 else torch.tensor(0) 
        first_lavit_idx = lavit_indices[0]
        labels[:first_lavit_idx] = -100
        target_h, target_w = image_grid_thw[1].item(), image_grid_thw[2].item()
        orig_w, orig_h = image.size
        v_top_path = item.get('v_top_path_abs')
        if v_top_path and not os.path.isabs(v_top_path):
            if self.dataset_root and v_top_path.startswith("data/"):
                v_top_path = os.path.join(self.dataset_root, v_top_path)
            elif self.v_top_dir:
                v_top_path = os.path.join(self.v_top_dir, v_top_path)
        if not v_top_path or not os.path.exists(v_top_path):
             logger.error(f"V_top path invalid: {v_top_path}")
             v_top_tensor = torch.zeros(target_h * target_w, 5120)
        else:
             v_top_tensor = torch.load(v_top_path, map_location='cpu')
        v_top_tensor = self._resize_feature(
            v_top_tensor, v_top_tensor.shape[0], target_h, target_w, orig_w, orig_h
        )
        attn_path = item.get('attention_path_abs')
        if attn_path and not os.path.isabs(attn_path):
            if self.dataset_root and attn_path.startswith("data/"):
                attn_path = os.path.join(self.dataset_root, attn_path)
            elif self.attention_dir:
                attn_path = os.path.join(self.attention_dir, attn_path)
        if not attn_path or not os.path.exists(attn_path):
            logger.error(f"Attention path invalid: {attn_path}")
            traj_tensor = torch.zeros(target_h * target_w)
        else:
            with open(attn_path, 'r') as f:
                attn_data = json.load(f)
                p_t_list = attn_data['steps'][0]['p_t']
                traj_tensor = torch.tensor(p_t_list, dtype=torch.float32)
        traj_tensor = self._resize_feature(
            traj_tensor, traj_tensor.shape[0], target_h, target_w, orig_w, orig_h
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "v_top_tensor": v_top_tensor,
            "traj_tensor": traj_tensor,
            "lavit_token_idx": lavit_idx,
            "vision_start_idx": vision_start_idx,
            "vision_end_idx": vision_end_idx,
            "lavit_start_idx": lavit_start_idx,
        }
class LaViTCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = [item['attention_mask'] for item in batch]
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
        image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch], dim=0)
        v_tops = [item['v_top_tensor'] for item in batch]
        trajs = [item['traj_tensor'] for item in batch]
        v_tops_padded = torch.nn.utils.rnn.pad_sequence(
            v_tops, batch_first=True, padding_value=0
        )
        trajs_padded = torch.nn.utils.rnn.pad_sequence(
            trajs, batch_first=True, padding_value=0
        )
        patch_lens = torch.tensor([v.shape[0] for v in v_tops], dtype=torch.long)
        lavit_indices = torch.tensor([item['lavit_token_idx'] for item in batch], dtype=torch.long)
        vision_start_indices = torch.tensor([item['vision_start_idx'] for item in batch], dtype=torch.long)
        vision_end_indices = torch.tensor([item['vision_end_idx'] for item in batch], dtype=torch.long)
        lavit_start_indices = torch.tensor([item['lavit_start_idx'] for item in batch], dtype=torch.long)
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "v_top_tensor": v_tops_padded,
            "traj_tensor": trajs_padded,
            "patch_lens": patch_lens,
            "lavit_token_idx": lavit_indices,
            "vision_start_idx": vision_start_indices,
            "vision_end_idx": vision_end_indices,
            "lavit_start_idx": lavit_start_indices,
        }