import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
class AttentionHookExtractor:
    def __init__(self, model, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.hooks = []
        self.layer_outputs = {}
        self._text_token_mask = None
        self._image_token_mask = None
        self._current_device = None
        self.attn_module_name = self._detect_attn_module_name()
        if self.verbose:
            print(f"[INFO] Detected attention module name: {self.attn_module_name}")
    def _detect_attn_module_name(self) -> str:
        for name, module in self.model.named_modules():
            if "layers.0" in name and "self_attn" in name:
                return "self_attn"
        return "self_attn"
    def _hook_fn(self, module, args, output, layer_idx):
        attn_weights = None
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
        if attn_weights is None:
            if self.verbose:
                print(f"[WARNING] Layer {layer_idx}: No attention weights found in output.")
            return
        processed_data = self._process_attention_weights(attn_weights)
        self.layer_outputs[layer_idx] = processed_data.detach().cpu()
        del attn_weights
    def _process_attention_weights(self, attn_weights: torch.Tensor) -> torch.Tensor:
        device = attn_weights.device
        if self._image_token_mask.device != device:
            self._image_token_mask = self._image_token_mask.to(device)
        if self._text_token_mask.device != device:
            self._text_token_mask = self._text_token_mask.to(device)
        image_mask = self._image_token_mask
        text_mask = self._text_token_mask
        image_positions = torch.where(image_mask)[0]
        if len(image_positions) == 0:
            return torch.zeros(image_mask.sum(), device='cpu')
        last_image_pos = image_positions[-1].item()
        seq_len = attn_weights.shape[2]
        answer_start = last_image_pos + 1
        answer_end = seq_len
        answer_to_image = attn_weights[0, :, answer_start:answer_end, :][:, :, image_mask]
        if answer_to_image.shape[1] > 0:
            avg_attention = answer_to_image.mean(dim=(0, 1))
        else:
            last_token_attn = attn_weights[0, :, -1, :]
            avg_attention = last_token_attn[:, image_mask].mean(dim=0)
        return avg_attention
    def register_hooks(self):
        self.hooks = []
        self.layer_outputs = {}
        layer_count = 0
        for name, module in self.model.named_modules():
            if "visual" in name:
                continue
            if "attn" in name.lower() or "attention" in name.lower():
                parts = name.split('.')
                layer_idx = -1
                for i, part in enumerate(parts):
                    if part == "layers" or part == "blocks":
                        if i + 1 < len(parts) and parts[i+1].isdigit():
                            layer_idx = int(parts[i+1])
                            break
                if layer_idx >= 0:
                    if "Qwen2_5_VLAttention" in module.__class__.__name__ or \
                       "LlamaAttention" in module.__class__.__name__ or \
                       "SelfAttention" in module.__class__.__name__ or \
                       name.endswith(self.attn_module_name):
                        if self.verbose:
                            print(f"[DEBUG] Hooking layer {layer_idx}: {name} ({module.__class__.__name__})")
                        hook = module.register_forward_hook(
                            lambda m, inp, out, idx=layer_idx: self._hook_fn(m, inp, out, idx)
                        )
                        self.hooks.append(hook)
                        layer_count += 1
        if layer_count == 0:
            print("[WARNING] No attention modules hooked! Check model structure.")
            print(f"[DEBUG] Model modules: {[n for n, _ in self.model.named_modules()][:20]}...")
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    @torch.no_grad()
    def extract(self, inputs, image_token_mask, text_token_mask):
        self._image_token_mask = image_token_mask
        self._text_token_mask = text_token_mask
        self.register_hooks()
        try:
            self.model(**inputs, output_attentions=True, output_hidden_states=False)
        finally:
            self.remove_hooks()
        if not self.layer_outputs:
            print("[ERROR] No attention data captured. Ensure attn_implementation='eager'.")
            raise RuntimeError("No attention data captured via hooks.")
        stacked_avgs = torch.stack(list(self.layer_outputs.values()), dim=0) 
        final_s = stacked_avgs.mean(dim=0) 
        return final_s
    def _hook_fn_modify(self, module, args, output, layer_idx):
        if not isinstance(output, tuple):
            return output
        attn_output = output[0]
        attn_weights = output[1] if len(output) > 1 else None
        if attn_weights is not None:
            processed = self._process_attention_weights(attn_weights)
            self.layer_outputs[layer_idx] = processed.detach().cpu()
            new_output = (attn_output, None) + output[2:]
            return new_output
        elif self.verbose and layer_idx < 2:
            print(f"[DEBUG] Layer {layer_idx}: hook triggered but no attention weights captured (output_attentions not enabled?)")
        return output
    def register_modifying_hooks(self):
        self.hooks = []
        self.layer_outputs = {}
        layer_count = 0
        for name, module in self.model.named_modules():
            if "visual" in name:
                continue
            if "attn" in name.lower() or "attention" in name.lower():
                parts = name.split('.')
                layer_idx = -1
                for i, part in enumerate(parts):
                    if part == "layers" or part == "blocks":
                        if i + 1 < len(parts) and parts[i+1].isdigit():
                            layer_idx = int(parts[i+1])
                            break
                if layer_idx >= 0:
                    if name.endswith("self_attn") or name.endswith("attention"):
                        if self.verbose and layer_count < 2:
                            print(f"[DEBUG] Hooking (modifying) layer {layer_idx}: {name}")
                        hook = module.register_forward_hook(
                            lambda m, inp, out, idx=layer_idx: self._hook_fn_modify(m, inp, out, idx)
                        )
                        self.hooks.append(hook)
                        layer_count += 1
        if layer_count == 0:
            print("[WARNING] No attention modules hooked! Check model structure.")
    @torch.no_grad()
    def extract_optimized(self, inputs, image_token_mask, text_token_mask):
        self._image_token_mask = image_token_mask
        self._text_token_mask = text_token_mask
        self.register_modifying_hooks()
        try:
            self.model(**inputs, output_attentions=True, output_hidden_states=False)
        finally:
            self.remove_hooks()
        if not self.layer_outputs:
            if self.verbose:
                print("[WARNING] Hook 未捕获到任何 attention tensor，返回全零向量")
            return torch.zeros(image_token_mask.sum(), device='cpu')
        layer_indices = sorted(self.layer_outputs.keys())
        n_layers = len(layer_indices)
        n_use = min(8, n_layers)
        selected_indices = layer_indices[-n_use:]
        selected_tensors = [self.layer_outputs[idx] for idx in selected_indices]
        if self.verbose:
            print(f"[INFO] 使用最后 {n_use} 层 (layers {selected_indices[0]}-{selected_indices[-1]}) 的注意力")
        stacked_avgs = torch.stack(selected_tensors, dim=0)
        final_s = stacked_avgs.mean(dim=0)
        return final_s