from __future__ import annotations
import gc
import os
from typing import Dict, Optional
import torch
class VTopLayerExtractor:
    def __init__(self, model, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.use_multi_gpu = hasattr(model, "hf_device_map") and model.hf_device_map is not None
        self._last_layer_hidden_states: Optional[torch.Tensor] = None
        self._hook = None
        self._last_layer_name: Optional[str] = None
    def _find_last_transformer_layer(self):
        if hasattr(self.model, "language_model"):
            lang_model = self.model.language_model
            if hasattr(lang_model, "model") and hasattr(lang_model.model, "layers"):
                layers = lang_model.model.layers
                if len(layers) > 0:
                    last_layer = layers[-1]
                    layer_name = "language_model.model.layers[-1]"
                    if self.verbose:
                        print(f"[VTopLayerExtractor] 最后一层: {layer_name}, 总层数: {len(layers)}")
                    return last_layer, layer_name
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            if len(layers) > 0:
                last_layer = layers[-1]
                layer_name = "model.model.layers[-1]"
                if self.verbose:
                    print(f"[VTopLayerExtractor] 最后一层: {layer_name}, 总层数: {len(layers)}")
                return last_layer, layer_name
        candidate_layers = []
        for name, module in self.model.named_modules():
            if "layers" in name.lower() and hasattr(module, "__len__"):
                try:
                    if len(module) > 0:
                        if hasattr(module[-1], "self_attn") or hasattr(module[-1], "mlp"):
                            candidate_layers.append((name, module))
                except Exception:
                    continue
        if candidate_layers:
            candidate_layers.sort(key=lambda x: len(x[0]), reverse=True)
            name, module = candidate_layers[0]
            last_layer = module[-1]
            layer_name = f"{name}[-1]"
            if self.verbose:
                print(f"[VTopLayerExtractor] 最后一层(候补): {layer_name}, 总层数: {len(module)}")
            return last_layer, layer_name
        if self.verbose:
            print("[VTopLayerExtractor] 未找到最后一层 Transformer，模型结构示例：")
            for i, (name, module) in enumerate(list(self.model.named_modules())[:20]):
                print(f"  {i}: {name} - {type(module).__name__}")
        raise RuntimeError("无法定位模型的最后一层 Transformer。")
    def _hook_last_layer(self, module, _, output):
        hidden_states = None
        if isinstance(output, tuple):
            for item in output:
                if torch.is_tensor(item) and item.dim() == 3:
                    hidden_states = item
                    break
        elif torch.is_tensor(output):
            hidden_states = output
        elif hasattr(output, "last_hidden_state"):
            hidden_states = output.last_hidden_state
        if hidden_states is not None and torch.is_tensor(hidden_states) and hidden_states.dim() == 3:
            self._last_layer_hidden_states = hidden_states.detach()
            if self.verbose:
                print(
                    f"[VTopLayerExtractor] 捕获最后一层输出: shape={tuple(hidden_states.shape)}, "
                    f"dtype={hidden_states.dtype}, device={hidden_states.device}"
                )
        elif self.verbose:
            print(f"[VTopLayerExtractor] 警告: 未能从输出中提取 hidden_states, output type={type(output)}")
    def _empty_cache_all_gpus(self):
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                for idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(idx):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
    @torch.no_grad()
    def capture(self, inputs: Dict[str, torch.Tensor], image_token_mask: torch.Tensor) -> torch.Tensor:
        if image_token_mask is None:
            raise ValueError("image_token_mask 不能为 None")
        self._last_layer_hidden_states = None
        last_layer, layer_name = self._find_last_transformer_layer()
        self._last_layer_name = layer_name
        if self.verbose:
            print(f"[VTopLayerExtractor] 在 {layer_name} 注册 hook")
        self._hook = last_layer.register_forward_hook(self._hook_last_layer)
        outputs = None
        try:
            outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False)
        finally:
            if self._hook is not None:
                self._hook.remove()
                self._hook = None
        if self._last_layer_hidden_states is None:
            raise RuntimeError("未捕获到最后一层输出，请检查 hook 注册是否成功。")
        hidden_states = self._last_layer_hidden_states
        if hidden_states.dim() != 3:
            raise RuntimeError(f"最后一层隐藏状态形状异常: {tuple(hidden_states.shape)}")
        if image_token_mask.device != hidden_states.device:
            if self.verbose:
                print(
                    f"[VTopLayerExtractor] 将 image_token_mask 从 {image_token_mask.device} "
                    f"搬到 {hidden_states.device}"
                )
            image_token_mask = image_token_mask.to(hidden_states.device)
        batch, seq_len, _ = hidden_states.shape
        if image_token_mask.shape[0] != seq_len:
            raise ValueError(
                f"image_token_mask 长度 ({image_token_mask.shape[0]}) 与序列长度 ({seq_len}) 不匹配"
            )
        if batch != 1:
            print("[WARNING] Batch size 不为 1，仅使用第一个样本的特征。")
        image_positions = image_token_mask.bool()
        v_top_layer = hidden_states[0][image_positions]
        if self.verbose:
            print(
                f"[VTopLayerExtractor] 提取 V_top_layer: shape={tuple(v_top_layer.shape)}, "
                f"N_img={v_top_layer.shape[0]}, d={v_top_layer.shape[1]}"
            )
        v_top_layer_cpu = v_top_layer.cpu()
        del hidden_states, v_top_layer, self._last_layer_hidden_states
        self._last_layer_hidden_states = None
        if outputs is not None:
            del outputs
        self._empty_cache_all_gpus()
        gc.collect()
        return v_top_layer_cpu
    def save(self, v_top_layer: torch.Tensor, save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.endswith(".npy"):
            import numpy as np
            np.save(save_path, v_top_layer.cpu().numpy())
        elif save_path.endswith((".pt", ".pth")):
            torch.save(v_top_layer.cpu(), save_path)
        else:
            import numpy as np
            actual_path = save_path + ".npy"
            np.save(actual_path, v_top_layer.cpu().numpy())
            save_path = actual_path
        if self.verbose:
            print(f"[VTopLayerExtractor] 已保存 V_top_layer 至 {save_path}")
        return save_path
    def capture_and_save(
        self,
        inputs: Dict[str, torch.Tensor],
        image_token_mask: torch.Tensor,
        save_path: str,
    ) -> str:
        features = self.capture(inputs, image_token_mask)
        return self.save(features, save_path)