from __future__ import annotations
import os, json, re, math, gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor,AutoModelForImageTextToText
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import build_prompt_with_chat_template, build_teacher_generation_prompt
def probe_image_token_module(model, processor, image, question, T=4, max_new_tokens=16):
    import torch
    import re
    model.eval()
    prompt = f"Image:\n<image>\n\nQuestion:\n{question}\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(next(model.parameters()).device)
    candidates = []
    hooks = []
    def is_img_token_tensor(t: torch.Tensor):
        if not torch.is_tensor(t) or t.dim() != 3 or not torch.is_floating_point(t):
            return False
        B, N, D = t.shape
        if B != 1: return False
        if not (16 <= N <= 8192): return False
        if not (128 <= D <= 8192): return False
        return True
    def make_hook(name):
        def _hook(module, inp, out):
            def try_tensor(x):
                if torch.is_tensor(x) and is_img_token_tensor(x):
                    candidates.append((name, tuple(x.shape)))
            if torch.is_tensor(out):
                try_tensor(out)
            elif isinstance(out, (tuple, list)):
                for item in out:
                    try_tensor(item)
        return _hook
    for name, m in model.named_modules():
        hooks.append(m.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        _ = model(**{k: v.to(next(model.parameters()).device) if torch.is_tensor(v) else v for k, v in inputs.items()})
    for h in hooks:
        h.remove()
    seen = set()
    uniq = []
    for name, shape in candidates:
        if (name, shape) not in seen:
            seen.add((name, shape))
            uniq.append((name, shape))
    uniq.sort(key=lambda x: (len(x[0]), x[1][1]), reverse=True)
    print("\n[PROBE] Candidate image-token modules (pick the top one as target):")
    for i, (n, shp) in enumerate(uniq[:20], 1):
        print(f"  {i:2d}. {n:60s}  shape={shp}")
    return uniq
STEP_PATTERN = re.compile(r"<step>\s*(.*?)\s*</step>", re.DOTALL | re.IGNORECASE)
def split_steps(text: str, T: int) -> List[str]:
    matches = STEP_PATTERN.findall(text)
    if not matches:
        raise ValueError("No <step>...</step> format blocks found in teacher output.")
    steps = []
    for content in matches:
        steps.append(f"<step>{content}</step>")
    if len(steps) < T:
        raise ValueError(f"Only found {len(steps)} steps, but need {T} steps.")
    if len(steps) > T:
        print(f"[WARNING] Found {len(steps)} steps, but only need {T}. Returning first {T} steps.")
        steps = steps[:T]
    return steps
def to_device(x, device):
    if isinstance(x, dict):
        return {k: (to_device(v, device)) for k, v in x.items()}
    return x.to(device) if hasattr(x, "to") else x
class ConcatAttentionBackend:
    def __init__(self, model, tokenizer, image_token_mask_getter=None):
        self.model = model
        self.tokenizer = tokenizer
        self.image_token_mask_getter = image_token_mask_getter
        self._attn_buffers = []
    def _hook_attn(self, module, input, output):
        if isinstance(output, tuple):
            for item in output:
                if torch.is_tensor(item) and item.dim() >= 3:
                    self._attn_buffers.append(item.detach())
                    break
        elif hasattr(output, "attn_probs"):
            self._attn_buffers.append(output.attn_probs.detach())
    def _register_hooks(self):
        self._hooks = []
        for name, m in self.model.named_modules():
            if any(k in name.lower() for k in ["attn", "attention"]) and hasattr(m, "forward"):
                self._hooks.append(m.register_forward_hook(self._hook_attn))
    def _remove_hooks(self):
        for h in getattr(self, "_hooks", []):
            h.remove()
        self._hooks = []
    @torch.no_grad()
    def compute_pt(
        self,
        inputs: Dict[str, torch.Tensor],
        image_token_mask: torch.Tensor,
        target_pos: int,
        tau: float = 0.15,
        agg: str = "last2"
    ) -> Optional[torch.Tensor]:
        self._attn_buffers.clear()
        self._register_hooks()
        try:
            outputs = self.model(**inputs, output_attentions=True)
        finally:
            self._remove_hooks()
        if not self._attn_buffers:
            return None
        attns = []
        for a in self._attn_buffers:
            if a.dim() == 4:
                attns.append(a)
        if not attns:
            return None
        A = torch.stack(attns, dim=0)
        A = A[:, 0]
        if target_pos >= A.shape[2]:
            target_pos = A.shape[2]-1
        A_q = A[:, :, target_pos, :]
        if agg == "last2" and A_q.shape[0] >= 2:
            A_agg = A_q[-2:].mean(dim=0)
        else:
            A_agg = A_q[-1]
        A_agg = A_agg.mean(dim=0)
        if image_token_mask is None or image_token_mask.sum() == 0:
            return None
        A_img = A_agg[image_token_mask.bool()]
        p = F.softmax(A_img / tau, dim=-1).detach().cpu()
        return p
class GradBackend:
    def __init__(self, model, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self._img_feat = None
        self._last_layer_name = None
        self._wrapped = False
        self._orig_get_image_features = None
        self.use_multi_gpu = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
    def _empty_cache_all_gpus(self):
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
    def _enable_wrap(self):
        if self._wrapped:
            return
        core = getattr(self.model, "model", None)
        if core is None or not hasattr(core, "get_image_features"):
            raise RuntimeError("model.model.get_image_features not found")
        self._orig_get_image_features = core.get_image_features
        def wrapped_get_image_features(*args, **kwargs):
            out = self._orig_get_image_features(*args, **kwargs)
            if isinstance(out, tuple):
                if len(out) == 2:
                    if torch.is_tensor(out[0]):
                        t = out[0]
                    elif isinstance(out[0], (list, tuple)) and torch.is_tensor(out[0][0]):
                        t = out[0][0]
                    else:
                        raise ValueError(f"[DEBUG] get_image_features returned non-tensor data: {type(out[0])}")
                    leaf = t.clone().detach().requires_grad_(True)
                    self._img_feat = leaf
                    self._last_layer_name = "model.get_image_features[0]"
                    try:
                        leaf.retain_grad()
                    except Exception:
                        pass
                    if isinstance(out[0], (list, tuple)):
                        modified_first = list(out[0])
                        modified_first[0] = leaf
                        out = (modified_first, out[1])
                    else:
                        out = (leaf, out[1])
                    return out
                elif len(out) == 1:
                    if torch.is_tensor(out[0]):
                        t = out[0]
                    elif torch.is_tensor(out[0][0]):
                        t = out[0][0]
                    else:
                        raise ValueError(f"[DEBUG] get_image_features returned non-tensor data: {type(out[0])}")
                    leaf = t.clone().detach().requires_grad_(True)
                    self._img_feat = leaf
                    self._last_layer_name = "model.get_image_features[0]"
                    try:
                        leaf.retain_grad()
                    except Exception:
                        pass
                    out = (leaf,)
                    print(f"[DEBUG] Visual feature tensor detected: {leaf.shape}, dtype={leaf.dtype}")
                    print(f"[DEBUG] Visual feature device: {leaf.device}")
                    return out
                else:
                    raise ValueError(f"[DEBUG] Unexpected tuple length: {len(out)}")
            def _print_tensor(tag, t):
                print(f"[GF dbg] {tag}: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"device={t.device} dim={t.dim()} req={t.requires_grad} "
                    f"leaf={t.is_leaf} grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}")
            print("[GF dbg] get_image_features type:", type(out).__name__)
            if torch.is_tensor(out):
                _print_tensor("out", out)
            elif isinstance(out, (tuple, list)):
                print("[GF dbg] len:", len(out))
                for i, it in enumerate(out):
                    if torch.is_tensor(it):
                        _print_tensor(f"out[{i}]", it)
                    else:
                        print(f"[GF dbg] out[{i}]:", type(it).__name__)
            elif isinstance(out, dict):
                print("[GF dbg] keys:", list(out.keys()))
                for k, v in out.items():
                    if torch.is_tensor(v):
                        _print_tensor(f"out['{k}']", v)
                    else:
                        print(f"[GF dbg] out['{k}']:", type(v).__name__)
            else:
                print("[GF dbg] (unrecognized container)")
            return out
        core.get_image_features = wrapped_get_image_features
        self._wrapped = True
    def _disable_wrap(self):
        if not self._wrapped:
            return
        core = getattr(self.model, "model", None)
        if core is not None and self._orig_get_image_features is not None:
            core.get_image_features = self._orig_get_image_features
        self._orig_get_image_features = None
        self._wrapped = False
    def _clear(self):
        self._img_feat = None
        self._last_layer_name = None
    def compute_pt(
        self,
        inputs: Dict[str, torch.Tensor],
        target_pos: int,
        tau: float = 0.05,
        target_id: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        self._clear()
        for k, v in inputs.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                v.requires_grad_(True)
        self._enable_wrap()
        try:
            outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False)
        finally:
            self._disable_wrap()
        logits = outputs.logits.float()
        B, S, V = logits.shape
        assert B == 1, "Currently assumes batch=1"
        target_pos = min(target_pos, S - 1)
        if self._img_feat is None:
            raise RuntimeError("get_image_features didn't return valid image tokens. Please check weights and processor input.")
        if self.verbose:
            print(f"[GradBackend] Hooked image tokens at: {self._last_layer_name}, feat shape={tuple(self._img_feat.shape)}, "
                  f"requires_grad={self._img_feat.requires_grad}, grad_fn={type(self._img_feat.grad_fn).__name__ if self._img_feat.grad_fn else None}")
        next_logits = logits[0, target_pos]
        if target_id is None:
            target_id = int(next_logits.argmax().item())
        nll = -F.log_softmax(next_logits, dim=-1)[target_id]
        grads = torch.autograd.grad(nll, self._img_feat, retain_graph=False, allow_unused=False)[0]
        g = grads.abs().mean(dim=-1).squeeze(0)
        p = F.softmax(g / tau, dim=-1).detach().cpu()
        metadata = {}
        if return_metadata:
            g_cpu = g.detach().cpu()
            metadata['attribution_max'] = g_cpu.max().item()
            metadata['attribution_mean'] = g_cpu.mean().item()
            metadata['attribution_sum'] = g_cpu.sum().item()
            p_for_entropy = p + 1e-10
            entropy = -(p_for_entropy * torch.log(p_for_entropy)).sum().item()
            metadata['entropy'] = entropy
            metadata['max_normalized'] = g_cpu.max().item() / (g_cpu.sum().item() + 1e-10)
        self._clear()
        del outputs, logits, next_logits, nll, grads, g
        self._empty_cache_all_gpus()
        gc.collect()
        if return_metadata:
            return p, metadata
        return p
from v_top_layer_extractor import VTopLayerExtractor
@dataclass
class Trajectory:
    steps: List[Dict]
    v_top_layer_path: Optional[str] = None
class TeacherTrajectoryExtractor:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16, tau_grad: float = 0.01):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        max_memory = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / 1024**3
                max_memory[i] = f"{int(total_gb * 0.9)}GiB"
            print(f"[INFO] Detected {torch.cuda.device_count()} GPUs, forcing multi-GPU mode")
            print(f"[INFO] max_memory set for each GPU: {max_memory}")
        elif torch.cuda.is_available():
            print(f"[INFO] Only detected 1 GPU, using single-GPU mode")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True
        )
        self.model.eval()
        self.use_multi_gpu = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
        if self.use_multi_gpu:
            print(f"[INFO] Detected multi-GPU mode, device distribution: {self.model.hf_device_map}")
        else:
            print(f"[INFO] Using single-GPU mode")
        self._input_device = self._get_model_input_device()
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'config'):
            config = self.model.language_model.config
            if hasattr(config, 'num_experts') and config.num_experts > 1:
                print(f"[INFO] Detected MoE model, number of experts: {config.num_experts}")
                self._empty_cache_all_gpus()
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.attn_backend = ConcatAttentionBackend(self.model, self.tokenizer)
        self.grad_backend = GradBackend(self.model, verbose=True)
        self.v_top_layer_extractor = VTopLayerExtractor(self.model, verbose=True)
    def _get_model_input_device(self) -> torch.device:
        if not self.use_multi_gpu:
            return torch.device(self.device)
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            if hasattr(self.model.language_model.model, 'embed_tokens'):
                embed_layer = self.model.language_model.model.embed_tokens
                if hasattr(embed_layer, 'weight'):
                    device = embed_layer.weight.device
                    print(f"[INFO] Detected model input device: {device}")
                    return device
        first_param = next(self.model.parameters())
        device = first_param.device
        print(f"[INFO] Using first parameter's device as input device: {device}")
        return device
    def _empty_cache_all_gpus(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
    @torch.no_grad()
    def teacher_generate_steps(self, image: Image.Image, question: str, T: int = 4, max_new_tokens: int = 4096) -> List[str]:
        prompt = build_teacher_generation_prompt(self.processor, question, T)
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        )
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        safe_max_new_tokens = min(max_new_tokens, 1024)
        if max_new_tokens > safe_max_new_tokens:
            print(f"[WARNING] max_new_tokens={max_new_tokens} is too large, limited to {safe_max_new_tokens} to avoid memory overflow")
        out = self.model.generate(
            **inputs,
            max_new_tokens=safe_max_new_tokens,
            do_sample=False,         
            temperature=0.01,        
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.processor.decode(out[0][input_length:], skip_special_tokens=True)
        print("teacher generate steps:", text)
        steps = split_steps(text, T)
        del inputs, out
        self._empty_cache_all_gpus()
        import gc
        gc.collect()
        return steps
    def _encode_with_image(self, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt"
        )
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        return inputs
    def _image_token_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        input_ids_flat = input_ids.squeeze(0)
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        try:
            start_idx = (input_ids_flat == vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
            end_idx = (input_ids_flat == vision_end_token_id).nonzero(as_tuple=True)[0][0].item()
            mask = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            mask[start_idx + 1:end_idx] = True
            return mask
        except IndexError:
            return None
    def extract_pt_per_step(
        self,
        image: Image.Image,
        question: str,
        teacher_steps: List[str],
        tau_grad: float,
        tau_attn: float = 0.15,
        topk: int = 8,
        skip_markup: bool = True,
        downsample: int = 2,
        max_tokens_per_step: int = 50,
        max_grad_calls_per_step: int = 10,
        aggregation_method: str = "weighted_by_strength",
        attribution_threshold: float = 0.1,
        min_entropy: float = 0.5,
        v_top_layer_save_path: Optional[str] = None,
        enable_v_top_layer: bool = True,
    ) -> Trajectory:
        import gc
        T = len(teacher_steps)
        traj = {"steps": []}
        def print_memory_usage(stage=""):
            if torch.cuda.is_available():
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                    total_allocated = sum(
                        torch.cuda.memory_allocated(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    total_reserved = sum(
                        torch.cuda.memory_reserved(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    print(f"[Memory {stage}] Allocated: {total_allocated:.2f}GB (all GPUs), Reserved: {total_reserved:.2f}GB (all GPUs)")
                    for i in range(torch.cuda.device_count()):
                        alloc = torch.cuda.memory_allocated(i) / 1024**3
                        reserv = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"  GPU {i}: Allocated={alloc:.2f}GB, Reserved={reserv:.2f}GB")
                else:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[Memory {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        print_memory_usage("start")
        v_top_layer_path = None
        if enable_v_top_layer and v_top_layer_save_path is not None:
            print("[INFO] Starting to capture V_top_layer (static features)...")
            prompt_initial = build_prompt_with_chat_template(
                self.processor, question, [], force_T=T
            )
            inputs_initial = self._encode_with_image(image, prompt_initial)
            image_token_mask = self._image_token_mask(inputs_initial["input_ids"])
            if image_token_mask is None:
                print("[WARNING] Unable to identify image token positions, skipping V_top_layer extraction")
            else:
                try:
                    v_top_layer = self.v_top_layer_extractor.capture(
                        inputs=inputs_initial,
                        image_token_mask=image_token_mask
                    )
                    print(f"[INFO] V_top_layer captured: shape={tuple(v_top_layer.shape)}, "
                          f"N_img={v_top_layer.shape[0]}, d={v_top_layer.shape[1]}")
                    print(f"[INFO] V_top_layer will be reused in all {T} steps (static features)")
                    v_top_layer_path = self.v_top_layer_extractor.save(
                        v_top_layer, v_top_layer_save_path
                    )
                    del v_top_layer
                except Exception as e:
                    print(f"[WARNING] V_top_layer capture failed, skipping: {str(e)}")
            del inputs_initial, prompt_initial
            self._empty_cache_all_gpus()
            gc.collect()
        elif not enable_v_top_layer:
            print("[INFO] V_top_layer capture disabled")
        elif v_top_layer_save_path is None:
            print("[INFO] No V_top_layer save path provided, skipping capture")
        print_memory_usage("after_v_top_layer_extraction")
        for t in range(1, T + 1):
            prompt_prefix = build_prompt_with_chat_template(
                self.processor, question, teacher_steps[:t-1], force_T=T
            )
            prompt_full = build_prompt_with_chat_template(
                self.processor, question, teacher_steps[:t],   force_T=T
            )
            inputs_prefix = self._encode_with_image(image, prompt_prefix)
            inputs_full   = self._encode_with_image(image, prompt_full)
            ids_full = inputs_full["input_ids"][0]
            S_prefix = inputs_prefix["input_ids"].shape[1]
            S_full   = ids_full.shape[0]
            span_start, span_end = S_prefix, S_full
            token_attributions = []
            grad_calls_made = 0
            tokens_to_process = min(span_end - span_start, max_tokens_per_step)
            print(f"[INFO] Step {t}: processing {tokens_to_process} tokens (original: {span_end - span_start})")
            for j in range(span_start, span_start + tokens_to_process):
                if grad_calls_made >= max_grad_calls_per_step:
                    print(f"[WARNING] Step {t}: reached maximum gradient calculation limit ({max_grad_calls_per_step}), stopping processing")
                    break
                if downsample > 1 and ((j - span_start) % downsample != 0):
                    continue
                if skip_markup:
                    tok_str = self.tokenizer.decode(ids_full[j:j+1], skip_special_tokens=False)
                    if ("<" in tok_str) or (">" in tok_str):
                        continue
                prev_pos = j - 1
                target_id = int(ids_full[j].item())
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                        total_allocated = sum(
                            torch.cuda.memory_allocated(i) / 1024**3 
                            for i in range(torch.cuda.device_count())
                        )
                        if total_allocated > 90:
                            print(f"[WARNING] Total memory usage too high ({total_allocated:.1f}GB), skipping token {j}")
                            continue
                    else:
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        if allocated > 90:
                            print(f"[WARNING] Memory usage too high ({allocated:.1f}GB), skipping token {j}")
                            continue
                p_j, metadata_j = self.grad_backend.compute_pt(
                    inputs=inputs_full,
                    target_pos=prev_pos,
                    tau=tau_grad,
                    target_id=target_id,
                    return_metadata=True
                )
                token_attributions.append((p_j, metadata_j, j))
                grad_calls_made += 1
                print(f"[INFO] Step {t}: completed gradient calculation {grad_calls_made}/{max_grad_calls_per_step}, "
                      f"attribution strength={metadata_j['attribution_max']:.4f}, entropy={metadata_j['entropy']:.4f}")
            if len(token_attributions) == 0:
                j = span_end - 1
                prev_pos  = j - 1
                target_id = int(ids_full[j].item())
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                gc.collect()
                p_j, metadata_j = self.grad_backend.compute_pt(
                    inputs=inputs_full,
                    target_pos=prev_pos,
                    tau=tau_grad,
                    target_id=target_id,
                    return_metadata=True
                )
                token_attributions.append((p_j, metadata_j, j))
            p_t = self._aggregate_attributions(
                token_attributions,
                method=aggregation_method,
                threshold=attribution_threshold,
                min_entropy=min_entropy
            )
            topk_idx = torch.topk(p_t, k=min(topk, p_t.numel())).indices.tolist()
            traj["steps"].append({
                "p_t": p_t.tolist(),
                "p_topk_idx": topk_idx,
                "tau_used": float(tau_grad),
                "method": f"grad_{aggregation_method}",
                "span_token_count": len(token_attributions),
                "downsample": int(downsample),
                "skip_markup": bool(skip_markup),
                "aggregation_method": aggregation_method,
            })
            del inputs_prefix, inputs_full, token_attributions, p_t
            if 'inputs' in locals():
                del inputs
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            print_memory_usage(f"step_{t}")
        return Trajectory(steps=traj["steps"], v_top_layer_path=v_top_layer_path)
    def _aggregate_attributions(
        self,
        token_attributions: List[Tuple[torch.Tensor, Dict, int]],
        method: str = "weighted_by_strength",
        threshold: float = 0.1,
        min_entropy: float = 0.5,
    ) -> torch.Tensor:
        if method == "simple_avg":
            P_sum = None
            for p_j, _, _ in token_attributions:
                P_sum = p_j if P_sum is None else (P_sum + p_j)
            return (P_sum / len(token_attributions)).cpu()
        elif method == "weighted_by_strength":
            weights = []
            p_list = []
            for p_j, metadata_j, _ in token_attributions:
                weight = metadata_j['max_normalized']
                weights.append(weight)
                p_list.append(p_j)
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / (weights.sum() + 1e-10)
            p_t = torch.zeros_like(p_list[0])
            for w, p_j in zip(weights, p_list):
                p_t += w * p_j
            return p_t.cpu()
        elif method == "filter_by_threshold":
            filtered = []
            for p_j, metadata_j, token_idx in token_attributions:
                if metadata_j['max_normalized'] >= threshold:
                    filtered.append(p_j)
                    print(f"[Filter] Token {token_idx}: attribution strength={metadata_j['max_normalized']:.4f} >= {threshold}, retained")
                else:
                    print(f"[Filter] Token {token_idx}: attribution strength={metadata_j['max_normalized']:.4f} < {threshold}, filtered")
            if len(filtered) == 0:
                print(f"[WARNING] All tokens were filtered, using simple average")
                return self._aggregate_attributions(token_attributions, "simple_avg", threshold, min_entropy)
            P_sum = None
            for p_j in filtered:
                P_sum = p_j if P_sum is None else (P_sum + p_j)
            return (P_sum / len(filtered)).cpu()
        elif method == "entropy_weighted":
            weights = []
            p_list = []
            for p_j, metadata_j, _ in token_attributions:
                entropy = metadata_j['entropy']
                if entropy >= min_entropy:
                    weights.append(entropy)
                    p_list.append(p_j)
            if len(weights) == 0:
                print(f"[WARNING] All tokens' entropy is below {min_entropy}, using simple average")
                return self._aggregate_attributions(token_attributions, "simple_avg", threshold, min_entropy)
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = F.softmax(weights, dim=0)
            p_t = torch.zeros_like(p_list[0])
            for w, p_j in zip(weights, p_list):
                p_t += w * p_j
            return p_t.cpu()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="./data/examples/images/level_3/316/map_3x3_step_0.png")
    parser.add_argument("--question", type=str, default="How can the character go to the gift and avoid obstacles?(e.g. water surface).")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save", type=str, default="traj.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to process per step, for controlling memory usage")
    parser.add_argument("--downsample", type=int, default=1, help="Token sampling interval, reducing computation")
    parser.add_argument("--max_grad_calls", type=int, default=50, help="Maximum number of gradient calculation per step, this is the true memory control parameter")
    parser.add_argument("--tau_grad", type=float, default=0.01, help="Temperature parameter for gradient attribution")
    parser.add_argument("--aggregation_method", type=str, default="weighted_by_strength", 
                        choices=["simple_avg", "weighted_by_strength", "filter_by_threshold", "entropy_weighted"],
                        help="Aggregation method: simple_avg(simple average), weighted_by_strength(weighted by strength, recommended), filter_by_threshold(threshold filtering), entropy_weighted(entropy weighted)")
    parser.add_argument("--attribution_threshold", type=float, default=0.1, 
                        help="Threshold for attribution strength for filter_by_threshold strategy")
    parser.add_argument("--min_entropy", type=float, default=0.5, 
                        help="Minimum entropy threshold for entropy_weighted strategy")
    args = parser.parse_args()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cache_dir = "../my_qwen_model"
    os.environ['HF_HOME'] = cache_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_directory = "/root/autodl-tmp/my_qwen_model/Qwen/Qwen2.5-VL-32B-Instruct"
    extractor = TeacherTrajectoryExtractor(local_model_directory, device=device, dtype=torch.bfloat16)
    img = Image.open(args.image).convert("RGB")
    teacher_steps = extractor.teacher_generate_steps(img, args.question, T=args.steps)
    print("Teacher steps:\n", "\n".join(teacher_steps))
    traj = extractor.extract_pt_per_step(
        img, args.question, teacher_steps,
        max_tokens_per_step=args.max_tokens,
        downsample=args.downsample,
        max_grad_calls_per_step=args.max_grad_calls,
        tau_grad=args.tau_grad,
        aggregation_method=args.aggregation_method,
        attribution_threshold=args.attribution_threshold,
        min_entropy=args.min_entropy
    )
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(traj.__dict__, f, ensure_ascii=False, indent=2)
    print(f"Saved trajectory to {args.save}")
if __name__ == "__main__":
    main()