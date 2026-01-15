from __future__ import annotations
import os, json, re, gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.lvr_prompt_utils import build_prompt_with_chat_template, build_teacher_generation_prompt
from v_top_layer_extractor import VTopLayerExtractor
from src.attention_hook_utils import AttentionHookExtractor
@dataclass
class Trajectory:
    steps: List[Dict]
    v_top_layer_path: Optional[str] = None
    image_path: Optional[str] = None
    question: Optional[str] = None
class AttentionBasedTrajectoryExtractor:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        preprocessor_path = os.path.join(model_path, "preprocessor_config.json")
        if not os.path.exists(preprocessor_path):
            print(f"[INFO] preprocessor_config.json not found in {model_path}. Loading from base model.")
            base_model_path = "/root/autodl-tmp/my_qwen_model/Qwen2.5-VL-3B-Instruct"
            if not os.path.exists(base_model_path):
                base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
            self.processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.processor.tokenizer = tokenizer
                print("[INFO] Successfully loaded tokenizer from checkpoint.")
            except Exception as e:
                print(f"[WARNING] Could not load tokenizer from checkpoint: {e}. Using base tokenizer.")
        else:
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
            print(f"[INFO] Max memory per GPU: {max_memory}")
        elif torch.cuda.is_available():
            print(f"[INFO] 只检测到1个GPU，使用单GPU模式")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.model.eval()
        self.v_top_layer_extractor = VTopLayerExtractor(self.model, verbose=True)
        self.attention_hook_extractor = AttentionHookExtractor(self.model, verbose=True)
        self.use_multi_gpu = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
        if self.use_multi_gpu:
            print(f"[INFO] 检测到多 GPU 模式，设备分布: {self.model.hf_device_map}")
        else:
            print(f"[INFO] 使用单 GPU 模式")
        self._input_device = self._get_model_input_device()
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
        print(f"[INFO] Using first parameter device as input device: {device}")
        return device
    def _collect_attention_configs(self) -> List:
        configs = []
        candidates = [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "model", None), "config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
            getattr(
                getattr(getattr(self.model, "model", None), "language_model", None),
                "config",
                None,
            ),
        ]
        seen = set()
        for cfg in candidates:
            if cfg is None or not hasattr(cfg, "output_attentions"):
                continue
            if id(cfg) in seen:
                continue
            seen.add(id(cfg))
            configs.append(cfg)
        return configs
    def _set_output_attentions_temporarily(self, enable: bool) -> List[Tuple[object, bool]]:
        configs = self._collect_attention_configs()
        prev_flags = []
        for cfg in configs:
            prev_value = getattr(cfg, "output_attentions", False)
            prev_flags.append((cfg, prev_value))
            cfg.output_attentions = enable
        return prev_flags
    def _restore_output_attentions(self, prev_flags) -> None:
        for cfg, value in prev_flags:
            cfg.output_attentions = value
    def _empty_cache_all_gpus(self):
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
    @torch.no_grad()
    def teacher_generate_steps(self, image: Image.Image, question: str, T: int = 4, max_new_tokens: int = 4096) -> List[str]:
        prompt = build_teacher_generation_prompt(self.processor, question, force_T=T)
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        )
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        safe_max_new_tokens = min(max_new_tokens, 1024)
        if max_new_tokens > safe_max_new_tokens:
            print(f"[WARNING] max_new_tokens={max_new_tokens} 过大，已限制为 {safe_max_new_tokens} 以避免显存溢出")
        out = self.model.generate(
            **inputs,
            max_new_tokens=safe_max_new_tokens,
            do_sample=False,         
            temperature=0.01,        
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True, 
        )
        text = self.processor.decode(out[0][input_length:], skip_special_tokens=True)
        print("teacher generate response (LVR format):", text)
        del inputs, out
        self._empty_cache_all_gpus()
        import gc
        gc.collect()
        return [text]
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
        try:
            start_idx = (input_ids_flat == vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
            end_idx = (input_ids_flat == vision_end_token_id).nonzero(as_tuple=True)[0][0].item()
            mask = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            mask[start_idx + 1:end_idx] = True
            return mask
        except IndexError:
            return None
    def _text_token_mask(self, image_token_mask: torch.Tensor) -> torch.Tensor:
        return ~image_token_mask
    @torch.no_grad()
    def compute_attention_based_attribution(
        self,
        inputs: Dict[str, torch.Tensor],
        image_token_mask: torch.Tensor,
        text_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_token_mask is None:
            text_token_mask = self._text_token_mask(image_token_mask)
        print("[INFO] 使用 Hook 机制流式提取 Attention...")
        prev_flags = self._set_output_attentions_temporarily(True)
        try:
            S = self.attention_hook_extractor.extract_optimized(
                inputs=inputs,
                image_token_mask=image_token_mask,
                text_token_mask=text_token_mask
            )
        finally:
            self._restore_output_attentions(prev_flags)
        return S
    def extract_pt_per_step(
        self,
        image: Image.Image,
        question: str,
        teacher_steps: List[str],
        topk: int = 8,
        v_top_layer_save_path: Optional[str] = None,
        enable_v_top_layer: bool = False,
        image_path: Optional[str] = None,
    ) -> Trajectory:
        T = len(teacher_steps)
        print(f"[INFO] 提取轨迹，共 {T} 个步骤（LVR格式）")
        traj = {"steps": []}
        def print_memory_usage(stage=""):
            if torch.cuda.is_available():
                if self.use_multi_gpu:
                    total_allocated = sum(
                        torch.cuda.memory_allocated(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    total_reserved = sum(
                        torch.cuda.memory_reserved(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    print(f"[Memory {stage}] Allocated: {total_allocated:.2f}GB (所有GPU), Reserved: {total_reserved:.2f}GB (所有GPU)")
                else:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[Memory {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        print_memory_usage("start")
        v_top_layer_path = None
        if enable_v_top_layer and v_top_layer_save_path is not None:
            print("[INFO] 开始抓取 V_top_layer（静态特征）...")
            prompt_initial = build_prompt_with_chat_template(
                self.processor, question, steps_prefix=[], force_T=T
            )
            inputs_initial = self._encode_with_image(image, prompt_initial)
            image_token_mask = self._image_token_mask(inputs_initial["input_ids"])
            if image_token_mask is None:
                print("[WARNING] 无法识别图像 token 位置，跳过 V_top_layer 提取")
            else:
                try:
                    v_top_layer = self.v_top_layer_extractor.capture(
                        inputs=inputs_initial,
                        image_token_mask=image_token_mask
                    )
                    print(f"[INFO] V_top_layer 抓取完成: shape={tuple(v_top_layer.shape)}, "
                          f"N_img={v_top_layer.shape[0]}, d={v_top_layer.shape[1]}")
                    v_top_layer_path = self.v_top_layer_extractor.save(
                        v_top_layer, v_top_layer_save_path
                    )
                    del v_top_layer
                except Exception as e:
                    print(f"[WARNING] V_top_layer 捕获失败，跳过: {str(e)}")
            del inputs_initial, prompt_initial
            self._empty_cache_all_gpus()
            gc.collect()
        elif not enable_v_top_layer:
            print("[INFO] V_top_layer 捕获已禁用")
        elif v_top_layer_save_path is None:
            print("[INFO] 未提供 V_top_layer 保存路径，跳过捕获")
        for t in range(1, T + 1):
            print(f"\n[INFO] 处理步骤 {t}/{T}")
            prompt_full = build_prompt_with_chat_template(
                self.processor, question, steps_prefix=teacher_steps[:t], force_T=T
            )
            inputs_full = self._encode_with_image(image, prompt_full)
            image_token_mask = self._image_token_mask(inputs_full["input_ids"])
            if image_token_mask is None:
                raise RuntimeError(f"步骤 {t}: 无法识别图像token位置")
            S = self.compute_attention_based_attribution(
                inputs=inputs_full,
                image_token_mask=image_token_mask
            )
            p_t = (S - S.min()) / (S.max() - S.min() + 1e-10)
            topk_idx = torch.topk(p_t, k=min(topk, p_t.numel())).indices.tolist()
            traj["steps"].append({
                "p_t": p_t.tolist(),
                "p_topk_idx": topk_idx,
                "tau_used": "attention",
                "method": "attention_based",
                "raw_attention_scores": S.tolist(),
            })
            print(f"[INFO] 步骤 {t}: 归因分布计算完成，N_img={len(p_t)}, "
                  f"最大注意力强度={S.max().item():.4f}, 平均注意力强度={S.mean().item():.4f}")
            del inputs_full, image_token_mask, S, p_t
            self._empty_cache_all_gpus()
            gc.collect()
            print_memory_usage(f"step_{t}")
        return Trajectory(
            steps=traj["steps"], 
            v_top_layer_path=v_top_layer_path,
            image_path=image_path,
            question=question
        )
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/root/autodl-tmp/ViLR/data/BLINK_output/test-00000-of-00001/test_Relative_Depth_1/124_1.jpg")
    parser.add_argument("--question", type=str, default="Which point is closer to the camera?")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save", type=str, default="/root/autodl-tmp/ViLR/output/traj_1.json")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--topk", type=int, default=8, help="返回top-k个最重要的图像token索引")
    parser.add_argument("--enable_v_top_layer", type=bool, default=False)
    parser.add_argument("--v_top_layer_save_path", type=str, default=None)
    args = parser.parse_args()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    cache_dir = "../my_qwen_model"
    os.environ['HF_HOME'] = cache_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_directory = "/root/autodl-tmp/ViLR/training/checkpoints/single_stage_4lvr-1400steps_1/checkpoint-1000"
    extractor = AttentionBasedTrajectoryExtractor(
        local_model_directory, 
        device=device, 
        dtype=torch.bfloat16
    )
    img = Image.open(args.image).convert("RGB")
    teacher_steps = extractor.teacher_generate_steps(img, args.question, T=args.steps)
    print("Teacher steps:\n", "\n".join(teacher_steps))
    traj = extractor.extract_pt_per_step(
        img, args.question, teacher_steps,
        topk=args.topk,
        image_path=args.image
    )
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(traj.__dict__, f, ensure_ascii=False, indent=2)
    print(f"Saved trajectory to {args.save}")
if __name__ == "__main__":
    main()