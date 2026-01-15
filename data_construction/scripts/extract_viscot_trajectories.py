import os
import json
import re
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../extraction')))
from teacher_traj_extractor_stepwise import TeacherTrajectoryExtractor
from teacher_traj_extractor_attention import AttentionBasedTrajectoryExtractor
from src.answer_validator import validate_sample
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def _copy_if_needed(src: str, dst: str, copy_images: bool) -> None:
    if not copy_images:
        return
    _ensure_dir(os.path.dirname(dst))
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
def parse_question(conversation_value: str) -> str:
    question = conversation_value.strip()
    if question.startswith("<image>"):
        question = question[7:]
        if question.startswith("\n"):
            question = question[1:]
    return question.strip()
def extract_answer(conversation_value: str) -> str:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, conversation_value, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return conversation_value.strip()
def load_viscot_data(json_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    print(f"正在加载数据文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    print(f"加载了 {len(data)} 个样本")
    return data
def process_sample(
    sample: Dict,
    extractor: Union[TeacherTrajectoryExtractor, AttentionBasedTrajectoryExtractor],
    base_image_dir: str,
    output_dir: str,
    vtop_output_dir: Optional[str] = None,
    extraction_method: str = "gradient",
    steps: int = 1,
    tau_grad: float = 0.01,
    skip_markup: bool = True,
    downsample: int = 2,
    max_tokens_per_step: int = 50,
    max_grad_calls_per_step: int = 10,
    aggregation_method: str = "simple_avg",
    attribution_threshold: float = 0.1,
    min_entropy: float = 0.5,
    topk: int = 8,
    enable_validation: bool = True,
    validation_api_key: Optional[str] = None,
    validation_model: Optional[str] = None,
    enable_v_top_layer: bool = True,
) -> Dict[str, Any]:
    try:
        image_paths = sample.get("image", [])
        if not image_paths:
            raise ValueError("样本中没有image字段")
        image_relative_path = image_paths[0]
        image_full_path = os.path.join(base_image_dir, image_relative_path)
        if not os.path.exists(image_full_path):
            raise FileNotFoundError(f"图像文件不存在: {image_full_path}")
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError("conversations字段至少需要2个元素")
        question_raw = conversations[0].get("value", "")
        question = parse_question(question_raw)
        answer_raw = conversations[1].get("value", "")
        ground_truth = extract_answer(answer_raw)
        print(f"  [样本 {sample.get('question_id', 'unknown')}] Ground Truth: {ground_truth}")
        image = Image.open(image_full_path).convert("RGB")
        print(f"  [样本 {sample.get('question_id', 'unknown')}] 生成教师步骤...")
        teacher_steps = extractor.teacher_generate_steps(
            image, question, T=steps, max_new_tokens=4096
        )
        teacher_full_response = "\n".join(teacher_steps)
        is_answer_correct = True
        final_answer = None
        validation_explanation = None
        if enable_validation:
            print(f"  [样本 {sample.get('question_id', 'unknown')}] 校验答案...")
            is_answer_correct, final_answer, validation_explanation = validate_sample(
                teacher_response=teacher_full_response,
                ground_truth=ground_truth,
                question=question,
                api_key=validation_api_key,
                model=validation_model,
            )
            if is_answer_correct:
                print(f"  [样本 {sample.get('question_id', 'unknown')}] 答案校验通过")
            else:
                print(f"  [样本 {sample.get('question_id', 'unknown')}] 答案校验未通过，跳过轨迹提取")
                return {
                    "question_id": sample.get("question_id", "unknown"),
                    "dataset": sample.get("dataset", None),
                    "split": sample.get("split", None),
                    "image_path": image_full_path,
                    "image_relative_path": image_relative_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "teacher_full_response": teacher_full_response,
                    "final_answer": final_answer,
                    "validation": {
                        "is_correct": False,
                        "explanation": validation_explanation,
                    },
                    "skipped": True,
                    "skip_reason": "答案校验未通过",
                }
        question_id = sample.get("question_id", "unknown")
        vtop_dir = vtop_output_dir or os.path.join(output_dir, "tensors")
        os.makedirs(vtop_dir, exist_ok=True)
        v_top_layer_filename = f"sample_{question_id:06d}_v_top_layer.pth"
        v_top_layer_save_path = os.path.join(vtop_dir, v_top_layer_filename)
        if vtop_output_dir:
            v_top_layer_relative_path = os.path.relpath(v_top_layer_save_path, vtop_output_dir)
        else:
            v_top_layer_relative_path = os.path.join("tensors", v_top_layer_filename)
        print(f"  [样本 {question_id}] 提取视觉轨迹（方法: {extraction_method}）...")
        if extraction_method == "gradient":
            trajectory = extractor.extract_pt_per_step(
                image=image,
                question=question,
                teacher_steps=teacher_steps,
                tau_grad=tau_grad,
                skip_markup=skip_markup,
                downsample=downsample,
                max_tokens_per_step=max_tokens_per_step,
                max_grad_calls_per_step=max_grad_calls_per_step,
                aggregation_method=aggregation_method,
                attribution_threshold=attribution_threshold,
                min_entropy=min_entropy,
                v_top_layer_save_path=v_top_layer_save_path if enable_v_top_layer else None,
                enable_v_top_layer=enable_v_top_layer,
            )
        elif extraction_method == "attention":
            trajectory = extractor.extract_pt_per_step(
                image=image,
                question=question,
                teacher_steps=teacher_steps,
                topk=topk,
                v_top_layer_save_path=v_top_layer_save_path if enable_v_top_layer else None,
                enable_v_top_layer=enable_v_top_layer,
            )
        else:
            raise ValueError(f"未知的提取方法: {extraction_method}，支持的方法: 'gradient', 'attention'")
        result = {
            "question_id": question_id,
            "dataset": sample.get("dataset", None),
            "split": sample.get("split", None),
            "image_path": image_full_path,
            "image_relative_path": image_relative_path,
            "question": question,
            "ground_truth": ground_truth,
            "teacher_full_response": teacher_full_response,
            "final_answer": final_answer,
            "v_top_layer_path": v_top_layer_relative_path,
            "v_top_layer_path_abs": v_top_layer_save_path,
            "trajectory": {
                "steps": trajectory.steps,
            },
            "parameters": {
                "extraction_method": extraction_method,
                "steps": steps,
                "tau_grad": tau_grad if extraction_method == "gradient" else None,
                "skip_markup": skip_markup if extraction_method == "gradient" else None,
                "downsample": downsample if extraction_method == "gradient" else None,
                "max_tokens_per_step": max_tokens_per_step if extraction_method == "gradient" else None,
                "max_grad_calls_per_step": max_grad_calls_per_step if extraction_method == "gradient" else None,
                "aggregation_method": aggregation_method if extraction_method == "gradient" else None,
                "attribution_threshold": attribution_threshold if extraction_method == "gradient" else None,
                "min_entropy": min_entropy if extraction_method == "gradient" else None,
                "topk": topk if extraction_method == "attention" else None,
            },
        }
        if enable_validation:
            result["validation"] = {
                "is_correct": is_answer_correct,
                "explanation": validation_explanation,
            }
        print(f"  [样本 {sample.get('question_id', 'unknown')}] 完成！")
        return result
    except Exception as e:
        print(f"  [样本 {sample.get('question_id', 'unknown')}] 处理失败: {str(e)}")
        if torch.cuda.is_available():
            if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        return {
            "question_id": sample.get("question_id", None),
            "error": str(e),
            "image_path": image_full_path if 'image_full_path' in locals() else None,
        }
def build_lavit15k_from_viscot(
    extractor: Union[TeacherTrajectoryExtractor, AttentionBasedTrajectoryExtractor],
    data_file: str,
    base_image_dir: str,
    output_root: str,
    extraction_method: str,
    steps: int,
    tau_grad: float,
    skip_markup: bool,
    downsample: int,
    max_tokens_per_step: int,
    max_grad_calls_per_step: int,
    aggregation_method: str,
    attribution_threshold: float,
    min_entropy: float,
    topk: int,
    enable_validation: bool,
    validation_api_key: Optional[str],
    validation_model: Optional[str],
    enable_v_top_layer: bool,
    max_samples: Optional[int],
    start_idx: int,
    random_seed: Optional[int],
    shuffle: bool,
    copy_images: bool,
    save_every: int,
) -> None:
    data = load_viscot_data(data_file, max_samples=None)
    used_seed = None
    if shuffle or random_seed is not None:
        if random_seed is not None:
            used_seed = random_seed
            random.seed(random_seed)
        else:
            import time
            used_seed = int(time.time())
            random.seed(used_seed)
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
    if start_idx > 0:
        data = data[start_idx:]
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    images_dir = os.path.join(output_root, "data", "images")
    vtop_dir = os.path.join(output_root, "data", "features", "vtop")
    traj_dir = os.path.join(output_root, "data", "features", "trajectories")
    metadata_dir = os.path.join(output_root, "data", "metadata")
    for d in (images_dir, vtop_dir, traj_dir, metadata_dir):
        _ensure_dir(d)
    results: List[Dict[str, Any]] = []
    failed = 0
    skipped = 0
    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] 处理样本...")
        result = process_sample(
            sample=sample,
            extractor=extractor,
            base_image_dir=base_image_dir,
            output_dir=traj_dir,
            vtop_output_dir=vtop_dir,
            extraction_method=extraction_method,
            steps=steps,
            tau_grad=tau_grad,
            skip_markup=skip_markup,
            downsample=downsample,
            max_tokens_per_step=max_tokens_per_step,
            max_grad_calls_per_step=max_grad_calls_per_step,
            aggregation_method=aggregation_method,
            attribution_threshold=attribution_threshold,
            min_entropy=min_entropy,
            topk=topk,
            enable_validation=enable_validation,
            validation_api_key=validation_api_key,
            validation_model=validation_model,
            enable_v_top_layer=enable_v_top_layer,
        )
        if "error" in result:
            failed += 1
            continue
        if result.get("skipped"):
            skipped += 1
            continue
        question_id = result.get("question_id", i)
        image_rel = result.get("image_relative_path", "")
        if image_rel.startswith("data/images/"):
            image_rel = image_rel[len("data/images/"):]
        v_top_path = result.get("v_top_layer_path_abs")
        if not v_top_path:
            v_top_rel = result.get("v_top_layer_path")
            if v_top_rel:
                v_top_path = os.path.join(vtop_dir, v_top_rel) if not os.path.isabs(v_top_rel) else v_top_rel
        traj_filename = f"sample_{question_id:06d}_attention.json" if isinstance(question_id, int) else f"sample_{question_id}_attention.json"
        traj_path = os.path.join(traj_dir, traj_filename)
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(result.get("trajectory", {}), f, ensure_ascii=False, indent=2)
        image_src = result.get("image_path")
        image_dst = os.path.join(images_dir, image_rel)
        if image_src:
            _copy_if_needed(image_src, image_dst, copy_images)
        v_top_filename = os.path.basename(v_top_path) if v_top_path else None
        if v_top_path and os.path.exists(v_top_path):
            vtop_dst = os.path.join(vtop_dir, v_top_filename)
            if os.path.abspath(v_top_path) != os.path.abspath(vtop_dst):
                _copy_if_needed(v_top_path, vtop_dst, True)
        image_path_value = f"data/images/{image_rel}" if copy_images else image_src
        results.append({
            "sample_id": f"lavit_{len(results)+1:06d}",
            "question_id": question_id,
            "source_dataset": result.get("dataset", "unknown"),
            "split": result.get("split", "train"),
            "question": result.get("question", ""),
            "ground_truth": result.get("ground_truth", ""),
            "image_path": image_path_value,
            "v_top_path": f"data/features/vtop/{v_top_filename}" if v_top_filename else None,
            "trajectory_path": f"data/features/trajectories/{traj_filename}",
        })
        if save_every and (len(results) % save_every == 0):
            checkpoint_path = os.path.join(metadata_dir, "lavit_15k_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    training_results: List[Dict[str, Any]] = []
    for item in results:
        image_path = item.get("image_path", "")
        if image_path.startswith("data/images/"):
            image_relative_path = image_path[len("data/images/"):]
        else:
            image_relative_path = image_path
        v_top_path = item.get("v_top_path")
        attention_path = item.get("trajectory_path")
        v_top_path_abs = os.path.join(output_root, v_top_path) if v_top_path and not os.path.isabs(v_top_path) else v_top_path
        attention_path_abs = os.path.join(output_root, attention_path) if attention_path and not os.path.isabs(attention_path) else attention_path
        training_results.append({
            "question_id": item.get("question_id"),
            "dataset": item.get("source_dataset"),
            "split": item.get("split"),
            "question": item.get("question", ""),
            "ground_truth_enriched": item.get("ground_truth", ""),
            "image_relative_path": image_relative_path,
            "v_top_path_abs": v_top_path_abs,
            "attention_path_abs": attention_path_abs,
        })
    training_json = os.path.join(metadata_dir, "lavit_15k_for_training.json")
    with open(training_json, "w", encoding="utf-8") as f:
        json.dump({"results": training_results}, f, ensure_ascii=False, indent=2)
def main():
    parser = argparse.ArgumentParser(
        description="从Visual-CoT数据集中提取视觉轨迹"
    )
    parser.add_argument("--data_file", type=str, default="./data/Visual-CoT-full/viscot_363k_lvr_formatted.json", help="Visual-CoT数据文件路径")
    parser.add_argument("--base_image_dir", type=str, default="./data/Visual-CoT-full", help="图像文件的基础目录")
    parser.add_argument("--max_samples", type=int, default=5, help="最大处理样本数量（None表示处理所有样本）")
    parser.add_argument("--start_idx", type=int, default=0, help="起始样本索引（在随机打乱后应用）")
    parser.add_argument("--random_seed", type=int, default=5, help="随机数种子，用于固定随机顺序（默认不使用随机）")
    parser.add_argument("--shuffle", action="store_true", default=True, help="随机打乱数据顺序（需要配合 --random_seed 使用以确保可复现）")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/my_qwen_model/Qwen/Qwen2.5-VL-32B-Instruct", help="模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备（cuda或cpu）")
    parser.add_argument("--method", type=str, default="attention", choices=["gradient", "attention"], 
                        help="提取方法: 'gradient' (梯度归因) 或 'attention' (注意力权重)")
    parser.add_argument("--steps", type=int, default=1, help="推理步骤数")
    parser.add_argument("--tau_grad", type=float, default=0.01, help="梯度归因的温度参数（仅用于gradient方法）")
    parser.add_argument("--skip_markup", action="store_true", help="跳过包含<>的标记token（默认启用，仅用于gradient方法）")
    parser.add_argument("--no_skip_markup", action="store_false", dest="skip_markup", help="禁用跳过标记token")
    parser.set_defaults(skip_markup=True)
    parser.add_argument("--downsample", type=int, default=1, help="token抽样间隔（仅用于gradient方法）")
    parser.add_argument("--max_tokens_per_step", type=int, default=50, help="每步最大处理token数量（仅用于gradient方法）")
    parser.add_argument("--max_grad_calls_per_step", type=int, default=10, help="每步最大梯度计算次数（仅用于gradient方法）")
    parser.add_argument("--aggregation_method", type=str, default="weighted_by_strength", choices=["simple_avg", "weighted_by_strength", "filter_by_threshold", "entropy_weighted"], help="聚合策略（仅用于gradient方法）")
    parser.add_argument("--attribution_threshold", type=float, default=0.1, help="归因强度阈值（用于filter_by_threshold，仅用于gradient方法）")
    parser.add_argument("--min_entropy", type=float, default=0.5, help="最小熵阈值（用于entropy_weighted，仅用于gradient方法）")
    parser.add_argument("--topk", type=int, default=8, help="返回top-k个最重要的图像token索引（仅用于attention方法）")
    parser.add_argument("--output_dir", type=str, default="./trajectories/viscot", help="输出目录")
    parser.add_argument("--output_lavit15k_root", type=str, default=None, help="输出 LaViT-15k 结构目录")
    parser.add_argument("--copy_images", action="store_true", default=True, help="复制图片到 LaViT-15k 目录")
    parser.add_argument("--no_copy_images", action="store_false", dest="copy_images")
    parser.add_argument("--save_every", type=int, default=50, help="定期保存metadata检查点")
    parser.add_argument("--save_format", type=str, default="json", choices=["json", "jsonl"], help="保存格式：json（单个文件）或jsonl（每行一个样本）")
    parser.add_argument("--enable_validation", action="store_true", default=True, help="启用答案校验（默认启用）")
    parser.add_argument("--disable_validation", action="store_false", dest="enable_validation", help="禁用答案校验")
    parser.add_argument("--validation_api_key", type=str, default=None, help="OpenRouter API 密钥（默认使用内置密钥）")
    parser.add_argument("--validation_model", type=str, default="mimo-v2-flash", help="用于校验的模型名称")
    parser.add_argument("--save_only_correct", action="store_true", default=False, help="只保存校验通过的样本（默认保存所有样本）")
    parser.add_argument("--enable_v_top_layer", action="store_true", default=True, help="启用 V_top_layer 捕获（默认启用）")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    data = load_viscot_data(args.data_file, max_samples=None)
    used_seed = None
    if args.shuffle or args.random_seed is not None:
        if args.random_seed is not None:
            used_seed = args.random_seed
            random.seed(args.random_seed)
            print(f"使用随机数种子: {args.random_seed}")
        else:
            import time
            used_seed = int(time.time())
            random.seed(used_seed)
            print(f"使用时间戳作为随机数种子: {used_seed}")
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
        print(f"数据已随机打乱，共 {len(data)} 个样本")
    else:
        print(f"保持原始数据顺序，共 {len(data)} 个样本")
    if args.start_idx > 0:
        data = data[args.start_idx:]
        print(f"从索引 {args.start_idx} 开始处理，剩余 {len(data)} 个样本")
    if args.max_samples is not None and args.max_samples > 0:
        original_count = len(data)
        data = data[:args.max_samples]
        print(f"限制处理数量为 {args.max_samples} 个样本（原始: {original_count}）")
    processing_params = {
        "extraction_method": args.method,
        "random_seed": used_seed,
        "shuffled": args.shuffle or args.random_seed is not None,
        "start_idx": args.start_idx,
        "max_samples": args.max_samples,
        "total_samples_loaded": len(data),
    }
    print(f"正在加载模型: {args.model_path}")
    print(f"使用提取方法: {args.method}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[GPU诊断] 检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name}, 总显存: {total_memory:.2f}GB")
    else:
        print("[GPU诊断] 未检测到CUDA设备")
    if args.method == "gradient":
        extractor = TeacherTrajectoryExtractor(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16
        )
    elif args.method == "attention":
        extractor = AttentionBasedTrajectoryExtractor(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"未知的提取方法: {args.method}")
    print("模型加载完成！")
    if torch.cuda.is_available():
        print(f"[GPU诊断] 模型加载后的GPU使用情况:")
        if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
            print(f"  设备分布: {extractor.model.hf_device_map}")
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserv = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
        else:
            print("  警告：模型未使用多GPU，只使用了单GPU模式")
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserv = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU 0: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
    if args.output_lavit15k_root:
        build_lavit15k_from_viscot(
            extractor=extractor,
            data_file=args.data_file,
            base_image_dir=args.base_image_dir,
            output_root=args.output_lavit15k_root,
            extraction_method=args.method,
            steps=args.steps,
            tau_grad=args.tau_grad,
            skip_markup=args.skip_markup,
            downsample=args.downsample,
            max_tokens_per_step=args.max_tokens_per_step,
            max_grad_calls_per_step=args.max_grad_calls_per_step,
            aggregation_method=args.aggregation_method,
            attribution_threshold=args.attribution_threshold,
            min_entropy=args.min_entropy,
            topk=args.topk,
            enable_validation=args.enable_validation,
            validation_api_key=args.validation_api_key,
            validation_model=args.validation_model,
            enable_v_top_layer=args.enable_v_top_layer,
            max_samples=args.max_samples,
            start_idx=args.start_idx,
            random_seed=args.random_seed,
            shuffle=args.shuffle,
            copy_images=args.copy_images,
            save_every=args.save_every,
        )
        return
    results = []
    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] 处理样本...")
        if torch.cuda.is_available():
            if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
                print(f"[显存状态-处理前] 所有GPU显存使用:")
                for gpu_id in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserv = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"  GPU {gpu_id}: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
            else:
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserv = torch.cuda.memory_reserved() / 1024**3
                print(f"[显存状态-处理前] GPU 0: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
        result = process_sample(
            sample=sample,
            extractor=extractor,
            base_image_dir=args.base_image_dir,
            output_dir=args.output_dir,
            extraction_method=args.method,
            steps=args.steps,
            tau_grad=args.tau_grad,
            skip_markup=args.skip_markup,
            downsample=args.downsample,
            max_tokens_per_step=args.max_tokens_per_step,
            max_grad_calls_per_step=args.max_grad_calls_per_step,
            aggregation_method=args.aggregation_method,
            attribution_threshold=args.attribution_threshold,
            min_entropy=args.min_entropy,
            topk=args.topk,
            enable_validation=args.enable_validation,
            validation_api_key=args.validation_api_key,
            validation_model=args.validation_model,
            enable_v_top_layer=args.enable_v_top_layer,
        )
        results.append(result)
        if torch.cuda.is_available():
            if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        if (i + 1) % 10 == 0:
            checkpoint_file = os.path.join(args.output_dir, "trajectories_checkpoint.json")
            checkpoint_data = {
                "processing_params": processing_params,
                "results": results,
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            print(f"已保存检查点到 {checkpoint_file} ({len(results)} 个样本)")
    print(f"\n正在保存结果到 {args.output_dir}...")
    if args.save_only_correct and args.enable_validation:
        filtered_results = [r for r in results if r.get("validation", {}).get("is_correct", False)]
        print(f"  过滤前: {len(results)} 个样本")
        print(f"  过滤后: {len(filtered_results)} 个样本（仅校验通过的）")
    else:
        filtered_results = results
    if args.save_format == "json":
        output_file = os.path.join(args.output_dir, "trajectories.json")
        output_data = {
            "processing_params": processing_params,
            "results": filtered_results,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(filtered_results)} 个样本到 {output_file}")
    else:
        output_file = os.path.join(args.output_dir, "trajectories.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"type": "metadata", "processing_params": processing_params}, ensure_ascii=False) + '\n')
            for result in filtered_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"已保存 {len(filtered_results)} 个样本到 {output_file}")
    successful = sum(1 for r in results if "error" not in r and r.get("skipped", False) == False)
    failed = sum(1 for r in results if "error" in r)
    skipped = sum(1 for r in results if r.get("skipped", False) == True)
    total = len(results)
    print(f"\n处理完成！")
    print(f"  成功（已保存轨迹）: {successful}")
    print(f"  跳过（答案校验未通过）: {skipped}")
    print(f"  失败（处理错误）: {failed}")
    print(f"  总计: {total}")
    if args.enable_validation:
        validated_correct = sum(1 for r in results if r.get("validation", {}).get("is_correct", False))
        validated_incorrect = sum(1 for r in results if "validation" in r and not r.get("validation", {}).get("is_correct", True))
        print(f"\n校验统计:")
        print(f"  校验通过: {validated_correct}")
        print(f"  校验未通过: {validated_incorrect}")
if __name__ == "__main__":
    main()