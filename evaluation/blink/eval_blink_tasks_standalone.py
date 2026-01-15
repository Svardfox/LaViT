import argparse
import os
import json
import sys
from tqdm import tqdm
from io import BytesIO
from typing import Optional, Dict, List
import pandas as pd
from PIL import Image
import torch
from transformers import Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration
import re
def extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r'<\|im_end\|>$', '', str(text)).strip()
    matches = re.findall(r'\(([A-Ha-h])\)', text)
    if matches:
        return matches[-1].lower()
    conclusion_patterns = [
        r'[Aa]nswer\s*:?\s*([A-Ha-h])\b',
        r'[Cc]hoice\s*:?\s*([A-Ha-h])\b',
        r'[Cc]onclusion\s*:?\s*([A-Ha-h])\b',
        r'[Ff]inal\s+answer\s*:?\s*([A-Ha-h])\b',
        r'[Tt]herefore.*?\bis\s+([A-Ha-h])\b',
        r'[Tt]he\s+correct\s+answer\s+is\s+([A-Ha-h])\b',
        r'([A-Ha-h])\s+is\s+the\s+correct\s+answer'
    ]
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    matches = re.findall(r'\b([A-Ha-h])\b', text)
    if matches:
        filtered = [m for m in matches if m.upper() in 'ABCDEFGH' and m != 'a']
        if filtered:
            return filtered[-1].lower()
        if matches[-1].lower() == 'a':
            return 'a'
    if len(text) < 10:
        match = re.search(r'([A-Ha-h])', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None
def check_answer_correct(predicted_answer: str, ground_truth: str) -> bool:
    pred_letter = extract_answer_letter(predicted_answer)
    gt_letter = extract_answer_letter(ground_truth)
    if pred_letter is None or gt_letter is None:
        return False
    return pred_letter == gt_letter
class BaseQwenEvaluator:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        print("Model loaded successfully.")
    def generate(self, image_path, question: str, max_new_tokens: int = 512):
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
def load_blink_task_data(data_root: str, task_name: str):
    parquet_path = os.path.join(data_root, task_name, "val-00000-of-00001.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(data_root, task_name, "test-00000-of-00001.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"BLINK {task_name} data not found. "
            f"Expected at: {os.path.join(data_root, task_name, 'val-00000-of-00001.parquet')} "
            f"or {os.path.join(data_root, task_name, 'test-00000-of-00001.parquet')}"
        )
    print(f"Loading BLINK {task_name} from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    data = []
    for idx, row in df.iterrows():
        image_objs = []
        for img_key in ['image_1', 'image_2', 'image_3', 'image_4']:
            if img_key in row and row[img_key] is not None:
                image_data = row[img_key]
                try:
                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        image_obj = Image.open(BytesIO(image_data['bytes'])).convert("RGB")
                        image_objs.append(image_obj)
                    elif isinstance(image_data, bytes):
                        image_obj = Image.open(BytesIO(image_data)).convert("RGB")
                        image_objs.append(image_obj)
                except Exception as e:
                    continue
        if len(image_objs) == 0:
            continue
        image_path = image_objs if len(image_objs) > 1 else image_objs[0]
        if 'prompt' in row and row['prompt']:
            prompt = row['prompt']
        else:
            choices = row.get('choices', [])
            prompt = row.get('question', '')
            if len(choices) > 0:
                prompt += "\nOptions:"
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for i, choice in enumerate(choices):
                    label = labels[i] if i < len(labels) else str(i)
                    prompt += f"\n({label}) {choice}"
                prompt += "\nAnswer with the option letter from the given choices directly."
        answer_text = row.get('answer', "")
        ground_truth = answer_text
        if answer_text.startswith('(') and answer_text.endswith(')'):
            ground_truth = answer_text[1:-1]
        elif answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            ground_truth = answer_text
        else:
            choices = row.get('choices', [])
            if answer_text in choices:
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                idx_ans = list(choices).index(answer_text)
                if idx_ans < len(labels):
                    ground_truth = labels[idx_ans]
        choices_list = row.get('choices', [])
        if hasattr(choices_list, 'tolist'):
            choices_list = choices_list.tolist()
        data.append({
            "id": row.get('idx', f"{task_name}_{idx}"),
            "image_path": image_path,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "meta": {"task": task_name, "choices": choices_list, "num_images": len(image_objs)}
        })
    print(f"Loaded {len(data)} samples.")
    return data
def evaluate_task(
    task_name: str,
    data_root: str,
    evaluator: BaseQwenEvaluator,
    output_file: str,
    max_samples: int = None
) -> Dict:
    print(f"\n{'=' * 60}")
    print(f"Task: {task_name}")
    print(f"{'=' * 60}")
    data = load_blink_task_data(data_root, task_name)
    if max_samples:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples for debugging.")
    correct_count = 0
    total_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for sample in tqdm(data, desc=f"Evaluating {task_name}"):
            output_text = evaluator.generate(
                sample['image_path'],
                sample['prompt']
            )
            if output_text is None:
                print(f"Skipping sample {sample['id']} due to generation error.")
                continue
            is_correct = check_answer_correct(output_text, sample['ground_truth'])
            if is_correct:
                correct_count += 1
            total_count += 1
            result = {
                "id": sample['id'],
                "prompt": sample['prompt'],
                "ground_truth": sample['ground_truth'],
                "model_output": output_text,
                "is_correct": is_correct,
                "meta": sample.get('meta', {})
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"\n{task_name} Results:")
    print(f"  Total samples: {total_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Wrong: {total_count - correct_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return {
        "task": task_name,
        "total": total_count,
        "correct": correct_count,
        "wrong": total_count - correct_count,
        "accuracy": accuracy,
        "output_file": output_file
    }
def evaluate_multiple_tasks(
    data_root: str,
    model_path: str,
    tasks: List[str],
    output_dir: str = None,
    max_samples: int = None,
    device: str = None
):
    print("=" * 60)
    print("Initializing model...")
    print("=" * 60)
    evaluator = BaseQwenEvaluator(model_path, device=device)
    if output_dir is None:
        model_name = os.path.basename(model_path.rstrip('/'))
        output_dir = f"eval_results_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for task_name in tasks:
        model_name = os.path.basename(model_path.rstrip('/'))
        output_file = os.path.join(output_dir, f"{task_name}_{model_name}.jsonl")
        try:
            result = evaluate_task(
                task_name=task_name,
                data_root=data_root,
                evaluator=evaluator,
                output_file=output_file,
                max_samples=max_samples
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "task": task_name,
                "error": str(e)
            })
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Task':<25} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 60)
    for result in all_results:
        if "error" in result:
            print(f"{result['task']:<25} {'ERROR':<8}")
        else:
            print(f"{result['task']:<25} {result['total']:<8} {result['correct']:<8} {result['accuracy']:.2f}%")
    print("=" * 60)
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")
    return all_results
def main():
    parser = argparse.ArgumentParser(
        description="独立的 BLINK 多任务评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="BLINK 数据集根目录"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径（Qwen2.5-VL 模型）"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        required=True,
        help="要评估的任务列表（如: Relative_Depth IQ_Test）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出结果目录。如果不指定，将使用默认名称。"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（用于调试，可选）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu）。如果不指定，将自动检测。"
    )
    args = parser.parse_args()
    if not os.path.exists(args.data_root):
        print(f"❌ 错误: 数据集根目录不存在: {args.data_root}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)
    try:
        results = evaluate_multiple_tasks(
            data_root=args.data_root,
            model_path=args.model_path,
            tasks=args.tasks,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            device=args.device
        )
        print("\n✅ 所有评估成功完成！")
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()