import re
import json
from typing import Optional, Tuple, List, Dict
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
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^([A-Ha-h])[\.\:\)]', line)
        if match:
            pass
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
def check_answer_correct(
    predicted_answer: str,
    ground_truth: str,
    method: str = "regex"
) -> bool:
    if method == "regex":
        pred_letter = extract_answer_letter(predicted_answer)
        gt_letter = extract_answer_letter(ground_truth)
        if pred_letter is None or gt_letter is None:
            return False
        return pred_letter == gt_letter
    else:
        raise ValueError(f"Unsupported method: {method}")
def calculate_accuracy(
    results_file: str,
    method: str = "regex"
) -> Dict:
    total = 0
    correct = 0
    failed_extraction = 0
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            pred = data.get('model_output', '')
            gt = data.get('ground_truth', '')
            if check_answer_correct(pred, gt, method=method):
                correct += 1
            else:
                pred_letter = extract_answer_letter(pred)
                gt_letter = extract_answer_letter(gt)
                if pred_letter is None or gt_letter is None:
                    failed_extraction += 1
    accuracy = correct / total * 100 if total > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "failed_extraction": failed_extraction
    }
def analyze_results(
    results_file: str,
    method: str = "regex",
    show_examples: int = 5
) -> None:
    stats = calculate_accuracy(results_file, method=method)
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total Samples: {stats['total']}")
    print(f"Correct: {stats['correct']}")
    print(f"Wrong: {stats['wrong']}")
    print(f"Accuracy: {stats['accuracy']:.2f}%")
    if stats['failed_extraction'] > 0:
        print(f"Failed Extraction: {stats['failed_extraction']}")
    print("=" * 60)
    correct_examples = []
    wrong_examples = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pred = data.get('model_output', '')
            gt = data.get('ground_truth', '')
            is_correct = check_answer_correct(pred, gt, method=method)
            example = {
                'id': data.get('id', ''),
                'prompt': data.get('prompt', '').split('\n')[0],
                'gt': gt,
                'pred': pred[:100] + '...' if len(pred) > 100 else pred
            }
            if is_correct:
                if len(correct_examples) < show_examples:
                    correct_examples.append(example)
            else:
                if len(wrong_examples) < show_examples:
                    wrong_examples.append(example)
            if len(correct_examples) >= show_examples and len(wrong_examples) >= show_examples:
                break
    if correct_examples:
        print(f"\nCorrect Examples (top {len(correct_examples)}):")
        for i, ex in enumerate(correct_examples, 1):
            print(f"\n  {i}. ID: {ex['id']}")
            print(f"     Question: {ex['prompt']}")
            print(f"     GT: {ex['gt']} | Pred: {ex['pred']}")
    if wrong_examples:
        print(f"\nWrong Examples (top {len(wrong_examples)}):")
        for i, ex in enumerate(wrong_examples, 1):
            print(f"\n  {i}. ID: {ex['id']}")
            print(f"     Question: {ex['prompt']}")
            print(f"     GT: {ex['gt']} | Pred: {ex['pred']}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate evaluation accuracy")
    parser.add_argument("--results_file", type=str, required=True, help="Path to evaluation results JSONL file")
    parser.add_argument("--method", type=str, default="regex", help="Evaluation method (default: regex)")
    parser.add_argument("--show_examples", type=int, default=5, help="Number of examples to show")
    args = parser.parse_args()
    analyze_results(args.results_file, method=args.method, show_examples=args.show_examples)