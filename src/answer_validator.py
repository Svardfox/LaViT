import re
from typing import Optional, Tuple
from openai import OpenAI
def extract_final_answer(teacher_response: str) -> Optional[str]:
    final_answer_marker = re.search(
        r"FINAL\s+ANSWER\s*:?\s*\n?", 
        teacher_response, 
        re.IGNORECASE
    )
    if final_answer_marker:
        start_pos = final_answer_marker.end()
        remaining_text = teacher_response[start_pos:].strip()
        if remaining_text:
            lines = remaining_text.split('\n')
            answer_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('<step') and not line.startswith('REASONING'):
                    answer_lines.append(line)
                    if line.startswith('---') or line.startswith('==='):
                        break
            if answer_lines:
                answer = ' '.join(answer_lines).strip()
                answer = re.sub(r"^\s*\[|\]\s*$", "", answer)
                answer = re.sub(r"^\"|\"$", "", answer)
                answer = answer.strip()
                if answer:
                    return answer
    lines = teacher_response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('<step') and not line.startswith('REASONING') and 'FINAL' not in line.upper():
            return line
    return None
def validate_answer_with_llm(
    predicted_answer: str,
    ground_truth: str,
    question: Optional[str] = None,
    api_key: str = "sk-ssi02jvclw4pw34b7wkxngtmkwt0g5x6b107wok1f4p0khnc",
    model: str = "mimo-v2-flash",
    base_url: str = "https://api.xiaomimimo.com/v1",
) -> Tuple[bool, str]:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    prompt_parts = []
    if question:
        prompt_parts.extend([
            "Question:",
            question,
            "",
        ])
    prompt_parts.extend([
        "Ground Truth Answer:",
        ground_truth,
        "",
        "Teacher's Response:",
        predicted_answer,
        "",
        "Please carefully compare the teacher's response with the ground truth answer and determine if they express the same meaning or provide the same result.",
        "If the teacher's response contains reasoning steps, focus on the final answer or conclusion.",
        "Consider the following:",
        "- Do both answers address the same question?",
        "- Do they convey the same information or conclusion?",
        "- Are they semantically equivalent (even if worded differently)?",
        "- If the teacher's response contains multiple parts, does the final answer or conclusion match the ground truth?",
        "",
        "Your response must start with either 'YES' (if the answers are consistent) or 'NO' (if they are inconsistent),",
        "followed by a brief explanation of your judgment.",
    ])
    prompt = "\n".join(prompt_parts)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
        )
        response_text = completion.choices[0].message.content.strip()
        response_upper = response_text.upper()
        is_correct = response_upper.startswith("YES")
        return is_correct, response_text
    except Exception as e:
        error_msg = f"API call failed: {str(e)}"
        print(f"[WARNING] {error_msg}")
        return False, error_msg
def validate_sample(
    teacher_response: str,
    ground_truth: str,
    question: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    final_answer = extract_final_answer(teacher_response)
    if final_answer is None:
        print("[WARNING] Failed to extract FINAL ANSWER, using full teacher response for validation")
        predicted_answer = teacher_response
        final_answer = None
    else:
        predicted_answer = final_answer
    if api_key is None:
        api_key = "sk-ssi02jvclw4pw34b7wkxngtmkwt0g5x6b107wok1f4p0khnc"
    if model is None:
        model = "mimo-v2-flash"
    if base_url is None:
        base_url = "https://api.xiaomimimo.com/v1"
    is_correct, explanation = validate_answer_with_llm(
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        question=question,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    return is_correct, final_answer, explanation
if __name__ == "__main__":
    test_ground_truth = "The person on the left has short hair."
    is_valid, final_answer, explanation = validate_sample(
        teacher_response=test_response,
        ground_truth=test_ground_truth,
        question="What is the hair style of the person on the left?",
    )
    print(f"Final Answer: {final_answer}")
    print(f"Is Correct: {is_valid}")
    print(f"Explanation: {explanation}")