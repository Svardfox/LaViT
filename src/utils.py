from typing import List, Dict, Any
import pdb
import torch
def build_prompt_with_chat_template(
    processor,
    question: str, 
    steps_prefix: List[str], 
    force_T: int
) -> str:

    test_instruction = (
        "Please carefully observe the provided image and answer the question accordingly."
        f"You must solve the question whatever it is in step by step in exactly {force_T} reasoning steps. Follow this EXACT format:\n\n"
        "REASONING:\n" +
        ("\n".join(["<step>analyze one aspect of the problem</step>" for i in range(force_T)])) +
        "\n\nFINAL ANSWER:\n\n\n"
         "IMPORTANT RULES:\n"
        "- Each <step> tag must contain only ONE specific analysis or observation\n"
        "- Do NOT include any reasoning outside the <step> tags\n"
        "- After all steps, provide ONLY the action sequence in the final answer\n"
        "- Do not explain your final answer - just give the moves"
    )
    messages = [
        {
            "role": "system",
            "content": test_instruction
        },
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}"}
            ]
        }
    ]
    if steps_prefix:
        assistant_response = "\n".join(steps_prefix)
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })  
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    if steps_prefix:
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"][0]
        last_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == 151645:
                last_idx = i
                break
        if last_idx != -1:
            new_input_ids = torch.cat([input_ids[:last_idx], input_ids[last_idx+1:]])
            inputs["input_ids"] = new_input_ids.unsqueeze(0)
        prompt = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return prompt
def build_teacher_generation_prompt(processor, question: str, force_T: int) -> str:

    test_instruction = (
        "Please carefully observe the provided image and answer the question accordingly."
        f"You must solve the question whatever it is in step by step in exactly {force_T} reasoning steps. Follow this EXACT format:\n\n"
        "REASONING:\n" +
        ("\n".join(["<step>analyze one aspect of the problem</step>" for i in range(force_T)])) +
        "\n\nFINAL ANSWER:\n\n\n"
         "IMPORTANT RULES:\n"
        "- Each <step> tag must contain only ONE specific analysis or observation\n"
        "- Do NOT include any reasoning outside the <step> tags\n"
        "- After all steps, provide ONLY the action sequence in the final answer\n"
        "- Do not explain your final answer - just give the moves"
    )
    messages = [
        {
            "role": "system",
            "content": test_instruction
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt