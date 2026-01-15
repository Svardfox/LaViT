from typing import List, Optional
def build_lavit_messages(
    image_path: str,
    question: str,
    system_message: Optional[str] = None
) -> List[dict]:
    messages = []
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": question
            },
        ],
    })
    return messages
def build_lavit_teacher_generation_messages(
    image_path: str,
    question: str,
    system_message: Optional[str] = None
) -> List[dict]:
    return build_lavit_messages(image_path, question, system_message)
def build_lavit_stepwise_messages(
    image_path: str,
    question: str,
    previous_steps: List[str],
    system_message: Optional[str] = None
) -> List[dict]:
    messages = []
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": question
            },
        ],
    })
    if previous_steps:
        assistant_response = "\n".join(previous_steps)
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
    return messages
def apply_lavit_chat_template(
    processor,
    messages: List[dict],
    add_generation_prompt: bool = True
) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )
def build_prompt_with_chat_template(
    processor,
    question: str,
    steps_prefix: List[str],
    force_T: int = None,
    image_path: str = None
) -> str:
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    })
    if steps_prefix:
        messages.append({
            "role": "assistant",
            "content": "\n".join(steps_prefix)
        })
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(not steps_prefix)
    )
def build_teacher_generation_prompt(
    processor,
    question: str,
    force_T: int = None,
    image_path: str = None
) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
build_lvr_messages = build_lavit_messages
build_lvr_teacher_generation_messages = build_lavit_teacher_generation_messages
build_lvr_stepwise_messages = build_lavit_stepwise_messages
apply_lvr_chat_template = apply_lavit_chat_template