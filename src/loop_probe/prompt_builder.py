from __future__ import annotations


def build_math_prompt(tokenizer, question: str) -> str:
    user_msg = f"{question}\n\nYou must put your final answer within \\boxed{{}}."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
