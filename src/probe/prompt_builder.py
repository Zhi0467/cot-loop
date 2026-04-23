from __future__ import annotations

from .prompt_format import format_user_prompt


def build_math_prompt(
    tokenizer,
    question: str,
    *,
    prompt_format: str = "auto",
    thinking_mode: str = "default",
) -> str:
    user_msg = f"{question}\n\nYou must put your final answer within \\boxed{{}}."
    return format_user_prompt(
        tokenizer,
        user_msg,
        prompt_format=prompt_format,
        thinking_mode=thinking_mode,
    )
