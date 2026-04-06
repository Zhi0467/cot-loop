from __future__ import annotations


VALID_PROMPT_FORMATS = ("auto", "raw", "chat_template")


def tokenizer_has_chat_template(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and bool(template.strip())


def resolve_prompt_format(tokenizer, prompt_format: str = "auto") -> str:
    if prompt_format not in VALID_PROMPT_FORMATS:
        raise ValueError(
            f"Unknown prompt format {prompt_format!r}. Valid choices: {VALID_PROMPT_FORMATS}."
        )
    if prompt_format == "auto":
        return "chat_template" if tokenizer_has_chat_template(tokenizer) else "raw"
    if prompt_format == "chat_template" and not tokenizer_has_chat_template(tokenizer):
        raise ValueError(
            "Requested prompt_format='chat_template', but tokenizer.chat_template is not set."
        )
    return prompt_format


def format_user_prompt(tokenizer, user_msg: str, prompt_format: str = "auto") -> str:
    resolved = resolve_prompt_format(tokenizer, prompt_format)
    if resolved == "raw":
        return user_msg
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
