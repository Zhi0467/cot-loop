from __future__ import annotations


VALID_PROMPT_FORMATS = ("auto", "raw", "chat_template")
VALID_THINKING_MODES = ("default", "on", "off")


def tokenizer_has_chat_template(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and bool(template.strip())


def tokenizer_supports_enable_thinking(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "enable_thinking" in template


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


def resolve_thinking_mode(
    tokenizer,
    *,
    prompt_format: str = "auto",
    thinking_mode: str = "default",
) -> str | None:
    if thinking_mode not in VALID_THINKING_MODES:
        raise ValueError(
            f"Unknown thinking_mode {thinking_mode!r}. Valid choices: {VALID_THINKING_MODES}."
        )
    resolved_prompt_format = resolve_prompt_format(tokenizer, prompt_format)
    if resolved_prompt_format != "chat_template":
        if thinking_mode != "default":
            raise ValueError(
                "thinking_mode requires prompt_format to resolve to chat_template."
            )
        return None
    if thinking_mode == "default":
        return "on" if tokenizer_supports_enable_thinking(tokenizer) else "default"
    if thinking_mode == "off" and not tokenizer_supports_enable_thinking(tokenizer):
        raise ValueError(
            "Requested thinking_mode='off', but tokenizer chat_template does not "
            "support enable_thinking."
        )
    return thinking_mode


def build_chat_template_prompt(
    tokenizer,
    messages,
    *,
    thinking_mode: str = "default",
) -> str:
    resolved_thinking_mode = resolve_thinking_mode(
        tokenizer,
        prompt_format="chat_template",
        thinking_mode=thinking_mode,
    )
    kwargs: dict[str, object] = {}
    if resolved_thinking_mode == "off":
        kwargs["enable_thinking"] = False
    elif (
        thinking_mode == "on"
        and tokenizer_supports_enable_thinking(tokenizer)
    ):
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def format_user_prompt(
    tokenizer,
    user_msg: str,
    prompt_format: str = "auto",
    *,
    thinking_mode: str = "default",
) -> str:
    resolved = resolve_prompt_format(tokenizer, prompt_format)
    if resolved == "raw":
        if thinking_mode != "default":
            raise ValueError(
                "thinking_mode requires prompt_format to resolve to chat_template."
            )
        return user_msg
    return build_chat_template_prompt(
        tokenizer,
        [{"role": "user", "content": user_msg}],
        thinking_mode=thinking_mode,
    )
