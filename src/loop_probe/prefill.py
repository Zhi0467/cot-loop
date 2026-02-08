import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .types import SampleRecord


def select_prefill_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _sdp_kernel_context():
    if torch.cuda.is_available():
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        )
    return contextlib.nullcontext()


def load_prefill_model_and_tokenizer(model_id: str, trust_remote_code: bool):
    dtype = select_prefill_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            **kwargs,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def extract_prefill_features(
    model,
    tokenizer,
    device: torch.device,
    records: list[SampleRecord],
    *,
    log_prefix: str,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    total = len(records)
    if total == 0:
        raise SystemExit(f"No records found for split '{log_prefix}'.")

    with torch.inference_mode():
        for idx, rec in enumerate(records, start=1):
            encoded = tokenizer(rec.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with _sdp_kernel_context():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            if out.hidden_states is None:
                raise RuntimeError("Model did not return hidden states during prefill.")

            vec = out.hidden_states[-1][:, -1, :].squeeze(0).float().cpu()
            features.append(vec)

            if idx == 1 or idx == total or idx % 50 == 0:
                print(f"[{log_prefix}] prefill {idx}/{total}", flush=True)

    return torch.stack(features, dim=0)
