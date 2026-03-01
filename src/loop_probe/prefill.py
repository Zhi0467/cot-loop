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
    batch_size: int = 1,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    total = len(records)
    if total == 0:
        raise SystemExit(f"No records found for split '{log_prefix}'.")
    if batch_size < 1:
        raise SystemExit("--prefill-batch-size must be >= 1.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise SystemExit(
                "Tokenizer has no pad_token/eos_token; cannot batch prefill prompts."
            )
        tokenizer.pad_token = tokenizer.eos_token

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_records = records[start:end]
            encoded = tokenizer(
                [rec.prompt for rec in batch_records],
                return_tensors="pt",
                padding=True,
            )
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

            hidden = out.hidden_states[-1]
            if attention_mask is None:
                last_token_idx = torch.full(
                    (hidden.size(0),),
                    hidden.size(1) - 1,
                    device=hidden.device,
                    dtype=torch.long,
                )
            else:
                token_positions = torch.arange(hidden.size(1), device=hidden.device)
                token_positions = token_positions.unsqueeze(0).expand(hidden.size(0), -1)
                masked_positions = token_positions.masked_fill(attention_mask == 0, -1)
                last_token_idx = masked_positions.max(dim=1).values
                if torch.any(last_token_idx < 0):
                    raise RuntimeError("Found an empty prompt after tokenization.")

            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            batch_vecs = hidden[batch_idx, last_token_idx].float().cpu()
            features.extend(batch_vecs.unbind(dim=0))

            if end == total or start == 0 or end % 50 == 0:
                print(f"[{log_prefix}] prefill {end}/{total}", flush=True)

    return torch.stack(features, dim=0)
