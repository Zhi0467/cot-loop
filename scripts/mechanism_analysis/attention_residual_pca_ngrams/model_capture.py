from __future__ import annotations

from types import MethodType

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .records import (
    VECTOR_ATTENTION_WRITE,
    VECTOR_FINAL_HIDDEN,
    VECTOR_POST_ATTENTION_RESIDUAL,
    CapturedVectors,
    SelectedRollout,
)


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no CUDA device is available.")
    if device.index is not None:
        return device
    return torch.device("cuda:0")


def load_model(model_id: str, *, device: torch.device):
    kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
    }
    implementations = (
        ["flash_attention_2", "sdpa", "eager"]
        if device.type == "cuda"
        else ["eager"]
    )
    last_error: Exception | None = None
    for attn_implementation in implementations:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - host/model dependent
            last_error = exc
    assert last_error is not None
    raise last_error


def analysis_input_ids(row: SelectedRollout) -> list[int]:
    max_start = max(row.rescan_ngram_start_positions)
    return row.prompt_token_ids + row.completion_token_ids[:max_start]


def capture_vectors(
    model,
    row: SelectedRollout,
    *,
    device: torch.device,
    include_final_hidden: bool,
) -> CapturedVectors:
    input_ids = analysis_input_ids(row)
    if len(input_ids) != row.replay_token_count:
        raise RuntimeError(
            f"Replay length mismatch for {row.selection_id}: "
            f"{len(input_ids)} != {row.replay_token_count}"
        )
    max_boundary = max(row.boundary_positions)
    if max_boundary >= len(input_ids):
        raise RuntimeError(
            f"Boundary {max_boundary} is outside replay prefix length {len(input_ids)} "
            f"for {row.selection_id}."
        )

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    boundary_tensor = torch.tensor(row.boundary_positions, dtype=torch.long, device=device)
    captured: dict[str, torch.Tensor] = {}
    final_layer = model.model.layers[-1]
    original_forward = final_layer.forward

    def _wrapped_forward(
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)
        attn_output, _attn_weights = layer.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        post_attention = residual + attn_output
        mlp_residual = post_attention
        mlp_input = layer.post_attention_layernorm(post_attention)
        mlp_output = layer.mlp(mlp_input)
        final_hidden = mlp_residual + mlp_output

        captured[VECTOR_ATTENTION_WRITE] = (
            attn_output[0, boundary_tensor, :].detach().float().cpu()
        )
        captured[VECTOR_POST_ATTENTION_RESIDUAL] = (
            post_attention[0, boundary_tensor, :].detach().float().cpu()
        )
        if include_final_hidden:
            captured[VECTOR_FINAL_HIDDEN] = (
                final_hidden[0, boundary_tensor, :].detach().float().cpu()
            )
        return final_hidden

    final_layer.forward = MethodType(_wrapped_forward, final_layer)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=boundary_tensor,
            )
    finally:
        final_layer.forward = original_forward

    missing = [
        name
        for name in (VECTOR_ATTENTION_WRITE, VECTOR_POST_ATTENTION_RESIDUAL)
        if name not in captured
    ]
    if missing:
        raise RuntimeError(f"Did not capture vectors {missing} for {row.selection_id}.")

    return _captured_vectors_from_logits(
        logits=outputs.logits[0].detach().float().cpu(),
        captured=captured,
        repeat_token_id=int(row.ngram_token_ids[0]),
    )


def _captured_vectors_from_logits(
    *,
    logits: torch.Tensor,
    captured: dict[str, torch.Tensor],
    repeat_token_id: int,
) -> CapturedVectors:
    if repeat_token_id < 0 or repeat_token_id >= logits.shape[-1]:
        raise RuntimeError(
            f"Repeat token id {repeat_token_id} is outside vocab size {logits.shape[-1]}."
        )
    repeat_token_logits_tensor = logits[:, repeat_token_id]
    probabilities = torch.softmax(logits, dim=-1)[:, repeat_token_id]
    top_k = torch.topk(logits, k=2, dim=-1)
    top_ids = top_k.indices[:, 0]
    top_logits = top_k.values[:, 0]
    second_logits = top_k.values[:, 1]
    other_max = torch.where(top_ids == repeat_token_id, second_logits, top_logits)
    margins = repeat_token_logits_tensor - other_max
    return CapturedVectors(
        vectors={name: _to_numpy(tensor) for name, tensor in captured.items()},
        repeat_probabilities=[float(v) for v in probabilities.tolist()],
        repeat_logit_margins=[float(v) for v in margins.tolist()],
        repeat_token_logits=[float(v) for v in repeat_token_logits_tensor.tolist()],
        top_token_ids=[int(v) for v in top_ids.tolist()],
        top_token_logits=[float(v) for v in top_logits.tolist()],
    )


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()
