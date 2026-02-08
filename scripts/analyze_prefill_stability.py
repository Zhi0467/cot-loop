#!/usr/bin/env python3
"""Analyze prefill activation stability and optional vLLM rollout loop stats."""

import argparse
import contextlib
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import build_prompt, has_ngram_loop, load_jsonl

ROLLOUTS = 10
LOOP_N = 30
LOOP_K = 20


def select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_model(model_id: str, dtype: torch.dtype, trust_remote_code: bool):
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": trust_remote_code,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            **kwargs,
        )


def sdp_kernel_context():
    if torch.cuda.is_available():
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        )
    return contextlib.nullcontext()


def tokenize_prompt(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def compute_prompt_similarity(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rollouts: int,
) -> Tuple[List[float], List[float]]:
    layer_vectors: List[List[torch.Tensor]] = []
    with torch.inference_mode():
        for _ in range(rollouts):
            with sdp_kernel_context():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            if not layer_vectors:
                layer_vectors = [[] for _ in range(len(hidden_states) - 1)]
            if len(hidden_states) - 1 != len(layer_vectors):
                raise RuntimeError("Hidden state layer count changed across rollouts.")
            for idx, layer_state in enumerate(hidden_states[1:]):
                vec = layer_state[:, -1, :].float().squeeze(0)
                layer_vectors[idx].append(vec)

    avg_cos = []
    min_cos = []
    for vectors in layer_vectors:
        stacked = torch.stack(vectors, dim=0)
        stacked = F.normalize(stacked, dim=-1)
        cosine = stacked @ stacked.T
        idx = torch.triu_indices(cosine.size(0), cosine.size(1), offset=1)
        upper = cosine[idx[0], idx[1]]
        if upper.numel() == 0:
            avg_cos.append(1.0)
            min_cos.append(1.0)
        else:
            avg_cos.append(upper.mean().item())
            min_cos.append(upper.min().item())
    return avg_cos, min_cos


def write_csv(path: str, avg_cos: List[float], min_cos: List[float]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_idx", "avg_pairwise_cos", "min_pairwise_cos"])
        for idx, (avg_val, min_val) in enumerate(zip(avg_cos, min_cos), start=1):
            writer.writerow([idx, avg_val, min_val])


def write_rollout_csv(
    path: str,
    rows: List[Dict[str, object]],
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_id",
                "temperature",
                "num_prompts",
                "rollouts_per_prompt",
                "num_samples",
                "looped",
                "not_looped",
                "loop_fraction",
                "max_tokens",
                "seed",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model_id"],
                    row["temperature"],
                    row["num_prompts"],
                    row["rollouts_per_prompt"],
                    row["num_samples"],
                    row["looped"],
                    row["not_looped"],
                    row["loop_fraction"],
                    row["max_tokens"],
                    row["seed"],
                ]
            )


def save_plot(path: str, avg_cos: List[float], min_cos: List[float]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    layers = list(range(1, len(avg_cos) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, avg_cos, marker="o", label="avg pairwise cosine")
    ax.plot(layers, min_cos, marker="o", label="min pairwise cosine")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Prefill activation stability (temp=0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_vllm_rollouts(
    tokenizer,
    model_id: str,
    prompts: List[str],
    trust_remote_code: bool,
    rollouts: int,
    max_tokens: int,
    tp: int,
    dtype: str,
    max_model_len: int,
    max_num_seqs: Optional[int],
    max_num_batched_tokens: Optional[int],
    seed: int,
) -> List[Dict[str, object]]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(
            "vLLM is required for rollout generation. "
            "Install it or skip --out-rollout-csv."
        ) from exc

    gen_config = GenerationConfig.from_pretrained(model_id)
    top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    top_k = gen_config.top_k if gen_config.top_k is not None else -1
    if top_k == 0:
        top_k = -1

    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tp,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "trust_remote_code": trust_remote_code,
    }
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    llm = LLM(**llm_kwargs)
    if not prompts:
        raise RuntimeError("No prompts were provided for vLLM rollouts.")

    looped = 0
    for rollout_idx in range(rollouts):
        # Keep greedy settings fixed at temperature=0 (purpose of this script).
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=1,
            repetition_penalty=1.0,
            seed=seed + rollout_idx,
        )
        outputs = llm.generate(prompts, sampling_params)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Expected {len(prompts)} outputs, but got {len(outputs)}."
            )

        for out in outputs:
            if len(out.outputs) != 1:
                raise RuntimeError(
                    f"Expected 1 sample per prompt, but got {len(out.outputs)}."
                )
            sample = out.outputs[0]
            token_ids = getattr(sample, "token_ids", None)
            if not token_ids:
                token_ids = tokenizer.encode(sample.text, add_special_tokens=False)
            if has_ngram_loop(token_ids, n=LOOP_N, k=LOOP_K):
                looped += 1

    num_samples = len(prompts) * rollouts
    not_looped = num_samples - looped
    return [
        {
            "model_id": model_id,
            "temperature": 0.0,
            "num_prompts": len(prompts),
            "rollouts_per_prompt": rollouts,
            "num_samples": num_samples,
            "looped": looped,
            "not_looped": not_looped,
            "loop_fraction": (looped / num_samples) if num_samples else 0.0,
            "max_tokens": max_tokens,
            "seed": seed,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", default="data/aime_2024_2025.jsonl")
    parser.add_argument("--out-rollout-csv", required=True)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    if args.rollouts < 1:
        raise SystemExit("--rollouts must be >= 1.")
    data = load_jsonl(args.data)
    if not data:
        raise SystemExit("Dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    prompts = [build_prompt(tokenizer, row["question"], 1) for row in data]
    rollout_rows = run_vllm_rollouts(
        tokenizer=tokenizer,
        model_id=args.model_id,
        prompts=prompts,
        trust_remote_code=args.trust_remote_code,
        rollouts=args.rollouts,
        max_tokens=args.max_tokens,
        tp=args.tp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
    )
    write_rollout_csv(
        path=args.out_rollout_csv,
        rows=rollout_rows,
    )


if __name__ == "__main__":
    main()
