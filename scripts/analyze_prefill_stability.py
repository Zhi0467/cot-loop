#!/usr/bin/env python3
"""Analyze prefill activation stability with repeated forward passes."""

import argparse
import contextlib
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import build_prompt, load_jsonl

ROLLOUTS = 10


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", default="data/aime_2024_2025.jsonl")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-plot", required=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    if not data:
        raise SystemExit("Dataset is empty.")

    dtype = select_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    model = load_model(args.model_id, dtype, args.trust_remote_code)
    model.eval()

    device = model.device
    sum_avg: List[float] = []
    sum_min: List[float] = []
    num_prompts = 0

    for row in data:
        prompt = build_prompt(tokenizer, row["question"], 1)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt, device)
        avg_cos, min_cos = compute_prompt_similarity(
            model, input_ids, attention_mask, ROLLOUTS
        )
        if not sum_avg:
            sum_avg = [0.0 for _ in avg_cos]
            sum_min = [0.0 for _ in min_cos]
        if len(sum_avg) != len(avg_cos):
            raise RuntimeError("Layer count mismatch across prompts.")
        for idx in range(len(avg_cos)):
            sum_avg[idx] += avg_cos[idx]
            sum_min[idx] += min_cos[idx]
        num_prompts += 1

    if num_prompts == 0:
        raise SystemExit("No prompts processed.")

    avg_by_layer = [val / num_prompts for val in sum_avg]
    min_by_layer = [val / num_prompts for val in sum_min]

    write_csv(args.out_csv, avg_by_layer, min_by_layer)
    save_plot(args.out_plot, avg_by_layer, min_by_layer)


if __name__ == "__main__":
    main()
