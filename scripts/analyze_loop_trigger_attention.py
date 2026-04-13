#!/usr/bin/env python3
"""Analyze attention patterns at the n-gram loop trigger point.

Loads a saved prompt_rollout_archive JSONL, picks one rollout, re-tokenizes
the completion text, runs the n-gram loop detector (with span tracking),
performs a single Hugging Face teacher-forcing forward pass with
output_attentions=True, and emits:
  - JSON with per-layer bin masses and provenance metadata
  - CSV with one scalar row per layer
  - optional matplotlib heatmap (layers x key positions)

Example:
  python scripts/analyze_loop_trigger_attention.py \\
      --model-id Qwen/Qwen3-1.7B \\
      --data outputs/probe_data/diagnostics/prompt_rollout_archive.jsonl \\
      --line-index 0 --rollout-index 0 \\
      --loop-n 30 --loop-k 20 \\
      --out-json outputs/loop_attn_pilot.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Inline n-gram loop detector with span tracking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopTriggerDetail:
    """Mirrors labeling.first_ngram_loop_prefix_length semantics plus spans."""
    n: int
    k: int
    first_loop_prefix: int
    pattern_start_positions: list[int]

    @property
    def pattern_end_exclusive(self) -> list[int]:
        return [s + self.n for s in self.pattern_start_positions]


def detect_ngram_loop_with_spans(
    token_ids: list[int],
    *,
    n: int = 30,
    k: int = 20,
) -> LoopTriggerDetail | None:
    """Same rolling-hash semantics as labeling.first_ngram_loop_prefix_length,
    but also records the completion-relative start indices of all k occurrences
    of the triggering n-gram."""
    if len(token_ids) < n:
        return None

    base = 1000003
    mask = (1 << 64) - 1
    pow_n = pow(base, n, 1 << 64)

    h = 0
    for t in token_ids[:n]:
        h = (h * base + (t + 1)) & mask

    positions: dict[int, list[int]] = defaultdict(list)
    positions[h].append(0)

    for i in range(n, len(token_ids)):
        out_t = token_ids[i - n] + 1
        in_t = token_ids[i] + 1
        h = (h * base + in_t - (out_t * pow_n)) & mask
        start = i - n + 1
        positions[h].append(start)
        if len(positions[h]) >= k:
            first_loop_prefix = i + 1
            starts = positions[h][-k:]
            return LoopTriggerDetail(
                n=n,
                k=k,
                first_loop_prefix=first_loop_prefix,
                pattern_start_positions=starts,
            )

    return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Attention analysis (torch deferred)
# ---------------------------------------------------------------------------

def bin_attention_masses(
    attn_row,  # torch.Tensor — imported lazily
    *,
    prompt_len: int,
    span_ranges: list[tuple[int, int]],
    seq_len: int,
) -> dict[str, float]:
    """Partition a 1-D attention distribution over key positions into bins.

    Bins:
      prompt        – [0, prompt_len)
      prior_repeats – union of span_ranges (completion-relative, shifted by prompt_len)
      rest          – everything else
    """
    import torch

    prompt_mass = float(attn_row[:prompt_len].sum())

    repeat_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn_row.device)
    for start, end in span_ranges:
        abs_start = prompt_len + start
        abs_end = min(prompt_len + end, seq_len)
        if abs_start < seq_len:
            repeat_mask[abs_start:abs_end] = True
    repeat_mask[:prompt_len] = False

    repeat_mass = float(attn_row[repeat_mask].sum())
    rest_mass = float(attn_row.sum()) - prompt_mass - repeat_mass

    return {
        "prompt_mass": prompt_mass,
        "prior_repeat_mass": repeat_mass,
        "rest_mass": rest_mass,
    }


def run_attention_analysis(
    model,
    tokenizer,
    device,
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    trigger: LoopTriggerDetail,
    query_mode: str,
    max_seq_len: int | None,
) -> dict:
    """Single forward pass; returns per-layer bin masses + raw attention slices."""
    import torch

    completion_truncated = completion_ids[: trigger.first_loop_prefix]
    full_ids = prompt_ids + completion_truncated
    if max_seq_len is not None and len(full_ids) > max_seq_len:
        raise SystemExit(
            f"Sequence length {len(full_ids)} exceeds --max-seq-len {max_seq_len}. "
            "Raise the cap or pick a shorter rollout."
        )

    prompt_len = len(prompt_ids)
    seq_len = len(full_ids)

    if query_mode == "end_of_kth_ngram":
        query_idx = seq_len - 1
    elif query_mode == "start_of_kth_ngram":
        kth_start = trigger.pattern_start_positions[-1]
        query_idx = prompt_len + kth_start
    else:
        raise SystemExit(f"Unknown --query-mode '{query_mode}'.")

    if query_idx < 0 or query_idx >= seq_len:
        raise SystemExit(
            f"Computed query index {query_idx} is out of bounds for "
            f"sequence length {seq_len}."
        )

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    attentions = outputs.attentions
    if not attentions:
        raise SystemExit(
            "outputs.attentions is empty or None. The chosen attention backend "
            "likely does not materialize attention weights. Try a different "
            "--attn-implementation (e.g. 'eager')."
        )

    prior_spans = [
        (s, s + trigger.n) for s in trigger.pattern_start_positions[:-1]
    ]
    kth_span = (
        trigger.pattern_start_positions[-1],
        trigger.pattern_start_positions[-1] + trigger.n,
    )

    num_layers = len(attentions)
    per_layer: list[dict] = []
    attn_heatmap_rows: list[list[float]] = []

    for layer_idx, layer_attn in enumerate(attentions):
        head_mean = layer_attn[0].mean(dim=0)
        query_row = head_mean[query_idx, :seq_len]

        bins = bin_attention_masses(
            query_row,
            prompt_len=prompt_len,
            span_ranges=prior_spans,
            seq_len=seq_len,
        )
        entropy = -float(
            (query_row * query_row.clamp(min=1e-12).log()).sum()
        )
        bins["entropy"] = entropy
        bins["layer"] = layer_idx
        per_layer.append(bins)

        attn_heatmap_rows.append(query_row.cpu().float().tolist())

    return {
        "num_layers": num_layers,
        "prompt_len": prompt_len,
        "completion_truncated_len": len(completion_truncated),
        "seq_len": seq_len,
        "query_idx": query_idx,
        "query_mode": query_mode,
        "prior_repeat_spans_completion_relative": prior_spans,
        "kth_span_completion_relative": kth_span,
        "per_layer": per_layer,
        "attn_heatmap_rows": attn_heatmap_rows,
    }


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def save_heatmap(
    result: dict,
    path: str,
    *,
    prompt_len: int,
    trigger: LoopTriggerDetail,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(result["attn_heatmap_rows"], dtype=np.float32)
    num_layers, seq_len = data.shape

    fig, ax = plt.subplots(figsize=(min(24, max(10, seq_len / 60)), 8))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"Attention from query idx {result['query_idx']} ({result['query_mode']})"
    )

    ax.axvline(
        x=prompt_len - 0.5, color="white", linewidth=0.8,
        linestyle="--", label="prompt end",
    )
    for span_start, span_end in result["prior_repeat_spans_completion_relative"]:
        abs_start = prompt_len + span_start
        abs_end = prompt_len + span_end
        ax.axvline(x=abs_start - 0.5, color="cyan", linewidth=0.5, linestyle=":")
        ax.axvline(x=abs_end - 0.5, color="cyan", linewidth=0.5, linestyle=":")

    kth_abs_start = prompt_len + result["kth_span_completion_relative"][0]
    kth_abs_end = prompt_len + result["kth_span_completion_relative"][1]
    ax.axvline(x=kth_abs_start - 0.5, color="red", linewidth=0.8, linestyle="--")
    ax.axvline(x=kth_abs_end - 0.5, color="red", linewidth=0.8, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze attention at n-gram loop trigger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-id", required=True,
                    help="HF model id or local path.")
    p.add_argument("--data", required=True,
                    help="JSONL file (prompt_rollout_archive format).")
    p.add_argument("--line-index", type=int, default=0,
                    help="0-based row index in the JSONL.")
    p.add_argument("--rollout-index", type=int, default=0,
                    help="Which rollout inside the chosen row.")
    p.add_argument("--loop-n", type=int, default=30,
                    help="N-gram length for loop detector.")
    p.add_argument("--loop-k", type=int, default=20,
                    help="Repeat count threshold for loop detector.")
    p.add_argument(
        "--query-mode",
        choices=["end_of_kth_ngram", "start_of_kth_ngram"],
        default="end_of_kth_ngram",
        help="Which token position to extract the attention row from.",
    )
    p.add_argument(
        "--attn-implementation", default="eager",
        help="HF attention backend string passed to from_pretrained(). "
             "Use a backend that returns attention weights with "
             "output_attentions=True. Fused backends (flash_attention_2, "
             "sdpa) typically do NOT materialize weights.",
    )
    p.add_argument("--max-seq-len", type=int, default=None,
                    help="Safety cap on total sequence length.")
    p.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--out-json", default=None,
                    help="Path for JSON output.")
    p.add_argument("--out-csv", default=None,
                    help="Path for CSV output (one row per layer).")
    p.add_argument("--heatmap-out", default=None,
                    help="Path for attention heatmap image.")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    rows = load_jsonl(args.data)
    if args.line_index < 0 or args.line_index >= len(rows):
        raise SystemExit(
            f"--line-index {args.line_index} out of range for "
            f"{len(rows)} rows in {args.data}."
        )
    row = rows[args.line_index]
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise SystemExit("Selected row has no 'rollouts' list.")
    if args.rollout_index < 0 or args.rollout_index >= len(rollouts):
        raise SystemExit(
            f"--rollout-index {args.rollout_index} out of range for "
            f"{len(rollouts)} rollouts."
        )
    rollout = rollouts[args.rollout_index]
    completion_text = rollout.get("completion_text")
    if not isinstance(completion_text, str) or not completion_text:
        raise SystemExit("Selected rollout has no completion_text.")

    saved_first_loop_prefix = rollout.get("first_loop_prefix_length")
    saved_loop_flag = rollout.get("loop_flag")

    # ── Tokenize (deferred heavy imports) ─────────────────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    prompt_text = row.get("prompt")
    saved_prompt_ids = row.get("prompt_token_ids")

    if saved_prompt_ids is not None and isinstance(saved_prompt_ids, list):
        prompt_ids = [int(t) for t in saved_prompt_ids]
    elif isinstance(prompt_text, str):
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    else:
        raise SystemExit("Row has neither 'prompt_token_ids' nor 'prompt' string.")

    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    print(
        f"Prompt tokens: {len(prompt_ids)}, "
        f"completion tokens (re-tokenized): {len(completion_ids)}"
    )

    # ── Detect loop ────────────────────────────────────────────────────
    trigger = detect_ngram_loop_with_spans(
        completion_ids, n=args.loop_n, k=args.loop_k
    )

    retokenize_loop_mismatch = None
    if trigger is None:
        print("No loop detected on re-tokenized completion.")
        if saved_loop_flag is not None and int(saved_loop_flag) == 1:
            print(
                "WARNING: saved archive has loop_flag=1 but re-tokenized "
                "stream has no loop. Likely a re-tokenization mismatch."
            )
            retokenize_loop_mismatch = True
        raise SystemExit("Cannot proceed without a detected loop.")
    else:
        print(
            f"Loop trigger at completion prefix {trigger.first_loop_prefix} "
            f"(n={trigger.n}, k={trigger.k}). "
            f"Pattern starts: {trigger.pattern_start_positions}"
        )
        if saved_first_loop_prefix is not None:
            saved_val = int(saved_first_loop_prefix)
            if saved_val != trigger.first_loop_prefix:
                print(
                    f"WARNING: re-tokenized first_loop_prefix "
                    f"{trigger.first_loop_prefix} != saved {saved_val}."
                )
                retokenize_loop_mismatch = True
            else:
                retokenize_loop_mismatch = False

    # ── Load model ─────────────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(
        f"Loading model {args.model_id} with "
        f"attn_implementation='{args.attn_implementation}' ..."
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    # ── Forward pass ───────────────────────────────────────────────────
    result = run_attention_analysis(
        model,
        tokenizer,
        device,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        trigger=trigger,
        query_mode=args.query_mode,
        max_seq_len=args.max_seq_len,
    )

    # ── Provenance ─────────────────────────────────────────────────────
    provenance = {
        "model_id": args.model_id,
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "data_file": args.data,
        "line_index": args.line_index,
        "rollout_index": args.rollout_index,
        "loop_n": args.loop_n,
        "loop_k": args.loop_k,
        "query_mode": args.query_mode,
        "max_seq_len": args.max_seq_len,
        "retokenize_loop_mismatch": retokenize_loop_mismatch,
        "prompt_token_count_retokenized": len(prompt_ids),
        "completion_token_count_retokenized": len(completion_ids),
        "caveats": [
            "Attention from a single teacher-forcing forward pass, not from "
            "the original vLLM autoregressive decode.",
            "Completion token IDs recovered via tokenizer.encode(completion_text); "
            "bitwise match with original generation stream is not guaranteed.",
        ],
    }
    trigger_dict = asdict(trigger)
    trigger_dict["pattern_end_exclusive"] = trigger.pattern_end_exclusive

    output = {
        "provenance": provenance,
        "trigger": trigger_dict,
        "num_layers": result["num_layers"],
        "prompt_len": result["prompt_len"],
        "seq_len": result["seq_len"],
        "query_idx": result["query_idx"],
        "query_mode": result["query_mode"],
        "prior_repeat_spans_completion_relative": result[
            "prior_repeat_spans_completion_relative"
        ],
        "kth_span_completion_relative": result["kth_span_completion_relative"],
        "per_layer": result["per_layer"],
    }

    # ── Write JSON ─────────────────────────────────────────────────────
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        print(f"JSON written to {args.out_json}")

    # ── Write CSV ──────────────────────────────────────────────────────
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fieldnames = [
            "layer", "prompt_mass", "prior_repeat_mass", "rest_mass", "entropy",
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for layer_row in result["per_layer"]:
                writer.writerow({k: layer_row[k] for k in fieldnames})
        print(f"CSV written to {args.out_csv}")

    # ── Print summary ──────────────────────────────────────────────────
    print("\n=== Per-layer bin masses ===")
    for layer_row in result["per_layer"]:
        print(
            f"  layer {layer_row['layer']:3d}: "
            f"prompt={layer_row['prompt_mass']:.4f}  "
            f"prior_repeat={layer_row['prior_repeat_mass']:.4f}  "
            f"rest={layer_row['rest_mass']:.4f}  "
            f"entropy={layer_row['entropy']:.2f}"
        )

    # ── Heatmap ────────────────────────────────────────────────────────
    if args.heatmap_out:
        save_heatmap(
            result,
            args.heatmap_out,
            prompt_len=result["prompt_len"],
            trigger=trigger,
        )

    if not args.out_json and not args.out_csv:
        print("\nHint: pass --out-json / --out-csv to persist results to disk.")


if __name__ == "__main__":
    main()
