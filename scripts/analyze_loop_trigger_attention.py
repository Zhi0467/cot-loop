#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.labeling import find_ngram_loop_trigger


DEFAULT_ARCHIVE_ROOTS = {
    "gpqa": "/data/scratch/murphy/outputs/cot-loop-detection/gpqa_mean_relative_from_archive_20260322/data",
    "aime": "/data/scratch/murphy/outputs/cot-loop-detection/aime_mean_relative_from_archive_20260322",
    "math500": "/data/scratch/murphy/outputs/cot-loop-detection/math_mean_relative_from_archive_20260323",
    "mmlu_pro": "/data/scratch/murphy/outputs/cot-loop-detection/mmlu_mean_relative_from_archive_20260323",
    "livecodebench": "/data/scratch/murphy/outputs/cot-loop-detection/livecodebench_mean_relative_from_archive_20260323",
}

QUERY_POSITION_MODES = ("trigger_end", "trigger_start", "pre_trigger_start")


@dataclass(frozen=True)
class SelectedRollout:
    dataset: str
    split: str
    sample_id: int
    rollout_index: int
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    prompt_token_count: int
    saved_completion_length: int
    reconstructed_completion_length: int
    first_loop_prefix_length: int
    total_prefix_length: int
    finish_reason: str
    length_diff: int
    loop_trigger: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_ARCHIVE_ROOTS),
        help=(
            "Comma-separated default dataset keys or explicit archive "
            "directories/files. Explicit paths may point to "
            "prompt_rollout_archive.jsonl(.gz), __rollout_archive.jsonl(.gz), "
            "or the aggregate rollout-stats JSON that owns a matching "
            "__rollout_archive.jsonl.gz sidecar."
        ),
    )
    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument("--max-trigger-prefix", type=int, default=4096)
    parser.add_argument("--max-samples-per-dataset", type=int, default=5)
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Only run reconstruction / trigger matching summary.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--query-position-mode",
        choices=QUERY_POSITION_MODES,
        default="trigger_end",
        help=(
            "Which token inside the triggering copy to use as the attention "
            "query."
        ),
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no CUDA device is available.")
    if device.index is not None:
        return device
    # `torch.cuda.set_device` requires an explicit index; default to the first
    # visible device so CUDA_VISIBLE_DEVICES-based sharding works as expected.
    return torch.device("cuda:0")


def _iter_archive_rows(root: Path):
    path = _resolve_archive_path(root)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _resolve_archive_path(root: Path) -> Path:
    if root.is_file():
        if root.suffix == ".json":
            stem = root.stem
            candidates: list[Path] = []
            if "__preexisting_" in stem:
                prefix, archived_tail = stem.split("__preexisting_", 1)
                candidates.append(
                    root.with_name(
                        f"{prefix}__rollout_archive.jsonl__preexisting_{archived_tail}.gz"
                    )
                )
            candidates.append(root.with_name(f"{stem}__rollout_archive.jsonl.gz"))
            if stem.endswith("__lcb_records"):
                candidates.append(
                    root.with_name(
                        f"{stem[:-len('__lcb_records')]}__rollout_archive.jsonl.gz"
                    )
                )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            raise SystemExit(
                "No rollout archive sidecar found for aggregate stats file "
                f"{root}."
            )
        return root

    prompt_profile_candidates = [
        root / "diagnostics" / "prompt_rollout_archive.jsonl",
        root / "diagnostics" / "prompt_rollout_archive.jsonl.gz",
        root / "prompt_rollout_archive.jsonl",
        root / "prompt_rollout_archive.jsonl.gz",
    ]
    for candidate in prompt_profile_candidates:
        if candidate.is_file():
            return candidate

    rollout_sidecars = sorted(root.glob("*__rollout_archive.jsonl.gz"))
    if len(rollout_sidecars) == 1:
        return rollout_sidecars[0]
    if len(rollout_sidecars) > 1:
        raise SystemExit(
            "Multiple rollout sidecars found under "
            f"{root}; pass the specific file path instead."
        )

    raise SystemExit(f"No rollout archive found under {root}.")


def _completion_token_ids_from_rollout(rollout: dict[str, Any], tokenizer) -> list[int]:
    saved_token_ids = rollout.get("completion_token_ids")
    if isinstance(saved_token_ids, list):
        return [int(token_id) for token_id in saved_token_ids]
    return tokenizer.encode(
        str(rollout["completion_text"]),
        add_special_tokens=False,
    )


def _saved_completion_length(rollout: dict[str, Any]) -> int:
    if "length" in rollout:
        return int(rollout["length"])
    if "completion_token_count" in rollout:
        return int(rollout["completion_token_count"])
    raise SystemExit("Rollout row is missing both 'length' and 'completion_token_count'.")


def _reconstruction_summary(
    archive_root: Path,
    dataset: str,
    tokenizer,
    *,
    loop_n: int,
    loop_k: int,
    max_trigger_prefix: int,
    max_samples_per_dataset: int,
) -> tuple[dict[str, Any], list[SelectedRollout]]:
    eos_ids: int | list[int] | None = None
    try:
        generation_config = GenerationConfig.from_pretrained(tokenizer.name_or_path)
    except OSError:
        generation_config = None
    if generation_config is not None:
        eos_ids = generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    elif eos_ids is None:
        eos_ids = []
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(int(tokenizer.eos_token_id))
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
                eos_ids.append(token_id)

    total_rollouts = 0
    loop_rows = 0
    exact_length_matches = 0
    length_diff_counter: Counter[int] = Counter()
    length_diff_by_finish: dict[str, Counter[int]] = defaultdict(Counter)
    prefix_match_counter: Counter[bool] = Counter()
    prefix_match_by_finish: dict[str, Counter[bool]] = defaultdict(Counter)
    match_if_allow_hidden_stop = 0
    usable_rows: list[SelectedRollout] = []

    for row in _iter_archive_rows(archive_root):
        prompt_token_ids = [int(v) for v in row["prompt_token_ids"]]
        split = str(row.get("split") or row.get("source_split") or "unknown")
        for rollout in row["rollouts"]:
            total_rollouts += 1
            finish_reason = str(rollout.get("finish_reason") or "unknown")
            completion_token_ids = _completion_token_ids_from_rollout(
                rollout,
                tokenizer,
            )
            saved_completion_length = _saved_completion_length(rollout)
            reconstructed_length = len(completion_token_ids)
            length_diff = saved_completion_length - reconstructed_length
            length_diff_counter[length_diff] += 1
            length_diff_by_finish[finish_reason][length_diff] += 1
            if length_diff == 0:
                exact_length_matches += 1
            if length_diff == 0 or (
                length_diff == 1 and finish_reason != "length"
            ):
                match_if_allow_hidden_stop += 1

            first_loop_prefix_length = rollout.get("first_loop_prefix_length")
            if first_loop_prefix_length is None:
                continue

            loop_rows += 1
            first_loop_prefix_length = int(first_loop_prefix_length)
            trigger = find_ngram_loop_trigger(
                completion_token_ids,
                n=loop_n,
                k=loop_k,
            )
            prefix_matches = (
                trigger is not None
                and trigger.first_loop_prefix == first_loop_prefix_length
            )
            prefix_match_counter[prefix_matches] += 1
            prefix_match_by_finish[finish_reason][prefix_matches] += 1
            if not prefix_matches:
                continue
            total_prefix_length = len(prompt_token_ids) + first_loop_prefix_length
            if total_prefix_length > max_trigger_prefix:
                continue
            selected = SelectedRollout(
                dataset=dataset,
                split=split,
                sample_id=int(row["sample_id"]),
                rollout_index=int(
                    rollout.get("rollout_index", rollout.get("generation_index", -1))
                ),
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                prompt_token_count=len(prompt_token_ids),
                saved_completion_length=saved_completion_length,
                reconstructed_completion_length=reconstructed_length,
                first_loop_prefix_length=first_loop_prefix_length,
                total_prefix_length=total_prefix_length,
                finish_reason=finish_reason,
                length_diff=length_diff,
                loop_trigger={
                    "ngram_start_positions": list(trigger.ngram_start_positions),
                    "trigger_start": int(trigger.trigger_start),
                    "trigger_end": int(trigger.trigger_end),
                    "ngram_token_ids": list(trigger.ngram_token_ids),
                    "hidden_stop_token_candidates": eos_ids,
                },
            )
            usable_rows.append(selected)

    usable_rows.sort(
        key=lambda row: (
            row.total_prefix_length,
            row.first_loop_prefix_length,
            row.sample_id,
            row.rollout_index,
        )
    )
    selected_rows = usable_rows[:max_samples_per_dataset]
    summary = {
        "dataset": dataset,
        "archive_root": str(archive_root),
        "total_rollouts": total_rollouts,
        "loop_rows": loop_rows,
        "exact_length_matches": exact_length_matches,
        "length_diff_top": length_diff_counter.most_common(8),
        "length_diff_by_finish_reason": {
            name: counter.most_common(6)
            for name, counter in sorted(length_diff_by_finish.items())
        },
        "prefix_match_counts": {
            "true": prefix_match_counter[True],
            "false": prefix_match_counter[False],
        },
        "prefix_match_by_finish_reason": {
            name: {
                "true": counter[True],
                "false": counter[False],
            }
            for name, counter in sorted(prefix_match_by_finish.items())
        },
        "match_if_allow_single_hidden_stop_token": match_if_allow_hidden_stop,
        "selected_rows": len(selected_rows),
        "selected_total_prefix_lengths": [
            row.total_prefix_length for row in selected_rows
        ],
    }
    return summary, selected_rows


def _index_positions(spans: list[tuple[int, int]]) -> list[int]:
    positions: list[int] = []
    for start, end in spans:
        positions.extend(range(start, end + 1))
    return positions


def _region_builder(
    row: SelectedRollout,
    *,
    recent_window: int,
    query_position_mode: str,
) -> dict[str, Any]:
    if query_position_mode not in QUERY_POSITION_MODES:
        raise ValueError(
            f"Unsupported query_position_mode={query_position_mode!r}; "
            f"expected one of {QUERY_POSITION_MODES}."
        )
    ngram_positions = row.loop_trigger["ngram_start_positions"]
    n = len(row.loop_trigger["ngram_token_ids"])
    prompt_len = row.prompt_token_count
    current_span = (
        prompt_len + int(row.loop_trigger["trigger_start"]),
        prompt_len + int(row.loop_trigger["trigger_end"]),
    )
    previous_spans = [
        (prompt_len + int(start), prompt_len + int(start) + n - 1)
        for start in ngram_positions[:-1]
    ]
    last_previous_span = previous_spans[-1] if previous_spans else None
    current_positions = set(range(current_span[0], current_span[1] + 1))
    previous_positions = set(_index_positions(previous_spans)) - current_positions
    last_previous_positions = (
        set(range(last_previous_span[0], last_previous_span[1] + 1))
        - current_positions
        if last_previous_span is not None
        else set()
    )
    prompt_positions = set(range(prompt_len))
    if query_position_mode == "trigger_end":
        query_position = current_span[1]
    elif query_position_mode == "trigger_start":
        query_position = current_span[0]
    else:
        query_position = current_span[0] - 1
    if query_position < 0:
        raise ValueError(
            "Computed a negative query position for "
            f"sample_id={row.sample_id}, rollout_index={row.rollout_index}."
        )

    recent_start = max(prompt_len, query_position - recent_window)
    recent_positions = {
        pos
        for pos in range(recent_start, query_position)
        if pos not in previous_positions and pos not in current_positions
    }
    accessible_positions = set(range(query_position + 1))
    other_completion_positions = sorted(
        accessible_positions
        - prompt_positions
        - previous_positions
        - current_positions
        - recent_positions
    )

    def region_for_index(index: int) -> str:
        if index in current_positions:
            return "current_trigger"
        if index in last_previous_positions:
            return "last_previous_loop"
        if index in previous_positions:
            return "earlier_previous_loop"
        if index in prompt_positions:
            return "prompt"
        if index in recent_positions:
            return "recent_nonloop"
        return "other_completion"

    return {
        "query_position": query_position,
        "query_position_mode": query_position_mode,
        "prompt_positions": sorted(prompt_positions),
        "previous_positions": sorted(previous_positions),
        "last_previous_positions": sorted(last_previous_positions),
        "current_positions": sorted(current_positions),
        "recent_positions": sorted(recent_positions),
        "other_completion_positions": other_completion_positions,
        "region_for_index": region_for_index,
    }


def _capture_attention(
    model,
    row: SelectedRollout,
    *,
    device: torch.device,
    recent_window: int,
    query_position_mode: str,
) -> list[dict[str, Any]]:
    region_info = _region_builder(
        row,
        recent_window=recent_window,
        query_position_mode=query_position_mode,
    )
    query_position = int(region_info["query_position"])
    input_ids = row.prompt_token_ids + row.completion_token_ids[: row.first_loop_prefix_length]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)

    layer_summaries: list[dict[str, Any]] = []
    original_forwards = []

    def _make_wrapped_forward(
        *,
        layer_idx: int,
        original_forward,
    ):
        def _wrapped_forward(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            query_states = self_attn.q_norm(
                self_attn.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self_attn.k_norm(
                self_attn.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
            key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
            selected_query = query_states[:, :, query_position : query_position + 1, :]
            attn_logits = (
                torch.matmul(selected_query, key_states.transpose(2, 3))
                * self_attn.scaling
            )
            future_positions = torch.arange(
                key_states.shape[-2],
                device=attn_logits.device,
            )
            future_mask = future_positions.view(1, 1, 1, -1) > query_position
            attn_logits = attn_logits.masked_fill(
                future_mask,
                torch.finfo(attn_logits.dtype).min,
            )
            if attention_mask is not None:
                if attention_mask.ndim == 4:
                    additive_mask = attention_mask[
                        :,
                        :,
                        query_position : query_position + 1,
                        : key_states.shape[-2],
                    ]
                    attn_logits = attn_logits + additive_mask
                elif attention_mask.ndim == 2:
                    key_padding_mask = attention_mask[
                        :,
                        : key_states.shape[-2],
                    ].view(attention_mask.shape[0], 1, 1, -1)
                    attn_logits = attn_logits.masked_fill(
                        key_padding_mask == 0,
                        torch.finfo(attn_logits.dtype).min,
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported attention_mask rank {attention_mask.ndim}."
                    )
            if self_attn.sliding_window is not None:
                window_cutoff = query_position - int(self_attn.sliding_window)
                if window_cutoff >= 0:
                    sliding_mask = future_positions.view(1, 1, 1, -1) <= window_cutoff
                    attn_logits = attn_logits.masked_fill(
                        sliding_mask,
                        torch.finfo(attn_logits.dtype).min,
                    )
            attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)[
                0, :, 0, :
            ]

            prev_positions = region_info["previous_positions"]
            last_prev_positions = region_info["last_previous_positions"]
            prompt_positions = region_info["prompt_positions"]
            current_positions = region_info["current_positions"]
            recent_positions = region_info["recent_positions"]
            other_completion_positions = region_info["other_completion_positions"]

            top1_indices = attn_weights.argmax(dim=-1).tolist()
            region_counter = Counter(
                region_info["region_for_index"](int(index)) for index in top1_indices
            )

            def _mass(indices: list[int]) -> float:
                if not indices:
                    return 0.0
                return float(attn_weights[:, indices].sum(dim=-1).mean().item())

            mean_attention = attn_weights.mean(dim=0)
            top_positions = torch.topk(
                mean_attention,
                k=min(10, mean_attention.shape[-1]),
            ).indices.tolist()

            layer_summaries.append(
                {
                    "layer": int(layer_idx),
                    "query_position": query_position,
                    "query_position_mode": str(region_info["query_position_mode"]),
                    "mean_prev_loop_mass": _mass(prev_positions),
                    "mean_last_prev_loop_mass": _mass(last_prev_positions),
                    "mean_prompt_mass": _mass(prompt_positions),
                    "mean_current_trigger_mass": _mass(current_positions),
                    "mean_recent_nonloop_mass": _mass(recent_positions),
                    "mean_other_completion_mass": _mass(other_completion_positions),
                    "top1_fraction_previous_loop": (
                        (
                            region_counter["last_previous_loop"]
                            + region_counter["earlier_previous_loop"]
                        )
                        / float(attn_weights.shape[0])
                    ),
                    "top1_fraction_last_previous_loop": (
                        region_counter["last_previous_loop"] / float(attn_weights.shape[0])
                    ),
                    "top1_fraction_prompt": (
                        region_counter["prompt"] / float(attn_weights.shape[0])
                    ),
                    "top1_fraction_current_trigger": (
                        region_counter["current_trigger"] / float(attn_weights.shape[0])
                    ),
                    "top1_fraction_recent_nonloop": (
                        region_counter["recent_nonloop"] / float(attn_weights.shape[0])
                    ),
                    "top1_fraction_other_completion": (
                        region_counter["other_completion"] / float(attn_weights.shape[0])
                    ),
                    "top_mean_positions": [int(v) for v in top_positions],
                    "top_mean_regions": [
                        region_info["region_for_index"](int(v)) for v in top_positions
                    ],
                }
            )
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return _wrapped_forward

    for layer_idx, layer in enumerate(model.model.layers):
        original_forward = layer.self_attn.forward
        original_forwards.append((layer.self_attn, original_forward))
        layer.self_attn.forward = MethodType(
            _make_wrapped_forward(
                layer_idx=layer_idx,
                original_forward=original_forward,
            ),
            layer.self_attn,
        )

    try:
        with torch.no_grad():
            model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=False,
                # The attention probe never consumes the logits tensor itself;
                # asking Qwen3 to keep only the final logit avoids a large
                # full-prefix lm_head allocation on long trigger prefixes.
                logits_to_keep=1,
            )
    finally:
        for module, original_forward in original_forwards:
            module.forward = original_forward

    return layer_summaries


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def _assign_shards(
    rows: list[SelectedRollout],
    *,
    num_shards: int,
) -> tuple[list[list[SelectedRollout]], list[int]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.total_prefix_length,
            row.prompt_token_count,
            row.dataset,
            row.sample_id,
            row.rollout_index,
        ),
        reverse=True,
    )
    shard_rows: list[list[SelectedRollout]] = [[] for _ in range(num_shards)]
    shard_loads = [0 for _ in range(num_shards)]
    for row in ordered_rows:
        target_shard = min(
            range(num_shards),
            key=lambda shard_idx: (shard_loads[shard_idx], len(shard_rows[shard_idx])),
        )
        shard_rows[target_shard].append(row)
        shard_loads[target_shard] += row.total_prefix_length
    for rows_for_shard in shard_rows:
        rows_for_shard.sort(
            key=lambda row: (
                row.dataset,
                row.split,
                row.total_prefix_length,
                row.first_loop_prefix_length,
                row.sample_id,
                row.rollout_index,
            )
        )
    return shard_rows, shard_loads


def _load_model(
    model_id: str,
    *,
    device: torch.device,
):
    kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
    }
    attn_implementations = (
        ["flash_attention_2", "sdpa", "eager"]
        if device.type == "cuda"
        else ["eager"]
    )
    last_error: Exception | None = None
    for attn_implementation in attn_implementations:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                **kwargs,
            )
            return model
        except Exception as exc:  # pragma: no cover - exercised on CUDA hosts
            last_error = exc
    assert last_error is not None
    raise last_error


def main() -> None:
    args = _parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards.")
    if "qwen3" not in args.model_id.lower():
        raise SystemExit(
            "scripts/analyze_loop_trigger_attention.py currently supports only "
            "Qwen3 checkpoints because the attention wrapper depends on "
            "Qwen3-specific internals."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    reconstruction_summaries: dict[str, Any] = {}
    selected_rows: list[SelectedRollout] = []
    for dataset in requested_datasets:
        archive_root = Path(DEFAULT_ARCHIVE_ROOTS.get(dataset, dataset))
        dataset_label = dataset
        summary, rows = _reconstruction_summary(
            archive_root,
            dataset_label,
            tokenizer,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            max_trigger_prefix=args.max_trigger_prefix,
            max_samples_per_dataset=args.max_samples_per_dataset,
        )
        reconstruction_summaries[dataset_label] = summary
        selected_rows.extend(rows)

    shard_rows, shard_loads = _assign_shards(
        selected_rows,
        num_shards=args.num_shards,
    )
    selected_rows = shard_rows[args.shard_index]

    _write_json(out_dir / "reconstruction_summary.json", reconstruction_summaries)
    _write_json(
        out_dir / "analysis_config.json",
        {
            "model_id": args.model_id,
            "datasets": requested_datasets,
            "loop_n": args.loop_n,
            "loop_k": args.loop_k,
            "max_trigger_prefix": args.max_trigger_prefix,
            "max_samples_per_dataset": args.max_samples_per_dataset,
            "recent_window": args.recent_window,
            "num_shards": args.num_shards,
            "query_position_mode": args.query_position_mode,
        },
    )
    _write_json(
        out_dir / "shard_manifest.json",
        {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "total_selected_rows": sum(len(rows) for rows in shard_rows),
            "rows_in_shard": len(selected_rows),
            "shard_total_prefix_loads": shard_loads,
            "shard_row_counts": [len(rows) for rows in shard_rows],
            "query_position_mode": args.query_position_mode,
        },
    )
    _write_jsonl(
        out_dir / "selected_rows.jsonl",
        [
            {
                "dataset": row.dataset,
                "split": row.split,
                "sample_id": row.sample_id,
                "rollout_index": row.rollout_index,
                "prompt_token_count": row.prompt_token_count,
                "saved_completion_length": row.saved_completion_length,
                "reconstructed_completion_length": row.reconstructed_completion_length,
                "first_loop_prefix_length": row.first_loop_prefix_length,
                "total_prefix_length": row.total_prefix_length,
                "finish_reason": row.finish_reason,
                "length_diff": row.length_diff,
                "loop_trigger": row.loop_trigger,
            }
            for row in selected_rows
        ],
    )
    if args.skip_attention or not selected_rows:
        return

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = _load_model(
        args.model_id,
        device=device,
    )
    model.to(device)
    model.eval()

    per_sample_rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        layer_summaries = _capture_attention(
            model,
            row,
            device=device,
            recent_window=args.recent_window,
            query_position_mode=args.query_position_mode,
        )
        per_sample_rows.append(
            {
                "dataset": row.dataset,
                "sample_id": row.sample_id,
                "rollout_index": row.rollout_index,
                "total_prefix_length": row.total_prefix_length,
                "first_loop_prefix_length": row.first_loop_prefix_length,
                "finish_reason": row.finish_reason,
                "length_diff": row.length_diff,
                "query_position_mode": args.query_position_mode,
                "layer_summaries": layer_summaries,
            }
        )
        for layer_summary in layer_summaries:
            per_layer_rows.append(
                {
                    "dataset": row.dataset,
                    "sample_id": row.sample_id,
                    "rollout_index": row.rollout_index,
                    **layer_summary,
                }
            )

    _write_json(out_dir / "attention_per_sample.json", per_sample_rows)

    grouped_by_dataset_and_layer: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_layer_rows:
        grouped_by_dataset_and_layer[(row["dataset"], int(row["layer"]))].append(row)

    layer_means: list[dict[str, Any]] = []
    for (dataset, layer), rows in sorted(grouped_by_dataset_and_layer.items()):
        layer_means.append(
            {
                "dataset": dataset,
                "layer": layer,
                "query_position_mode": args.query_position_mode,
                "num_rows": len(rows),
                "mean_prev_loop_mass": _mean([float(r["mean_prev_loop_mass"]) for r in rows]),
                "mean_last_prev_loop_mass": _mean([float(r["mean_last_prev_loop_mass"]) for r in rows]),
                "mean_prompt_mass": _mean([float(r["mean_prompt_mass"]) for r in rows]),
                "mean_current_trigger_mass": _mean([float(r["mean_current_trigger_mass"]) for r in rows]),
                "mean_recent_nonloop_mass": _mean([float(r["mean_recent_nonloop_mass"]) for r in rows]),
                "mean_other_completion_mass": _mean([float(r["mean_other_completion_mass"]) for r in rows]),
                "top1_fraction_previous_loop": _mean([float(r["top1_fraction_previous_loop"]) for r in rows]),
                "top1_fraction_last_previous_loop": _mean([float(r["top1_fraction_last_previous_loop"]) for r in rows]),
                "top1_fraction_prompt": _mean([float(r["top1_fraction_prompt"]) for r in rows]),
                "top1_fraction_current_trigger": _mean([float(r["top1_fraction_current_trigger"]) for r in rows]),
                "top1_fraction_recent_nonloop": _mean([float(r["top1_fraction_recent_nonloop"]) for r in rows]),
                "top1_fraction_other_completion": _mean([float(r["top1_fraction_other_completion"]) for r in rows]),
            }
        )

    with (out_dir / "attention_layer_means.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_means[0].keys()))
        writer.writeheader()
        writer.writerows(layer_means)

    overall_summary: dict[str, Any] = {}
    for dataset in requested_datasets:
        dataset_rows = [row for row in per_layer_rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        summary_layer = max(int(row["layer"]) for row in dataset_rows)
        summary_rows = [
            row for row in dataset_rows if int(row["layer"]) == summary_layer
        ]
        overall_summary[dataset] = {
            "num_selected_rows": len(
                {(row["sample_id"], row["rollout_index"]) for row in summary_rows}
            ),
            "summary_layer": summary_layer,
            "query_position_mode": args.query_position_mode,
            "mean_prev_loop_mass": _mean([float(r["mean_prev_loop_mass"]) for r in summary_rows]),
            "mean_last_prev_loop_mass": _mean([float(r["mean_last_prev_loop_mass"]) for r in summary_rows]),
            "mean_prompt_mass": _mean([float(r["mean_prompt_mass"]) for r in summary_rows]),
            "mean_current_trigger_mass": _mean([float(r["mean_current_trigger_mass"]) for r in summary_rows]),
            "mean_recent_nonloop_mass": _mean([float(r["mean_recent_nonloop_mass"]) for r in summary_rows]),
            "mean_other_completion_mass": _mean([float(r["mean_other_completion_mass"]) for r in summary_rows]),
            "top1_fraction_previous_loop": _mean([float(r["top1_fraction_previous_loop"]) for r in summary_rows]),
            "top1_fraction_last_previous_loop": _mean([float(r["top1_fraction_last_previous_loop"]) for r in summary_rows]),
            "top1_fraction_prompt": _mean([float(r["top1_fraction_prompt"]) for r in summary_rows]),
            "top1_fraction_current_trigger": _mean([float(r["top1_fraction_current_trigger"]) for r in summary_rows]),
            "top1_fraction_recent_nonloop": _mean([float(r["top1_fraction_recent_nonloop"]) for r in summary_rows]),
            "top1_fraction_other_completion": _mean([float(r["top1_fraction_other_completion"]) for r in summary_rows]),
        }

    _write_json(out_dir / "attention_summary.json", overall_summary)


if __name__ == "__main__":
    main()
