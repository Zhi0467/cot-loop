#!/usr/bin/env python3
"""Collect rollout statistics for a dataset/model pair into a rollout_bundle.v1.

The collector writes a single gzipped JSONL bundle of per-prompt rows
(prompt text, prompt token ids, per-rollout completion text, completion
token ids, flags, and task-kind grading) plus a small sidecar JSON with the
run metadata and aggregate metrics. See ``src/probe/bundle_io.py`` for
the reader/writer primitives and ``docs/reference/rollout-bundle-v1-schema.md`` for the
row/sidecar schema.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import signal
import shlex
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from transformers import AutoTokenizer

from probe.adapters import (
    codegen_ungraded,
    livecodebench_codegen,
    math_freeform,
    multiple_choice_gpqa,
    multiple_choice_mmlupro,
    taco_codegen,
)
from probe.adapters._common import parse_row_filter_json
from probe.adapters.livecodebench_codegen import LcbSampleRecord
from probe.bundle_io import (
    BUNDLE_SCHEMA,
    BundleWriter,
    bundle_paths,
    completed_sample_ids,
    concat_bundles,
    iter_bundle_rows,
    rewrite_bundle,
    write_bundle_sidecar,
)
from probe.collector import (
    ALL_STATISTICS,
    CollectorConfig,
    WorkerAggregator,
    compute_metrics,
    merge_aggregators,
)
from probe.configs import RolloutConfig
from probe.labeling import (
    aggregate_prompt_profile,
    find_ngram_loop_trigger,
    profile_target_name,
)
from probe.prompt_format import (
    VALID_PROMPT_FORMATS,
    VALID_THINKING_MODES,
    resolve_prompt_format,
    resolve_thinking_mode,
)
from probe.prompt_builder import build_math_prompt
from probe.rollout import resolve_sampling_defaults
from probe.types import DatasetSpec
from utils import get_visible_devices, suppress_sem_unlink_errors

TASK_CHOICES = (
    "math_freeform",
    "codegen_ungraded",
    "taco_codegen",
    "multiple_choice_gpqa",
    "multiple_choice_mmlupro",
    "livecodebench_codegen",
)
CPU_POSTHOC_TASK_KINDS = frozenset({"livecodebench_codegen", "taco_codegen"})
STATS_CONTRACT_VERSION = "rollout_stats_v2"


@dataclass(frozen=True)
class PromptWorkItem:
    sample_id: int
    prompt: str
    gold_answer: str | None = None
    gold_index: int | None = None
    question_id: str | None = None
    record_id: str | None = None
    source_split: str | None = None
    prompt_style: str | None = None
    choices: tuple[str, ...] | None = None
    record_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ExclusionArchive:
    prompts: frozenset[str]
    sample_ids: frozenset[int]

    @property
    def has_any(self) -> bool:
        return bool(self.prompts or self.sample_ids)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-kind", required=True, choices=TASK_CHOICES)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--starter-code-field", default="starter_code")
    parser.add_argument("--record-id-field", default="")
    parser.add_argument(
        "--metadata-fields",
        default="",
        help="Comma-separated raw dataset fields to preserve in each bundle row.",
    )
    parser.add_argument(
        "--row-filter-json",
        default="",
        help=(
            "Optional JSON object describing row filters before max-sample truncation, "
            "for example '{\"field_in\": {\"difficulty\": [\"HARD\", \"VERY_HARD\"]}}'."
        ),
    )
    parser.add_argument(
        "--exclude-prompt-jsonl",
        default="",
        help=(
            "Optional JSONL archive with top-level 'prompt' and/or 'sample_id' fields. "
            "Matching prompts or archived sample indices are excluded before max-sample "
            "truncation. Intended for disjoint follow-up screens or curated benchmark subsets."
        ),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=81920)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--statistics",
        default=",".join(ALL_STATISTICS),
        help="Comma-separated statistics to compute, or 'all'.",
    )
    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument(
        "--profile-tail-threshold",
        type=float,
        default=0.5,
        help="Prompt-level tail threshold used for majority_s_t screening summaries.",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--livecodebench-repo", default="")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--lm-style-override", default=None)
    parser.add_argument(
        "--prompt-format",
        default="auto",
        choices=VALID_PROMPT_FORMATS,
        help=(
            "How to serialize non-LiveCodeBench prompts before generation. "
            "'auto' uses the tokenizer chat template when available and falls "
            "back to raw plain text otherwise."
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        default="default",
        choices=VALID_THINKING_MODES,
        help=(
            "Explicit thinking-mode control for chat-template prompts. "
            "'default' preserves the tokenizer's native behavior."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a prior run by skipping sample_ids already present in the "
            "bundle file at the resolved out path. All other arguments must match "
            "the prior run."
        ),
    )
    parser.add_argument(
        "--defer-cpu-finalize",
        action="store_true",
        help=(
            "For CPU-only post-hoc graders such as LiveCodeBench and TACO, stop "
            "after writing generated rollout rows and mark grading as deferred. "
            "Run --finalize-only later with GPUs hidden to fill pass/fail labels "
            "and rewrite the sidecar."
        ),
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Finalize an existing bundle without generating. This runs CPU-only "
            "post-hoc grading and sidecar rewriting and does not require vLLM."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    return parser.parse_args()


def _parse_statistics(raw: str) -> list[str]:
    value = raw.strip()
    if not value or value.lower() == "all":
        return list(ALL_STATISTICS)
    seen: set[str] = set()
    stats: list[str] = []
    for name in (part.strip() for part in value.split(",")):
        if not name or name in seen:
            continue
        seen.add(name)
        stats.append(name)
    if not stats:
        return list(ALL_STATISTICS)
    unknown = sorted(set(stats) - set(ALL_STATISTICS))
    if unknown:
        raise SystemExit(
            f"Unknown statistic(s): {unknown}. Valid choices: {list(ALL_STATISTICS)}"
        )
    return stats


def _parse_metadata_fields(raw: str) -> list[str] | None:
    fields = [part.strip() for part in raw.split(",") if part.strip()]
    return fields or None


def _thinking_mode_metadata(
    *,
    requested: str,
    resolved: str | None,
) -> dict[str, object]:
    if resolved is None:
        return {}
    return {
        "thinking_mode_requested": requested,
        "thinking_mode_resolved": resolved,
    }


def _slugify(value: str) -> str:
    text = value.strip()
    if not text:
        return "default"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "default"


def _load_excluded_prompts(path: str | None) -> ExclusionArchive:
    source = (path or "").strip()
    if not source:
        return ExclusionArchive(prompts=frozenset(), sample_ids=frozenset())
    if not os.path.exists(source):
        raise SystemExit(f"--exclude-prompt-jsonl path does not exist: {source}")
    prompts: set[str] = set()
    sample_ids: set[int] = set()
    with open(source, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in --exclude-prompt-jsonl at line {lineno}: {exc}"
                ) from exc
            prompt = payload.get("prompt")
            if isinstance(prompt, str):
                prompts.add(prompt)
            sample_id = payload.get("sample_id")
            if isinstance(sample_id, int) and sample_id >= 0:
                sample_ids.add(int(sample_id))
    return ExclusionArchive(
        prompts=frozenset(prompts),
        sample_ids=frozenset(sample_ids),
    )


def _derive_output_path(args: argparse.Namespace) -> str:
    dataset_label = os.path.basename(args.dataset) if os.path.isfile(args.dataset) else args.dataset
    parts = [_slugify(dataset_label)]
    if args.dataset_config:
        parts.append(_slugify(args.dataset_config))
    parts.append(_slugify(args.split))
    parts.append(_slugify(args.model_id))
    filename = "__".join(parts) + ".json"
    return os.path.join(ROOT, "outputs", "model_stats", filename)


def _archive_preexisting(path: str) -> None:
    if not os.path.exists(path):
        return
    stem, ext = os.path.splitext(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived_path = f"{stem}__preexisting_{timestamp}{ext}"
    counter = 1
    while os.path.exists(archived_path):
        archived_path = f"{stem}__preexisting_{timestamp}_{counter}{ext}"
        counter += 1
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.replace(path, archived_path)
    print(f"Archived preexisting file from {path} to {archived_path}", flush=True)


def _rank_bundle_path(bundle_path: str, rank: int) -> str:
    if bundle_path.endswith(".jsonl.gz"):
        base = bundle_path[: -len(".jsonl.gz")]
    else:
        base, _ = os.path.splitext(bundle_path)
    return f"{base}__rank{rank}.jsonl.gz"


def _completed_sample_ids_for_resume(paths: Iterable[str]) -> set[int]:
    """Return completed sample ids from valid rows, repairing canceled gzip tails."""
    completed: set[int] = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        rows: list[dict[str, Any]] = []
        try:
            for row in iter_bundle_rows(path):
                rows.append(row)
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            if rows:
                _archive_preexisting(path)
                rewrite_bundle(path, rows)
                print(
                    f"Rewrote resumable prefix with {len(rows)} row(s) after "
                    f"truncated/corrupt bundle tail in {path}: {exc}",
                    flush=True,
                )
            else:
                _archive_preexisting(path)
                print(
                    f"Archived unreadable empty/invalid resume bundle {path}: {exc}",
                    flush=True,
                )
        for row in rows:
            sid = row.get("sample_id")
            if isinstance(sid, int):
                completed.add(int(sid))
    return completed


def _normalize_finish_reason(reason: object) -> str:
    if hasattr(reason, "value"):
        reason = getattr(reason, "value")
    if reason is None:
        return "unknown"
    text = str(reason).strip()
    if not text:
        return "unknown"
    return text.split(".")[-1].lower()


def _effective_max_tokens(prompt_len: int, rollout_cfg: RolloutConfig) -> int:
    return min(rollout_cfg.max_tokens, max(0, rollout_cfg.max_model_len - prompt_len))


def _total_token_count(prompt_len: int, token_count: int) -> int:
    return prompt_len + token_count


def _hit_max_model_len(
    *,
    prompt_len: int,
    token_count: int,
    finish_reason: str,
    rollout_cfg: RolloutConfig,
) -> bool:
    if finish_reason != "length":
        return False
    return _total_token_count(prompt_len, token_count) >= rollout_cfg.max_model_len


def _run_dataset_preflight(args: argparse.Namespace) -> None:
    if (
        not os.path.exists(args.dataset)
        and args.dataset.lower().endswith((".jsonl", ".json", ".csv"))
    ):
        raise SystemExit(f"Dataset path does not exist: {args.dataset}")

    if args.task_kind == "multiple_choice_gpqa" and not os.path.isfile(args.dataset):
        hf_token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
            "HUGGINGFACE_TOKEN", ""
        ).strip()
        if not hf_token:
            raise SystemExit(
                "multiple_choice_gpqa requires HF_TOKEN for gated dataset access."
            )
        os.environ.setdefault("HF_TOKEN", hf_token)

    if args.task_kind == "livecodebench_codegen":
        if not os.environ.get("HF_DATASETS_CACHE", "").strip():
            raise SystemExit(
                "livecodebench_codegen requires HF_DATASETS_CACHE to be set before startup."
            )
        if not args.livecodebench_repo:
            raise SystemExit(
                "livecodebench_codegen requires --livecodebench-repo."
            )
        if not os.path.isdir(args.livecodebench_repo):
            raise SystemExit(
                f"LiveCodeBench repo path does not exist: {args.livecodebench_repo}"
            )


def _run_dependency_preflight(
    args: argparse.Namespace,
    *,
    require_vllm: bool,
) -> None:
    if require_vllm:
        try:
            import vllm  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "vLLM is required for collect_model_stats. Install vLLM first."
            ) from exc

    if args.task_kind == "math_freeform":
        math_freeform.preflight()
    elif args.task_kind == "taco_codegen":
        taco_codegen.preflight()
    elif args.task_kind == "livecodebench_codegen":
        try:
            livecodebench_codegen.preflight(
                args.livecodebench_repo,
                args.release_version,
            )
        except Exception as exc:
            raise SystemExit(
                "LiveCodeBench preflight failed before model init."
            ) from exc


def _build_rollout_config(args: argparse.Namespace) -> RolloutConfig:
    if args.dp < 1:
        raise SystemExit("--dp must be >= 1.")
    if args.tp < 1:
        raise SystemExit("--tp must be >= 1.")
    if args.dp > 1 and args.tp != 1:
        raise SystemExit("Data-parallel collection requires --tp 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")
    if args.loop_n < 1:
        raise SystemExit("--loop-n must be >= 1.")
    if args.loop_k < 2:
        raise SystemExit("--loop-k must be >= 2.")
    if not (0.0 < args.profile_tail_threshold <= 1.0):
        raise SystemExit("--profile-tail-threshold must be in (0, 1].")
    if args.num_generations < 1:
        raise SystemExit("--num-generations must be >= 1.")
    if args.top_p is not None and not (0.0 < args.top_p <= 1.0):
        raise SystemExit("--top-p must be in (0, 1] when provided.")
    if args.top_k is not None and args.top_k < -1:
        raise SystemExit("--top-k must be >= -1 when provided.")

    return RolloutConfig(
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_generations=args.num_generations,
        max_tokens=args.max_tokens,
        tp=args.tp,
        dp=args.dp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
    )


def _load_prompt_items(
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> tuple[list[PromptWorkItem], dict[str, object], Any]:
    exclusion_archive = _load_excluded_prompts(args.exclude_prompt_jsonl)
    if args.task_kind == "livecodebench_codegen":
        if args.prompt_format == "chat_template":
            raise SystemExit(
                "livecodebench_codegen uses raw prompt strings and does not support "
                "--prompt-format chat_template."
            )
        resolved_prompt_format = "raw"
        resolved_thinking_mode = None
    else:
        if tokenizer is None:
            raise SystemExit(
                "Tokenizer is required to resolve prompt formatting for this task."
            )
        try:
            resolved_prompt_format = resolve_prompt_format(tokenizer, args.prompt_format)
            resolved_thinking_mode = resolve_thinking_mode(
                tokenizer,
                prompt_format=resolved_prompt_format,
                thinking_mode=args.thinking_mode,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if exclusion_archive.has_any and args.task_kind != "livecodebench_codegen":
        raise SystemExit(
            "--exclude-prompt-jsonl is currently only supported for "
            "livecodebench_codegen."
        )

    spec = DatasetSpec(
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
    )
    metadata_fields = _parse_metadata_fields(args.metadata_fields)
    row_filter = parse_row_filter_json(args.row_filter_json)
    record_id_field = args.record_id_field.strip() or None

    if args.task_kind == "math_freeform":
        if tokenizer is None:
            raise SystemExit("Tokenizer is required for math_freeform prompt building.")
        samples = math_freeform.load_samples(
            spec,
            question_field=args.question_field,
            answer_field=args.answer_field,
            record_id_field=record_id_field,
            metadata_fields=metadata_fields,
            row_filter=row_filter,
        )
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=build_math_prompt(
                    tokenizer,
                    record.prompt,
                    prompt_format=resolved_prompt_format,
                    thinking_mode=args.thinking_mode,
                ),
                gold_answer=gold_answer,
                record_id=record.record_id,
                source_split=record.source_split,
                prompt_style=record.prompt_style,
                choices=record.choices,
                record_metadata=record.metadata,
            )
            for record, gold_answer in samples
        ]
        return items, {
            "prompt_format_requested": args.prompt_format,
            "prompt_format_resolved": resolved_prompt_format,
            **_thinking_mode_metadata(
                requested=args.thinking_mode,
                resolved=resolved_thinking_mode,
            ),
            "record_id_field": record_id_field,
            "metadata_fields": metadata_fields,
            **({"row_filter": row_filter} if row_filter is not None else {}),
        }, None

    if args.task_kind == "codegen_ungraded":
        samples = codegen_ungraded.load_samples(
            spec,
            question_field=args.question_field,
            starter_code_field=args.starter_code_field or None,
            record_id_field=record_id_field,
            metadata_fields=metadata_fields,
            row_filter=row_filter,
        )
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=codegen_ungraded.build_codegen_prompt(
                    tokenizer,
                    record.prompt,
                    starter_code=(
                        None
                        if record.metadata is None
                            else str(record.metadata.get(args.starter_code_field, "") or "")
                    ),
                    prompt_format=resolved_prompt_format,
                    thinking_mode=args.thinking_mode,
                ),
                record_id=record.record_id,
                source_split=record.source_split,
                prompt_style=record.prompt_style,
                choices=record.choices,
                record_metadata=record.metadata,
            )
            for record in samples
        ]
        return items, {
            "prompt_format_requested": args.prompt_format,
            "prompt_format_resolved": resolved_prompt_format,
            **_thinking_mode_metadata(
                requested=args.thinking_mode,
                resolved=resolved_thinking_mode,
            ),
            "starter_code_field": args.starter_code_field or None,
            "record_id_field": record_id_field,
            "metadata_fields": metadata_fields,
            **({"row_filter": row_filter} if row_filter is not None else {}),
        }, None

    if args.task_kind == "taco_codegen":
        samples = taco_codegen.load_samples(
            spec,
            question_field=args.question_field,
            starter_code_field=args.starter_code_field or None,
            record_id_field=record_id_field,
            metadata_fields=metadata_fields,
            row_filter=row_filter,
        )
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=taco_codegen.build_codegen_prompt(
                    tokenizer,
                    record.prompt,
                    starter_code=(
                        None
                        if record.metadata is None
                            else str(record.metadata.get(args.starter_code_field, "") or "")
                    ),
                    prompt_format=resolved_prompt_format,
                    thinking_mode=args.thinking_mode,
                ),
                record_id=record.record_id,
                source_split=record.source_split,
                prompt_style=record.prompt_style,
                choices=record.choices,
                record_metadata=record.metadata,
            )
            for record in samples
        ]
        return items, {
            "prompt_format_requested": args.prompt_format,
            "prompt_format_resolved": resolved_prompt_format,
            **_thinking_mode_metadata(
                requested=args.thinking_mode,
                resolved=resolved_thinking_mode,
            ),
            "starter_code_field": args.starter_code_field or None,
            "record_id_field": record_id_field,
            "metadata_fields": metadata_fields,
            **({"row_filter": row_filter} if row_filter is not None else {}),
        }, None

    if args.task_kind == "multiple_choice_gpqa":
        if tokenizer is None:
            raise SystemExit(
                "Tokenizer is required for multiple_choice_gpqa prompt building."
            )
        samples = multiple_choice_gpqa.load_and_shuffle(spec, args.seed)
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=multiple_choice_gpqa.build_mcq_prompt(
                    tokenizer,
                    record.prompt,
                    options,
                    prompt_format=resolved_prompt_format,
                    thinking_mode=args.thinking_mode,
                ),
                gold_answer=gold_letter,
                record_id=record.record_id,
                source_split=record.source_split,
                prompt_style=record.prompt_style,
                choices=record.choices,
                record_metadata=record.metadata,
            )
            for record, options, gold_letter in samples
        ]
        metadata = {
            "prompt_format_requested": args.prompt_format,
            "prompt_format_resolved": resolved_prompt_format,
            **_thinking_mode_metadata(
                requested=args.thinking_mode,
                resolved=resolved_thinking_mode,
            ),
            "shuffle_policy": {
                "kind": "seed_xor_sample_id",
                "base_seed": args.seed,
            }
        }
        return items, metadata, None

    if args.task_kind == "multiple_choice_mmlupro":
        if tokenizer is None:
            raise SystemExit(
                "Tokenizer is required for multiple_choice_mmlupro prompt building."
            )
        samples = multiple_choice_mmlupro.load_samples(spec)
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=multiple_choice_mmlupro.build_mcq_prompt(
                    tokenizer,
                    record.prompt,
                    options,
                    prompt_format=resolved_prompt_format,
                    thinking_mode=args.thinking_mode,
                ),
                gold_answer=gold_answer,
                gold_index=gold_index,
                record_id=record.record_id,
                source_split=record.source_split,
                prompt_style=record.prompt_style,
                choices=record.choices,
                record_metadata=record.metadata,
            )
            for record, options, gold_answer, gold_index in samples
        ]
        return items, {
            "prompt_format_requested": args.prompt_format,
            "prompt_format_resolved": resolved_prompt_format,
            **_thinking_mode_metadata(
                requested=args.thinking_mode,
                resolved=resolved_thinking_mode,
            ),
        }, None

    benchmark, format_prompt = livecodebench_codegen.load_benchmark(
        args.livecodebench_repo,
        args.release_version,
    )
    prompt_records, lm_style = livecodebench_codegen.build_prompts(
        benchmark,
        format_prompt,
        repo_path=args.livecodebench_repo,
        model_id=args.model_id,
        lm_style_override=args.lm_style_override,
        thinking_mode=args.thinking_mode,
    )
    resolved_prompt_format = livecodebench_codegen.prompt_format_for_lm_style(lm_style)
    resolved_thinking_mode = None
    if resolved_prompt_format == "chat_template":
        lcb_tokenizer = tokenizer
        if lcb_tokenizer is None:
            lcb_tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                trust_remote_code=args.trust_remote_code,
                use_fast=True,
            )
        try:
            resolved_thinking_mode = resolve_thinking_mode(
                lcb_tokenizer,
                prompt_format=resolved_prompt_format,
                thinking_mode=args.thinking_mode,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.thinking_mode != "default":
        raise SystemExit(
            "livecodebench_codegen thinking_mode requires a chat-template prompt "
            "surface, for example --lm-style-override HFChatTemplate."
        )
    prompt_rows = list(zip(benchmark, prompt_records, strict=True))
    excluded_count = 0
    if exclusion_archive.has_any:
        filtered_rows = [
            (instance, record)
            for original_idx, (instance, record) in enumerate(prompt_rows)
            if record[1] not in exclusion_archive.prompts
            and original_idx not in exclusion_archive.sample_ids
        ]
        excluded_count = len(prompt_rows) - len(filtered_rows)
        prompt_rows = filtered_rows
    if args.max_samples is not None:
        prompt_rows = prompt_rows[: args.max_samples]
    selected_benchmark = [instance for instance, _record in prompt_rows]
    items = [
        PromptWorkItem(
            sample_id=idx,
            prompt=record[1],
            question_id=record[0],
            record_id=record[0],
            source_split=args.split,
            prompt_style="livecodebench_codegen",
        )
        for idx, (_instance, record) in enumerate(prompt_rows)
    ]
    metadata = {
        "lm_style": lm_style,
        "prompt_format_requested": args.prompt_format,
        "prompt_format_resolved": resolved_prompt_format,
        **_thinking_mode_metadata(
            requested=args.thinking_mode,
            resolved=resolved_thinking_mode,
        ),
    }
    if exclusion_archive.has_any:
        metadata["exclude_prompt_jsonl"] = args.exclude_prompt_jsonl
        metadata["excluded_prompt_count"] = int(excluded_count)
        metadata["excluded_archive_prompt_count"] = int(len(exclusion_archive.prompts))
        metadata["excluded_archive_sample_id_count"] = int(
            len(exclusion_archive.sample_ids)
        )
    return items, metadata, selected_benchmark


def _grade_qa_response(
    *,
    task_kind: str,
    item: PromptWorkItem,
    response_text: str,
) -> bool:
    if task_kind == "math_freeform":
        return math_freeform.grade(response_text, item.gold_answer or "")
    elif task_kind == "multiple_choice_gpqa":
        return multiple_choice_gpqa.grade(response_text, item.gold_answer or "")
    elif task_kind == "multiple_choice_mmlupro":
        return multiple_choice_mmlupro.grade(
            response_text,
            item.gold_answer or "",
            item.gold_index,
        )
    raise ValueError(f"Unsupported QA task_kind '{task_kind}'.")


def _update_qa_stats(
    agg: WorkerAggregator,
    *,
    correct: bool,
    token_count: int,
    loop_flag: bool,
    max_length_hit: bool,
) -> None:
    agg.num_graded += 1
    if correct:
        agg.num_correct += 1
        agg.correct_length_sum += token_count
        if loop_flag:
            agg.num_correct_and_looped += 1
        if max_length_hit:
            agg.num_correct_and_max_length_hit += 1
        if loop_flag and max_length_hit:
            agg.num_correct_and_looped_and_max_length_hit += 1
    else:
        agg.num_wrong += 1
        agg.wrong_length_sum += token_count


def _sanitize_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonable(item) for key, item in value.items()}
    return str(value)


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _build_rollout_dict(
    *,
    rollout_index: int,
    completion_text: str,
    completion_token_ids: list[int],
    prompt_len: int,
    finish_reason: str,
    loop_flag: bool,
    first_loop_prefix: int | None,
    loop_trigger: dict[str, Any] | None,
    max_length_hit: bool,
    effective_max_tokens: int,
    tail_threshold: float,
    grading: dict[str, Any] | None,
) -> dict[str, Any]:
    token_count = len(completion_token_ids)
    relative_length = (
        float(token_count) / float(effective_max_tokens)
        if effective_max_tokens > 0
        else None
    )
    tail_hit = int(
        effective_max_tokens > 0
        and relative_length is not None
        and relative_length >= float(tail_threshold)
    )
    return {
        "rollout_index": int(rollout_index),
        "completion_text": completion_text,
        "completion_token_ids": completion_token_ids,
        "completion_token_count": int(token_count),
        "total_token_count": int(_total_token_count(prompt_len, token_count)),
        "finish_reason": finish_reason,
        "loop_flag": bool(loop_flag),
        "max_length_hit": bool(max_length_hit),
        "first_loop_prefix_length": first_loop_prefix,
        "loop_trigger": loop_trigger,
        "length": int(token_count),
        "relative_length": relative_length,
        "cap_hit": int(max_length_hit),
        "tail_hit": tail_hit,
        "grading": grading,
    }


def _build_prompt_profile(
    rollouts: list[dict[str, Any]],
    *,
    effective_max_tokens: int,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
) -> dict[str, Any]:
    profile = aggregate_prompt_profile(
        [rollout["completion_token_ids"] for rollout in rollouts],
        effective_max_tokens=effective_max_tokens,
        loop_n=loop_n,
        loop_k=loop_k,
        tail_threshold=tail_threshold,
        finish_reasons=[str(rollout["finish_reason"]) for rollout in rollouts],
    )
    return profile


def _bundle_row_sorted(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(row: dict[str, Any]) -> int:
        sid = row.get("sample_id")
        return int(sid) if isinstance(sid, int) else -1

    return sorted(rows, key=_sort_key)


def _collect_worker_stats(
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    *,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
    bundle_path: str,
    skip_sample_ids: set[int] | None = None,
    rank: int = 0,
) -> WorkerAggregator:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(
            "vLLM is required for collect_model_stats. Install vLLM first."
        ) from exc

    agg = WorkerAggregator()
    if not items:
        return agg
    skip_sample_ids = skip_sample_ids or set()
    target_name = profile_target_name("majority_tail", tail_threshold=tail_threshold)

    rollout_cfg = collector_cfg.rollout_cfg
    if rollout_cfg.max_num_seqs is not None and rollout_cfg.max_num_seqs < 1:
        raise SystemExit("--max-num-seqs must be >= 1 when provided.")
    if (
        rollout_cfg.max_num_seqs is not None
        and rollout_cfg.max_num_seqs < rollout_cfg.num_generations
    ):
        raise SystemExit(
            "--max-num-seqs must be >= --num-generations when sampling multiple "
            "generations per prompt."
        )

    top_p, top_k = resolve_sampling_defaults(
        rollout_cfg.model_id,
        top_p=rollout_cfg.top_p,
        top_k=rollout_cfg.top_k,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
        use_fast=True,
    )

    llm_kwargs = {
        "model": rollout_cfg.model_id,
        "tensor_parallel_size": rollout_cfg.tp,
        "dtype": rollout_cfg.dtype,
        "max_model_len": rollout_cfg.max_model_len,
        "trust_remote_code": rollout_cfg.trust_remote_code,
    }
    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "").strip()
    if gpu_mem_util:
        try:
            gpu_mem_value = float(gpu_mem_util)
        except Exception as exc:
            raise SystemExit(
                "VLLM_GPU_MEMORY_UTILIZATION must be a float in (0, 1]."
            ) from exc
        if not (0.0 < gpu_mem_value <= 1.0):
            raise SystemExit("VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1].")
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_value
    if rollout_cfg.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = rollout_cfg.max_num_seqs
    if rollout_cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = rollout_cfg.max_num_batched_tokens

    pending_items = [item for item in items if int(item.sample_id) not in skip_sample_ids]
    if skip_sample_ids:
        print(
            f"[collect-dp-rank {rank}] skipping {len(items) - len(pending_items)} "
            f"sample(s) already present in {bundle_path}",
            flush=True,
        )
    if not pending_items:
        return agg

    llm = LLM(**llm_kwargs)

    if rollout_cfg.max_num_seqs is not None:
        chunk_size = max(1, rollout_cfg.max_num_seqs // rollout_cfg.num_generations)
    else:
        chunk_size = len(pending_items)

    with BundleWriter(bundle_path) as writer:
        for start in range(0, len(pending_items), chunk_size):
            end = min(start + chunk_size, len(pending_items))
            batch_items = pending_items[start:end]
            batch_prompts = [item.prompt for item in batch_items]
            prompt_input_ids = tokenizer(
                batch_prompts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            valid_items: list[tuple[PromptWorkItem, list[int], int, int]] = []
            for item, input_ids in zip(batch_items, prompt_input_ids):
                agg.num_samples_seen += 1
                token_ids = [int(tok) for tok in input_ids]
                prompt_len = len(token_ids)
                agg.prompt_length_sum += prompt_len
                agg.prompt_length_min = (
                    prompt_len
                    if agg.prompt_length_min is None
                    else min(agg.prompt_length_min, prompt_len)
                )
                agg.prompt_length_max = (
                    prompt_len
                    if agg.prompt_length_max is None
                    else max(agg.prompt_length_max, prompt_len)
                )
                effective_max = _effective_max_tokens(prompt_len, rollout_cfg)
                if effective_max < 1:
                    agg.num_prompt_too_long += 1
                    writer.write_row(
                        _build_prompt_too_long_row(
                            item,
                            prompt_token_ids=token_ids,
                            effective_max_tokens=effective_max,
                            max_model_len=rollout_cfg.max_model_len,
                            target_name=target_name,
                        )
                    )
                    continue
                valid_items.append((item, token_ids, prompt_len, effective_max))

            for item, prompt_token_ids, prompt_len, effective_max in valid_items:
                sampling_params = SamplingParams(
                    temperature=rollout_cfg.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=effective_max,
                    n=rollout_cfg.num_generations,
                    repetition_penalty=1.0,
                    seed=collector_cfg.seed + item.sample_id,
                )
                outputs = llm.generate([item.prompt], sampling_params)
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"Expected 1 prompt output, got {len(outputs)}."
                    )
                output = outputs[0]
                if len(output.outputs) != rollout_cfg.num_generations:
                    raise RuntimeError(
                        "Expected "
                        f"{rollout_cfg.num_generations} output(s) per prompt, got "
                        f"{len(output.outputs)}."
                    )
                rollouts: list[dict[str, Any]] = []
                for generation_index, sample in enumerate(output.outputs):
                    text = str(getattr(sample, "text", ""))
                    token_ids = getattr(sample, "token_ids", None)
                    if not token_ids:
                        token_ids = tokenizer.encode(text, add_special_tokens=False)
                    token_ids = [int(token_id) for token_id in token_ids]
                    token_count = len(token_ids)
                    finish_reason = _normalize_finish_reason(
                        getattr(sample, "finish_reason", None)
                        or getattr(output, "finish_reason", None)
                        or getattr(sample, "stop_reason", None)
                    )
                    loop_trigger = find_ngram_loop_trigger(
                        token_ids,
                        n=loop_n,
                        k=loop_k,
                    )
                    first_loop_prefix = (
                        loop_trigger.first_loop_prefix
                        if loop_trigger is not None
                        else None
                    )
                    loop_flag = first_loop_prefix is not None
                    max_length_hit = _hit_max_model_len(
                        prompt_len=prompt_len,
                        token_count=token_count,
                        finish_reason=finish_reason,
                        rollout_cfg=rollout_cfg,
                    )

                    agg.num_generated += 1
                    agg.length_sum += token_count
                    agg.length_sq_sum += token_count * token_count
                    if loop_flag and first_loop_prefix is not None:
                        agg.num_looped += 1
                        agg.loop_length_sum += token_count
                        agg.first_loop_prefix_sum += first_loop_prefix
                        agg.first_loop_prefix_count += 1
                    if max_length_hit:
                        agg.num_max_length_hits += 1
                    if loop_flag and max_length_hit:
                        agg.num_looped_and_max_length_hit += 1

                    grading: dict[str, Any] | None = None
                    if collector_cfg.task_kind == "livecodebench_codegen":
                        code_output = livecodebench_codegen.extract_code_output(
                            text,
                            repo_path=collector_cfg.livecodebench_repo or "",
                            model_id=rollout_cfg.model_id,
                            lm_style_override=collector_cfg.lm_style_override,
                        )
                        # LCB pass/fail is applied post-hoc (batch grader); for now
                        # only store the extracted code. ``grading.passed`` will be
                        # filled in during the finalize pass.
                        grading = {"code_output": code_output, "passed": None}
                    elif collector_cfg.task_kind == "taco_codegen":
                        # TACO grading is also post-hoc (subprocess test runner).
                        # The finalize pass fills ``grading``.
                        grading = None
                    elif collector_cfg.task_kind in {
                        "math_freeform",
                        "multiple_choice_gpqa",
                        "multiple_choice_mmlupro",
                    }:
                        correct = _grade_qa_response(
                            task_kind=collector_cfg.task_kind,
                            item=item,
                            response_text=text,
                        )
                        _update_qa_stats(
                            agg,
                            correct=correct,
                            token_count=token_count,
                            loop_flag=loop_flag,
                            max_length_hit=max_length_hit,
                        )
                        grading = {"correct": int(bool(correct))}
                    else:  # codegen_ungraded and anything else
                        grading = None

                    rollouts.append(
                        _build_rollout_dict(
                            rollout_index=generation_index,
                            completion_text=text,
                            completion_token_ids=token_ids,
                            prompt_len=prompt_len,
                            finish_reason=finish_reason,
                            loop_flag=loop_flag,
                            first_loop_prefix=first_loop_prefix,
                            loop_trigger=(
                                asdict(loop_trigger) if loop_trigger is not None else None
                            ),
                            max_length_hit=max_length_hit,
                            effective_max_tokens=effective_max,
                            tail_threshold=tail_threshold,
                            grading=grading,
                        )
                    )

                profile = _build_prompt_profile(
                    rollouts,
                    effective_max_tokens=effective_max,
                    loop_n=loop_n,
                    loop_k=loop_k,
                    tail_threshold=tail_threshold,
                )
                writer.write_row(
                    _build_bundle_row(
                        item,
                        prompt_token_ids=prompt_token_ids,
                        effective_max_tokens=effective_max,
                        max_model_len=rollout_cfg.max_model_len,
                        rollouts=rollouts,
                        profile=profile,
                        target_name=target_name,
                    )
                )
            print(
                f"[collect-dp-rank {rank}] processed {end}/{len(pending_items)} prompts",
                flush=True,
            )

    return agg


def _build_bundle_row(
    item: PromptWorkItem,
    *,
    prompt_token_ids: list[int],
    effective_max_tokens: int,
    max_model_len: int,
    rollouts: list[dict[str, Any]],
    profile: dict[str, Any],
    target_name: str,
) -> dict[str, Any]:
    record_metadata = (
        _sanitize_jsonable(item.record_metadata)
        if item.record_metadata is not None
        else None
    )
    majority_tail = int(profile["majority_tail"])
    row: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "sample_id": int(item.sample_id),
        "split": item.source_split,
        "record_id": item.record_id,
        "question_id": item.question_id,
        "prompt_style": item.prompt_style,
        "choices": list(item.choices) if item.choices is not None else None,
        "gold_answer": item.gold_answer,
        "gold_index": item.gold_index,
        "prompt": item.prompt,
        "prompt_token_ids": list(prompt_token_ids),
        "prompt_token_count": int(len(prompt_token_ids)),
        "effective_max_tokens": int(effective_max_tokens),
        "max_model_len": int(max_model_len),
        "prompt_too_long": False,
        "prompt_profile": {
            "target_name": target_name,
            "target_value": majority_tail,
            "target_kind": "binary",
            "p_cap": float(profile["p_cap"]),
            "p_loop": float(profile["p_loop"]),
            "fraction_loop": float(profile["fraction_loop"]),
            "fraction_len_0.5": float(profile["fraction_len_0.5"]),
            "loop_budget_share": float(profile["loop_budget_share"]),
            "mu_log_rel": float(profile["mu_log_rel"]),
            "mean_length": float(profile["mean_length"]),
            "mean_relative_length": float(profile["mean_relative_length"]),
            "tail_threshold": float(profile["tail_threshold"]),
            "tail_hit_count": int(profile["tail_hit_count"]),
            "majority_tail": majority_tail,
            "num_rollouts": int(profile["num_rollouts"]),
            "lengths": profile["lengths"],
            "relative_lengths": profile["relative_lengths"],
            "cap_hits": profile["cap_hits"],
            "loop_flags": profile["loop_flags"],
            "tail_hits": profile["tail_hits"],
            "first_loop_prefix_lengths": profile["first_loop_prefix_lengths"],
        },
        "rollouts": rollouts,
        "record_metadata": record_metadata,
    }
    return row


def _build_prompt_too_long_row(
    item: PromptWorkItem,
    *,
    prompt_token_ids: list[int],
    effective_max_tokens: int,
    max_model_len: int,
    target_name: str,
) -> dict[str, Any]:
    record_metadata = (
        _sanitize_jsonable(item.record_metadata)
        if item.record_metadata is not None
        else None
    )
    return {
        "schema": BUNDLE_SCHEMA,
        "sample_id": int(item.sample_id),
        "split": item.source_split,
        "record_id": item.record_id,
        "question_id": item.question_id,
        "prompt_style": item.prompt_style,
        "choices": list(item.choices) if item.choices is not None else None,
        "gold_answer": item.gold_answer,
        "gold_index": item.gold_index,
        "prompt": item.prompt,
        "prompt_token_ids": list(prompt_token_ids),
        "prompt_token_count": int(len(prompt_token_ids)),
        "effective_max_tokens": int(effective_max_tokens),
        "max_model_len": int(max_model_len),
        "prompt_too_long": True,
        "prompt_profile": None,
        "rollouts": [],
        "record_metadata": record_metadata,
    }


def _worker_main(
    rank: int,
    device: str,
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
    bundle_path: str,
    skip_sample_ids: set[int],
    out_queue,
) -> None:
    suppress_sem_unlink_errors()
    try:
        os.setsid()
    except OSError:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        agg = _collect_worker_stats(
            items,
            collector_cfg,
            loop_n=loop_n,
            loop_k=loop_k,
            tail_threshold=tail_threshold,
            bundle_path=bundle_path,
            skip_sample_ids=skip_sample_ids,
            rank=rank,
        )
    except Exception:
        tb = traceback.format_exc()
        try:
            out_queue.put((rank, None, tb))
        except Exception:
            pass
        raise
    out_queue.put((rank, agg, None))


def _cleanup_worker_group(worker_pid: int, rank: int) -> None:
    if worker_pid <= 0:
        return
    try:
        os.killpg(worker_pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        print(
            f"[collect-dp-rank {rank}] warning: cannot probe process group {worker_pid}: {exc}",
            flush=True,
        )
        return

    for sig, wait_s in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
        try:
            os.killpg(worker_pid, sig)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            print(
                f"[collect-dp-rank {rank}] warning: cannot signal process group {worker_pid}: {exc}",
                flush=True,
            )
            return
        time.sleep(wait_s)
        try:
            os.killpg(worker_pid, 0)
        except ProcessLookupError:
            return


def _run_collection(
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    *,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
    bundle_path: str,
    skip_sample_ids: set[int],
    preserve_rank_bundles: bool = False,
) -> WorkerAggregator:
    if collector_cfg.rollout_cfg.dp == 1:
        return _collect_worker_stats(
            items,
            collector_cfg,
            loop_n=loop_n,
            loop_k=loop_k,
            tail_threshold=tail_threshold,
            bundle_path=bundle_path,
            skip_sample_ids=skip_sample_ids,
        )

    devices = get_visible_devices()
    worker_count = min(collector_cfg.rollout_cfg.dp, len(items))
    if len(devices) < worker_count:
        raise SystemExit(
            f"Requested dp={collector_cfg.rollout_cfg.dp}, but only {len(devices)} visible GPU(s)."
        )

    rank_paths = [_rank_bundle_path(bundle_path, rank) for rank in range(worker_count)]
    if not preserve_rank_bundles:
        for path in rank_paths:
            _archive_preexisting(path)

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    processes = []
    for rank in range(worker_count):
        shard_items = items[rank::worker_count]
        p = ctx.Process(
            target=_worker_main,
            args=(
                rank,
                devices[rank],
                shard_items,
                collector_cfg,
                loop_n,
                loop_k,
                tail_threshold,
                rank_paths[rank],
                skip_sample_ids,
                out_queue,
            ),
        )
        p.start()
        processes.append(p)

    aggregators: dict[int, WorkerAggregator] = {}
    failures: list[tuple[int, str | int]] = []
    try:
        while len(aggregators) < worker_count:
            try:
                worker_rank, agg, error = out_queue.get(timeout=30)
            except queue_module.Empty:
                dead_missing = []
                for proc_rank, process in enumerate(processes):
                    if proc_rank in aggregators:
                        continue
                    if process.exitcode is not None:
                        dead_missing.append((proc_rank, process.exitcode))
                if dead_missing:
                    raise SystemExit(
                        f"DP worker(s) exited before reporting stats: {dead_missing}"
                    )
                continue

            worker_rank = int(worker_rank)
            if error is not None:
                raise SystemExit(f"DP worker {worker_rank} failed:\n{error}")
            if agg is None:
                raise SystemExit(f"DP worker {worker_rank} failed: missing aggregator")
            aggregators[worker_rank] = agg

        for proc_rank, process in enumerate(processes):
            process.join(timeout=30)
            if process.is_alive():
                if proc_rank in aggregators:
                    process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    print(
                        f"[collect-dp-rank {proc_rank}] terminated after stats were received (teardown hang)",
                        flush=True,
                    )
                else:
                    process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    failures.append((proc_rank, "alive_without_stats"))

            if process.exitcode not in (0, None) and proc_rank not in aggregators:
                failures.append((proc_rank, process.exitcode))

        if failures:
            failure_lines = [f"rank={rank}: {detail}" for rank, detail in failures]
            raise SystemExit("DP worker(s) failed:\n" + "\n".join(failure_lines))

        merged = merge_aggregators(aggregators[rank] for rank in range(worker_count))
        concat_sources = [bundle_path] + rank_paths if os.path.isfile(bundle_path) else list(rank_paths)
        concat_bundles(concat_sources, bundle_path)
        return merged
    finally:
        for rank, process in enumerate(processes):
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
            _cleanup_worker_group(process.pid, rank)
        try:
            out_queue.close()
            out_queue.join_thread()
        except Exception:
            pass


def _finalize_lcb_grading(
    bundle_path: str,
    *,
    benchmark,
    repo_path: str,
    release_version: str,
) -> dict[str, Any]:
    """Run the LCB batch grader over the bundle and rewrite with ``grading`` filled."""
    rows = list(iter_bundle_rows(bundle_path))
    records: list[LcbSampleRecord] = []
    for row in rows:
        question_id = str(row.get("question_id") or row.get("record_id") or "")
        if not question_id:
            continue
        if bool(row.get("prompt_too_long")):
            records.append(
                LcbSampleRecord(
                    question_id=question_id,
                    generation_index=-1,
                    code_output="",
                    prompt_too_long=True,
                )
            )
            continue
        for rollout in row.get("rollouts") or []:
            if not isinstance(rollout, dict):
                continue
            grading = rollout.get("grading") or {}
            code_output = str(grading.get("code_output", ""))
            records.append(
                LcbSampleRecord(
                    question_id=question_id,
                    generation_index=int(rollout.get("rollout_index", -1)),
                    code_output=code_output,
                    prompt_too_long=False,
                )
            )

    native_metrics, grading_by_record_key = livecodebench_codegen.evaluate_records(
        benchmark,
        records,
        repo_path=repo_path,
        release_version=release_version,
    )

    for row in rows:
        if bool(row.get("prompt_too_long")):
            continue
        question_id = str(row.get("question_id") or row.get("record_id") or "")
        for rollout in row.get("rollouts") or []:
            if not isinstance(rollout, dict):
                continue
            rollout_index = int(rollout.get("rollout_index", -1))
            passed = grading_by_record_key.get((question_id, rollout_index))
            grading = rollout.get("grading") or {}
            grading["passed"] = None if passed is None else bool(passed)
            grading.setdefault("code_output", "")
            rollout["grading"] = grading

    rewrite_bundle(bundle_path, rows)
    return native_metrics


def _finalize_taco_grading(bundle_path: str) -> dict[str, Any]:
    """Run the TACO grader over the bundle and rewrite with ``grading`` filled."""
    rows = list(iter_bundle_rows(bundle_path))
    native_metrics, grading_by_record_key = taco_codegen.evaluate_prompt_archive_rows(rows)
    for row in rows:
        if bool(row.get("prompt_too_long")):
            continue
        sample_id = int(row.get("sample_id", -1))
        for rollout in row.get("rollouts") or []:
            if not isinstance(rollout, dict):
                continue
            rollout_index = int(rollout.get("rollout_index", -1))
            passed = grading_by_record_key.get((sample_id, rollout_index))
            if passed is None:
                rollout["grading"] = None
                continue
            rollout["grading"] = {"passed": bool(passed)}
    rewrite_bundle(bundle_path, rows)
    return native_metrics


def _rollout_token_count(rollout: dict[str, Any]) -> int:
    value = rollout.get("completion_token_count")
    if value is None:
        value = rollout.get("length")
    if value is None:
        token_ids = rollout.get("completion_token_ids")
        if isinstance(token_ids, list):
            return len(token_ids)
        return 0
    return int(value)


def _rollout_passed(rollout: dict[str, Any]) -> bool | None:
    grading = rollout.get("grading")
    passed: Any = None
    if isinstance(grading, dict):
        passed = grading.get("passed")
        if passed is None:
            passed = grading.get("correct")
    if passed is None:
        passed = rollout.get("correct")
    if passed is None:
        return None
    if isinstance(passed, str):
        normalized = passed.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    return bool(passed)


def _aggregate_bundle_counters(bundle_path: str) -> WorkerAggregator:
    """Rebuild all sidecar counters from the completed bundle rows."""
    agg = WorkerAggregator()
    for row in iter_bundle_rows(bundle_path):
        agg.num_samples_seen += 1
        prompt_token_count = row.get("prompt_token_count")
        if isinstance(prompt_token_count, int):
            agg.prompt_length_sum += int(prompt_token_count)
            agg.prompt_length_min = (
                int(prompt_token_count)
                if agg.prompt_length_min is None
                else min(agg.prompt_length_min, int(prompt_token_count))
            )
            agg.prompt_length_max = (
                int(prompt_token_count)
                if agg.prompt_length_max is None
                else max(agg.prompt_length_max, int(prompt_token_count))
            )
        if bool(row.get("prompt_too_long")):
            agg.num_prompt_too_long += 1
            continue
        for rollout in row.get("rollouts") or []:
            if not isinstance(rollout, dict):
                continue
            token_count = _rollout_token_count(rollout)
            loop_flag = bool(rollout.get("loop_flag", False))
            max_length_hit = bool(rollout.get("max_length_hit", rollout.get("cap_hit", 0)))

            agg.num_generated += 1
            agg.length_sum += token_count
            agg.length_sq_sum += token_count * token_count
            if loop_flag:
                agg.num_looped += 1
                agg.loop_length_sum += token_count
                first_loop_prefix = rollout.get("first_loop_prefix_length")
                if first_loop_prefix is not None:
                    agg.first_loop_prefix_sum += int(first_loop_prefix)
                    agg.first_loop_prefix_count += 1
            if max_length_hit:
                agg.num_max_length_hits += 1
            if loop_flag and max_length_hit:
                agg.num_looped_and_max_length_hit += 1

            passed = _rollout_passed(rollout)
            if passed is None:
                continue
            agg.num_graded += 1
            if passed:
                agg.num_correct += 1
                agg.correct_length_sum += token_count
                if loop_flag:
                    agg.num_correct_and_looped += 1
                if max_length_hit:
                    agg.num_correct_and_max_length_hit += 1
                if loop_flag and max_length_hit:
                    agg.num_correct_and_looped_and_max_length_hit += 1
            else:
                agg.num_wrong += 1
                agg.wrong_length_sum += token_count
    return agg


def _prompt_token_summary(bundle_path: str) -> dict[str, float | int | None]:
    prompt_lengths: list[int] = []
    for row in iter_bundle_rows(bundle_path):
        ptc = row.get("prompt_token_count")
        if isinstance(ptc, int):
            prompt_lengths.append(int(ptc))
    if not prompt_lengths:
        return {
            "avg_prompt_token_count": None,
            "median_prompt_token_count": None,
            "min_prompt_token_count": None,
            "max_prompt_token_count": None,
        }
    return {
        "avg_prompt_token_count": float(sum(prompt_lengths)) / float(len(prompt_lengths)),
        "median_prompt_token_count": _median(prompt_lengths),
        "min_prompt_token_count": int(min(prompt_lengths)),
        "max_prompt_token_count": int(max(prompt_lengths)),
    }


def _prompt_profile_summary(
    bundle_path: str,
    *,
    tail_threshold: float,
) -> dict[str, Any]:
    prompt_count_total = 0
    prompt_count_too_long = 0
    profiled_prompt_count = 0
    positive_count = 0
    any_loop_positive_count = 0
    majority_loop_positive_count = 0
    any_len_0_5_positive_count = 0
    majority_len_0_5_positive_count = 0
    fraction_loop_sum = 0.0
    fraction_len_0_5_sum = 0.0
    total_len_0_5_hits = 0.0
    total_tail_hits = 0
    total_rollouts = 0
    rollout_lengths: list[int] = []
    prompt_lengths: list[int] = []
    for row in iter_bundle_rows(bundle_path):
        prompt_count_total += 1
        ptc = row.get("prompt_token_count")
        if isinstance(ptc, int):
            prompt_lengths.append(int(ptc))
        if bool(row.get("prompt_too_long")):
            prompt_count_too_long += 1
            continue
        profile = row.get("prompt_profile")
        if not isinstance(profile, dict) or int(profile.get("num_rollouts", 0)) <= 0:
            continue
        profiled_prompt_count += 1
        num_rollouts = int(profile.get("num_rollouts", 0))
        fraction_loop = float(profile.get("fraction_loop", profile.get("p_loop", 0.0)))
        fraction_len_0_5 = float(profile.get("fraction_len_0.5", 0.0))
        positive_count += int(profile.get("majority_tail", 0))
        any_loop_positive_count += int(fraction_loop > 0.0)
        majority_loop_positive_count += int(fraction_loop > 0.5)
        any_len_0_5_positive_count += int(fraction_len_0_5 > 0.0)
        majority_len_0_5_positive_count += int(fraction_len_0_5 > 0.5)
        fraction_loop_sum += fraction_loop
        fraction_len_0_5_sum += fraction_len_0_5
        total_len_0_5_hits += fraction_len_0_5 * float(num_rollouts)
        total_tail_hits += int(profile.get("tail_hit_count", 0))
        total_rollouts += num_rollouts
        for length in profile.get("lengths") or []:
            rollout_lengths.append(int(length))
    return {
        "tail_threshold": float(tail_threshold),
        "target_name": profile_target_name("majority_tail", tail_threshold=tail_threshold),
        "prompt_count_total": prompt_count_total,
        "prompt_count_profiled": profiled_prompt_count,
        "prompt_count_too_long": prompt_count_too_long,
        "prompt_positive_count": positive_count,
        "prompt_any_loop_positive_count": any_loop_positive_count,
        "prompt_majority_loop_positive_count": majority_loop_positive_count,
        "prompt_any_len_0_5_positive_count": any_len_0_5_positive_count,
        "prompt_majority_len_0_5_positive_count": majority_len_0_5_positive_count,
        "prompt_positive_rate": (
            float(positive_count) / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "prompt_any_loop_positive_rate": (
            float(any_loop_positive_count) / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "prompt_majority_loop_positive_rate": (
            float(majority_loop_positive_count) / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "prompt_any_len_0_5_positive_rate": (
            float(any_len_0_5_positive_count) / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "prompt_majority_len_0_5_positive_rate": (
            float(majority_len_0_5_positive_count) / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "prompt_mean_fraction_loop": (
            fraction_loop_sum / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "prompt_mean_fraction_len_0_5": (
            fraction_len_0_5_sum / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "completion_tail_fraction": (
            float(total_tail_hits) / float(total_rollouts)
            if total_rollouts
            else None
        ),
        "rollout_len_0_5_fraction": (
            float(total_len_0_5_hits) / float(total_rollouts)
            if total_rollouts
            else None
        ),
        "avg_generation_length": (
            float(sum(rollout_lengths)) / float(len(rollout_lengths))
            if rollout_lengths
            else None
        ),
        "median_generation_length": _median(rollout_lengths),
        "avg_prompt_token_count": (
            float(sum(prompt_lengths)) / float(len(prompt_lengths))
            if prompt_lengths
            else None
        ),
        "median_prompt_token_count": _median(prompt_lengths),
    }


def _sort_bundle(bundle_path: str) -> None:
    rows = list(iter_bundle_rows(bundle_path))
    rewrite_bundle(bundle_path, _bundle_row_sorted(rows))


def _load_sidecar_metadata(sidecar_path: str) -> dict[str, Any] | None:
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else None


def _resolved_generation_config(rollout_cfg: RolloutConfig) -> dict[str, Any]:
    resolved_top_p, resolved_top_k = resolve_sampling_defaults(
        rollout_cfg.model_id,
        top_p=rollout_cfg.top_p,
        top_k=rollout_cfg.top_k,
    )
    resolved_generation_config = rollout_cfg.to_dict()
    resolved_generation_config["top_p"] = resolved_top_p
    resolved_generation_config["top_k"] = resolved_top_k
    return resolved_generation_config


def _select_lcb_benchmark_from_bundle(
    bundle_path: str,
    *,
    repo_path: str,
    release_version: str,
) -> Any:
    question_ids: list[str] = []
    for row in iter_bundle_rows(bundle_path):
        question_id = str(row.get("question_id") or row.get("record_id") or "")
        if question_id:
            question_ids.append(question_id)
    if not question_ids:
        raise SystemExit(f"No LiveCodeBench question_id values found in {bundle_path}.")

    benchmark, _format_prompt = livecodebench_codegen.load_benchmark(
        repo_path,
        release_version,
    )
    benchmark_by_question_id = {
        str(instance.question_id): instance
        for instance in benchmark
    }
    missing = [
        question_id
        for question_id in question_ids
        if question_id not in benchmark_by_question_id
    ]
    if missing:
        raise SystemExit(
            "Bundle contains LiveCodeBench question_id values that are absent from "
            f"{release_version}: {missing[:10]}"
        )
    return [benchmark_by_question_id[question_id] for question_id in question_ids]


def _run_cpu_posthoc_grading(
    args: argparse.Namespace,
    *,
    bundle_path: str,
    lcb_benchmark: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lcb_native_metrics: dict[str, Any] = {}
    taco_native_metrics: dict[str, Any] = {}
    if args.task_kind == "livecodebench_codegen":
        benchmark = lcb_benchmark
        if benchmark is None:
            benchmark = _select_lcb_benchmark_from_bundle(
                bundle_path,
                repo_path=args.livecodebench_repo,
                release_version=args.release_version,
            )
        lcb_native_metrics = _finalize_lcb_grading(
            bundle_path,
            benchmark=benchmark,
            repo_path=args.livecodebench_repo,
            release_version=args.release_version,
        )
    elif args.task_kind == "taco_codegen":
        taco_native_metrics = _finalize_taco_grading(bundle_path)
    return lcb_native_metrics, taco_native_metrics


def _posthoc_grading_metadata(
    args: argparse.Namespace,
    *,
    status: str | None,
) -> dict[str, Any] | None:
    if args.task_kind not in CPU_POSTHOC_TASK_KINDS:
        return None
    command_parts = [
        "uv",
        "run",
        "python",
        "scripts/rollout/collect_model_stats.py",
        "--task-kind",
        args.task_kind,
        "--out",
        args.out or _derive_output_path(args),
        "--finalize-only",
        "--statistics",
        args.statistics,
        "--loop-n",
        str(args.loop_n),
        "--loop-k",
        str(args.loop_k),
        "--profile-tail-threshold",
        str(args.profile_tail_threshold),
    ]
    if args.livecodebench_repo:
        command_parts.extend(["--livecodebench-repo", args.livecodebench_repo])
    if args.task_kind == "livecodebench_codegen":
        command_parts.extend(["--release-version", args.release_version])
    return {
        "required": True,
        "status": status or "finalized",
        "device_class": "cpu",
        "finalize_command": (
            "CUDA_VISIBLE_DEVICES='' "
            + " ".join(shlex.quote(part) for part in command_parts)
        ),
    }


def _build_sidecar_payload(
    args: argparse.Namespace,
    *,
    rollout_cfg: RolloutConfig,
    statistics: list[str],
    bundle_path: str,
    task_metadata: dict[str, object] | None,
    lcb_native_metrics: dict[str, Any],
    taco_native_metrics: dict[str, Any],
    base_metadata: dict[str, Any] | None = None,
    posthoc_grading_status: str | None = None,
) -> dict[str, Any]:
    agg = _aggregate_bundle_counters(bundle_path)
    metrics = compute_metrics(agg, statistics)
    prompt_profile_summary = _prompt_profile_summary(
        bundle_path,
        tail_threshold=args.profile_tail_threshold,
    )

    if base_metadata is not None:
        sidecar_metadata = dict(base_metadata)
        sidecar_metadata.update(
            {
                "schema": BUNDLE_SCHEMA,
                "bundle_file": os.path.basename(bundle_path),
                "statistics": statistics,
                "stats_contract_version": STATS_CONTRACT_VERSION,
                "loop_detector": {"n": args.loop_n, "k": args.loop_k},
                "tail_threshold": float(args.profile_tail_threshold),
                "prompt_token_summary": _prompt_token_summary(bundle_path),
                "prompt_profile_summary": prompt_profile_summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    else:
        sidecar_metadata = {
            "schema": BUNDLE_SCHEMA,
            "bundle_file": os.path.basename(bundle_path),
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "task_kind": args.task_kind,
            "model_id": rollout_cfg.model_id,
            "generation_config": _resolved_generation_config(rollout_cfg),
            "stats_contract_version": STATS_CONTRACT_VERSION,
            "seed": args.seed,
            "statistics": statistics,
            "loop_detector": {"n": args.loop_n, "k": args.loop_k},
            "tail_threshold": float(args.profile_tail_threshold),
            "prompt_token_summary": _prompt_token_summary(bundle_path),
            "prompt_profile_summary": prompt_profile_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **({"max_samples": args.max_samples} if args.max_samples is not None else {}),
            **(
                {"release_version": args.release_version}
                if args.task_kind == "livecodebench_codegen"
                else {}
            ),
            **(task_metadata or {}),
        }

    if lcb_native_metrics:
        sidecar_metadata["lcb_native_metrics"] = lcb_native_metrics
    if taco_native_metrics:
        sidecar_metadata["taco_native_metrics"] = taco_native_metrics
    posthoc_metadata = _posthoc_grading_metadata(
        args,
        status=posthoc_grading_status,
    )
    if posthoc_metadata is not None:
        sidecar_metadata["posthoc_grading"] = posthoc_metadata

    return {
        "schema": BUNDLE_SCHEMA,
        "metadata": sidecar_metadata,
        "counts": {
            "num_samples": agg.num_samples_seen,
            "num_generated": agg.num_generated,
            "num_graded": agg.num_graded,
            "num_correct": agg.num_correct,
            "num_wrong": agg.num_wrong,
            "num_looped": agg.num_looped,
            "num_max_length_hits": agg.num_max_length_hits,
            "num_prompt_too_long": agg.num_prompt_too_long,
            "num_looped_and_max_length_hit": agg.num_looped_and_max_length_hit,
            "num_correct_and_looped": agg.num_correct_and_looped,
            "num_correct_and_max_length_hit": agg.num_correct_and_max_length_hit,
            "num_correct_and_looped_and_max_length_hit": (
                agg.num_correct_and_looped_and_max_length_hit
            ),
            "num_prompt_profiled": int(prompt_profile_summary["prompt_count_profiled"]),
            "num_prompt_majority_tail_positive": int(
                prompt_profile_summary["prompt_positive_count"]
            ),
            "num_prompt_any_loop_positive": int(
                prompt_profile_summary["prompt_any_loop_positive_count"]
            ),
            "num_prompt_majority_loop_positive": int(
                prompt_profile_summary["prompt_majority_loop_positive_count"]
            ),
            "num_prompt_any_len_0_5_positive": int(
                prompt_profile_summary["prompt_any_len_0_5_positive_count"]
            ),
            "num_prompt_majority_len_0_5_positive": int(
                prompt_profile_summary["prompt_majority_len_0_5_positive_count"]
            ),
        },
        "metrics": {
            **metrics,
            "majority_s_0.5_positive_rate": prompt_profile_summary["prompt_positive_rate"],
            "prompt_any_loop_positive_rate": prompt_profile_summary[
                "prompt_any_loop_positive_rate"
            ],
            "prompt_majority_loop_positive_rate": prompt_profile_summary[
                "prompt_majority_loop_positive_rate"
            ],
            "prompt_mean_fraction_loop": prompt_profile_summary[
                "prompt_mean_fraction_loop"
            ],
            "prompt_any_len_0_5_positive_rate": prompt_profile_summary[
                "prompt_any_len_0_5_positive_rate"
            ],
            "prompt_majority_len_0_5_positive_rate": prompt_profile_summary[
                "prompt_majority_len_0_5_positive_rate"
            ],
            "prompt_mean_fraction_len_0_5": prompt_profile_summary[
                "prompt_mean_fraction_len_0_5"
            ],
            "completion_tail_fraction": prompt_profile_summary["completion_tail_fraction"],
            "rollout_len_0_5_fraction": prompt_profile_summary[
                "rollout_len_0_5_fraction"
            ],
            "median_generation_length": prompt_profile_summary["median_generation_length"],
        },
    }


def _finalize_existing_bundle(args: argparse.Namespace) -> None:
    if args.task_kind not in CPU_POSTHOC_TASK_KINDS:
        raise SystemExit(
            "--finalize-only is only meaningful for CPU post-hoc task kinds: "
            f"{sorted(CPU_POSTHOC_TASK_KINDS)}"
        )
    _run_dataset_preflight(args)
    _run_dependency_preflight(args, require_vllm=False)
    rollout_cfg = _build_rollout_config(args)
    statistics = _parse_statistics(args.statistics)
    if not args.out and not args.dataset:
        raise SystemExit("--finalize-only requires --out unless --dataset is provided.")
    out_path = args.out or _derive_output_path(args)
    sidecar_path, bundle_path = bundle_paths(out_path)
    if not os.path.exists(bundle_path):
        raise SystemExit(f"Bundle path does not exist: {bundle_path}")

    base_metadata = _load_sidecar_metadata(sidecar_path)
    _sort_bundle(bundle_path)
    lcb_native_metrics, taco_native_metrics = _run_cpu_posthoc_grading(
        args,
        bundle_path=bundle_path,
    )
    if os.path.exists(sidecar_path):
        _archive_preexisting(sidecar_path)
    sidecar_payload = _build_sidecar_payload(
        args,
        rollout_cfg=rollout_cfg,
        statistics=statistics,
        bundle_path=bundle_path,
        task_metadata=None,
        lcb_native_metrics=lcb_native_metrics,
        taco_native_metrics=taco_native_metrics,
        base_metadata=base_metadata,
        posthoc_grading_status="finalized",
    )
    write_bundle_sidecar(sidecar_path, sidecar_payload)
    print(f"Finalized rollout bundle at {bundle_path}", flush=True)
    print(f"Wrote rollout sidecar to {sidecar_path}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.finalize_only:
        _finalize_existing_bundle(args)
        return
    if not args.dataset:
        raise SystemExit("--dataset is required unless --finalize-only is set.")
    _run_dataset_preflight(args)
    _run_dependency_preflight(args, require_vllm=True)
    rollout_cfg = _build_rollout_config(args)
    statistics = _parse_statistics(args.statistics)
    out_path = args.out or _derive_output_path(args)
    sidecar_path, bundle_path = bundle_paths(out_path)

    skip_sample_ids: set[int] = set()
    if args.resume:
        resume_paths = [bundle_path]
        if rollout_cfg.dp > 1:
            resume_paths.extend(
                _rank_bundle_path(bundle_path, rank)
                for rank in range(rollout_cfg.dp)
            )
        skip_sample_ids = _completed_sample_ids_for_resume(resume_paths)
        if skip_sample_ids:
            print(
                f"Resuming: {len(skip_sample_ids)} sample(s) already present across "
                f"{len(resume_paths)} bundle path(s)",
                flush=True,
            )
        _archive_preexisting(sidecar_path)
    else:
        _archive_preexisting(sidecar_path)
        _archive_preexisting(bundle_path)

    tokenizer = AutoTokenizer.from_pretrained(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
        use_fast=True,
    )
    items, task_metadata, lcb_benchmark = _load_prompt_items(args, tokenizer)
    if not items:
        raise SystemExit("No prompt items were loaded for collection.")

    collector_cfg = CollectorConfig(
        rollout_cfg=rollout_cfg,
        seed=args.seed,
        task_kind=args.task_kind,
        statistics=statistics,
        livecodebench_repo=args.livecodebench_repo or None,
        release_version=args.release_version,
        lm_style_override=args.lm_style_override,
        progress_metadata=None,
    )

    agg = _run_collection(
        items,
        collector_cfg,
        loop_n=args.loop_n,
        loop_k=args.loop_k,
        tail_threshold=args.profile_tail_threshold,
        bundle_path=bundle_path,
        skip_sample_ids=skip_sample_ids,
        preserve_rank_bundles=args.resume,
    )

    # Normalize row order so consumers can assume ascending sample_id.
    _sort_bundle(bundle_path)

    lcb_native_metrics: dict[str, Any] = {}
    taco_native_metrics: dict[str, Any] = {}
    posthoc_grading_status: str | None = None
    should_defer_cpu_finalize = (
        args.defer_cpu_finalize and args.task_kind in CPU_POSTHOC_TASK_KINDS
    )
    if should_defer_cpu_finalize:
        posthoc_grading_status = "deferred"
        print(
            f"Deferred CPU-only post-hoc grading for {args.task_kind}; "
            "run --finalize-only with CUDA_VISIBLE_DEVICES='' after generation.",
            flush=True,
        )
    elif args.task_kind in CPU_POSTHOC_TASK_KINDS:
        lcb_native_metrics, taco_native_metrics = _run_cpu_posthoc_grading(
            args,
            bundle_path=bundle_path,
            lcb_benchmark=lcb_benchmark,
        )
        posthoc_grading_status = "finalized"

    sidecar_payload = _build_sidecar_payload(
        args,
        rollout_cfg=rollout_cfg,
        statistics=statistics,
        bundle_path=bundle_path,
        task_metadata=task_metadata,
        lcb_native_metrics=lcb_native_metrics,
        taco_native_metrics=taco_native_metrics,
        posthoc_grading_status=posthoc_grading_status,
    )

    write_bundle_sidecar(sidecar_path, sidecar_payload)
    print(f"Wrote rollout bundle to {bundle_path}", flush=True)
    print(f"Wrote rollout sidecar to {sidecar_path}", flush=True)


if __name__ == "__main__":
    main()
