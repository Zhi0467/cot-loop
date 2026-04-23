#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import shutil
import signal
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from transformers import AutoTokenizer

from loop_probe.adapters import (
    codegen_ungraded,
    livecodebench_codegen,
    math_freeform,
    multiple_choice_gpqa,
    multiple_choice_mmlupro,
    taco_codegen,
)
from loop_probe.adapters._common import parse_row_filter_json
from loop_probe.collector import (
    ALL_STATISTICS,
    CollectorConfig,
    LcbSampleRecord,
    WorkerAggregator,
    compute_metrics,
    merge_aggregators,
)
from loop_probe.configs import RolloutConfig
from loop_probe.labeling import (
    aggregate_prompt_profile,
    find_ngram_loop_trigger,
    profile_target_name,
)
from loop_probe.prompt_format import (
    VALID_PROMPT_FORMATS,
    VALID_THINKING_MODES,
    resolve_prompt_format,
    resolve_thinking_mode,
)
from loop_probe.prompt_builder import build_math_prompt
from loop_probe.rollout import resolve_sampling_defaults
from loop_probe.types import DatasetSpec
from utils import get_visible_devices, suppress_sem_unlink_errors

TASK_CHOICES = (
    "math_freeform",
    "codegen_ungraded",
    "taco_codegen",
    "multiple_choice_gpqa",
    "multiple_choice_mmlupro",
    "livecodebench_codegen",
)
STATS_CONTRACT_VERSION = "rollout_stats_v2"
PROMPT_PROFILE_ARCHIVE_SCHEMA = "prompt_rollout_archive.v2"
PROMPT_PROFILE_SUMMARY_SCHEMA = "prompt_profile_summary.v1"


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
    parser.add_argument("--dataset", required=True)
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
        help="Comma-separated raw dataset fields to preserve in the prompt archive.",
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
            "truncation. Intended for disjoint follow-up screens such as "
            "LiveCodeBench-extra."
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
        "--resume-lcb-records-checkpoint",
        default="",
        help=(
            "Reuse an existing LiveCodeBench __lcb_records checkpoint and skip "
            "generation. Only valid with --task-kind livecodebench_codegen."
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


def _lcb_records_checkpoint_path(out_path: str) -> str:
    base, ext = os.path.splitext(out_path)
    return f"{base}__lcb_records{ext or '.json'}"


def _prompt_profile_path(out_path: str) -> str:
    base, _ext = os.path.splitext(out_path)
    return f"{base}__prompt_profile.jsonl"


def _prompt_rollout_archive_path(out_path: str) -> str:
    base, _ext = os.path.splitext(out_path)
    return f"{base}__prompt_rollout_archive.jsonl"


def _progress_path(out_path: str) -> str:
    base, _ext = os.path.splitext(out_path)
    return f"{base}__progress.json"


def _prompt_rollout_replay_path(out_path: str) -> str:
    base, _ = os.path.splitext(out_path)
    return f"{base}__rollout_archive.jsonl.gz"


def _prompt_rollout_replay_path_for_lcb_checkpoint(checkpoint_path: str) -> str:
    base, _ = os.path.splitext(checkpoint_path)
    suffix = "__lcb_records"
    archived_suffix = f"{suffix}__preexisting_"
    if base.endswith(suffix):
        prefix = base[:-len(suffix)]
        archived_tail = ""
    elif archived_suffix in base:
        prefix, archived_tail = base.split(archived_suffix, 1)
        archived_tail = f"__preexisting_{archived_tail}"
    else:
        return ""
    return f"{prefix}__rollout_archive.jsonl{archived_tail}.gz"


def _archive_preexisting_output(path: str) -> None:
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
    print(
        f"Archived preexisting output from {path} to {archived_path}",
        flush=True,
    )


def _prepare_output_paths(
    args: argparse.Namespace,
    out_path: str,
    *,
    preserve_lcb_checkpoint_path: str | None = None,
    preserve_prompt_rollout_replay_path: str | None = None,
) -> None:
    _archive_preexisting_output(out_path)
    _archive_preexisting_output(_prompt_profile_path(out_path))
    _archive_preexisting_output(_prompt_rollout_archive_path(out_path))
    _archive_preexisting_output(_progress_path(out_path))
    prompt_rollout_replay_path = _prompt_rollout_replay_path(out_path)
    if not (
        preserve_prompt_rollout_replay_path
        and os.path.abspath(prompt_rollout_replay_path)
        == os.path.abspath(preserve_prompt_rollout_replay_path)
    ):
        _archive_preexisting_output(prompt_rollout_replay_path)
    if args.task_kind == "livecodebench_codegen":
        checkpoint_path = _lcb_records_checkpoint_path(out_path)
        if preserve_lcb_checkpoint_path and os.path.abspath(
            checkpoint_path
        ) == os.path.abspath(preserve_lcb_checkpoint_path):
            return
        _archive_preexisting_output(checkpoint_path)


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
    require_vllm: bool = True,
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


def _build_prompt_profile_rows(
    item: PromptWorkItem,
    *,
    prompt: str,
    prompt_token_ids: list[int],
    effective_max_tokens: int,
    prompt_rollout_rows: list[dict[str, object]],
    profile: dict[str, object],
    target_name: str,
    target_value: int,
) -> tuple[dict[str, object], dict[str, object]]:
    record_metadata = (
        _sanitize_jsonable(item.record_metadata)
        if item.record_metadata is not None
        else None
    )
    summary_row = {
        "schema_name": PROMPT_PROFILE_SUMMARY_SCHEMA,
        "split": item.source_split,
        "sample_id": int(item.sample_id),
        "record_id": item.record_id,
        "question_id": item.question_id,
        "prompt_style": item.prompt_style,
        "choices": list(item.choices) if item.choices is not None else None,
        "prompt_token_count": int(len(prompt_token_ids)),
        "effective_max_tokens": int(effective_max_tokens),
        "target_kind": "binary",
        "target_name": target_name,
        "target_value": int(target_value),
        target_name: int(target_value),
        "p_cap": float(profile["p_cap"]),
        "p_loop": float(profile["p_loop"]),
        "loop_budget_share": float(profile["loop_budget_share"]),
        "mu_log_rel": float(profile["mu_log_rel"]),
        "mean_length": float(profile["mean_length"]),
        "mean_relative_length": float(profile["mean_relative_length"]),
        "tail_threshold": float(profile["tail_threshold"]),
        "tail_hit_count": int(profile["tail_hit_count"]),
        "majority_tail": int(profile["majority_tail"]),
        "num_rollouts": int(profile["num_rollouts"]),
        "lengths": profile["lengths"],
        "relative_lengths": profile["relative_lengths"],
        "cap_hits": profile["cap_hits"],
        "loop_flags": profile["loop_flags"],
        "tail_hits": profile["tail_hits"],
        "first_loop_prefix_lengths": profile["first_loop_prefix_lengths"],
        "finish_reasons": [row["finish_reason"] for row in prompt_rollout_rows],
        "correct_flags": [row.get("correct") for row in prompt_rollout_rows],
        "record_metadata": record_metadata,
    }
    archive_row = {
        "schema_name": PROMPT_PROFILE_ARCHIVE_SCHEMA,
        "split": item.source_split,
        "sample_id": int(item.sample_id),
        "record_id": item.record_id,
        "question_id": item.question_id,
        "prompt_style": item.prompt_style,
        "choices": list(item.choices) if item.choices is not None else None,
        "prompt": prompt,
        "prompt_token_ids": list(prompt_token_ids),
        "prompt_token_count": int(len(prompt_token_ids)),
        "effective_max_tokens": int(effective_max_tokens),
        "target_kind": "binary",
        "target_name": target_name,
        "target_value": int(target_value),
        target_name: int(target_value),
        "p_cap": float(profile["p_cap"]),
        "p_loop": float(profile["p_loop"]),
        "loop_budget_share": float(profile["loop_budget_share"]),
        "mu_log_rel": float(profile["mu_log_rel"]),
        "mean_length": float(profile["mean_length"]),
        "mean_relative_length": float(profile["mean_relative_length"]),
        "tail_threshold": float(profile["tail_threshold"]),
        "tail_hit_count": int(profile["tail_hit_count"]),
        "majority_tail": int(profile["majority_tail"]),
        "num_rollouts": int(profile["num_rollouts"]),
        "rollouts": prompt_rollout_rows,
        "record_metadata": record_metadata,
    }
    return summary_row, archive_row


def _build_prompt_too_long_rows(
    item: PromptWorkItem,
    *,
    prompt: str,
    prompt_token_ids: list[int],
    effective_max_tokens: int,
    target_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    record_metadata = (
        _sanitize_jsonable(item.record_metadata)
        if item.record_metadata is not None
        else None
    )
    common = {
        "split": item.source_split,
        "sample_id": int(item.sample_id),
        "record_id": item.record_id,
        "question_id": item.question_id,
        "prompt_style": item.prompt_style,
        "choices": list(item.choices) if item.choices is not None else None,
        "prompt_token_count": int(len(prompt_token_ids)),
        "effective_max_tokens": int(effective_max_tokens),
        "target_kind": "binary",
        "target_name": target_name,
        "target_value": None,
        target_name: None,
        "num_rollouts": 0,
        "prompt_too_long": True,
        "record_metadata": record_metadata,
    }
    summary_row = {
        "schema_name": PROMPT_PROFILE_SUMMARY_SCHEMA,
        **common,
        "finish_reasons": [],
        "correct_flags": [],
    }
    archive_row = {
        "schema_name": PROMPT_PROFILE_ARCHIVE_SCHEMA,
        **common,
        "prompt": prompt,
        "prompt_token_ids": list(prompt_token_ids),
        "rollouts": [],
    }
    return summary_row, archive_row


def _write_jsonl_rows(path: str, rows: list[dict[str, object]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_row(path: str, row: dict[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_profile_summary(
    rows: list[dict[str, object]],
    *,
    tail_threshold: float,
) -> dict[str, object]:
    profiled_rows = [
        row for row in rows if not bool(row.get("prompt_too_long")) and int(row.get("num_rollouts", 0)) > 0
    ]
    rollout_lengths: list[int] = []
    prompt_lengths = [int(row["prompt_token_count"]) for row in rows if isinstance(row.get("prompt_token_count"), int)]
    total_tail_hits = 0
    total_rollouts = 0
    positive_count = 0
    for row in profiled_rows:
        positive_count += int(row.get("majority_tail", 0))
        total_tail_hits += int(row.get("tail_hit_count", 0))
        total_rollouts += int(row.get("num_rollouts", 0))
        rollout_lengths.extend(int(length) for length in row.get("lengths", []))
    profiled_prompt_count = len(profiled_rows)
    return {
        "schema_name": PROMPT_PROFILE_SUMMARY_SCHEMA,
        "tail_threshold": float(tail_threshold),
        "target_name": profile_target_name("majority_tail", tail_threshold=tail_threshold),
        "prompt_count_total": len(rows),
        "prompt_count_profiled": profiled_prompt_count,
        "prompt_count_too_long": int(sum(bool(row.get("prompt_too_long")) for row in rows)),
        "prompt_positive_count": positive_count,
        "prompt_positive_rate": (
            float(positive_count) / float(profiled_prompt_count)
            if profiled_prompt_count
            else None
        ),
        "completion_tail_fraction": (
            float(total_tail_hits) / float(total_rollouts)
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


def _write_progress_checkpoint(
    agg: WorkerAggregator,
    collector_cfg: CollectorConfig,
    *,
    tail_threshold: float,
    out_path: str,
) -> str:
    prompt_profile_summary = _prompt_profile_summary(
        agg.prompt_profile_rows,
        tail_threshold=tail_threshold,
    )
    metrics = compute_metrics(agg, collector_cfg.statistics)
    payload = {
        "status": "in_progress",
        "metadata": {
            "task_kind": collector_cfg.task_kind,
            "model_id": collector_cfg.rollout_cfg.model_id,
            "seed": collector_cfg.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(collector_cfg.progress_metadata or {}),
            "prompt_profile_summary": prompt_profile_summary,
            "prompt_profile_file": os.path.basename(_prompt_profile_path(out_path)),
            "prompt_rollout_archive_file": os.path.basename(_prompt_rollout_archive_path(out_path)),
            "prompt_rollout_archive_schema": PROMPT_PROFILE_ARCHIVE_SCHEMA,
        },
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
        },
        "metrics": {
            **metrics,
            "majority_s_0.5_positive_rate": prompt_profile_summary["prompt_positive_rate"],
            "completion_tail_fraction": prompt_profile_summary["completion_tail_fraction"],
            "median_generation_length": prompt_profile_summary["median_generation_length"],
        },
    }
    path = _progress_path(out_path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _collect_worker_stats(
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    *,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
    progress_out_path: str | None = None,
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

    llm = LLM(**llm_kwargs)

    if rollout_cfg.max_num_seqs is not None:
        chunk_size = max(1, rollout_cfg.max_num_seqs // rollout_cfg.num_generations)
    else:
        chunk_size = len(items)
    # Keep archive memory bounded even when vLLM is allowed to process the full
    # dataset as one logical chunk.
    prompt_rollout_spill_rows = max(1, min(chunk_size, 16))
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        batch_items = items[start:end]
        batch_prompts = [item.prompt for item in batch_items]
        prompt_input_ids = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        valid_items: list[tuple[PromptWorkItem, list[int], int, int]] = []
        for item, input_ids in zip(batch_items, prompt_input_ids):
            agg.num_samples_seen += 1
            prompt_len = len(input_ids)
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
                profile_row, archive_row = _build_prompt_too_long_rows(
                    item,
                    prompt=item.prompt,
                    prompt_token_ids=[int(tok) for tok in input_ids],
                    effective_max_tokens=effective_max,
                    target_name=target_name,
                )
                agg.prompt_profile_rows.append(profile_row)
                agg.prompt_rollout_archive_rows.append(archive_row)
                if progress_out_path:
                    _append_jsonl_row(_prompt_profile_path(progress_out_path), profile_row)
                    _append_jsonl_row(_prompt_rollout_archive_path(progress_out_path), archive_row)
                    _write_progress_checkpoint(
                        agg,
                        collector_cfg,
                        tail_threshold=tail_threshold,
                        out_path=progress_out_path,
                    )
                agg.prompt_rollout_records.append(
                    {
                        "sample_id": int(item.sample_id),
                        "prompt": item.prompt,
                        "prompt_token_ids": [int(tok) for tok in input_ids],
                        "prompt_token_count": int(prompt_len),
                        "effective_max_tokens": int(effective_max),
                        "gold_answer": item.gold_answer,
                        "gold_index": item.gold_index,
                        "question_id": item.question_id,
                        "prompt_too_long": True,
                        "rollouts": [],
                    }
                )
                if len(agg.prompt_rollout_records) >= prompt_rollout_spill_rows:
                    _spill_prompt_rollout_records(
                        agg,
                        rank=rank,
                    )
                if collector_cfg.task_kind == "livecodebench_codegen":
                    agg.lcb_sample_records.append(
                        LcbSampleRecord(
                            question_id=item.question_id or "",
                            generation_index=-1,
                            code_output="",
                            token_count=0,
                            prompt_token_count=prompt_len,
                            total_token_count=prompt_len,
                            effective_max_tokens=effective_max,
                            max_model_len=rollout_cfg.max_model_len,
                            loop_flag=False,
                            max_length_hit=False,
                            finish_reason="prompt_too_long",
                            prompt_too_long=True,
                            first_loop_prefix_length=None,
                        )
                    )
                continue
            valid_items.append((item, [int(tok) for tok in input_ids], prompt_len, effective_max))
            valid_items.append(
                (item, [int(tok) for tok in input_ids], prompt_len, effective_max)
            )

        if valid_items:
            for item, prompt_token_ids, prompt_len, effective_max in valid_items:
                sampling_params = SamplingParams(
                    temperature=rollout_cfg.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=rollout_cfg.max_tokens,
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
                prompt_rollout_rows: list[dict[str, object]] = []
                prompt_rollouts: list[dict[str, Any]] = []
                prompt_rollout_rows: list[dict[str, object]] = []
                for generation_index, sample in enumerate(output.outputs):
                    text = str(getattr(sample, "text", ""))
                    token_ids = getattr(sample, "token_ids", None)
                    if not token_ids:
                        token_ids = tokenizer.encode(text, add_special_tokens=False)
                    token_ids = [int(token_id) for token_id in token_ids]
                    token_ids = [int(token_id) for token_id in token_ids]
                    token_count = len(token_ids)
                    total_token_count = _total_token_count(prompt_len, token_count)
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

                    prompt_rollouts.append(
                        {
                            "generation_index": int(generation_index),
                            "completion_text": text,
                            "completion_token_ids": token_ids,
                            "completion_token_count": int(token_count),
                            "total_token_count": int(total_token_count),
                            "finish_reason": finish_reason,
                            "loop_flag": bool(loop_flag),
                            "max_length_hit": bool(max_length_hit),
                            "first_loop_prefix_length": first_loop_prefix,
                            "loop_trigger": (
                                asdict(loop_trigger) if loop_trigger is not None else None
                            ),
                        }
                    )
                    correct_flag: int | None = None
                    if collector_cfg.task_kind == "livecodebench_codegen":
                        agg.lcb_sample_records.append(
                            LcbSampleRecord(
                                question_id=item.question_id or "",
                                generation_index=generation_index,
                                code_output=livecodebench_codegen.extract_code_output(
                                    text,
                                    repo_path=collector_cfg.livecodebench_repo or "",
                                    model_id=rollout_cfg.model_id,
                                    lm_style_override=collector_cfg.lm_style_override,
                                ),
                                token_count=token_count,
                                prompt_token_count=prompt_len,
                                total_token_count=total_token_count,
                                effective_max_tokens=effective_max,
                                max_model_len=rollout_cfg.max_model_len,
                                loop_flag=loop_flag,
                                max_length_hit=max_length_hit,
                                finish_reason=finish_reason,
                                prompt_too_long=False,
                                first_loop_prefix_length=first_loop_prefix,
                            )
                        )
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
                        correct_flag = int(correct)
                        _update_qa_stats(
                            agg,
                            correct=correct,
                            token_count=token_count,
                            loop_flag=loop_flag,
                            max_length_hit=max_length_hit,
                        )
                    prompt_rollout_row = {
                        "rollout_index": int(generation_index),
                        "completion_text": text,
                        "completion_token_ids": token_ids,
                        "finish_reason": finish_reason,
                        "length": int(token_count),
                        "relative_length": (
                            float(token_count) / float(effective_max)
                            if effective_max > 0
                            else None
                        ),
                        "cap_hit": int(max_length_hit),
                        "loop_flag": int(loop_flag),
                        "tail_hit": int(
                            effective_max > 0
                            and (float(token_count) / float(effective_max)) >= float(tail_threshold)
                        ),
                        "first_loop_prefix_length": first_loop_prefix,
                        "correct": correct_flag,
                    }
                    if collector_cfg.task_kind == "livecodebench_codegen":
                        prompt_rollout_row["question_id"] = item.question_id
                    prompt_rollout_rows.append(prompt_rollout_row)

                profile = aggregate_prompt_profile(
                    [row["completion_token_ids"] for row in prompt_rollout_rows],
                    effective_max_tokens=effective_max,
                    loop_n=loop_n,
                    loop_k=loop_k,
                    tail_threshold=tail_threshold,
                    finish_reasons=[str(row["finish_reason"]) for row in prompt_rollout_rows],
                )
                prompt_profile_row, archive_row = _build_prompt_profile_rows(
                    item,
                    prompt=item.prompt,
                    prompt_token_ids=prompt_token_ids,
                    effective_max_tokens=effective_max,
                    prompt_rollout_rows=prompt_rollout_rows,
                    profile=profile,
                    target_name=target_name,
                    target_value=int(profile["majority_tail"]),
                )
                agg.prompt_profile_rows.append(prompt_profile_row)
                agg.prompt_rollout_archive_rows.append(archive_row)
                if progress_out_path:
                    _append_jsonl_row(_prompt_profile_path(progress_out_path), prompt_profile_row)
                    _append_jsonl_row(_prompt_rollout_archive_path(progress_out_path), archive_row)
                    _write_progress_checkpoint(
                        agg,
                        collector_cfg,
                        tail_threshold=tail_threshold,
                        out_path=progress_out_path,
                    )
                    if collector_cfg.task_kind == "livecodebench_codegen":
                        _write_lcb_records_checkpoint(agg, progress_out_path)

                agg.prompt_rollout_records.append(
                    {
                        "sample_id": int(item.sample_id),
                        "prompt": item.prompt,
                        "prompt_token_ids": prompt_token_ids,
                        "prompt_token_count": int(prompt_len),
                        "effective_max_tokens": int(effective_max),
                        "gold_answer": item.gold_answer,
                        "gold_index": item.gold_index,
                        "question_id": item.question_id,
                        "prompt_too_long": False,
                        "rollouts": prompt_rollouts,
                    }
                )
                if len(agg.prompt_rollout_records) >= prompt_rollout_spill_rows:
                    _spill_prompt_rollout_records(
                        agg,
                        rank=rank,
                    )

        _spill_prompt_rollout_records(
            agg,
            rank=rank,
        )
        print(
            f"[collect-dp-rank {rank}] processed {end}/{len(items)} prompts",
            flush=True,
        )

    return agg


def _worker_main(
    rank: int,
    device: str,
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
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
            rank=rank,
        )
        _spill_prompt_rollout_records(
            agg,
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


def _spill_prompt_rollout_records(
    agg: WorkerAggregator,
    *,
    rank: int,
) -> None:
    if not agg.prompt_rollout_records:
        return

    fd, part_path = tempfile.mkstemp(
        prefix=f"collect_model_stats_prompt_rollouts_rank{rank}_",
        suffix=".jsonl.gz",
    )
    os.close(fd)
    try:
        with gzip.open(part_path, "wt", encoding="utf-8") as handle:
            for row in agg.prompt_rollout_records:
                json.dump(row, handle, ensure_ascii=True)
                handle.write("\n")
    except Exception:
        try:
            os.unlink(part_path)
        except OSError:
            pass
        raise

    agg.prompt_rollout_part_paths.append(part_path)
    agg.prompt_rollout_records.clear()
    print(
        f"[collect-dp-rank {rank}] spooled prompt rollout archive chunk to {part_path}",
        flush=True,
    )


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
    progress_out_path: str | None = None,
) -> WorkerAggregator:
    if collector_cfg.rollout_cfg.dp == 1:
        return _collect_worker_stats(
            items,
            collector_cfg,
            loop_n=loop_n,
            loop_k=loop_k,
            tail_threshold=tail_threshold,
            progress_out_path=progress_out_path,
        )

    devices = get_visible_devices()
    worker_count = min(collector_cfg.rollout_cfg.dp, len(items))
    if len(devices) < worker_count:
        raise SystemExit(
            f"Requested dp={collector_cfg.rollout_cfg.dp}, but only {len(devices)} visible GPU(s)."
        )

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
            failure_lines = []
            for rank, detail in failures:
                failure_lines.append(f"rank={rank}: {detail}")
            raise SystemExit("DP worker(s) failed:\n" + "\n".join(failure_lines))

        merged = merge_aggregators(aggregators[rank] for rank in range(worker_count))
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


def _load_lcb_records_checkpoint(checkpoint_path: str) -> WorkerAggregator:
    if not checkpoint_path:
        raise SystemExit("Missing --resume-lcb-records-checkpoint path.")
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"LiveCodeBench checkpoint path does not exist: {checkpoint_path}"
        )

    with open(checkpoint_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(
            f"LiveCodeBench checkpoint must contain a JSON object: {checkpoint_path}"
        )

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise SystemExit(
            f"LiveCodeBench checkpoint is missing a records list: {checkpoint_path}"
        )

    agg = WorkerAggregator()
    agg.lcb_sample_records = [
        LcbSampleRecord.from_dict(record) for record in raw_records
    ]

    prompt_lengths: dict[str, int] = {}
    for index, record in enumerate(agg.lcb_sample_records):
        if not record.question_id:
            raise SystemExit(
                "LiveCodeBench checkpoint record is missing question_id at "
                f"records[{index}]."
            )
        existing_prompt_length = prompt_lengths.get(record.question_id)
        if existing_prompt_length is None:
            prompt_lengths[record.question_id] = record.prompt_token_count
        elif existing_prompt_length != record.prompt_token_count:
            raise SystemExit(
                "LiveCodeBench checkpoint has inconsistent prompt_token_count for "
                f"question_id {record.question_id!r}."
            )

        if record.prompt_too_long:
            agg.num_prompt_too_long += 1
            continue

        agg.num_generated += 1
        agg.length_sum += record.token_count
        agg.length_sq_sum += record.token_count * record.token_count
        if record.loop_flag:
            agg.num_looped += 1
            agg.loop_length_sum += record.token_count
            if record.first_loop_prefix_length is not None:
                agg.first_loop_prefix_sum += record.first_loop_prefix_length
                agg.first_loop_prefix_count += 1
        if record.max_length_hit:
            agg.num_max_length_hits += 1
        if record.loop_flag and record.max_length_hit:
            agg.num_looped_and_max_length_hit += 1

    agg.num_samples_seen = len(prompt_lengths)
    agg.prompt_length_sum = sum(prompt_lengths.values())
    if prompt_lengths:
        prompt_length_values = list(prompt_lengths.values())
        agg.prompt_length_min = min(prompt_length_values)
        agg.prompt_length_max = max(prompt_length_values)

    expected_counts = {
        "num_samples": agg.num_samples_seen,
        "num_generated": agg.num_generated,
        "num_prompt_too_long": agg.num_prompt_too_long,
    }
    for field_name, actual in expected_counts.items():
        expected = payload.get(field_name)
        if expected is None:
            continue
        if int(expected) != actual:
            raise SystemExit(
                f"LiveCodeBench checkpoint {checkpoint_path} has inconsistent "
                f"{field_name}: payload={expected} reconstructed={actual}."
            )

    return agg


def _apply_lcb_grades(
    agg: WorkerAggregator,
    benchmark,
    *,
    repo_path: str,
    release_version: str,
) -> tuple[dict[str, Any], dict[tuple[str, int], bool]]:
    native_metrics, grading_by_record_key = livecodebench_codegen.evaluate_records(
        benchmark,
        agg.lcb_sample_records,
        repo_path=repo_path,
        release_version=release_version,
    )
    records_by_key = {}
    for record in agg.lcb_sample_records:
        if record.prompt_too_long:
            continue
        record_key = (record.question_id, record.generation_index)
        if record_key in records_by_key:
            raise RuntimeError(f"Duplicate LiveCodeBench record key: {record_key}")
        records_by_key[record_key] = record

    missing_keys = sorted(set(records_by_key) - set(grading_by_record_key))
    if missing_keys:
        raise RuntimeError(
            "Missing LiveCodeBench grades for generated records: "
            f"{missing_keys[:10]}"
        )

    agg.num_graded = 0
    agg.num_correct = 0
    agg.num_wrong = 0
    agg.correct_length_sum = 0
    agg.wrong_length_sum = 0
    agg.num_correct_and_looped = 0
    agg.num_correct_and_max_length_hit = 0
    agg.num_correct_and_looped_and_max_length_hit = 0

    for record_key, passed in grading_by_record_key.items():
        record = records_by_key[record_key]
        agg.num_graded += 1
        if passed:
            agg.num_correct += 1
            agg.correct_length_sum += record.token_count
            if record.loop_flag:
                agg.num_correct_and_looped += 1
            if record.max_length_hit:
                agg.num_correct_and_max_length_hit += 1
            if record.loop_flag and record.max_length_hit:
                agg.num_correct_and_looped_and_max_length_hit += 1
        else:
            agg.num_wrong += 1
            agg.wrong_length_sum += record.token_count
    return native_metrics, grading_by_record_key


def _annotate_lcb_prompt_rows_with_grades(
    agg: WorkerAggregator,
    *,
    grading_by_record_key: dict[tuple[str, int], bool],
) -> None:
    for row in agg.prompt_rollout_archive_rows:
        question_id = row.get("question_id")
        rollouts = row.get("rollouts")
        if not isinstance(question_id, str) or not isinstance(rollouts, list):
            continue
        correct_flags: list[int | None] = []
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                correct_flags.append(None)
                continue
            rollout_index = int(rollout.get("rollout_index", len(correct_flags)))
            passed = grading_by_record_key.get((question_id, rollout_index))
            rollout["correct"] = None if passed is None else int(bool(passed))
            correct_flags.append(rollout["correct"])
        row["correct_flags"] = correct_flags

    by_question_id = {
        str(row.get("question_id")): row for row in agg.prompt_rollout_archive_rows if isinstance(row, dict)
    }
    for row in agg.prompt_profile_rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str):
            continue
        archive_row = by_question_id.get(question_id)
        if not isinstance(archive_row, dict):
            continue
        correct_flags = archive_row.get("correct_flags")
        row["correct_flags"] = correct_flags


def _apply_taco_grades(
    agg: WorkerAggregator,
) -> tuple[dict[str, object], dict[tuple[int, int], bool]]:
    native_metrics, grading_by_record_key = taco_codegen.evaluate_prompt_archive_rows(
        agg.prompt_rollout_archive_rows,
    )
    agg.num_graded = 0
    agg.num_correct = 0
    agg.num_wrong = 0
    agg.correct_length_sum = 0
    agg.wrong_length_sum = 0
    agg.num_correct_and_looped = 0
    agg.num_correct_and_max_length_hit = 0
    agg.num_correct_and_looped_and_max_length_hit = 0

    for row in agg.prompt_rollout_archive_rows:
        sample_id = row.get("sample_id")
        rollouts = row.get("rollouts")
        if not isinstance(sample_id, int) or not isinstance(rollouts, list):
            continue
        correct_flags: list[int | None] = []
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                correct_flags.append(None)
                continue
            rollout_index = int(rollout.get("rollout_index", len(correct_flags)))
            passed = grading_by_record_key.get((sample_id, rollout_index))
            rollout["correct"] = None if passed is None else int(bool(passed))
            correct_flags.append(rollout["correct"])
            if passed is None:
                continue
            agg.num_graded += 1
            token_count = int(rollout.get("length", 0))
            loop_flag = bool(int(rollout.get("loop_flag", 0)))
            max_length_hit = bool(int(rollout.get("cap_hit", 0)))
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
        row["correct_flags"] = correct_flags

    by_sample_id = {
        int(row.get("sample_id")): row
        for row in agg.prompt_rollout_archive_rows
        if isinstance(row, dict) and isinstance(row.get("sample_id"), int)
    }
    for row in agg.prompt_profile_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, int):
            continue
        archive_row = by_sample_id.get(sample_id)
        if not isinstance(archive_row, dict):
            continue
        row["correct_flags"] = archive_row.get("correct_flags")
    return native_metrics, grading_by_record_key


def _write_lcb_records_checkpoint(agg: WorkerAggregator, out_path: str) -> str:
    checkpoint_path = _lcb_records_checkpoint_path(out_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "num_samples": agg.num_samples_seen,
        "num_generated": agg.num_generated,
        "num_prompt_too_long": agg.num_prompt_too_long,
        "records": [asdict(record) for record in agg.lcb_sample_records],
    }
    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote LiveCodeBench records checkpoint to {checkpoint_path}", flush=True)
    return checkpoint_path


def _write_prompt_rollout_archive(agg: WorkerAggregator, out_path: str) -> str:
    if not agg.prompt_rollout_records and not agg.prompt_rollout_part_paths:
        return ""
    archive_path = _prompt_rollout_replay_path(out_path)
    archive_dir = os.path.dirname(archive_path)
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)
    with gzip.open(archive_path, "wt", encoding="utf-8") as handle:
        for row in agg.prompt_rollout_records:
            json.dump(row, handle, ensure_ascii=True)
            handle.write("\n")
        for part_path in agg.prompt_rollout_part_paths:
            with gzip.open(part_path, "rt", encoding="utf-8") as source_handle:
                for line in source_handle:
                    handle.write(line)
            try:
                os.unlink(part_path)
            except OSError:
                pass
    print(f"Wrote prompt rollout archive to {archive_path}", flush=True)
    return archive_path


def _restore_prompt_rollout_archive_for_resume(
    archive_path: str,
    out_path: str,
) -> str:
    if not archive_path:
        return ""
    if not os.path.isfile(archive_path):
        return ""

    destination_path = _prompt_rollout_replay_path(out_path)
    if os.path.abspath(archive_path) == os.path.abspath(destination_path):
        return archive_path

    destination_dir = os.path.dirname(destination_path)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copyfile(archive_path, destination_path)
    print(
        "Copied prompt rollout archive from resumed checkpoint to "
        f"{destination_path}",
        flush=True,
    )
    return destination_path


def _prompt_token_summary(agg: WorkerAggregator) -> dict[str, float | int | None]:
    prompt_lengths = [
        int(row["prompt_token_count"])
        for row in agg.prompt_profile_rows
        if isinstance(row, dict) and isinstance(row.get("prompt_token_count"), int)
    ]
    avg_prompt_length = (
        float(agg.prompt_length_sum) / float(agg.num_samples_seen)
        if agg.num_samples_seen
        else None
    )
    return {
        "avg_prompt_token_count": avg_prompt_length,
        "median_prompt_token_count": _median(prompt_lengths),
        "min_prompt_token_count": agg.prompt_length_min,
        "max_prompt_token_count": agg.prompt_length_max,
    }


def main() -> None:
    args = _parse_args()
    resume_lcb_checkpoint = args.resume_lcb_records_checkpoint.strip()
    resume_prompt_rollout_archive = (
        _prompt_rollout_replay_path_for_lcb_checkpoint(resume_lcb_checkpoint)
        if resume_lcb_checkpoint
        else ""
    )
    if resume_lcb_checkpoint and args.task_kind != "livecodebench_codegen":
        raise SystemExit(
            "--resume-lcb-records-checkpoint is only valid with "
            "--task-kind livecodebench_codegen."
        )
    _run_dataset_preflight(args)
    _run_dependency_preflight(
        args,
        require_vllm=not bool(resume_lcb_checkpoint),
    )
    rollout_cfg = _build_rollout_config(args)
    statistics = _parse_statistics(args.statistics)
    out_path = args.out or _derive_output_path(args)
    _prepare_output_paths(
        args,
        out_path,
        preserve_lcb_checkpoint_path=resume_lcb_checkpoint or None,
        preserve_prompt_rollout_replay_path=resume_prompt_rollout_archive or None,
    )

    tokenizer = None
    if not (
        resume_lcb_checkpoint and args.task_kind == "livecodebench_codegen"
    ):
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
        progress_metadata={
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "generation_config": {
                **rollout_cfg.to_dict(),
                "top_p": resolve_sampling_defaults(
                    rollout_cfg.model_id,
                    top_p=rollout_cfg.top_p,
                    top_k=rollout_cfg.top_k,
                )[0],
                "top_k": resolve_sampling_defaults(
                    rollout_cfg.model_id,
                    top_p=rollout_cfg.top_p,
                    top_k=rollout_cfg.top_k,
                )[1],
            },
            "statistics": statistics,
            "loop_detector": {"n": args.loop_n, "k": args.loop_k},
            **({"max_samples": args.max_samples} if args.max_samples is not None else {}),
            **(
                {"release_version": args.release_version}
                if args.task_kind == "livecodebench_codegen"
                else {}
            ),
            **task_metadata,
        },
    )

    if resume_lcb_checkpoint:
        agg = _load_lcb_records_checkpoint(resume_lcb_checkpoint)
        if agg.num_samples_seen != len(items):
            raise SystemExit(
                "LiveCodeBench checkpoint prompt count does not match the current "
                f"dataset selection ({agg.num_samples_seen} vs {len(items)})."
            )
        task_metadata = {
            **task_metadata,
            "resumed_from_lcb_records_checkpoint": True,
        }
        if agg.first_loop_prefix_count != agg.num_looped:
            task_metadata = {
                **task_metadata,
                "lcb_checkpoint_missing_first_loop_prefix_length": True,
            }
    else:
        agg = _run_collection(
            items,
            collector_cfg,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            tail_threshold=args.profile_tail_threshold,
            progress_out_path=out_path,
        )

    prompt_rollout_replay_path = _write_prompt_rollout_archive(agg, out_path)
    if not prompt_rollout_replay_path and resume_prompt_rollout_archive:
        prompt_rollout_replay_path = _restore_prompt_rollout_archive_for_resume(
            resume_prompt_rollout_archive,
            out_path,
        )

    lcb_native_metrics: dict[str, Any] = {}
    taco_native_metrics: dict[str, object] = {}
    if args.task_kind == "livecodebench_codegen":
        if not resume_lcb_checkpoint:
            _write_lcb_records_checkpoint(agg, out_path)
        lcb_native_metrics, grading_by_record_key = _apply_lcb_grades(
            agg,
            lcb_benchmark,
            repo_path=args.livecodebench_repo,
            release_version=args.release_version,
        )
        _annotate_lcb_prompt_rows_with_grades(
            agg,
            grading_by_record_key=grading_by_record_key,
        )
    elif args.task_kind == "taco_codegen":
        taco_native_metrics, _grading_by_record_key = _apply_taco_grades(agg)
    agg.prompt_profile_rows.sort(key=lambda row: int(row.get("sample_id", -1)))
    agg.prompt_rollout_archive_rows.sort(key=lambda row: int(row.get("sample_id", -1)))

    metrics = compute_metrics(agg, statistics)
    prompt_profile_summary = _prompt_profile_summary(
        agg.prompt_profile_rows,
        tail_threshold=args.profile_tail_threshold,
    )
    prompt_profile_relpath = (
        os.path.basename(_prompt_profile_path(out_path))
        if agg.prompt_profile_rows
        else None
    )
    prompt_rollout_archive_relpath = (
        os.path.basename(_prompt_rollout_archive_path(out_path))
        if agg.prompt_rollout_archive_rows
        else None
    )
    resolved_generation_config = rollout_cfg.to_dict()
    resolved_top_p, resolved_top_k = resolve_sampling_defaults(
        rollout_cfg.model_id,
        top_p=rollout_cfg.top_p,
        top_k=rollout_cfg.top_k,
    )
    resolved_generation_config["top_p"] = resolved_top_p
    resolved_generation_config["top_k"] = resolved_top_k

    payload = {
        "metadata": {
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "task_kind": args.task_kind,
            "model_id": rollout_cfg.model_id,
            "generation_config": resolved_generation_config,
            "stats_contract_version": STATS_CONTRACT_VERSION,
            "seed": args.seed,
            "statistics": statistics,
            "loop_detector": {"n": args.loop_n, "k": args.loop_k},
            "prompt_token_summary": _prompt_token_summary(agg),
            "prompt_profile_summary": prompt_profile_summary,
            "prompt_profile_file": prompt_profile_relpath,
            "prompt_rollout_archive_file": prompt_rollout_archive_relpath,
            "prompt_rollout_archive_schema": (
                PROMPT_PROFILE_ARCHIVE_SCHEMA
                if prompt_rollout_archive_relpath is not None
                else None
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **({"max_samples": args.max_samples} if args.max_samples is not None else {}),
            **(
                {"release_version": args.release_version}
                if args.task_kind == "livecodebench_codegen"
                else {}
            ),
            **(
                {
                    "rollout_replay_archive_file": os.path.basename(
                        prompt_rollout_replay_path
                    )
                }
                if prompt_rollout_replay_path
                else {}
            ),
            **task_metadata,
        },
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
        },
        "metrics": metrics,
    }
    payload["counts"]["num_prompt_profiled"] = int(prompt_profile_summary["prompt_count_profiled"])
    payload["counts"]["num_prompt_majority_tail_positive"] = int(
        prompt_profile_summary["prompt_positive_count"]
    )
    payload["metrics"]["majority_s_0.5_positive_rate"] = prompt_profile_summary[
        "prompt_positive_rate"
    ]
    payload["metrics"]["completion_tail_fraction"] = prompt_profile_summary[
        "completion_tail_fraction"
    ]
    payload["metrics"]["median_generation_length"] = prompt_profile_summary[
        "median_generation_length"
    ]
    if lcb_native_metrics:
        payload["metadata"]["lcb_native_metrics"] = lcb_native_metrics
    if taco_native_metrics:
        payload["metadata"]["taco_native_metrics"] = taco_native_metrics

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if agg.prompt_profile_rows:
        _write_jsonl_rows(_prompt_profile_path(out_path), agg.prompt_profile_rows)
    if agg.prompt_rollout_archive_rows:
        _write_jsonl_rows(
            _prompt_rollout_archive_path(out_path),
            agg.prompt_rollout_archive_rows,
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote model stats to {out_path}", flush=True)


if __name__ == "__main__":
    main()
