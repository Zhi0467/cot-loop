#!/usr/bin/env python3
"""Build loop-probe train/test datasets from Hugging Face datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import zlib
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.configs import get_rollout_config, preset_choices
from loop_probe.hf_data import load_prompt_records, specs_equal, split_records
from loop_probe.labeling import (
    LABEL_TARGET_CHOICES,
    PROMPT_PROFILE_TARGET_CHOICES,
    aggregate_prompt_profile,
    labels_from_rollouts,
    profile_target_name,
    profile_target_value,
)
from loop_probe.adapters import (
    livecodebench_codegen,
    multiple_choice_gpqa,
    multiple_choice_mmlupro,
)
from loop_probe.prefill import (
    FEATURE_POOLING_CHOICES,
    extract_prefill_features_multi,
    load_prefill_model_and_tokenizer,
)
from loop_probe.rollout import generate_grouped_rollouts, generate_rollout_token_ids
from loop_probe.serialization import save_split_shards, write_manifest
from loop_probe.types import DatasetSpec, SampleRecord
from utils import build_prompt

DEFAULT_TEST_DATASET = "data/aime_2024_2025.jsonl"

RATIO_SPLIT_SOURCES = {
    "single_dataset_split",
    "same_train_test_spec_split",
    "same_train_test_source_ratio_split",
}
FEATURE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
BALANCE_CHOICES = ("none", "downsample")
ROLLOUT_LAST_TOKEN_ALL_LAYERS_MEAN = "rollout_last_token_all_layers_mean"
COMPLETION_POOLING_CHOICES = (ROLLOUT_LAST_TOKEN_ALL_LAYERS_MEAN,)
ALL_FEATURE_POOLING_CHOICES = FEATURE_POOLING_CHOICES + COMPLETION_POOLING_CHOICES
TARGET_KIND_CHOICES = ("binary", "probability", "regression")
BINARY_TARGET_MODE_CHOICES = ("rollout_label", "prompt_majority_tail")
TASK_KIND_CHOICES = (
    "math_freeform",
    "multiple_choice_gpqa",
    "multiple_choice_mmlupro",
    "livecodebench_codegen",
)
PROFILE_TARGET_CHOICES = PROMPT_PROFILE_TARGET_CHOICES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-max-samples", type=int, default=None)

    parser.add_argument(
        "--test-dataset",
        default="",
        help=(
            "Optional test dataset (HF dataset id or local JSONL path). "
            f"If omitted, defaults to '{DEFAULT_TEST_DATASET}'."
        ),
    )
    parser.add_argument("--test-config", default=None)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--test-max-samples", type=int, default=None)

    parser.add_argument("--prompt-field", required=True)
    parser.add_argument(
        "--answer-field",
        default="answer",
        help=(
            "Gold-answer field for math-style datasets. This does not affect prompt "
            "construction, but it is recorded in the manifest so downstream "
            "correctness reconstruction can reuse the right column."
        ),
    )
    parser.add_argument(
        "--task-kind",
        choices=TASK_KIND_CHOICES,
        default="math_freeform",
        help=(
            "Prompt/task formatting path. Use the multiple-choice modes so "
            "GPQA or MMLU-Pro prompts include answer options instead of the "
            "boxed-answer math template."
        ),
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.1,
        help=(
            "Used for ratio-based splitting when train/test come from one source."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--livecodebench-repo", default="")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--lm-style-override", default=None)

    parser.add_argument("--model-preset", choices=preset_choices(), default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)

    parser.set_defaults(trust_remote_code=None)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )

    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument(
        "--target-kind",
        choices=TARGET_KIND_CHOICES,
        default="binary",
        help=(
            "Target family to build. 'binary' preserves the legacy rollout-level "
            "loop label unless --binary-target-mode switches it to a prompt-level "
            "repeated-rollout head; 'probability' aggregates repeated rollouts into "
            "a prompt-level soft target."
        ),
    )
    parser.add_argument(
        "--binary-target-mode",
        choices=BINARY_TARGET_MODE_CHOICES,
        default="rollout_label",
        help=(
            "Binary-label surface. 'rollout_label' keeps the legacy one-rollout label. "
            "'prompt_majority_tail' uses repeated rollouts and labels each prompt with "
            "a strict majority vote over 1[L / E >= --profile-tail-threshold]."
        ),
    )
    parser.add_argument(
        "--label-target",
        choices=LABEL_TARGET_CHOICES,
        default="eventual_loop",
        help=(
            "How to derive binary labels from rollouts. "
            "'eventual_loop' matches the legacy eventual-loop bit; "
            "'loop_by_horizon' marks prompts whose rollout enters a loop "
            "within the first --label-horizon generated tokens."
        ),
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=None,
        help=(
            "Positive token horizon used when --label-target=loop_by_horizon."
        ),
    )
    parser.add_argument(
        "--profile-tail-threshold",
        type=float,
        default=0.9,
        help=(
            "Tail threshold used by prompt-profile targets. "
            "For probability targets it applies when "
            "--profile-target=s_tail and the emitted scalar is s_t = P(L / E >= t); "
            "it is ignored by direct rate targets such as p_loop / p_cap. "
            "for prompt-profile binary targets it defines the per-rollout "
            "tail event used by the strict-majority label."
        ),
    )
    parser.add_argument(
        "--profile-target",
        choices=PROFILE_TARGET_CHOICES,
        default=None,
        help=(
            "Prompt-profile scalar to supervise when --target-kind is "
            "'probability'/'regression' or when --binary-target-mode switches "
            "binary labels onto a prompt-profile head. Defaults to "
            "'majority_tail' for prompt-profile binary, 's_tail' for probability, "
            "and 'mean_relative_length' for regression. Probability heads also "
            "support direct prompt-level rates such as 'p_loop' and 'p_cap'; "
            "regression heads also support 'loop_budget_share'."
        ),
    )
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for rollout-completion feature extraction. "
            "Keep small for long generated trajectories."
        ),
    )
    parser.add_argument(
        "--feature-pooling",
        choices=ALL_FEATURE_POOLING_CHOICES,
        default="last_token_all_layers_stack",
        help="How to pool token activations into one feature tensor per prompt.",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=-1,
        help=(
            "Transformer layer index for features "
            "(0 = first layer, -1 = final layer)."
        ),
    )
    parser.add_argument(
        "--feature-key",
        default=None,
        help=(
            "Optional key for the primary feature view. "
            "If omitted, it is derived from pooling/layer."
        ),
    )
    parser.add_argument(
        "--extra-feature-view",
        action="append",
        default=[],
        metavar="KEY:POOLING:LAYER",
        help=(
            "Additional feature view to materialize into the same dataset. "
            "Repeat this flag for multiple views."
        ),
    )
    parser.add_argument(
        "--reuse-if-compatible",
        action="store_true",
        help="Skip rebuilding when out-dir has a compatible manifest and shard files.",
    )
    parser.add_argument(
        "--balance-train",
        choices=BALANCE_CHOICES,
        default="none",
        help=(
            "Optional balancing policy for the train split after loop labels are built. "
            "'downsample' keeps equal positives/negatives by random majority-class subsampling."
        ),
    )
    parser.add_argument(
        "--balance-test",
        choices=BALANCE_CHOICES,
        default="none",
        help=(
            "Optional balancing policy for the test split after loop labels are built. "
            "'downsample' keeps equal positives/negatives by random majority-class subsampling."
        ),
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=None,
        help="Optional balancing RNG seed. Defaults to --seed when omitted.",
    )
    parser.add_argument("--out-dir", required=True)

    return parser.parse_args()


def _layer_tag(feature_layer: int) -> str:
    if feature_layer == -1:
        return "final"
    if feature_layer >= 0:
        return f"layer{feature_layer}"
    return f"layer_m{abs(feature_layer)}"


def _default_feature_key(*, pooling: str, feature_layer: int) -> str:
    return f"{pooling}_{_layer_tag(feature_layer)}"


def _parse_feature_view(raw: str) -> tuple[str, dict[str, object]]:
    key, sep, remainder = raw.partition(":")
    if sep == "":
        raise SystemExit(
            "--extra-feature-view must match KEY:POOLING:LAYER, "
            f"got '{raw}'."
        )
    if ":" not in remainder:
        raise SystemExit(
            "--extra-feature-view must match KEY:POOLING:LAYER, "
            f"got '{raw}'."
        )

    pooling, _, layer_text = remainder.rpartition(":")
    key = key.strip()
    pooling = pooling.strip()
    layer_text = layer_text.strip()

    if not key:
        raise SystemExit("Feature view key cannot be empty.")
    if not FEATURE_KEY_PATTERN.fullmatch(key):
        raise SystemExit(
            "Feature view key must match [A-Za-z0-9_.-]+, "
            f"got '{key}'."
        )
    if pooling not in ALL_FEATURE_POOLING_CHOICES:
        raise SystemExit(
            f"Unknown feature pooling '{pooling}' for view '{key}'. "
            f"Valid: {ALL_FEATURE_POOLING_CHOICES}"
        )
    try:
        layer = int(layer_text)
    except Exception as exc:
        raise SystemExit(
            f"Feature layer for view '{key}' is not an integer: '{layer_text}'."
        ) from exc
    return key, {"pooling": pooling, "layer": layer}


def _feature_stage(pooling: str) -> str:
    if pooling in COMPLETION_POOLING_CHOICES:
        return "completion"
    return "prefill"


def _validate_layer_for_pooling(*, pooling: str, layer: int, key: str) -> None:
    if pooling in COMPLETION_POOLING_CHOICES and layer != -1:
        raise SystemExit(
            "Rollout-completion all-layer pooling ignores --feature-layer and "
            f"requires -1 for feature view '{key}', got {layer}."
        )
    if pooling in (
        "last_token_all_layers_mean",
        "last_token_all_layers_concat",
        "last_token_all_layers_stack",
        "last16_all_layers_concat",
        "last8_prev8_delta_all_layers_concat",
        "last16_mid16_delta_all_layers_concat",
        "last16_plus_delta8_all_layers_concat",
    ) and layer != -1:
        raise SystemExit(
            "All-layer prefill pooling ignores --feature-layer and requires -1 for "
            f"feature view '{key}', got {layer}."
        )


def _resolve_feature_views(args: argparse.Namespace) -> tuple[str, dict[str, dict[str, object]]]:
    if args.feature_key is not None:
        primary_key = args.feature_key.strip()
    else:
        primary_key = ""
    if not primary_key:
        primary_key = _default_feature_key(
            pooling=args.feature_pooling,
            feature_layer=args.feature_layer,
        )
    if not FEATURE_KEY_PATTERN.fullmatch(primary_key):
        raise SystemExit(
            "Primary feature key must match [A-Za-z0-9_.-]+, "
            f"got '{primary_key}'."
        )

    primary_pooling = args.feature_pooling
    primary_layer = int(args.feature_layer)
    _validate_layer_for_pooling(
        pooling=primary_pooling,
        layer=primary_layer,
        key=primary_key,
    )

    feature_views: dict[str, dict[str, object]] = {
        primary_key: {
            "pooling": primary_pooling,
            "layer": primary_layer,
            "stage": _feature_stage(primary_pooling),
        }
    }

    for raw in args.extra_feature_view:
        key, spec = _parse_feature_view(raw)
        spec["stage"] = _feature_stage(str(spec["pooling"]))
        _validate_layer_for_pooling(
            pooling=str(spec["pooling"]),
            layer=int(spec["layer"]),
            key=key,
        )
        prior = feature_views.get(key)
        if prior is not None and prior != spec:
            raise SystemExit(
                f"Duplicate feature view key '{key}' has conflicting settings."
            )
        feature_views[key] = spec

    return primary_key, feature_views


def _split_feature_views_by_stage(
    feature_views: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    prefill: dict[str, dict[str, object]] = {}
    completion: dict[str, dict[str, object]] = {}
    for key, spec in feature_views.items():
        stage = str(spec.get("stage", "prefill"))
        if stage == "completion":
            completion[key] = spec
        else:
            prefill[key] = spec
    return prefill, completion


def _sample_ids(records: list[SampleRecord]) -> list[int]:
    return [rec.sample_id for rec in records]


def _prompts(records: list[SampleRecord]) -> list[str]:
    return [rec.prompt for rec in records]


def _prompt_token_lengths(tokenizer, records: list[SampleRecord]) -> list[int]:
    return [len(token_ids) for token_ids in _prompt_token_ids(tokenizer, records)]


def _prompt_token_ids(tokenizer, records: list[SampleRecord]) -> list[list[int]]:
    if not records:
        return []
    encoded = tokenizer(
        [rec.prompt for rec in records],
        add_special_tokens=False,
    )["input_ids"]
    if len(encoded) != len(records):
        raise RuntimeError(
            "Tokenizer returned a mismatched number of prompt tokenizations."
        )
    return [list(token_ids) for token_ids in encoded]


def _apply_chat_prompt(
    tokenizer,
    records: list[SampleRecord],
    *,
    num_repetition: int = 1,
) -> list[SampleRecord]:
    formatted: list[SampleRecord] = []
    for rec in records:
        if rec.prompt_style == "math_freeform":
            prompt_text = build_prompt(tokenizer, rec.prompt, num_repetition)
        elif rec.prompt_style == "multiple_choice_gpqa":
            if not rec.choices:
                raise SystemExit("GPQA record is missing answer choices.")
            if num_repetition != 1:
                raise SystemExit("GPQA prompt formatting does not support num_repetition != 1.")
            prompt_text = multiple_choice_gpqa.build_mcq_prompt(
                tokenizer,
                rec.prompt,
                list(rec.choices),
            )
        elif rec.prompt_style == "multiple_choice_mmlupro":
            if not rec.choices:
                raise SystemExit("MMLU-Pro record is missing answer choices.")
            if num_repetition != 1:
                raise SystemExit(
                    "MMLU-Pro prompt formatting does not support num_repetition != 1."
                )
            prompt_text = multiple_choice_mmlupro.build_mcq_prompt(
                tokenizer,
                rec.prompt,
                list(rec.choices),
            )
        elif rec.prompt_style == "livecodebench_codegen":
            if num_repetition != 1:
                raise SystemExit(
                    "LiveCodeBench prompt formatting does not support num_repetition != 1."
                )
            prompt_text = rec.prompt
        else:
            raise SystemExit(f"Unsupported prompt_style '{rec.prompt_style}'.")
        formatted.append(
            SampleRecord(
                sample_id=rec.sample_id,
                prompt=prompt_text,
                source_split=rec.source_split,
                prompt_style=rec.prompt_style,
                choices=rec.choices,
            )
        )
    return formatted


def _load_task_records(
    spec: DatasetSpec,
    *,
    prompt_field: str,
    task_kind: str,
    seed: int,
    rollout_model_id: str,
    livecodebench_repo: str,
    release_version: str,
    lm_style_override: str | None,
) -> list[SampleRecord]:
    if task_kind == "math_freeform":
        return load_prompt_records(spec, prompt_field)
    if task_kind == "multiple_choice_gpqa":
        raw_records = multiple_choice_gpqa.load_and_shuffle(spec, seed)
        return [
            SampleRecord(
                sample_id=record.sample_id,
                prompt=record.prompt,
                source_split=record.source_split,
                prompt_style="multiple_choice_gpqa",
                choices=tuple(options),
            )
            for record, options, _gold_letter in raw_records
        ]
    if task_kind == "multiple_choice_mmlupro":
        raw_records = multiple_choice_mmlupro.load_samples(spec)
        return [
            SampleRecord(
                sample_id=record.sample_id,
                prompt=record.prompt,
                source_split=record.source_split,
                prompt_style="multiple_choice_mmlupro",
                choices=tuple(options),
            )
            for record, options, _gold_answer, _gold_index in raw_records
        ]
    if task_kind == "livecodebench_codegen":
        if not livecodebench_repo:
            raise SystemExit(
                "--livecodebench-repo is required when --task-kind=livecodebench_codegen."
            )
        effective_release_version = (
            str(spec.config) if spec.config not in (None, "") else release_version
        )
        benchmark, format_prompt = livecodebench_codegen.load_benchmark(
            livecodebench_repo,
            effective_release_version,
        )
        normalized_split = str(spec.split).strip().lower()
        if normalized_split not in ("", "all", "train", "validation", "val", "test"):
            raise SystemExit(
                "Unsupported LiveCodeBench split "
                f"'{spec.split}'. Use one of train/validation/test/all."
            )
        if normalized_split not in ("", "all"):
            def _partition_bucket(question_id: object) -> int:
                return zlib.crc32(str(question_id).encode("utf-8")) % 10

            if normalized_split == "train":
                benchmark = [
                    row for row in benchmark if _partition_bucket(row.question_id) < 8
                ]
            elif normalized_split in ("validation", "val"):
                benchmark = [
                    row for row in benchmark if _partition_bucket(row.question_id) == 8
                ]
            else:
                benchmark = [
                    row for row in benchmark if _partition_bucket(row.question_id) == 9
                ]
        prompt_records, _lm_style = livecodebench_codegen.build_prompts(
            benchmark,
            format_prompt,
            repo_path=livecodebench_repo,
            model_id=rollout_model_id,
            lm_style_override=lm_style_override,
            max_samples=spec.max_samples,
        )
        return [
            SampleRecord(
                sample_id=idx,
                prompt=prompt,
                source_split=spec.split,
                prompt_style="livecodebench_codegen",
            )
            for idx, (_question_id, prompt) in enumerate(prompt_records)
        ]
    raise SystemExit(f"Unsupported --task-kind '{task_kind}'.")


def _same_data_source(a: DatasetSpec, b: DatasetSpec) -> bool:
    if os.path.isfile(a.dataset) and os.path.isfile(b.dataset):
        try:
            return os.path.samefile(a.dataset, b.dataset)
        except OSError:
            return os.path.abspath(a.dataset) == os.path.abspath(b.dataset)

    return (
        a.dataset == b.dataset
        and a.config == b.config
        and a.split == b.split
    )


def _same_task_source(
    a: DatasetSpec,
    b: DatasetSpec,
    *,
    task_kind: str,
) -> bool:
    if task_kind == "livecodebench_codegen":
        return _same_data_source(a, b)
    return _same_data_source(a, b)


def _with_max_samples(spec: DatasetSpec, max_samples: int | None) -> DatasetSpec:
    return DatasetSpec(
        dataset=spec.dataset,
        config=spec.config,
        split=spec.split,
        max_samples=max_samples,
    )


def _split_source_uses_ratio(split_source: str) -> bool:
    return split_source in RATIO_SPLIT_SOURCES


def _make_specs(args: argparse.Namespace) -> tuple[DatasetSpec, DatasetSpec | None]:
    train_spec = DatasetSpec(
        dataset=args.train_dataset,
        config=args.train_config,
        split=args.train_split,
        max_samples=args.train_max_samples,
    )

    if args.test_dataset:
        return train_spec, DatasetSpec(
            dataset=args.test_dataset,
            config=args.test_config,
            split=args.test_split,
            max_samples=args.test_max_samples,
        )

    if args.task_kind != "math_freeform":
        explicit_test_split = (
            args.test_config is not None
            or args.test_split != "test"
            or args.test_max_samples is not None
        )
        if explicit_test_split:
            return train_spec, DatasetSpec(
                dataset=args.train_dataset,
                config=args.test_config
                if args.test_config is not None
                else args.train_config,
                split=args.test_split,
                max_samples=args.test_max_samples,
            )
        return train_spec, None

    test_dataset = DEFAULT_TEST_DATASET
    if not os.path.isfile(test_dataset):
        raise SystemExit(
            f"Default test dataset '{test_dataset}' was not found. "
            "Pass --test-dataset explicitly or create the default file."
        )

    return train_spec, DatasetSpec(
        dataset=test_dataset,
        config=args.test_config,
        split=args.test_split,
        max_samples=args.test_max_samples,
    )


def _resolve_splits(
    args: argparse.Namespace,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec | None,
    *,
    rollout_model_id: str,
) -> tuple[list[SampleRecord], list[SampleRecord], str]:
    if test_spec is None:
        merged_records = _load_task_records(
            train_spec,
            prompt_field=args.prompt_field,
            task_kind=str(args.task_kind),
            seed=args.seed,
            rollout_model_id=rollout_model_id,
            livecodebench_repo=str(args.livecodebench_repo),
            release_version=str(args.release_version),
            lm_style_override=args.lm_style_override,
        )
        train_records, test_records = split_records(
            merged_records,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
        split_source = "single_dataset_split"
        return train_records, test_records, split_source

    if specs_equal(train_spec, test_spec):
        merged_records = _load_task_records(
            train_spec,
            prompt_field=args.prompt_field,
            task_kind=str(args.task_kind),
            seed=args.seed,
            rollout_model_id=rollout_model_id,
            livecodebench_repo=str(args.livecodebench_repo),
            release_version=str(args.release_version),
            lm_style_override=args.lm_style_override,
        )
        train_records, test_records = split_records(
            merged_records,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
        split_source = "same_train_test_spec_split"
        return train_records, test_records, split_source

    if _same_task_source(train_spec, test_spec, task_kind=str(args.task_kind)):
        train_max = train_spec.max_samples
        test_max = test_spec.max_samples

        # When both split sizes are explicitly requested from the same source,
        # build a single shuffled pool and carve out disjoint train/test subsets.
        if train_max is not None and test_max is not None:
            requested_total = train_max + test_max
            merged_spec = _with_max_samples(train_spec, requested_total)
            merged_records = _load_task_records(
                merged_spec,
                prompt_field=args.prompt_field,
                task_kind=str(args.task_kind),
                seed=args.seed,
                rollout_model_id=rollout_model_id,
                livecodebench_repo=str(args.livecodebench_repo),
                release_version=str(args.release_version),
                lm_style_override=args.lm_style_override,
            )
            if len(merged_records) < requested_total:
                raise SystemExit(
                    "Requested disjoint split sizes exceed available rows: "
                    f"need train_max_samples + test_max_samples = {requested_total}, "
                    f"got {len(merged_records)}."
                )
            work = list(merged_records)
            rng = random.Random(args.seed)
            rng.shuffle(work)
            test_records = work[:test_max]
            train_records = work[test_max : test_max + train_max]
            split_source = "same_train_test_source_count_split"
            return train_records, test_records, split_source

        merged_spec = _with_max_samples(train_spec, None)
        merged_records = _load_task_records(
            merged_spec,
            prompt_field=args.prompt_field,
            task_kind=str(args.task_kind),
            seed=args.seed,
            rollout_model_id=rollout_model_id,
            livecodebench_repo=str(args.livecodebench_repo),
            release_version=str(args.release_version),
            lm_style_override=args.lm_style_override,
        )
        train_records, test_records = split_records(
            merged_records,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
        if train_max is not None:
            train_records = train_records[:train_max]
        if test_max is not None:
            test_records = test_records[:test_max]
        if not train_records:
            raise SystemExit("Train split is empty.")
        if not test_records:
            raise SystemExit("Test split is empty.")
        split_source = "same_train_test_source_ratio_split"
        return train_records, test_records, split_source

    train_records = _load_task_records(
        train_spec,
        prompt_field=args.prompt_field,
        task_kind=str(args.task_kind),
        seed=args.seed,
        rollout_model_id=rollout_model_id,
        livecodebench_repo=str(args.livecodebench_repo),
        release_version=str(args.release_version),
        lm_style_override=args.lm_style_override,
    )
    test_records = _load_task_records(
        test_spec,
        prompt_field=args.prompt_field,
        task_kind=str(args.task_kind),
        seed=args.seed,
        rollout_model_id=rollout_model_id,
        livecodebench_repo=str(args.livecodebench_repo),
        release_version=str(args.release_version),
        lm_style_override=args.lm_style_override,
    )
    if not train_records:
        raise SystemExit("Train split is empty.")
    if not test_records:
        raise SystemExit("Test split is empty.")
    split_source = "separate_specs"
    return train_records, test_records, split_source


def _resolve_split_source(
    train_spec: DatasetSpec,
    test_spec: DatasetSpec | None,
    *,
    task_kind: str,
) -> str:
    if test_spec is None:
        return "single_dataset_split"
    if specs_equal(train_spec, test_spec):
        return "same_train_test_spec_split"
    if _same_task_source(train_spec, test_spec, task_kind=task_kind):
        if train_spec.max_samples is not None and test_spec.max_samples is not None:
            return "same_train_test_source_count_split"
        return "same_train_test_source_ratio_split"
    return "separate_specs"


def _task_loader_config(args: argparse.Namespace) -> dict[str, object] | None:
    if args.task_kind != "livecodebench_codegen":
        return None

    repo_path = str(args.livecodebench_repo).strip()
    return {
        "livecodebench_repo": os.path.realpath(repo_path) if repo_path else "",
        "release_version": str(args.release_version),
        "lm_style_override": (
            ""
            if args.lm_style_override in (None, "")
            else str(args.lm_style_override)
        ),
    }


def _manifest_answer_field(task_kind: str, answer_field: str) -> str | None:
    if task_kind == "math_freeform":
        return str(answer_field)
    return None


def _split_shards_exist(out_dir: str, split_info: object) -> bool:
    if not isinstance(split_info, dict):
        return False
    shard_paths = split_info.get("shards")
    if not isinstance(shard_paths, list) or not shard_paths:
        return False
    for rel_path in shard_paths:
        if not isinstance(rel_path, str):
            return False
        if not os.path.exists(os.path.join(out_dir, rel_path)):
            return False
    return True


def _view_shards_exist(
    manifest: dict[str, object],
    out_dir: str,
    *,
    feature_key: str,
) -> bool:
    feature_views = manifest.get("feature_views")
    if not isinstance(feature_views, dict):
        return False
    view_info = feature_views.get(feature_key)
    if not isinstance(view_info, dict):
        return False
    return _split_shards_exist(out_dir, view_info.get("train")) and _split_shards_exist(
        out_dir, view_info.get("test")
    )


def _probe_cache_status(
    out_dir: str,
    *,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec | None,
    prompt_field: str,
    task_kind: str,
    split_source: str,
    split_ratio: float,
    loop_n: int,
    loop_k: int,
    target_spec: dict[str, object],
    label_spec: dict[str, object],
    rollout_config: dict[str, object],
    requested_primary_feature_key: str,
    requested_feature_views: dict[str, dict[str, object]],
    seed: int,
    balance_train: str,
    balance_test: str,
    balance_seed: int,
    answer_field: str,
    task_loader_config: dict[str, object] | None,
) -> tuple[bool, str]:
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False, "manifest.json not found"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return False, f"failed to read manifest.json: {exc}"

    expected = {
        "prompt_field": prompt_field,
        "answer_field": _manifest_answer_field(task_kind, answer_field),
        "task_kind": task_kind,
        "split_source": split_source,
        "loop_detector": {"n": loop_n, "k": loop_k},
        "rollout_config": rollout_config,
        "train_spec": asdict(train_spec),
        "test_spec": asdict(test_spec) if test_spec else None,
        "task_loader_config": task_loader_config,
    }
    if _split_source_uses_ratio(split_source):
        expected["split_ratio"] = split_ratio
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"manifest mismatch on '{key}'"

    manifest_target_spec = manifest.get("target_spec")
    if manifest_target_spec is None:
        manifest_target_spec = {
            "kind": "binary",
            "name": label_spec.get("target", "eventual_loop"),
            "horizon": label_spec.get("horizon"),
        }
    if manifest_target_spec != target_spec:
        return False, "manifest mismatch on 'target_spec'"

    if _target_source(target_spec) == "rollout_label":
        manifest_label_spec = manifest.get(
            "label_spec",
            {
                "target": "eventual_loop",
                "horizon": None,
            },
        )
        if manifest_label_spec != label_spec:
            return False, "manifest mismatch on 'label_spec'"

    expected_balancing = {
        "train": balance_train,
        "test": balance_test,
        "seed": balance_seed,
    }
    manifest_balancing = manifest.get("balancing")
    if manifest_balancing is None:
        if balance_train != "none" or balance_test != "none":
            return False, "manifest missing balancing metadata"
    elif manifest_balancing != expected_balancing:
        return False, "manifest mismatch on 'balancing'"

    manifest_seed = manifest.get("seed", None)
    if manifest_seed is not None:
        try:
            seed_value = int(manifest_seed)
        except Exception:
            return False, "manifest seed is not an integer"
        if seed_value != seed:
            return False, f"manifest seed={seed_value} != requested seed={seed}"

    feature_views = manifest.get("feature_views")
    if isinstance(feature_views, dict):
        manifest_default_feature_key = manifest.get("default_feature_key")
        if not isinstance(manifest_default_feature_key, str) or not manifest_default_feature_key:
            return False, "manifest missing default_feature_key for multi-view dataset"
        if manifest_default_feature_key != requested_primary_feature_key:
            return (
                False,
                "manifest mismatch on 'default_feature_key' "
                f"(cached='{manifest_default_feature_key}', requested='{requested_primary_feature_key}')",
            )

        for key, requested in requested_feature_views.items():
            view_info = feature_views.get(key)
            if not isinstance(view_info, dict):
                return False, f"missing requested feature view '{key}'"
            expected_stage = str(requested.get("stage", "prefill"))
            cached_stage = view_info.get("stage")
            if cached_stage is None:
                if expected_stage != "prefill":
                    return (
                        False,
                        f"feature view '{key}' missing stage metadata (expected '{expected_stage}')",
                    )
            elif cached_stage != expected_stage:
                return False, f"feature view '{key}' mismatch on 'stage'"
            expected_view = {
                "pooling": requested["pooling"],
                "layer": requested["layer"],
            }
            for view_field, expected_value in expected_view.items():
                if view_info.get(view_field) != expected_value:
                    return (
                        False,
                        f"feature view '{key}' mismatch on '{view_field}'",
                    )
            if not _view_shards_exist(manifest, out_dir, feature_key=key):
                return False, f"feature view '{key}' shards are missing"
    else:
        if len(requested_feature_views) != 1:
            return False, "manifest lacks multi-view data for requested feature set"
        only_view = next(iter(requested_feature_views.values()))
        if str(only_view.get("stage", "prefill")) != "prefill":
            return False, "legacy single-view manifest cannot satisfy completion-stage feature request"
        expected_feature = {
            "pooling": only_view["pooling"],
            "layer": only_view["layer"],
        }
        manifest_feature = manifest.get("feature_extraction")
        resolved_feature = expected_feature
        if manifest_feature is None:
            legacy_default_feature = {
                "pooling": "last_token",
                "layer": -1,
            }
            if expected_feature != legacy_default_feature:
                return (
                    False,
                    "legacy manifest missing feature_extraction metadata cannot satisfy "
                    "non-default feature view request",
                )
            resolved_feature = legacy_default_feature
        elif manifest_feature != expected_feature:
            return False, "manifest mismatch on 'feature_extraction'"
        expected_primary_key = _default_feature_key(
            pooling=resolved_feature["pooling"],
            feature_layer=resolved_feature["layer"],
        )
        if requested_primary_feature_key != expected_primary_key:
            return (
                False,
                "legacy manifest cannot satisfy "
                f"primary feature key '{requested_primary_feature_key}'",
            )
        if not _split_shards_exist(out_dir, manifest.get("train")):
            return False, "train shards are missing"
        if not _split_shards_exist(out_dir, manifest.get("test")):
            return False, "test shards are missing"

    if _target_source(target_spec) == "prompt_profile":
        prompt_profile_files = manifest.get("prompt_profile_files")
        if not isinstance(prompt_profile_files, dict):
            return False, "manifest missing prompt_profile_files for prompt-profile target"
        for split_name in ("train", "test"):
            rel_path = prompt_profile_files.get(split_name)
            if not isinstance(rel_path, str) or not rel_path:
                return False, f"manifest missing prompt-profile diagnostics for split '{split_name}'"
            if not os.path.exists(os.path.join(out_dir, rel_path)):
                return False, f"prompt-profile diagnostics missing for split '{split_name}'"
        prompt_rollout_archive_file = manifest.get("prompt_rollout_archive_file")
        if not isinstance(prompt_rollout_archive_file, str) or not prompt_rollout_archive_file:
            return False, "manifest missing prompt_rollout_archive_file for prompt-profile target"
        if not os.path.exists(os.path.join(out_dir, prompt_rollout_archive_file)):
            return False, "prompt_rollout_archive_file is missing"

    if manifest_seed is None:
        return True, "compatible legacy manifest match (no seed recorded)"
    return True, "manifest and shards match requested config"


def _build_split(
    split_name: str,
    records: list[SampleRecord],
    *,
    model,
    tokenizer,
    device,
    prefill_batch_size: int,
    feature_views: dict[str, dict[str, object]],
):
    prompt_count = len(records)
    sample_ids = _sample_ids(records)
    print(f"[{split_name}] extracting prefill features for {prompt_count} prompts", flush=True)
    view_specs = {
        key: (
            str(spec["pooling"]),
            int(spec["layer"]),
        )
        for key, spec in feature_views.items()
    }
    features_by_key = extract_prefill_features_multi(
        model,
        tokenizer,
        device,
        records,
        feature_views=view_specs,
        log_prefix=split_name,
        batch_size=prefill_batch_size,
    )
    return features_by_key, sample_ids


def _label_split(
    split_name: str,
    prompts: list[str],
    *,
    rollout_cfg,
    seed: int,
    loop_n: int,
    loop_k: int,
    label_target: str,
    label_horizon: int | None,
    return_rollout_token_ids: bool = True,
) -> tuple[list[int], list[list[int]] | None]:
    print(f"[{split_name}] running rollouts for {len(prompts)} prompts", flush=True)
    rollout_token_ids = generate_rollout_token_ids(
        prompts,
        rollout_cfg,
        seed=seed,
    )
    labels = labels_from_rollouts(
        rollout_token_ids,
        loop_n=loop_n,
        loop_k=loop_k,
        label_target=label_target,
        label_horizon=label_horizon,
    )
    if not return_rollout_token_ids:
        return labels, None
    return labels, rollout_token_ids


def _target_source(target_spec: dict[str, object]) -> str:
    kind = str(target_spec.get("kind", "binary"))
    if kind in ("probability", "regression"):
        return "prompt_profile"
    return str(target_spec.get("source", "rollout_label"))


def _build_prompt_profile_targets(
    split_name: str,
    records: list[SampleRecord],
    prompt_token_ids: list[list[int]],
    sample_ids: list[int],
    split_names: list[str],
    *,
    rollout_cfg,
    seed: int,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
    profile_target: str,
    target_kind: str,
) -> tuple[list[int | float], list[dict[str, object]], list[dict[str, object]]]:
    if (
        len(records) != len(prompt_token_ids)
        or len(records) != len(sample_ids)
        or len(records) != len(split_names)
    ):
        raise SystemExit(
            "Prompt-profile target builder got mismatched record/token/sample_id counts."
        )
    prompts = _prompts(records)
    print(
        f"[{split_name}] running grouped rollouts for {len(prompts)} prompts "
        f"(n={rollout_cfg.num_generations})",
        flush=True,
    )
    grouped_rollouts = generate_grouped_rollouts(
        prompts,
        rollout_cfg,
        seed=seed,
    )
    if len(grouped_rollouts) != len(prompts):
        raise RuntimeError(
            "Grouped rollout generator returned a mismatched number of prompt groups."
        )

    target_name = profile_target_name(
        profile_target,
        tail_threshold=tail_threshold,
    )
    targets: list[int | float] = []
    rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []
    for rec, prompt_ids, sample_id, split, prompt_rollouts in zip(
        records,
        prompt_token_ids,
        sample_ids,
        split_names,
        grouped_rollouts,
        strict=True,
    ):
        prompt_len = len(prompt_ids)
        effective_max_tokens = int(rollout_cfg.max_tokens)
        if rollout_cfg.max_model_len is not None and rollout_cfg.max_model_len > 0:
            effective_max_tokens = min(
                effective_max_tokens,
                int(rollout_cfg.max_model_len) - int(prompt_len),
            )
        if effective_max_tokens < 1:
            raise SystemExit(
                "Prompt exceeds the available generation budget under the current "
                "max_model_len / max_tokens setting."
            )

        profile = aggregate_prompt_profile(
            [rollout.token_ids for rollout in prompt_rollouts],
            effective_max_tokens=effective_max_tokens,
            loop_n=loop_n,
            loop_k=loop_k,
            tail_threshold=tail_threshold,
        )
        raw_target_value = profile_target_value(
            profile,
            profile_target=profile_target,
        )
        target_value = (
            int(raw_target_value)
            if target_kind == "binary"
            else float(raw_target_value)
        )
        targets.append(target_value)
        rows.append(
            {
                "split": split,
                "sample_id": int(sample_id),
                "prompt_token_count": int(prompt_len),
                "effective_max_tokens": int(effective_max_tokens),
                "target_kind": target_kind,
                "target_name": target_name,
                "target_value": target_value,
                target_name: target_value,
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
                "finish_reasons": [rollout.finish_reason for rollout in prompt_rollouts],
            }
        )
        rollout_rows = []
        for rollout_index, rollout, length, relative_length, cap_hit, loop_flag, tail_hit, first_loop_prefix in zip(
            range(len(prompt_rollouts)),
            prompt_rollouts,
            profile["lengths"],
            profile["relative_lengths"],
            profile["cap_hits"],
            profile["loop_flags"],
            profile["tail_hits"],
            profile["first_loop_prefix_lengths"],
            strict=True,
        ):
            rollout_rows.append(
                {
                    "rollout_index": int(rollout_index),
                    "completion_text": rollout.text,
                    "finish_reason": rollout.finish_reason,
                    "length": int(length),
                    "relative_length": float(relative_length),
                    "cap_hit": int(cap_hit),
                    "loop_flag": int(loop_flag),
                    "tail_hit": int(tail_hit),
                    "first_loop_prefix_length": first_loop_prefix,
                }
            )
        archive_rows.append(
            {
                "split": split,
                "sample_id": int(sample_id),
                "source_split": rec.source_split,
                "prompt_style": rec.prompt_style,
                "choices": list(rec.choices) if rec.choices is not None else None,
                "prompt": rec.prompt,
                "prompt_token_ids": list(prompt_ids),
                "prompt_token_count": int(prompt_len),
                "effective_max_tokens": int(effective_max_tokens),
                "target_kind": target_kind,
                "target_name": target_name,
                "target_value": target_value,
                target_name: target_value,
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
                "rollouts": rollout_rows,
            }
        )
    return targets, rows, archive_rows


def _extract_completion_features(
    split_name: str,
    records: list[SampleRecord],
    rollout_token_ids: list[list[int]],
    *,
    model,
    tokenizer,
    device,
    feature_views: dict[str, dict[str, object]],
    batch_size: int,
    max_model_len: int | None,
) -> dict[str, torch.Tensor]:
    if not feature_views:
        return {}
    if batch_size < 1:
        raise SystemExit("--completion-batch-size must be >= 1.")
    if len(records) != len(rollout_token_ids):
        raise SystemExit(
            "Completion feature extraction got mismatched record/rollout counts: "
            f"{len(records)} vs {len(rollout_token_ids)}."
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise SystemExit(
                "Tokenizer has no pad_token/eos_token; cannot batch completion prompts."
            )
        tokenizer.pad_token = tokenizer.eos_token

    features_by_key: dict[str, list[torch.Tensor]] = {k: [] for k in feature_views}
    total = len(records)
    pad_id = int(tokenizer.pad_token_id)

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_sequences: list[list[int]] = []
            batch_records = records[start:end]
            batch_rollouts = rollout_token_ids[start:end]
            batch_prompt_ids = tokenizer(
                [rec.prompt for rec in batch_records],
                add_special_tokens=False,
            )["input_ids"]
            if len(batch_prompt_ids) != len(batch_records):
                raise RuntimeError(
                    "Tokenizer returned a mismatched number of completion prompts."
                )
            for idx, (prompt_ids, gen_ids) in enumerate(
                zip(batch_prompt_ids, batch_rollouts),
                start=start,
            ):
                merged = list(prompt_ids) + [int(tok) for tok in gen_ids]
                if max_model_len is not None and max_model_len > 0 and len(merged) > max_model_len:
                    merged = merged[-max_model_len:]
                if not merged:
                    raise RuntimeError(
                        f"Encountered empty prompt+rollout sequence at index {idx}."
                    )
                batch_sequences.append(merged)
            max_len = max(len(seq) for seq in batch_sequences)

            input_ids = torch.full(
                (len(batch_sequences), max_len),
                fill_value=pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch_sequences), max_len),
                dtype=torch.long,
                device=device,
            )
            for row_idx, seq in enumerate(batch_sequences):
                seq_t = torch.tensor(seq, dtype=torch.long, device=device)
                seq_len = int(seq_t.numel())
                input_ids[row_idx, :seq_len] = seq_t
                attention_mask[row_idx, :seq_len] = 1

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            if out.hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden states during completion feature extraction."
                )
            num_hidden_layers = len(out.hidden_states) - 1
            if num_hidden_layers < 1:
                raise RuntimeError("Model returned no hidden layers during completion extraction.")

            last_token_idx = attention_mask.sum(dim=1) - 1
            if torch.any(last_token_idx < 0):
                raise RuntimeError(
                    "Found an empty completion sequence during hidden-state extraction."
                )
            batch_idx = torch.arange(input_ids.size(0), device=device)
            per_layer_last_token = torch.stack(
                [
                    out.hidden_states[layer_idx + 1][batch_idx, last_token_idx]
                    for layer_idx in range(num_hidden_layers)
                ],
                dim=1,
            )

            for key, spec in feature_views.items():
                pooling = str(spec["pooling"])
                if pooling == ROLLOUT_LAST_TOKEN_ALL_LAYERS_MEAN:
                    batch_vecs = per_layer_last_token.mean(dim=1).float().cpu()
                else:
                    raise SystemExit(
                        f"Unsupported completion feature pooling '{pooling}' for view '{key}'. "
                        f"Valid completion poolings: {COMPLETION_POOLING_CHOICES}"
                    )
                features_by_key[key].extend(batch_vecs.unbind(dim=0))

            if end == total or start == 0 or end % 50 == 0:
                print(f"[{split_name}] completion {end}/{total}", flush=True)

    return {
        key: torch.stack(view_features, dim=0)
        for key, view_features in features_by_key.items()
    }


def _balanced_indices(
    labels: list[int],
    *,
    split_name: str,
    mode: str,
    seed: int,
) -> list[int]:
    if mode == "none":
        return list(range(len(labels)))

    if mode != "downsample":
        raise SystemExit(f"Unsupported balance mode '{mode}' for split '{split_name}'.")

    positive_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]
    negative_idx = [idx for idx, label in enumerate(labels) if int(label) == 0]
    if not positive_idx or not negative_idx:
        raise SystemExit(
            f"Cannot downsample-balance split '{split_name}' because one class is missing "
            f"(pos={len(positive_idx)}, neg={len(negative_idx)})."
        )

    target_per_class = min(len(positive_idx), len(negative_idx))
    rng = random.Random(seed)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)
    keep = positive_idx[:target_per_class] + negative_idx[:target_per_class]
    keep.sort()
    return keep


def _subset_list(values: list[int], keep_indices: list[int]) -> list[int]:
    return [values[idx] for idx in keep_indices]


def _subset_float_list(values: list[float], keep_indices: list[int]) -> list[float]:
    return [float(values[idx]) for idx in keep_indices]


def _write_jsonl_rows(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _sample_shape_from_features(features: torch.Tensor) -> list[int]:
    if features.ndim not in (2, 3):
        raise SystemExit(
            "Expected flat or stacked features when materializing the manifest, "
            f"got shape {tuple(features.shape)}."
        )
    return [int(dim) for dim in features.shape[1:]]


def _resolve_label_spec(args: argparse.Namespace) -> dict[str, object]:
    label_target = str(args.label_target)
    label_horizon = args.label_horizon
    if label_target == "loop_by_horizon":
        if label_horizon is None or label_horizon < 1:
            raise SystemExit(
                "--label-horizon must be a positive integer when "
                "--label-target=loop_by_horizon."
            )
    elif label_horizon is not None:
        raise SystemExit(
            "--label-horizon is only valid when "
            "--label-target=loop_by_horizon."
        )
    return {
        "target": label_target,
        "horizon": label_horizon,
    }


def _resolve_target_spec(
    args: argparse.Namespace,
    *,
    rollout_cfg,
    label_spec: dict[str, object],
) -> dict[str, object]:
    target_kind = str(args.target_kind)
    if target_kind == "binary":
        binary_target_mode = str(args.binary_target_mode)
        if binary_target_mode == "rollout_label":
            if args.profile_target not in (None, ""):
                raise SystemExit(
                    "--profile-target is only valid for prompt-profile targets."
                )
            if rollout_cfg.num_generations != 1:
                raise SystemExit(
                    "Legacy rollout-level binary targets require --num-generations=1. "
                    "Use --binary-target-mode=prompt_majority_tail for repeated-rollout "
                    "prompt-level binary labels."
                )
            return {
                "kind": "binary",
                "name": str(label_spec["target"]),
                "horizon": label_spec["horizon"],
            }
        if binary_target_mode != "prompt_majority_tail":
            raise SystemExit(
                f"Unsupported --binary-target-mode '{binary_target_mode}'. "
                f"Valid: {BINARY_TARGET_MODE_CHOICES}"
            )
        if rollout_cfg.num_generations < 2:
            raise SystemExit(
                "--binary-target-mode=prompt_majority_tail requires "
                "--num-generations >= 2 so majority labels are estimated from "
                "repeated rollouts."
            )
        tail_threshold = float(args.profile_tail_threshold)
        if not 0.0 < tail_threshold <= 1.0:
            raise SystemExit("--profile-tail-threshold must be in (0, 1].")
        profile_target = args.profile_target
        if profile_target is None or profile_target == "":
            profile_target = "majority_tail"
        if profile_target != "majority_tail":
            raise SystemExit(
                "--binary-target-mode=prompt_majority_tail currently expects "
                "--profile-target=majority_tail."
            )
        target_name = profile_target_name(
            profile_target,
            tail_threshold=tail_threshold,
        )
        return {
            "kind": "binary",
            "source": "prompt_profile",
            "name": target_name,
            "profile_target": profile_target,
            "tail_threshold": tail_threshold,
            "num_generations": int(rollout_cfg.num_generations),
            "positive_rule": "strict_majority",
        }
    if target_kind not in ("probability", "regression"):
        raise SystemExit(f"Unsupported --target-kind '{target_kind}'.")
    if rollout_cfg.num_generations < 2:
        raise SystemExit(
            "--target-kind=probability/regression requires --num-generations >= 2 "
            "so the target is estimated from repeated rollouts rather than "
            "collapsing back to a single sample."
        )
    tail_threshold = float(args.profile_tail_threshold)
    if not 0.0 < tail_threshold <= 1.0:
        raise SystemExit("--profile-tail-threshold must be in (0, 1].")
    profile_target = args.profile_target
    if profile_target is None or profile_target == "":
        profile_target = "s_tail" if target_kind == "probability" else "mean_relative_length"
    if profile_target not in PROFILE_TARGET_CHOICES:
        raise SystemExit(
            f"Unsupported --profile-target '{profile_target}'. "
            f"Valid: {PROFILE_TARGET_CHOICES}"
        )
    if target_kind == "probability" and profile_target not in {
        "s_tail",
        "p_loop",
        "p_cap",
    }:
        raise SystemExit(
            "--target-kind=probability currently expects "
            "--profile-target in {s_tail, p_loop, p_cap}."
        )
    if target_kind == "regression" and profile_target not in {
        "mean_relative_length",
        "loop_budget_share",
    }:
        raise SystemExit(
            "--target-kind=regression currently expects "
            "--profile-target in {mean_relative_length, loop_budget_share}."
        )
    target_name = profile_target_name(
        profile_target,
        tail_threshold=tail_threshold,
    )
    if target_kind == "probability":
        return {
            "kind": "probability",
            "name": target_name,
            "profile_target": profile_target,
            "tail_threshold": tail_threshold,
            "num_generations": int(rollout_cfg.num_generations),
            "loss": "soft_bce",
        }
    return {
        "kind": "regression",
        "name": target_name,
        "profile_target": profile_target,
        "tail_threshold": tail_threshold,
        "num_generations": int(rollout_cfg.num_generations),
        "loss": "sigmoid_mse",
    }


def main() -> None:
    args = _parse_args()

    try:
        rollout_cfg = get_rollout_config(
            args.model_preset,
            model_id=args.model_id,
            temperature=args.temperature,
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
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    train_spec, test_spec = _make_specs(args)
    split_source = _resolve_split_source(
        train_spec,
        test_spec,
        task_kind=str(args.task_kind),
    )
    primary_feature_key, feature_views = _resolve_feature_views(args)
    prefill_feature_views, completion_feature_views = _split_feature_views_by_stage(
        feature_views
    )
    label_spec = _resolve_label_spec(args)
    target_spec = _resolve_target_spec(
        args,
        rollout_cfg=rollout_cfg,
        label_spec=label_spec,
    )
    target_source = _target_source(target_spec)
    task_loader_config = _task_loader_config(args)
    balance_seed = args.seed if args.balance_seed is None else args.balance_seed
    if target_source == "prompt_profile":
        if completion_feature_views:
            raise SystemExit(
                "Prompt-profile targets currently support prefill feature views only."
            )
        if float(rollout_cfg.temperature) <= 0.0:
            raise SystemExit(
                "Prompt-profile targets require stochastic repeated rollouts; "
                "pass --temperature > 0."
            )
        if target_spec["kind"] != "binary" and (
            args.balance_train != "none" or args.balance_test != "none"
        ):
            raise SystemExit(
                "Balancing is only supported for binary targets in this builder."
            )

    if args.reuse_if_compatible:
        cache_hit, reason = _probe_cache_status(
            args.out_dir,
            train_spec=train_spec,
            test_spec=test_spec,
            prompt_field=args.prompt_field,
            task_kind=str(args.task_kind),
            split_source=split_source,
            split_ratio=args.split_ratio,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            target_spec=target_spec,
            label_spec=label_spec,
            rollout_config=rollout_cfg.to_dict(),
            requested_primary_feature_key=primary_feature_key,
            requested_feature_views=feature_views,
            seed=args.seed,
            balance_train=args.balance_train,
            balance_test=args.balance_test,
            balance_seed=balance_seed,
            answer_field=str(args.answer_field),
            task_loader_config=task_loader_config,
        )
        if cache_hit:
            print(
                f"Reusing cached probe dataset at {args.out_dir}: {reason}",
                flush=True,
            )
            return
        print(
            f"Cache miss at {args.out_dir}: {reason}. Rebuilding dataset.",
            flush=True,
        )

    train_records, test_records, split_source = _resolve_splits(
        args,
        train_spec,
        test_spec,
        rollout_model_id=rollout_cfg.model_id,
    )
    target_desc = str(target_spec["name"])
    if target_spec["kind"] == "binary" and target_spec.get("horizon") is not None:
        target_desc = f"{target_desc}@{target_spec['horizon']}"

    print(
        f"Building probe dataset with model={rollout_cfg.model_id}, "
        f"train={len(train_records)}, test={len(test_records)}, "
        f"feature_views={list(feature_views.keys())}, "
        f"target={target_desc}, "
        f"task_kind={args.task_kind}",
        flush=True,
    )

    model, tokenizer, device = load_prefill_model_and_tokenizer(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
    )

    # Use a shared chat-prompt construction path across detector builders and eval scripts.
    train_records = _apply_chat_prompt(tokenizer, train_records, num_repetition=1)
    test_records = _apply_chat_prompt(tokenizer, test_records, num_repetition=1)

    train_ids = _sample_ids(train_records)
    test_ids = _sample_ids(test_records)
    train_prompt_token_ids = _prompt_token_ids(tokenizer, train_records)
    test_prompt_token_ids = _prompt_token_ids(tokenizer, test_records)
    train_prompt_token_lengths = [len(token_ids) for token_ids in train_prompt_token_ids]
    test_prompt_token_lengths = [len(token_ids) for token_ids in test_prompt_token_ids]
    train_features_by_key: dict[str, torch.Tensor] = {}
    test_features_by_key: dict[str, torch.Tensor] = {}

    if prefill_feature_views:
        train_prefill_features_by_key, train_ids = _build_split(
            "train",
            train_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefill_batch_size=args.prefill_batch_size,
            feature_views=prefill_feature_views,
        )
        test_prefill_features_by_key, test_ids = _build_split(
            "test",
            test_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefill_batch_size=args.prefill_batch_size,
            feature_views=prefill_feature_views,
        )
        train_features_by_key.update(train_prefill_features_by_key)
        test_features_by_key.update(test_prefill_features_by_key)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    train_prompts = _prompts(train_records)
    test_prompts = _prompts(test_records)
    all_prompts = train_prompts + test_prompts
    split_at = len(train_prompts)
    train_profile_rows: list[dict[str, object]] = []
    test_profile_rows: list[dict[str, object]] = []
    prompt_rollout_archive_rows: list[dict[str, object]] = []

    if target_source == "rollout_label":
        need_rollout_tokens = bool(completion_feature_views)
        all_labels, all_rollout_token_ids = _label_split(
            "all",
            all_prompts,
            rollout_cfg=rollout_cfg,
            seed=args.seed,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            label_target=str(label_spec["target"]),
            label_horizon=(
                int(label_spec["horizon"])
                if label_spec["horizon"] is not None
                else None
            ),
            return_rollout_token_ids=need_rollout_tokens,
        )
        train_labels: list[int | float] = all_labels[:split_at]
        test_labels: list[int | float] = all_labels[split_at:]

        if completion_feature_views:
            if all_rollout_token_ids is None:
                raise RuntimeError(
                    "Completion views requested but rollout token IDs were not retained."
                )
            train_rollout_token_ids = all_rollout_token_ids[:split_at]
            test_rollout_token_ids = all_rollout_token_ids[split_at:]
            completion_model, completion_tokenizer, completion_device = (
                load_prefill_model_and_tokenizer(
                    rollout_cfg.model_id,
                    trust_remote_code=rollout_cfg.trust_remote_code,
                )
            )
            train_completion_features_by_key = _extract_completion_features(
                "train",
                train_records,
                train_rollout_token_ids,
                model=completion_model,
                tokenizer=completion_tokenizer,
                device=completion_device,
                feature_views=completion_feature_views,
                batch_size=args.completion_batch_size,
                max_model_len=rollout_cfg.max_model_len,
            )
            test_completion_features_by_key = _extract_completion_features(
                "test",
                test_records,
                test_rollout_token_ids,
                model=completion_model,
                tokenizer=completion_tokenizer,
                device=completion_device,
                feature_views=completion_feature_views,
                batch_size=args.completion_batch_size,
                max_model_len=rollout_cfg.max_model_len,
            )
            train_features_by_key.update(train_completion_features_by_key)
            test_features_by_key.update(test_completion_features_by_key)

            del train_rollout_token_ids
            del test_rollout_token_ids
            del all_rollout_token_ids

            del completion_model
            del completion_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        train_keep_idx = _balanced_indices(
            train_labels,
            split_name="train",
            mode=args.balance_train,
            seed=balance_seed,
        )
        test_keep_idx = _balanced_indices(
            test_labels,
            split_name="test",
            mode=args.balance_test,
            seed=balance_seed + 1,
        )
        train_labels = _subset_list(train_labels, train_keep_idx)
        test_labels = _subset_list(test_labels, test_keep_idx)
    else:
        all_records = train_records + test_records
        all_prompt_token_ids = train_prompt_token_ids + test_prompt_token_ids
        all_ids = train_ids + test_ids
        all_split_names = (["train"] * len(train_records)) + (["test"] * len(test_records))
        all_targets, all_profile_rows, prompt_rollout_archive_rows = _build_prompt_profile_targets(
            "all",
            all_records,
            all_prompt_token_ids,
            all_ids,
            all_split_names,
            rollout_cfg=rollout_cfg,
            seed=args.seed,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            tail_threshold=float(target_spec["tail_threshold"]),
            profile_target=str(target_spec["profile_target"]),
            target_kind=str(target_spec["kind"]),
        )
        train_labels = all_targets[:split_at]
        test_labels = all_targets[split_at:]
        train_profile_rows = all_profile_rows[:split_at]
        test_profile_rows = all_profile_rows[split_at:]
        if target_spec["kind"] == "binary":
            train_keep_idx = _balanced_indices(
                train_labels,
                split_name="train",
                mode=args.balance_train,
                seed=balance_seed,
            )
            test_keep_idx = _balanced_indices(
                test_labels,
                split_name="test",
                mode=args.balance_test,
                seed=balance_seed + 1,
            )
            train_labels = _subset_list(train_labels, train_keep_idx)
            test_labels = _subset_list(test_labels, test_keep_idx)
        else:
            train_keep_idx = list(range(len(train_labels)))
            test_keep_idx = list(range(len(test_labels)))
            train_labels = _subset_float_list(train_labels, train_keep_idx)
            test_labels = _subset_float_list(test_labels, test_keep_idx)

    train_ids = _subset_list(train_ids, train_keep_idx)
    test_ids = _subset_list(test_ids, test_keep_idx)
    train_prompt_token_lengths = _subset_list(train_prompt_token_lengths, train_keep_idx)
    test_prompt_token_lengths = _subset_list(test_prompt_token_lengths, test_keep_idx)
    if target_source == "prompt_profile":
        train_profile_rows = [train_profile_rows[idx] for idx in train_keep_idx]
        test_profile_rows = [test_profile_rows[idx] for idx in test_keep_idx]
    train_keep_idx_t = torch.tensor(train_keep_idx, dtype=torch.long)
    test_keep_idx_t = torch.tensor(test_keep_idx, dtype=torch.long)

    feature_views_manifest: dict[str, dict[str, object]] = {}
    primary_train_meta: dict[str, object] | None = None
    primary_test_meta: dict[str, object] | None = None
    primary_input_dim: int | None = None

    for feature_key, feature_spec in feature_views.items():
        train_features = train_features_by_key.get(feature_key)
        test_features = test_features_by_key.get(feature_key)
        if train_features is None or test_features is None:
            raise SystemExit(
                f"Missing extracted features for requested view '{feature_key}'."
            )
        train_features = train_features.index_select(0, train_keep_idx_t)
        test_features = test_features.index_select(0, test_keep_idx_t)

        train_sample_shape = _sample_shape_from_features(train_features)
        test_sample_shape = _sample_shape_from_features(test_features)
        if test_sample_shape != train_sample_shape:
            raise SystemExit(
                "Sample shape mismatch between train/test for feature "
                f"'{feature_key}': {train_sample_shape} vs {test_sample_shape}"
            )
        input_dim = int(train_sample_shape[-1])
        if train_features.size(0) != len(train_labels):
            raise SystemExit(
                "Mismatched feature/label counts for split 'train' "
                f"feature '{feature_key}': {train_features.size(0)} vs {len(train_labels)}"
            )
        if test_features.size(0) != len(test_labels):
            raise SystemExit(
                "Mismatched feature/label counts for split 'test' "
                f"feature '{feature_key}': {test_features.size(0)} vs {len(test_labels)}"
            )

        if feature_key == primary_feature_key:
            train_split_name = "train"
            test_split_name = "test"
        else:
            train_split_name = os.path.join("features", feature_key, "train")
            test_split_name = os.path.join("features", feature_key, "test")

        train_meta = save_split_shards(
            args.out_dir,
            train_split_name,
            train_features,
            train_labels,
            train_ids,
            shard_size=args.shard_size,
            target_kind=str(target_spec["kind"]),
        )
        test_meta = save_split_shards(
            args.out_dir,
            test_split_name,
            test_features,
            test_labels,
            test_ids,
            shard_size=args.shard_size,
            target_kind=str(target_spec["kind"]),
        )

        feature_views_manifest[feature_key] = {
            "stage": feature_spec.get("stage", "prefill"),
            "pooling": feature_spec["pooling"],
            "layer": feature_spec["layer"],
            "input_dim": input_dim,
            "sample_shape": train_sample_shape,
            "train": train_meta,
            "test": test_meta,
        }

        if feature_key == primary_feature_key:
            primary_train_meta = train_meta
            primary_test_meta = test_meta
            primary_input_dim = input_dim

    if primary_train_meta is None or primary_test_meta is None or primary_input_dim is None:
        raise SystemExit(f"Primary feature view '{primary_feature_key}' was not materialized.")

    prompt_profile_files: dict[str, str] = {}
    prompt_rollout_archive_file = ""
    if target_source == "prompt_profile":
        train_profile_path = os.path.join("diagnostics", "train_prompt_profile.jsonl")
        test_profile_path = os.path.join("diagnostics", "test_prompt_profile.jsonl")
        prompt_rollout_archive_path = os.path.join(
            "diagnostics",
            "prompt_rollout_archive.jsonl",
        )
        _write_jsonl_rows(
            os.path.join(args.out_dir, train_profile_path),
            train_profile_rows,
        )
        _write_jsonl_rows(
            os.path.join(args.out_dir, test_profile_path),
            test_profile_rows,
        )
        prompt_profile_files = {
            "train": train_profile_path,
            "test": test_profile_path,
        }
        _write_jsonl_rows(
            os.path.join(args.out_dir, prompt_rollout_archive_path),
            prompt_rollout_archive_rows,
        )
        prompt_rollout_archive_file = prompt_rollout_archive_path

    manifest = {
        "version": 7,
        "input_dim": primary_input_dim,
        "sample_shape": feature_views_manifest[primary_feature_key]["sample_shape"],
        "default_feature_key": primary_feature_key,
        "feature_extraction": {
            "stage": feature_views[primary_feature_key].get("stage", "prefill"),
            "pooling": feature_views[primary_feature_key]["pooling"],
            "layer": feature_views[primary_feature_key]["layer"],
        },
        "feature_views": feature_views_manifest,
        "task_kind": str(args.task_kind),
        "prompt_field": args.prompt_field,
        "answer_field": _manifest_answer_field(
            str(args.task_kind),
            str(args.answer_field),
        ),
        "prompt_template": {
            "source": (
            "loop_probe.adapters.multiple_choice_gpqa.build_mcq_prompt"
            if args.task_kind == "multiple_choice_gpqa"
            else (
                "loop_probe.adapters.multiple_choice_mmlupro.build_mcq_prompt"
                if args.task_kind == "multiple_choice_mmlupro"
                else (
                    "loop_probe.adapters.livecodebench_codegen.build_prompts"
                    if args.task_kind == "livecodebench_codegen"
                    else "utils.build_prompt"
                )
            )
        ),
            "num_repetition": 1,
            "chat_template": True,
        },
        "task_loader_config": task_loader_config,
        "split_source": split_source,
        "split_ratio": args.split_ratio if _split_source_uses_ratio(split_source) else None,
        "seed": args.seed,
        "loop_detector": {
            "n": args.loop_n,
            "k": args.loop_k,
        },
        "target_spec": target_spec,
        "label_spec": label_spec if target_source == "rollout_label" else None,
        "balancing": {
            "train": args.balance_train,
            "test": args.balance_test,
            "seed": balance_seed,
        },
        "rollout_config": rollout_cfg.to_dict(),
        "prompt_profile_files": prompt_profile_files or None,
        "prompt_rollout_archive_file": prompt_rollout_archive_file or None,
        "train_spec": asdict(train_spec),
        "test_spec": asdict(test_spec) if test_spec else None,
        "train": primary_train_meta,
        "test": primary_test_meta,
    }
    write_manifest(args.out_dir, manifest)

    print(f"Wrote probe dataset to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
