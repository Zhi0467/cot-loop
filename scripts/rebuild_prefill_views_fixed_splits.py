#!/usr/bin/env python3
"""Re-extract prefill feature views on fixed labels/sample IDs from a reference dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import zlib

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loop_probe.prefill import (
    FEATURE_POOLING_CHOICES,
    extract_prefill_features_multi,
    load_prefill_model_and_tokenizer,
)
from loop_probe.labeling import PROMPT_PROFILE_TARGET_CHOICES
from loop_probe.serialization import save_split_shards, write_manifest
from loop_probe.adapters import (
    livecodebench_codegen,
    multiple_choice_gpqa,
    multiple_choice_mmlupro,
)
from loop_probe.types import SampleRecord
from utils import build_prompt

FEATURE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _is_prompt_profile_target_spec(target_spec: object) -> bool:
    if not isinstance(target_spec, dict):
        return False
    profile_target = target_spec.get("profile_target")
    if isinstance(profile_target, str) and profile_target in PROMPT_PROFILE_TARGET_CHOICES:
        return True
    target_name = target_spec.get("name")
    if not isinstance(target_name, str):
        return False
    return (
        target_name in PROMPT_PROFILE_TARGET_CHOICES
        or target_name.startswith("s_")
        or target_name.startswith("majority_s_")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--train-pool-jsonl", required=True)
    parser.add_argument("--eval-pool-jsonl", required=True)
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    parser.set_defaults(trust_remote_code=None)
    parser.add_argument("--num-repetition", type=int, default=None)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=None,
    )
    parser.add_argument("--feature-key", default=None)
    parser.add_argument(
        "--feature-pooling",
        choices=FEATURE_POOLING_CHOICES,
        required=True,
    )
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument(
        "--extra-feature-view",
        action="append",
        default=[],
        metavar="KEY:POOLING:LAYER",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _default_feature_key(*, pooling: str, feature_layer: int) -> str:
    if feature_layer == -1:
        layer_tag = "final"
    elif feature_layer >= 0:
        layer_tag = f"layer{feature_layer}"
    else:
        layer_tag = f"layer_m{abs(feature_layer)}"
    return f"{pooling}_{layer_tag}"


def _parse_feature_view(raw: str) -> tuple[str, dict[str, object]]:
    key, sep, remainder = raw.partition(":")
    if sep == "" or ":" not in remainder:
        raise SystemExit(
            "--extra-feature-view must match KEY:POOLING:LAYER, "
            f"got '{raw}'."
        )
    pooling, _, layer_text = remainder.rpartition(":")
    key = key.strip()
    pooling = pooling.strip()
    layer_text = layer_text.strip()
    if not key or not FEATURE_KEY_PATTERN.fullmatch(key):
        raise SystemExit(
            "Feature view key must match [A-Za-z0-9_.-]+, "
            f"got '{key}'."
        )
    if pooling not in FEATURE_POOLING_CHOICES:
        raise SystemExit(
            f"Unknown feature pooling '{pooling}' for view '{key}'. "
            f"Valid: {FEATURE_POOLING_CHOICES}"
        )
    try:
        layer = int(layer_text)
    except Exception as exc:
        raise SystemExit(
            f"Feature layer for view '{key}' is not an integer: '{layer_text}'."
        ) from exc
    return key, {"pooling": pooling, "layer": layer, "stage": "prefill"}


def _resolve_feature_views(args: argparse.Namespace) -> tuple[str, dict[str, dict[str, object]]]:
    primary_key = args.feature_key.strip() if args.feature_key else ""
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
    _validate_layer_for_pooling(
        pooling=str(args.feature_pooling),
        layer=int(args.feature_layer),
        key=primary_key,
    )
    feature_views: dict[str, dict[str, object]] = {
        primary_key: {
            "pooling": args.feature_pooling,
            "layer": int(args.feature_layer),
            "stage": "prefill",
        }
    }
    for raw in args.extra_feature_view:
        key, spec = _parse_feature_view(raw)
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


def _validate_layer_for_pooling(*, pooling: str, layer: int, key: str) -> None:
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


def _sample_shape_from_features(features: torch.Tensor) -> list[int]:
    if features.ndim not in (2, 3):
        raise SystemExit(
            "Expected flat or stacked features when materializing the manifest, "
            f"got shape {tuple(features.shape)}."
        )
    return [int(dim) for dim in features.shape[1:]]


def _load_jsonl_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_num}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected JSON object rows in {path}:{line_num}")
            rows.append(row)
    return rows


def _reference_prompt_profile_rows_match_records(
    *,
    reference_data_dir: str,
    reference_manifest: dict[str, object],
    records_by_split: dict[str, list[SampleRecord]],
) -> bool:
    rel_path = reference_manifest.get("prompt_rollout_archive_file")
    if not isinstance(rel_path, str) or not rel_path:
        return False
    archive_rows = _load_jsonl_rows(os.path.join(reference_data_dir, rel_path))
    prompt_by_key: dict[tuple[str, int], str] = {}
    for idx, row in enumerate(archive_rows):
        split = row.get("split")
        prompt = row.get("prompt")
        raw_sample_id = row.get("sample_id")
        if not isinstance(split, str) or not isinstance(prompt, str):
            return False
        try:
            sample_id = int(raw_sample_id)
        except Exception as exc:
            raise SystemExit(
                f"Reference prompt-profile archive row {idx} has invalid sample_id={raw_sample_id!r}."
            ) from exc
        key = (split, sample_id)
        existing = prompt_by_key.get(key)
        if existing is not None and existing != prompt:
            raise SystemExit(
                "Reference prompt-profile archive has conflicting prompt text for "
                f"split='{split}', sample_id={sample_id}."
            )
        prompt_by_key[key] = prompt
    for split, records in records_by_split.items():
        for record in records:
            if prompt_by_key.get((split, int(record.sample_id))) != record.prompt:
                return False
    return True


def _rows_by_sample_id(pool_rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    lookup: dict[int, dict[str, object]] = {}
    for idx, row in enumerate(pool_rows):
        raw_sample_id = row.get("_source_sample_id", idx)
        try:
            sample_id = int(raw_sample_id)
        except Exception as exc:
            raise SystemExit(
                f"Pool row {idx} has invalid _source_sample_id={raw_sample_id!r}."
            ) from exc
        if sample_id < 0:
            raise SystemExit(f"Pool row {idx} has negative _source_sample_id={sample_id}.")
        if sample_id in lookup:
            raise SystemExit(
                f"Duplicate _source_sample_id={sample_id} found while indexing pool rows."
            )
        lookup[sample_id] = row
    return lookup


def _normalize_livecodebench_split(raw_split: object) -> str:
    normalized = str(raw_split).strip().lower()
    if normalized not in ("", "all", "train", "validation", "val", "test"):
        raise SystemExit(
            "Unsupported LiveCodeBench split "
            f"'{raw_split}'. Use one of train/validation/test/all."
        )
    return normalized


def _partition_livecodebench_benchmark(benchmark, raw_split: object):
    normalized_split = _normalize_livecodebench_split(raw_split)
    if normalized_split in ("", "all"):
        return benchmark

    def _partition_bucket(question_id: object) -> int:
        return zlib.crc32(str(question_id).encode("utf-8")) % 10

    if normalized_split == "train":
        return [row for row in benchmark if _partition_bucket(row.question_id) < 8]
    if normalized_split in ("validation", "val"):
        return [row for row in benchmark if _partition_bucket(row.question_id) == 8]
    return [row for row in benchmark if _partition_bucket(row.question_id) == 9]


def _reference_split_spec(
    reference_manifest: dict[str, object],
    *,
    split: str,
) -> dict[str, object]:
    spec_payload = reference_manifest.get(f"{split}_spec")
    if isinstance(spec_payload, dict):
        return spec_payload
    if split == "test":
        fallback = reference_manifest.get("train_spec")
        if isinstance(fallback, dict):
            fallback_payload = dict(fallback)
            fallback_payload["split"] = "test"
            return fallback_payload
    raise SystemExit(f"Reference manifest is missing a usable {split}_spec payload.")


def _build_livecodebench_prompt_lookup(
    *,
    reference_manifest: dict[str, object],
    split: str,
    model_id: str,
) -> dict[int, str]:
    task_loader_config = reference_manifest.get("task_loader_config")
    if not isinstance(task_loader_config, dict):
        raise SystemExit(
            "LiveCodeBench fixed-split rebuild requires task_loader_config in the "
            "reference manifest."
        )
    repo_path = str(task_loader_config.get("livecodebench_repo", "")).strip()
    if not repo_path:
        raise SystemExit(
            "LiveCodeBench fixed-split rebuild requires "
            "task_loader_config.livecodebench_repo in the reference manifest."
        )
    default_release_version = str(
        task_loader_config.get("release_version", "")
    ).strip()
    if not default_release_version:
        raise SystemExit(
            "LiveCodeBench fixed-split rebuild requires "
            "task_loader_config.release_version in the reference manifest."
        )
    lm_style_override_raw = task_loader_config.get("lm_style_override")
    lm_style_override = (
        None
        if lm_style_override_raw in (None, "")
        else str(lm_style_override_raw)
    )

    split_spec = _reference_split_spec(reference_manifest, split=split)
    effective_release_version = (
        str(split_spec.get("config", "")).strip() or default_release_version
    )
    benchmark, format_prompt = livecodebench_codegen.load_benchmark(
        repo_path,
        effective_release_version,
    )
    benchmark = _partition_livecodebench_benchmark(
        benchmark,
        split_spec.get("split", split),
    )
    max_samples_raw = split_spec.get("max_samples")
    if max_samples_raw in (None, ""):
        max_samples = None
    else:
        try:
            max_samples = int(max_samples_raw)
        except Exception as exc:
            raise SystemExit(
                "LiveCodeBench fixed-split rebuild requires an integer max_samples "
                f"in the reference manifest, got {max_samples_raw!r}."
            ) from exc
    prompt_records, _lm_style = livecodebench_codegen.build_prompts(
        benchmark,
        format_prompt,
        repo_path=repo_path,
        model_id=model_id,
        lm_style_override=lm_style_override,
        max_samples=max_samples,
    )
    return {idx: prompt for idx, (_question_id, prompt) in enumerate(prompt_records)}


def _load_reference_manifest(reference_data_dir: str) -> dict[str, object]:
    manifest_path = os.path.join(reference_data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_reference_target_kind(
    manifest: dict[str, object],
    *,
    split: str,
) -> str:
    split_info = manifest.get(split)
    if isinstance(split_info, dict):
        split_target_kind = split_info.get("target_kind")
        if isinstance(split_target_kind, str) and split_target_kind:
            target_kind = split_target_kind
        else:
            target_spec = manifest.get("target_spec")
            target_kind = (
                str(target_spec.get("kind", "binary"))
                if isinstance(target_spec, dict)
                else "binary"
            )
    else:
        target_kind = "binary"
    if target_kind not in ("binary", "probability", "regression"):
        raise SystemExit(
            f"Unsupported target kind '{target_kind}' in reference split '{split}'."
        )
    return target_kind


def _resolve_built_splits(
    manifest: dict[str, object],
    requested_splits: list[str] | tuple[str, ...] | None,
) -> list[str]:
    available_splits = [
        split for split in ("train", "test") if isinstance(manifest.get(split), dict)
    ]
    if not available_splits:
        raise SystemExit(
            "Reference manifest does not expose any top-level train/test splits."
        )

    if not requested_splits:
        return available_splits

    built_splits = list(dict.fromkeys(requested_splits))
    missing = [split for split in built_splits if split not in available_splits]
    if missing:
        raise SystemExit(
            "Requested split(s) are unavailable in the reference manifest: "
            f"{missing}. Available splits: {available_splits}"
        )
    return built_splits


def _load_reference_split(
    *,
    reference_data_dir: str,
    manifest: dict[str, object],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    split_info = manifest.get(split)
    if not isinstance(split_info, dict):
        raise SystemExit(f"Reference manifest missing split '{split}'.")
    target_kind = _resolve_reference_target_kind(manifest, split=split)
    shard_paths = split_info.get("shards")
    if not isinstance(shard_paths, list) or not shard_paths:
        raise SystemExit(f"Reference split '{split}' has no shard files.")

    labels = []
    sample_ids = []
    for rel_path in shard_paths:
        if not isinstance(rel_path, str):
            raise SystemExit(f"Invalid shard path entry for split '{split}'.")
        shard = torch.load(os.path.join(reference_data_dir, rel_path), map_location="cpu")
        label_dtype = (
            torch.float32 if target_kind in ("probability", "regression") else torch.uint8
        )
        labels.append(shard["y"].to(dtype=label_dtype))
        sample_ids.append(shard["sample_ids"].to(dtype=torch.int64))
    return torch.cat(labels, dim=0), torch.cat(sample_ids, dim=0)


def _build_records(
    *,
    pool_rows_by_sample_id: dict[int, dict[str, object]],
    sample_ids: torch.Tensor,
    prompt_field: str,
    task_kind: str,
    seed: int,
    tokenizer,
    num_repetition: int,
    split: str,
    livecodebench_prompt_lookup: dict[int, str] | None = None,
) -> list[SampleRecord]:
    records = []
    for sample_id in sample_ids.tolist():
        sample_id_int = int(sample_id)
        row = pool_rows_by_sample_id.get(sample_id_int)
        if row is None:
            raise SystemExit(
                f"sample_id={sample_id_int} not found in the fixed-split source pool."
            )
        if task_kind == "multiple_choice_gpqa":
            prompt = row.get("Question")
            if not isinstance(prompt, str):
                raise SystemExit(
                    f"Prompt field 'Question' missing/invalid for sample_id={sample_id_int}"
                )
            if num_repetition != 1:
                raise SystemExit(
                    "GPQA prompt formatting does not support num_repetition != 1."
                )
            required = (
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            )
            missing = [name for name in required if row.get(name) is None]
            if missing:
                raise SystemExit(
                    f"GPQA row for sample_id={sample_id_int} is missing {missing}."
                )
            shuffled = [(str(row["Correct Answer"]), True)] + [
                (str(row[f"Incorrect Answer {idx}"]), False) for idx in range(1, 4)
            ]
            rng = random.Random(seed ^ sample_id_int)
            rng.shuffle(shuffled)
            prompt_text = multiple_choice_gpqa.build_mcq_prompt(
                tokenizer,
                prompt,
                [option for option, _ in shuffled],
            )
        elif task_kind == "multiple_choice_mmlupro":
            prompt = row.get("question")
            if not isinstance(prompt, str):
                raise SystemExit(
                    f"Prompt field 'question' missing/invalid for sample_id={sample_id_int}"
                )
            if num_repetition != 1:
                raise SystemExit(
                    "MMLU-Pro prompt formatting does not support num_repetition != 1."
                )
            options = row.get("options")
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        "MMLU-Pro row has string 'options' but it is not valid JSON."
                    ) from exc
            if not isinstance(options, list) or not options:
                raise SystemExit(
                    f"MMLU-Pro row for sample_id={sample_id_int} has invalid options."
                )
            prompt_text = multiple_choice_mmlupro.build_mcq_prompt(
                tokenizer,
                prompt,
                [str(option) for option in options],
            )
        elif task_kind == "math_freeform":
            prompt = row.get(prompt_field)
            if not isinstance(prompt, str):
                raise SystemExit(
                    f"Prompt field '{prompt_field}' missing/invalid for sample_id={sample_id_int}"
                )
            prompt_text = build_prompt(tokenizer, prompt, num_repetition)
        elif task_kind == "livecodebench_codegen":
            if num_repetition != 1:
                raise SystemExit(
                    "LiveCodeBench prompt formatting does not support num_repetition != 1."
                )
            if livecodebench_prompt_lookup is None:
                raise SystemExit(
                    "LiveCodeBench fixed-split rebuild requires a prompt lookup "
                    "built from the original formatter."
                )
            prompt_text = livecodebench_prompt_lookup.get(sample_id_int)
            if not isinstance(prompt_text, str):
                raise SystemExit(
                    "LiveCodeBench prompt lookup is missing "
                    f"sample_id={sample_id_int}."
                )
        else:
            raise SystemExit(
                "Fixed-split rebuild currently supports task_kind in "
                "{math_freeform, multiple_choice_gpqa, multiple_choice_mmlupro, "
                "livecodebench_codegen}; "
                f"got '{task_kind}'."
            )
        records.append(
            SampleRecord(
                sample_id=sample_id_int,
                prompt=prompt_text,
                source_split=split,
            )
        )
    return records


def _copy_reference_relpath(
    *,
    reference_data_dir: str,
    out_dir: str,
    rel_path: str,
) -> None:
    source_path = os.path.join(reference_data_dir, rel_path)
    if not os.path.exists(source_path):
        raise SystemExit(
            f"Reference prompt-profile diagnostics are missing: {source_path}"
        )
    dest_path = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(source_path, dest_path)


def _save_view_split(
    *,
    out_dir: str,
    primary_key: str,
    feature_key: str,
    split: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    shard_size: int,
    target_kind: str,
) -> dict[str, object]:
    if feature_key == primary_key:
        view_dir = out_dir
        prefix = ""
    else:
        view_dir = os.path.join(out_dir, "features", feature_key)
        prefix = os.path.join("features", feature_key)
    split_meta = save_split_shards(
        view_dir,
        split,
        features,
        labels.tolist(),
        sample_ids.tolist(),
        shard_size=shard_size,
        target_kind=target_kind,
    )
    if prefix:
        split_meta["shards"] = [
            os.path.join(prefix, rel_path) for rel_path in split_meta["shards"]
        ]
    return split_meta


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    reference_manifest = _load_reference_manifest(args.reference_data_dir)
    prompt_field = args.prompt_field
    if prompt_field is None:
        prompt_field = reference_manifest.get("prompt_field")
    if not isinstance(prompt_field, str) or not prompt_field:
        raise SystemExit("Could not resolve prompt field; pass --prompt-field explicitly.")

    rollout_config = reference_manifest.get("rollout_config")
    model_id = args.model_id
    if model_id is None and isinstance(rollout_config, dict):
        candidate = rollout_config.get("model_id")
        if isinstance(candidate, str) and candidate:
            model_id = candidate
    if not isinstance(model_id, str) or not model_id:
        raise SystemExit("Could not resolve model id; pass --model-id explicitly.")

    trust_remote_code = args.trust_remote_code
    if trust_remote_code is None and isinstance(rollout_config, dict):
        candidate = rollout_config.get("trust_remote_code")
        if isinstance(candidate, bool):
            trust_remote_code = candidate
    if trust_remote_code is None:
        trust_remote_code = True

    prompt_template = reference_manifest.get("prompt_template")
    num_repetition = args.num_repetition
    if num_repetition is None and isinstance(prompt_template, dict):
        candidate = prompt_template.get("num_repetition")
        if isinstance(candidate, int) and candidate >= 1:
            num_repetition = candidate
    if num_repetition is None:
        num_repetition = 1
    reference_num_repetition = 1
    reference_prompt_template = reference_manifest.get("prompt_template")
    if isinstance(reference_prompt_template, dict):
        candidate = reference_prompt_template.get("num_repetition")
        if isinstance(candidate, int) and candidate >= 1:
            reference_num_repetition = candidate
    reference_prompt_field = reference_manifest.get("prompt_field")
    reference_model_id: str | None = None
    if isinstance(rollout_config, dict):
        candidate = rollout_config.get("model_id")
        if isinstance(candidate, str) and candidate:
            reference_model_id = candidate
    prompt_profile_contract_compatible = (
        isinstance(reference_prompt_field, str)
        and prompt_field == reference_prompt_field
        and reference_model_id is not None
        and model_id == reference_model_id
        and num_repetition == reference_num_repetition
    )

    task_kind = str(reference_manifest.get("task_kind", "math_freeform"))
    manifest_seed = reference_manifest.get("seed", 0)
    try:
        seed = int(manifest_seed)
    except Exception:
        seed = 0

    primary_key, feature_views = _resolve_feature_views(args)
    feature_view_specs = {
        key: (str(spec["pooling"]), int(spec["layer"]))
        for key, spec in feature_views.items()
    }

    train_pool_rows_by_sample_id = _rows_by_sample_id(_load_jsonl_rows(args.train_pool_jsonl))
    eval_pool_rows_by_sample_id = _rows_by_sample_id(_load_jsonl_rows(args.eval_pool_jsonl))

    model, tokenizer, device = load_prefill_model_and_tokenizer(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
    )

    manifest_views: dict[str, dict[str, object]] = {
        key: {
            "layer": int(spec["layer"]),
            "pooling": str(spec["pooling"]),
            "stage": "prefill",
        }
        for key, spec in feature_views.items()
    }
    records_by_split: dict[str, list[SampleRecord]] = {}

    built_splits = _resolve_built_splits(reference_manifest, args.splits)
    print(
        f"Rebuilding splits {built_splits} from {args.reference_data_dir}",
        flush=True,
    )

    for split in built_splits:
        target_kind = _resolve_reference_target_kind(reference_manifest, split=split)
        labels, sample_ids = _load_reference_split(
            reference_data_dir=args.reference_data_dir,
            manifest=reference_manifest,
            split=split,
        )
        pool_rows_by_sample_id = (
            train_pool_rows_by_sample_id
            if split == "train"
            else eval_pool_rows_by_sample_id
        )
        livecodebench_prompt_lookup = None
        if task_kind == "livecodebench_codegen":
            livecodebench_prompt_lookup = _build_livecodebench_prompt_lookup(
                reference_manifest=reference_manifest,
                split=split,
                model_id=model_id,
            )
        records = _build_records(
            pool_rows_by_sample_id=pool_rows_by_sample_id,
            sample_ids=sample_ids,
            prompt_field=prompt_field,
            task_kind=task_kind,
            seed=seed,
            tokenizer=tokenizer,
            num_repetition=num_repetition,
            split=split,
            livecodebench_prompt_lookup=livecodebench_prompt_lookup,
        )
        records_by_split[split] = records
        features_by_key = extract_prefill_features_multi(
            model,
            tokenizer,
            device,
            records,
            feature_views=feature_view_specs,
            log_prefix=split,
            batch_size=args.prefill_batch_size,
        )
        for key, features in features_by_key.items():
            split_meta = _save_view_split(
                out_dir=args.out_dir,
                primary_key=primary_key,
                feature_key=key,
                split=split,
                features=features,
                labels=labels,
                sample_ids=sample_ids,
                shard_size=args.shard_size,
                target_kind=target_kind,
            )
            sample_shape = _sample_shape_from_features(features)
            prior_sample_shape = manifest_views[key].get("sample_shape")
            if prior_sample_shape is not None and list(prior_sample_shape) != sample_shape:
                raise SystemExit(
                    f"Sample shape mismatch across splits for feature '{key}': "
                    f"{prior_sample_shape} vs {sample_shape}"
                )
            manifest_views[key][split] = split_meta
            manifest_views[key]["input_dim"] = int(sample_shape[-1])
            manifest_views[key]["sample_shape"] = sample_shape

    prompt_profile_metadata_compatible = (
        prompt_profile_contract_compatible
        and _reference_prompt_profile_rows_match_records(
            reference_data_dir=args.reference_data_dir,
            reference_manifest=reference_manifest,
            records_by_split=records_by_split,
        )
    )
    if (
        not prompt_profile_metadata_compatible
        and _is_prompt_profile_target_spec(reference_manifest.get("target_spec"))
    ):
        raise SystemExit(
            "Cannot rebuild a prompt-profile target dataset unless the rebuilt "
            "prompts still match the archived prompt-profile contract. Reuse the "
            "original source pools or relabel from a fresh prompt-profile archive."
        )

    prompt_template_payload = reference_manifest.get("prompt_template")
    if isinstance(prompt_template_payload, dict):
        prompt_template_payload = dict(prompt_template_payload)
        prompt_template_payload["num_repetition"] = num_repetition
    else:
        prompt_template_payload = {
            "chat_template": True,
            "num_repetition": num_repetition,
            "source": "utils.build_prompt",
        }

    if prompt_profile_metadata_compatible:
        prompt_profile_files = reference_manifest.get("prompt_profile_files")
        if isinstance(prompt_profile_files, dict):
            for rel_path in prompt_profile_files.values():
                if isinstance(rel_path, str) and rel_path:
                    _copy_reference_relpath(
                        reference_data_dir=args.reference_data_dir,
                        out_dir=args.out_dir,
                        rel_path=rel_path,
                    )
        prompt_rollout_archive_file = reference_manifest.get("prompt_rollout_archive_file")
        if isinstance(prompt_rollout_archive_file, str) and prompt_rollout_archive_file:
            _copy_reference_relpath(
                reference_data_dir=args.reference_data_dir,
                out_dir=args.out_dir,
                rel_path=prompt_rollout_archive_file,
            )

    payload = {
        "version": 5,
        "created_from": args.reference_data_dir,
        "default_feature_key": primary_key,
        "feature_key": primary_key,
        "feature_stage": "prefill",
        "feature_pooling": feature_views[primary_key]["pooling"],
        "input_dim": int(manifest_views[primary_key]["input_dim"]),
        "sample_shape": manifest_views[primary_key]["sample_shape"],
        "feature_views": manifest_views,
        "prompt_field": prompt_field,
        "prompt_template": prompt_template_payload,
        "prefill_extraction": {
            "model_id": model_id,
            "trust_remote_code": trust_remote_code,
            "prefill_batch_size": args.prefill_batch_size,
        },
    }
    for split in built_splits:
        payload[split] = manifest_views[primary_key][split]
    for key in (
        "balancing",
        "label_spec",
        "loop_detector",
        "rollout_config",
        "seed",
        "selection",
        "split_ratio",
        "split_source",
        "task_kind",
        "test_spec",
        "target_spec",
        "train_spec",
        "answer_field",
        "prompt_profile_files",
        "prompt_rollout_archive_file",
        "task_loader_config",
    ):
        if (
            key in {"prompt_profile_files", "prompt_rollout_archive_file"}
            and not prompt_profile_metadata_compatible
        ):
            continue
        value = reference_manifest.get(key)
        if value is not None:
            payload[key] = value

    write_manifest(args.out_dir, payload)
    print(f"Wrote fixed-split prefill views to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
