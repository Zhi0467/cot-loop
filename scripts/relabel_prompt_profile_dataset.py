#!/usr/bin/env python3
"""Re-label a saved prompt-profile dataset from its rollout archive."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.dataloader import ActivationDataset, read_manifest, resolve_feature_key
from loop_probe.serialization import save_split_shards, write_manifest

TARGET_KIND_CHOICES = ("binary", "probability", "regression")
BALANCE_CHOICES = ("none", "downsample")
PROFILE_TARGET_CHOICES = (
    "s_tail",
    "p_loop",
    "p_cap",
    "mean_relative_length",
    "loop_budget_share",
    "majority_tail",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--target-kind",
        choices=TARGET_KIND_CHOICES,
        required=True,
    )
    parser.add_argument(
        "--profile-target",
        choices=PROFILE_TARGET_CHOICES,
        default=None,
    )
    parser.add_argument("--profile-tail-threshold", type=float, default=0.9)
    parser.add_argument(
        "--balance-train",
        choices=BALANCE_CHOICES,
        default="none",
    )
    parser.add_argument(
        "--balance-test",
        choices=BALANCE_CHOICES,
        default="none",
    )
    parser.add_argument("--balance-seed", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_rows(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _source_shard_size(
    source_dir: str,
    split_meta: object,
) -> int:
    if isinstance(split_meta, dict):
        shard_paths = split_meta.get("shards")
        if isinstance(shard_paths, list) and shard_paths:
            first_rel_path = shard_paths[0]
            if isinstance(first_rel_path, str) and first_rel_path:
                shard_path = os.path.join(source_dir, first_rel_path)
                if os.path.exists(shard_path):
                    shard = torch.load(shard_path, map_location="cpu")
                    x = shard.get("x")
                    if isinstance(x, torch.Tensor) and x.ndim >= 1 and int(x.shape[0]) >= 1:
                        return int(x.shape[0])
        num_samples = split_meta.get("num_samples")
        if isinstance(num_samples, int) and num_samples >= 1:
            return min(num_samples, 2048)
    return 2048


def _resolve_target_spec(
    *,
    target_kind: str,
    profile_target: str | None,
    tail_threshold: float,
    num_generations: int,
) -> dict[str, object]:
    if not 0.0 < tail_threshold <= 1.0:
        raise SystemExit("--profile-tail-threshold must be in (0, 1].")

    resolved_profile_target = profile_target
    if resolved_profile_target in (None, ""):
        if target_kind == "binary":
            resolved_profile_target = "majority_tail"
        elif target_kind == "probability":
            resolved_profile_target = "s_tail"
        else:
            resolved_profile_target = "mean_relative_length"

    if target_kind == "binary":
        if resolved_profile_target != "majority_tail":
            raise SystemExit(
                "--target-kind=binary currently expects "
                "--profile-target=majority_tail."
            )
        return {
            "kind": "binary",
            "source": "prompt_profile",
            "name": _profile_target_name(
                resolved_profile_target,
                tail_threshold=tail_threshold,
            ),
            "profile_target": resolved_profile_target,
            "tail_threshold": tail_threshold,
            "num_generations": num_generations,
            "positive_rule": "strict_majority",
        }

    if target_kind == "probability":
        if resolved_profile_target not in {"s_tail", "p_loop", "p_cap"}:
            raise SystemExit(
                "--target-kind=probability currently expects "
                "--profile-target in {s_tail, p_loop, p_cap}."
            )
        return {
            "kind": "probability",
            "name": _profile_target_name(
                resolved_profile_target,
                tail_threshold=tail_threshold,
            ),
            "profile_target": resolved_profile_target,
            "tail_threshold": tail_threshold,
            "num_generations": num_generations,
            "loss": "soft_bce",
        }

    if target_kind == "regression":
        if resolved_profile_target not in {
            "mean_relative_length",
            "loop_budget_share",
        }:
            raise SystemExit(
                "--target-kind=regression currently expects "
                "--profile-target in {mean_relative_length, loop_budget_share}."
            )
        return {
            "kind": "regression",
            "name": _profile_target_name(
                resolved_profile_target,
                tail_threshold=tail_threshold,
            ),
            "profile_target": resolved_profile_target,
            "tail_threshold": tail_threshold,
            "num_generations": num_generations,
            "loss": "sigmoid_mse",
        }

    raise SystemExit(f"Unsupported --target-kind '{target_kind}'.")


def _profile_target_name(
    profile_target: str,
    *,
    tail_threshold: float,
) -> str:
    if profile_target == "s_tail":
        return f"s_{format(float(tail_threshold), 'g')}"
    if profile_target == "mean_relative_length":
        return "mean_relative_length"
    if profile_target == "loop_budget_share":
        return "loop_budget_share"
    if profile_target == "p_loop":
        return "p_loop"
    if profile_target == "p_cap":
        return "p_cap"
    if profile_target == "majority_tail":
        return f"majority_s_{format(float(tail_threshold), 'g')}"
    raise SystemExit(
        f"Unknown prompt-profile target '{profile_target}'. "
        f"Valid: {PROFILE_TARGET_CHOICES}"
    )


def _profile_target_value(
    profile: dict[str, object],
    *,
    profile_target: str,
) -> float:
    if profile_target == "s_tail":
        return float(profile["s_tail"])
    if profile_target == "mean_relative_length":
        return float(profile["mean_relative_length"])
    if profile_target == "loop_budget_share":
        return float(profile["loop_budget_share"])
    if profile_target == "p_loop":
        return float(profile["p_loop"])
    if profile_target == "p_cap":
        return float(profile["p_cap"])
    if profile_target == "majority_tail":
        return float(profile["majority_tail"])
    raise SystemExit(
        f"Unknown prompt-profile target '{profile_target}'. "
        f"Valid: {PROFILE_TARGET_CHOICES}"
    )


def _balanced_indices(
    labels: list[int | float],
    *,
    split_name: str,
    mode: str,
    seed: int,
) -> list[int]:
    if mode == "none":
        return list(range(len(labels)))
    if mode != "downsample":
        raise SystemExit(f"Unsupported balancing mode '{mode}'.")

    positive_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]
    negative_idx = [idx for idx, label in enumerate(labels) if int(label) == 0]
    if not positive_idx or not negative_idx:
        raise SystemExit(
            f"Cannot downsample split '{split_name}' without both classes "
            f"(pos={len(positive_idx)}, neg={len(negative_idx)})."
        )

    target_per_class = min(len(positive_idx), len(negative_idx))
    rng = random.Random(seed)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)
    keep = positive_idx[:target_per_class] + negative_idx[:target_per_class]
    keep.sort()
    return keep


def _subset_rows(rows: list[dict[str, object]], keep_indices: list[int]) -> list[dict[str, object]]:
    return [rows[idx] for idx in keep_indices]


def _subset_labels(values: list[int | float], keep_indices: list[int]) -> list[int | float]:
    return [values[idx] for idx in keep_indices]


def _load_archive_index(
    source_dir: str,
    manifest: dict[str, object],
) -> tuple[dict[tuple[str, int], dict[str, object]], str]:
    rel_path = manifest.get("prompt_rollout_archive_file")
    if not isinstance(rel_path, str) or not rel_path:
        raise SystemExit("Source manifest is missing prompt_rollout_archive_file.")
    archive_path = os.path.join(source_dir, rel_path)
    if not os.path.exists(archive_path):
        raise SystemExit(f"Prompt rollout archive not found: {archive_path}")

    index: dict[tuple[str, int], dict[str, object]] = {}
    for row in _read_jsonl(archive_path):
        split = row.get("split")
        sample_id = row.get("sample_id")
        if not isinstance(split, str):
            raise SystemExit("Archive row missing string 'split'.")
        if not isinstance(sample_id, int):
            raise SystemExit("Archive row missing integer 'sample_id'.")
        key = (split, sample_id)
        if key in index:
            raise SystemExit(f"Duplicate archive row for split/sample_id {key}.")
        index[key] = row
    return index, rel_path


def _profile_from_archive_row(
    row: dict[str, object],
    *,
    tail_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rollout_rows = row.get("rollouts")
    if not isinstance(rollout_rows, list) or not rollout_rows:
        raise SystemExit("Archive row is missing non-empty 'rollouts'.")

    effective_max_tokens = int(row["effective_max_tokens"])
    lengths: list[int] = []
    relative_lengths: list[float] = []
    cap_hits: list[int] = []
    loop_flags: list[int] = []
    tail_hits: list[int] = []
    first_loop_prefix_lengths: list[int | None] = []
    finish_reasons: list[str | None] = []
    rebuilt_rollouts: list[dict[str, object]] = []

    for rollout in rollout_rows:
        if not isinstance(rollout, dict):
            raise SystemExit("Archive rollout entry must be an object.")
        length = int(rollout["length"])
        relative_length = float(rollout.get("relative_length", length / effective_max_tokens))
        cap_hit = int(rollout.get("cap_hit", int(length >= effective_max_tokens)))
        first_loop_prefix = rollout.get("first_loop_prefix_length")
        if first_loop_prefix is not None:
            first_loop_prefix = int(first_loop_prefix)
        loop_flag = int(rollout.get("loop_flag", int(first_loop_prefix is not None)))
        tail_hit = int(relative_length >= tail_threshold)
        finish_reason = rollout.get("finish_reason")

        lengths.append(length)
        relative_lengths.append(relative_length)
        cap_hits.append(cap_hit)
        loop_flags.append(loop_flag)
        tail_hits.append(tail_hit)
        first_loop_prefix_lengths.append(first_loop_prefix)
        finish_reasons.append(finish_reason if isinstance(finish_reason, str) else None)

        rebuilt_rollouts.append(
            {
                "rollout_index": int(rollout.get("rollout_index", len(rebuilt_rollouts))),
                "completion_text": rollout.get("completion_text"),
                "finish_reason": finish_reason,
                "length": length,
                "relative_length": relative_length,
                "cap_hit": cap_hit,
                "loop_flag": loop_flag,
                "tail_hit": tail_hit,
                "first_loop_prefix_length": first_loop_prefix,
            }
        )

    num_rollouts = len(lengths)
    tail_hit_count = int(sum(tail_hits))
    profile = {
        "num_rollouts": num_rollouts,
        "effective_max_tokens": effective_max_tokens,
        "lengths": lengths,
        "relative_lengths": relative_lengths,
        "cap_hits": cap_hits,
        "loop_flags": loop_flags,
        "first_loop_prefix_lengths": first_loop_prefix_lengths,
        "tail_hits": tail_hits,
        "tail_hit_count": tail_hit_count,
        "majority_tail": int(tail_hit_count > (num_rollouts / 2.0)),
        "mean_length": sum(lengths) / float(num_rollouts),
        "mean_relative_length": sum(relative_lengths) / float(num_rollouts),
        "loop_budget_share": sum(
            loop_flag * relative_length
            for loop_flag, relative_length in zip(loop_flags, relative_lengths, strict=True)
        )
        / float(num_rollouts),
        "p_cap": sum(cap_hits) / float(num_rollouts),
        "p_loop": sum(loop_flags) / float(num_rollouts),
        "mu_log_rel": sum(math.log1p(value) for value in relative_lengths) / float(num_rollouts),
        "tail_threshold": float(tail_threshold),
        "s_tail": sum(tail_hits) / float(num_rollouts),
        "finish_reasons": finish_reasons,
    }
    return profile, rebuilt_rollouts


def _rows_for_split(
    split_name: str,
    sample_ids: list[int],
    archive_index: dict[tuple[str, int], dict[str, object]],
    *,
    target_spec: dict[str, object],
) -> tuple[list[int | float], list[dict[str, object]], list[dict[str, object]]]:
    labels: list[int | float] = []
    profile_rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []
    target_name = str(target_spec["name"])
    target_kind = str(target_spec["kind"])
    profile_target = str(target_spec["profile_target"])
    tail_threshold = float(target_spec["tail_threshold"])

    for sample_id in sample_ids:
        key = (split_name, int(sample_id))
        source_row = archive_index.get(key)
        if source_row is None:
            raise SystemExit(f"Archive is missing row for split/sample_id {key}.")

        profile, rebuilt_rollouts = _profile_from_archive_row(
            source_row,
            tail_threshold=tail_threshold,
        )
        raw_target_value = _profile_target_value(
            profile,
            profile_target=profile_target,
        )
        target_value: int | float
        if target_kind == "binary":
            target_value = int(raw_target_value)
        else:
            target_value = float(raw_target_value)
        labels.append(target_value)

        prompt_token_count = int(source_row["prompt_token_count"])
        profile_row = {
            "split": split_name,
            "sample_id": int(sample_id),
            "prompt_token_count": prompt_token_count,
            "effective_max_tokens": int(profile["effective_max_tokens"]),
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
            "lengths": list(profile["lengths"]),
            "relative_lengths": list(profile["relative_lengths"]),
            "cap_hits": list(profile["cap_hits"]),
            "loop_flags": list(profile["loop_flags"]),
            "tail_hits": list(profile["tail_hits"]),
            "first_loop_prefix_lengths": list(profile["first_loop_prefix_lengths"]),
            "finish_reasons": list(profile["finish_reasons"]),
        }
        profile_rows.append(profile_row)

        archive_row = {
            "split": split_name,
            "sample_id": int(sample_id),
            "source_split": source_row.get("source_split"),
            "prompt_style": source_row.get("prompt_style"),
            "choices": source_row.get("choices"),
            "prompt": source_row.get("prompt"),
            "prompt_token_ids": source_row.get("prompt_token_ids"),
            "prompt_token_count": prompt_token_count,
            "effective_max_tokens": int(profile["effective_max_tokens"]),
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
            "rollouts": rebuilt_rollouts,
        }
        archive_rows.append(archive_row)

    return labels, profile_rows, archive_rows


def _feature_keys(manifest: dict[str, object]) -> list[str | None]:
    feature_views = manifest.get("feature_views")
    if isinstance(feature_views, dict) and feature_views:
        return [str(key) for key in sorted(feature_views.keys())]
    return [None]


def _view_manifest_entry(
    source_manifest: dict[str, object],
    *,
    feature_key: str | None,
    train_meta: dict[str, object],
    test_meta: dict[str, object],
) -> dict[str, object]:
    if feature_key is None:
        feature_extraction = source_manifest.get("feature_extraction")
        if not isinstance(feature_extraction, dict):
            raise SystemExit("Legacy manifest is missing feature_extraction metadata.")
        return {
            "stage": feature_extraction.get("stage", "prefill"),
            "pooling": feature_extraction["pooling"],
            "layer": feature_extraction["layer"],
            "input_dim": int(source_manifest["input_dim"]),
            "sample_shape": list(source_manifest["sample_shape"]),
            "train": train_meta,
            "test": test_meta,
        }

    feature_views = source_manifest.get("feature_views")
    if not isinstance(feature_views, dict):
        raise SystemExit("Source manifest is missing feature_views.")
    view_info = feature_views.get(feature_key)
    if not isinstance(view_info, dict):
        raise SystemExit(f"Source manifest is missing feature view '{feature_key}'.")
    return {
        "stage": view_info.get("stage", "prefill"),
        "pooling": view_info["pooling"],
        "layer": view_info["layer"],
        "input_dim": int(view_info["input_dim"]),
        "sample_shape": list(view_info["sample_shape"]),
        "train": train_meta,
        "test": test_meta,
    }


def main() -> None:
    args = _parse_args()
    source_dir = os.path.abspath(args.source_dir)
    out_dir = os.path.abspath(args.out_dir)

    source_manifest = read_manifest(source_dir)
    archive_index, archive_rel_path = _load_archive_index(source_dir, source_manifest)

    if args.target_kind != "binary" and (
        args.balance_train != "none" or args.balance_test != "none"
    ):
        raise SystemExit(
            "Balancing is only supported for binary targets in this relabel helper."
        )

    source_rollout_cfg = source_manifest.get("rollout_config")
    if not isinstance(source_rollout_cfg, dict):
        raise SystemExit("Source manifest is missing rollout_config.")
    num_generations = int(source_rollout_cfg.get("num_generations", 0))
    if num_generations < 2:
        raise SystemExit(
            "Prompt-profile relabeling expects a repeated-rollout source dataset "
            "with rollout_config.num_generations >= 2."
        )

    target_spec = _resolve_target_spec(
        target_kind=args.target_kind,
        profile_target=args.profile_target,
        tail_threshold=args.profile_tail_threshold,
        num_generations=num_generations,
    )

    default_feature_key = resolve_feature_key(source_manifest, None)
    base_train_dataset = ActivationDataset(source_dir, "train", feature_key=default_feature_key)
    base_test_dataset = ActivationDataset(source_dir, "test", feature_key=default_feature_key)
    train_sample_ids = [int(x) for x in base_train_dataset.sample_ids.tolist()]
    test_sample_ids = [int(x) for x in base_test_dataset.sample_ids.tolist()]

    train_labels, train_profile_rows, train_archive_rows = _rows_for_split(
        "train",
        train_sample_ids,
        archive_index,
        target_spec=target_spec,
    )
    test_labels, test_profile_rows, test_archive_rows = _rows_for_split(
        "test",
        test_sample_ids,
        archive_index,
        target_spec=target_spec,
    )

    if args.target_kind == "binary":
        train_keep_idx = _balanced_indices(
            train_labels,
            split_name="train",
            mode=args.balance_train,
            seed=args.balance_seed,
        )
        test_keep_idx = _balanced_indices(
            test_labels,
            split_name="test",
            mode=args.balance_test,
            seed=args.balance_seed + 1,
        )
    else:
        train_keep_idx = list(range(len(train_labels)))
        test_keep_idx = list(range(len(test_labels)))

    train_labels = _subset_labels(train_labels, train_keep_idx)
    test_labels = _subset_labels(test_labels, test_keep_idx)
    train_profile_rows = _subset_rows(train_profile_rows, train_keep_idx)
    test_profile_rows = _subset_rows(test_profile_rows, test_keep_idx)
    train_archive_rows = _subset_rows(train_archive_rows, train_keep_idx)
    test_archive_rows = _subset_rows(test_archive_rows, test_keep_idx)
    kept_train_ids = _subset_labels(train_sample_ids, train_keep_idx)
    kept_test_ids = _subset_labels(test_sample_ids, test_keep_idx)

    feature_views_manifest: dict[str, dict[str, object]] = {}
    primary_train_meta: dict[str, object] | None = None
    primary_test_meta: dict[str, object] | None = None
    primary_input_dim: int | None = None
    train_shard_size = _source_shard_size(source_dir, source_manifest.get("train"))
    test_shard_size = _source_shard_size(source_dir, source_manifest.get("test"))

    for feature_key in _feature_keys(source_manifest):
        train_dataset = ActivationDataset(source_dir, "train", feature_key=feature_key)
        test_dataset = ActivationDataset(source_dir, "test", feature_key=feature_key)
        if [int(x) for x in train_dataset.sample_ids.tolist()] != train_sample_ids:
            raise SystemExit(
                f"Feature view '{feature_key}' has train sample_ids in a different order."
            )
        if [int(x) for x in test_dataset.sample_ids.tolist()] != test_sample_ids:
            raise SystemExit(
                f"Feature view '{feature_key}' has test sample_ids in a different order."
            )

        train_features = train_dataset.x[train_keep_idx]
        test_features = test_dataset.x[test_keep_idx]
        train_meta = save_split_shards(
            out_dir,
            "train" if feature_key == default_feature_key or feature_key is None else os.path.join("features", feature_key, "train"),
            train_features,
            train_labels,
            kept_train_ids,
            shard_size=train_shard_size,
            target_kind=args.target_kind,
        )
        test_meta = save_split_shards(
            out_dir,
            "test" if feature_key == default_feature_key or feature_key is None else os.path.join("features", feature_key, "test"),
            test_features,
            test_labels,
            kept_test_ids,
            shard_size=test_shard_size,
            target_kind=args.target_kind,
        )

        view_manifest = _view_manifest_entry(
            source_manifest,
            feature_key=feature_key,
            train_meta=train_meta,
            test_meta=test_meta,
        )

        if feature_key is None:
            primary_train_meta = train_meta
            primary_test_meta = test_meta
            primary_input_dim = int(view_manifest["input_dim"])
        else:
            feature_views_manifest[feature_key] = view_manifest
            if feature_key == default_feature_key:
                primary_train_meta = train_meta
                primary_test_meta = test_meta
                primary_input_dim = int(view_manifest["input_dim"])

    if primary_train_meta is None or primary_test_meta is None or primary_input_dim is None:
        raise SystemExit("Failed to resolve primary feature metadata from source manifest.")

    train_profile_path = os.path.join("diagnostics", "train_prompt_profile.jsonl")
    test_profile_path = os.path.join("diagnostics", "test_prompt_profile.jsonl")
    prompt_rollout_archive_path = os.path.join("diagnostics", "prompt_rollout_archive.jsonl")
    _write_jsonl_rows(os.path.join(out_dir, train_profile_path), train_profile_rows)
    _write_jsonl_rows(os.path.join(out_dir, test_profile_path), test_profile_rows)
    _write_jsonl_rows(
        os.path.join(out_dir, prompt_rollout_archive_path),
        train_archive_rows + test_archive_rows,
    )

    manifest = {
        "version": int(source_manifest.get("version", 7)),
        "input_dim": primary_input_dim,
        "sample_shape": (
            list(source_manifest["sample_shape"])
            if "sample_shape" in source_manifest
            else list(feature_views_manifest[default_feature_key]["sample_shape"])
        ),
        "default_feature_key": default_feature_key,
        "feature_extraction": source_manifest.get("feature_extraction"),
        "feature_views": feature_views_manifest or None,
        "task_kind": source_manifest.get("task_kind"),
        "prompt_field": source_manifest.get("prompt_field"),
        "answer_field": source_manifest.get("answer_field"),
        "prompt_template": source_manifest.get("prompt_template"),
        "split_source": source_manifest.get("split_source"),
        "split_ratio": source_manifest.get("split_ratio"),
        "seed": source_manifest.get("seed"),
        "loop_detector": source_manifest.get("loop_detector"),
        "target_spec": target_spec,
        "label_spec": None,
        "balancing": {
            "train": args.balance_train,
            "test": args.balance_test,
            "seed": args.balance_seed,
        },
        "rollout_config": source_rollout_cfg,
        "prompt_profile_files": {
            "train": train_profile_path,
            "test": test_profile_path,
        },
        "prompt_rollout_archive_file": prompt_rollout_archive_path,
        "prompt_profile_source_dir": source_dir,
        "prompt_profile_source_archive_file": archive_rel_path,
        "train_spec": source_manifest.get("train_spec"),
        "test_spec": source_manifest.get("test_spec"),
        "task_loader_config": source_manifest.get("task_loader_config"),
        "train": primary_train_meta,
        "test": primary_test_meta,
    }

    write_manifest(out_dir, manifest)
    print(
        f"Wrote relabeled dataset to {out_dir} "
        f"with target={target_spec['name']} ({args.target_kind}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
