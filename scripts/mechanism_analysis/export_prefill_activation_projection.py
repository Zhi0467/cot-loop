#!/usr/bin/env python3
"""Export prompt- and rollout-level 2D projections from saved prefill datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probe.adapters import multiple_choice_gpqa, multiple_choice_mmlupro
from probe.types import DatasetSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join saved prefill feature shards with repeated-rollout diagnostics "
            "and export prompt/rollout 2D projection tables."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Dataset build directory.")
    parser.add_argument("--out-dir", required=True, help="Where to write CSV/JSON outputs.")
    parser.add_argument(
        "--projection-view",
        choices=("last_layer", "all_layers_flat"),
        default="last_layer",
        help=(
            "How to project stacked [layer, hidden] prefill features. "
            "'last_layer' matches the default last-prefill-token final-layer view."
        ),
    )
    parser.add_argument(
        "--source-dataset",
        default="",
        help=(
            "Optional override for the dataset path/id used to reconstruct correctness. "
            "Defaults to manifest train/test specs."
        ),
    )
    parser.add_argument(
        "--source-config",
        default="",
        help="Optional override for the source dataset config when reconstructing correctness.",
    )
    parser.add_argument(
        "--source-split",
        default="",
        help="Optional override for the source dataset split when reconstructing correctness.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_projection_vectors(features: torch.Tensor, projection_view: str) -> np.ndarray:
    if features.ndim == 2:
        return features.float().cpu().numpy()
    if features.ndim != 3:
        raise SystemExit(
            "Expected rank-2 or rank-3 feature tensor, "
            f"got shape {tuple(features.shape)}."
        )
    if projection_view == "last_layer":
        return features[:, -1, :].float().cpu().numpy()
    if projection_view == "all_layers_flat":
        flat = features.reshape(features.shape[0], -1)
        return flat.float().cpu().numpy()
    raise SystemExit(f"Unsupported projection view '{projection_view}'.")


def project_2d(vectors: np.ndarray) -> tuple[np.ndarray, list[float]]:
    num_points = int(vectors.shape[0])
    if num_points < 1:
        raise SystemExit("Expected at least one feature row for projection.")
    if num_points == 1:
        return np.zeros((1, 2), dtype=float), []

    n_components = min(2, num_points, int(vectors.shape[1]))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vectors)
    if coords.shape[1] == 1:
        coords = np.concatenate(
            [coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)],
            axis=1,
        )
    return coords, [float(x) for x in pca.explained_variance_ratio_]


def load_split_features(
    data_dir: Path,
    split_name: str,
    split_meta: dict[str, Any],
    projection_view: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_path in split_meta["shards"]:
        shard_path = data_dir / rel_path
        shard = torch.load(shard_path, map_location="cpu")
        vectors = resolve_projection_vectors(shard["x"], projection_view)
        sample_ids = shard["sample_ids"].tolist()
        targets = shard["y"].tolist()
        for sample_id, target_value, vector in zip(
            sample_ids,
            targets,
            vectors,
            strict=True,
        ):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": int(sample_id),
                    "target_value": float(target_value),
                    "vector": vector,
                    "feature_norm": float(np.linalg.norm(vector)),
                }
            )
    return rows


def spec_from_manifest(spec_payload: dict[str, Any], *, args: argparse.Namespace) -> DatasetSpec:
    dataset = args.source_dataset or str(spec_payload["dataset"])
    config = args.source_config or spec_payload.get("config")
    split = args.source_split or str(spec_payload["split"])
    max_samples = spec_payload.get("max_samples")
    if max_samples is not None:
        try:
            max_samples = int(max_samples)
        except Exception as exc:
            raise SystemExit(
                f"Invalid max_samples in manifest spec payload: {max_samples!r}"
            ) from exc
    return DatasetSpec(
        dataset=dataset,
        config=config,
        split=split,
        max_samples=max_samples,
    )


def unique_specs(manifest: dict[str, Any], *, args: argparse.Namespace) -> list[DatasetSpec]:
    seen: set[tuple[str, str | None, str]] = set()
    specs: list[DatasetSpec] = []
    for key in ("train_spec", "test_spec"):
        payload = manifest.get(key)
        if payload is None:
            continue
        spec = spec_from_manifest(payload, args=args)
        dedupe_key = (spec.dataset, spec.config, spec.split)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        specs.append(spec)
    return specs


def spec_payload_for_split(manifest: dict[str, Any], split_name: str) -> dict[str, Any] | None:
    split_source = str(manifest.get("split_source", ""))
    if split_source == "same_train_test_source_count_split":
        train_payload = manifest.get("train_spec")
        test_payload = manifest.get("test_spec")
        shared_payload = train_payload if isinstance(train_payload, dict) else test_payload
        if isinstance(shared_payload, dict):
            merged_payload = dict(shared_payload)
            train_max = train_payload.get("max_samples") if isinstance(train_payload, dict) else None
            test_max = test_payload.get("max_samples") if isinstance(test_payload, dict) else None
            if train_max is not None and test_max is not None:
                merged_payload["max_samples"] = int(train_max) + int(test_max)
            return merged_payload
        return None
    if split_source == "same_train_test_source_ratio_split":
        shared_payload = manifest.get("train_spec") or manifest.get("test_spec")
        if isinstance(shared_payload, dict):
            merged_payload = dict(shared_payload)
            merged_payload["max_samples"] = None
            return merged_payload
        return None
    if split_source in {
        "single_dataset_split",
        "same_train_test_spec_split",
    }:
        shared_payload = manifest.get("train_spec") or manifest.get("test_spec")
        if isinstance(shared_payload, dict):
            return shared_payload
        return None
    payload = manifest.get(f"{split_name}_spec")
    if isinstance(payload, dict):
        return payload
    return None


def build_correctness_lookup(
    manifest: dict[str, Any],
    *,
    sample_keys: set[tuple[str, int]],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    task_kind = str(manifest.get("task_kind", ""))
    sample_ids_by_split: dict[str, set[int]] = {}
    for split_name, sample_id in sample_keys:
        sample_ids_by_split.setdefault(split_name, set()).add(sample_id)
    try:
        gpqa_seed = int(manifest.get("seed", 0))
    except Exception:
        gpqa_seed = 0
    if task_kind == "multiple_choice_gpqa":
        lookup: dict[tuple[str, int], dict[str, Any]] = {}
        for split_name in ("train", "test"):
            spec_payload = spec_payload_for_split(manifest, split_name)
            if spec_payload is None:
                continue
            sample_ids = sample_ids_by_split.get(split_name, set())
            if not sample_ids:
                continue
            spec = spec_from_manifest(spec_payload, args=args)
            for record, _options, gold_letter in multiple_choice_gpqa.load_and_shuffle(
                spec,
                gpqa_seed,
            ):
                if record.sample_id in sample_ids:
                    lookup[(split_name, record.sample_id)] = {"gold_letter": gold_letter}
        return lookup, task_kind
    if task_kind == "multiple_choice_mmlupro":
        lookup = {}
        for split_name in ("train", "test"):
            spec_payload = spec_payload_for_split(manifest, split_name)
            if spec_payload is None:
                continue
            sample_ids = sample_ids_by_split.get(split_name, set())
            if not sample_ids:
                continue
            spec = spec_from_manifest(spec_payload, args=args)
            for record, _options, gold_answer, gold_index in multiple_choice_mmlupro.load_samples(
                spec
            ):
                if record.sample_id in sample_ids:
                    lookup[(split_name, record.sample_id)] = {
                        "gold_answer": gold_answer,
                        "gold_index": gold_index,
                    }
        return lookup, task_kind
    return {}, task_kind


def grade_rollout(
    task_kind: str,
    metadata: dict[str, Any] | None,
    completion_text: str,
) -> int | None:
    if metadata is None:
        return None
    if task_kind == "multiple_choice_gpqa":
        return int(
            multiple_choice_gpqa.grade(
                completion_text,
                str(metadata["gold_letter"]),
            )
        )
    if task_kind == "multiple_choice_mmlupro":
        return int(
            multiple_choice_mmlupro.grade(
                completion_text,
                str(metadata["gold_answer"]),
                metadata["gold_index"],
            )
        )
    return None


def prompt_preview(prompt: str, limit: int = 120) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest = read_json(data_dir / "manifest.json")

    feature_rows = (
        load_split_features(
            data_dir,
            "train",
            manifest["train"],
            args.projection_view,
        )
        + load_split_features(
            data_dir,
            "test",
            manifest["test"],
            args.projection_view,
        )
    )
    if not feature_rows:
        raise SystemExit(f"No feature shards found under '{data_dir}'.")

    vectors = np.stack([row["vector"] for row in feature_rows], axis=0)
    coords, explained_variance_ratio = project_2d(vectors)
    for row, coord in zip(feature_rows, coords, strict=True):
        row["pc1"] = float(coord[0])
        row["pc2"] = float(coord[1])
        del row["vector"]

    prompt_profile_rows = (
        read_jsonl(data_dir / manifest["prompt_profile_files"]["train"])
        + read_jsonl(data_dir / manifest["prompt_profile_files"]["test"])
    )
    prompt_profile_lookup = {
        (str(row["split"]), int(row["sample_id"])): row for row in prompt_profile_rows
    }
    archive_rows = read_jsonl(data_dir / manifest["prompt_rollout_archive_file"])
    archive_lookup = {
        (str(row["split"]), int(row["sample_id"])): row for row in archive_rows
    }

    sample_key_set = {
        (str(row["split"]), int(row["sample_id"])) for row in feature_rows
    }
    correctness_lookup, task_kind = build_correctness_lookup(
        manifest,
        sample_keys=sample_key_set,
        args=args,
    )

    prompt_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_profile: list[tuple[str, int]] = []
    missing_archive: list[tuple[str, int]] = []

    for feature_row in feature_rows:
        key = (str(feature_row["split"]), int(feature_row["sample_id"]))
        profile_row = prompt_profile_lookup.get(key)
        archive_row = archive_lookup.get(key)
        if profile_row is None:
            missing_profile.append(key)
            continue
        if archive_row is None:
            missing_archive.append(key)
            continue

        sample_meta = correctness_lookup.get(key)
        correct_flags: list[int] = []
        for rollout in archive_row["rollouts"]:
            correct_flag = grade_rollout(
                task_kind,
                sample_meta,
                str(rollout["completion_text"]),
            )
            if correct_flag is not None:
                correct_flags.append(correct_flag)
            rollout_rows.append(
                {
                    "split": str(feature_row["split"]),
                    "sample_id": int(feature_row["sample_id"]),
                    "rollout_index": int(rollout["rollout_index"]),
                    "pc1": float(feature_row["pc1"]),
                    "pc2": float(feature_row["pc2"]),
                    "cap_hit": int(rollout["cap_hit"]),
                    "loop_flag": int(rollout["loop_flag"]),
                    "correct": "" if correct_flag is None else int(correct_flag),
                    "finish_reason": str(rollout["finish_reason"]),
                    "length": int(rollout["length"]),
                    "relative_length": float(rollout["relative_length"]),
                    "first_loop_prefix_length": (
                        ""
                        if rollout["first_loop_prefix_length"] is None
                        else int(rollout["first_loop_prefix_length"])
                    ),
                }
            )

        prompt_rows.append(
            {
                "split": str(feature_row["split"]),
                "sample_id": int(feature_row["sample_id"]),
                "pc1": float(feature_row["pc1"]),
                "pc2": float(feature_row["pc2"]),
                "feature_norm": float(feature_row["feature_norm"]),
                "target_value": float(feature_row["target_value"]),
                "prompt_token_count": int(profile_row["prompt_token_count"]),
                "effective_max_tokens": int(profile_row["effective_max_tokens"]),
                "mean_length": float(profile_row["mean_length"]),
                "mean_relative_length": float(profile_row["mean_relative_length"]),
                "mu_log_rel": float(profile_row["mu_log_rel"]),
                "p_cap": float(profile_row["p_cap"]),
                "p_loop": float(profile_row["p_loop"]),
                "correct_rate": (
                    ""
                    if not correct_flags
                    else float(sum(correct_flags) / len(correct_flags))
                ),
                "rollout_count": int(profile_row["num_rollouts"]),
                "prompt_preview": prompt_preview(str(archive_row["prompt"])),
            }
        )

    if missing_profile:
        raise SystemExit(
            "Missing prompt-profile rows for feature samples: "
            + ", ".join(f"{split}:{sample_id}" for split, sample_id in missing_profile)
        )
    if missing_archive:
        raise SystemExit(
            "Missing rollout-archive rows for feature samples: "
            + ", ".join(f"{split}:{sample_id}" for split, sample_id in missing_archive)
        )

    prompt_rows.sort(key=lambda row: (row["split"], row["sample_id"]))
    rollout_rows.sort(
        key=lambda row: (row["split"], row["sample_id"], row["rollout_index"])
    )

    prompt_fieldnames = [
        "split",
        "sample_id",
        "pc1",
        "pc2",
        "feature_norm",
        "target_value",
        "prompt_token_count",
        "effective_max_tokens",
        "mean_length",
        "mean_relative_length",
        "mu_log_rel",
        "p_cap",
        "p_loop",
        "correct_rate",
        "rollout_count",
        "prompt_preview",
    ]
    rollout_fieldnames = [
        "split",
        "sample_id",
        "rollout_index",
        "pc1",
        "pc2",
        "cap_hit",
        "loop_flag",
        "correct",
        "finish_reason",
        "length",
        "relative_length",
        "first_loop_prefix_length",
    ]

    write_csv(out_dir / "prompt_projection.csv", prompt_fieldnames, prompt_rows)
    write_csv(out_dir / "rollout_projection.csv", rollout_fieldnames, rollout_rows)

    rollout_correct_values = [
        int(row["correct"])
        for row in rollout_rows
        if str(row["correct"]).strip() != ""
    ]
    summary = {
        "data_dir": str(data_dir),
        "projection_view": args.projection_view,
        "task_kind": task_kind,
        "manifest_default_feature_key": manifest.get("default_feature_key"),
        "num_prompts": len(prompt_rows),
        "num_rollouts": len(rollout_rows),
        "explained_variance_ratio": explained_variance_ratio,
        "split_counts": {
            "prompts": {
                "train": sum(1 for row in prompt_rows if row["split"] == "train"),
                "test": sum(1 for row in prompt_rows if row["split"] == "test"),
            },
            "rollouts": {
                "train": sum(1 for row in rollout_rows if row["split"] == "train"),
                "test": sum(1 for row in rollout_rows if row["split"] == "test"),
            },
        },
        "prompt_stats": {
            "mean_p_cap": float(np.mean([row["p_cap"] for row in prompt_rows])),
            "mean_p_loop": float(np.mean([row["p_loop"] for row in prompt_rows])),
            "mean_mean_relative_length": float(
                np.mean([row["mean_relative_length"] for row in prompt_rows])
            ),
        },
        "rollout_stats": {
            "cap_hit_rate": float(np.mean([row["cap_hit"] for row in rollout_rows])),
            "loop_rate": float(np.mean([row["loop_flag"] for row in rollout_rows])),
            "correct_rate": (
                None
                if not rollout_correct_values
                else float(np.mean(rollout_correct_values))
            ),
        },
    }
    write_json(out_dir / "projection_summary.json", summary)

    print(
        json.dumps(
            {
                "prompt_csv": str(out_dir / "prompt_projection.csv"),
                "rollout_csv": str(out_dir / "rollout_projection.csv"),
                "summary_json": str(out_dir / "projection_summary.json"),
                "explained_variance_ratio": summary["explained_variance_ratio"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
