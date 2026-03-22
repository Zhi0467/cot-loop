#!/usr/bin/env python3
"""Export prompt-level activation projections and separability summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_probe.adapters import math_freeform, multiple_choice_gpqa, multiple_choice_mmlupro
from loop_probe.types import DatasetSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join saved prompt-profile datasets with prefill activations and export "
            "prompt-level 2D projections plus quantitative separability summaries."
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
        "--tail-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.9],
        help="Relative-length thresholds to summarize on top of the saved archive.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=6,
        help="Maximum k to consider when picking the unsupervised prompt clustering.",
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
        for sample_id, target_value, vector in zip(sample_ids, targets, vectors):
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
    return DatasetSpec(
        dataset=dataset,
        config=config,
        split=split,
        max_samples=None,
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


def build_correctness_lookup(
    manifest: dict[str, Any],
    *,
    sample_ids: set[int],
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], str]:
    task_kind = str(manifest.get("task_kind", ""))
    if task_kind == "multiple_choice_gpqa":
        lookup: dict[int, dict[str, Any]] = {}
        for spec in unique_specs(manifest, args=args):
            for record, _options, gold_letter in multiple_choice_gpqa.load_and_shuffle(
                spec,
                int(manifest["seed"]),
            ):
                if record.sample_id in sample_ids:
                    lookup[record.sample_id] = {"gold_letter": gold_letter}
        return lookup, task_kind
    if task_kind == "multiple_choice_mmlupro":
        lookup = {}
        for spec in unique_specs(manifest, args=args):
            for record, _options, gold_answer, gold_index in multiple_choice_mmlupro.load_samples(
                spec
            ):
                if record.sample_id in sample_ids:
                    lookup[record.sample_id] = {
                        "gold_answer": gold_answer,
                        "gold_index": gold_index,
                    }
        return lookup, task_kind
    if task_kind == "math_freeform":
        lookup = {}
        for spec in unique_specs(manifest, args=args):
            for record, gold_answer in math_freeform.load_samples(spec):
                if record.sample_id in sample_ids:
                    lookup[record.sample_id] = {"gold_answer": gold_answer}
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
    if task_kind == "math_freeform":
        return int(
            math_freeform.grade(
                completion_text,
                str(metadata["gold_answer"]),
            )
        )
    return None


def prompt_preview(prompt: str, limit: int = 120) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def format_threshold(threshold: float) -> str:
    return format(float(threshold), "g")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    xr = rankdata(x)
    yr = rankdata(y)
    corr = np.corrcoef(xr, yr)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def choose_clusters(coords: np.ndarray, *, max_clusters: int) -> tuple[np.ndarray, dict[str, Any]]:
    num_points = coords.shape[0]
    if num_points < 2:
        return np.zeros(num_points, dtype=int), {
            "k": 1,
            "silhouette": None,
            "inertia": 0.0,
            "candidate_scores": [],
        }

    max_k = max(2, min(int(max_clusters), num_points - 1))
    best: dict[str, Any] | None = None
    candidate_scores: list[dict[str, Any]] = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=0)
        labels = model.fit_predict(coords)
        if len(set(labels)) < 2:
            silhouette = None
        else:
            silhouette = float(silhouette_score(coords, labels))
        candidate = {
            "k": int(k),
            "labels": labels,
            "inertia": float(model.inertia_),
            "silhouette": silhouette,
        }
        candidate_scores.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
            }
        )
        score = -math.inf if silhouette is None else float(silhouette)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "labels": labels,
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
            }

    if best is None:
        return np.zeros(num_points, dtype=int), {
            "k": 1,
            "silhouette": None,
            "inertia": 0.0,
            "candidate_scores": candidate_scores,
        }
    return np.asarray(best["labels"], dtype=int), {
        "k": int(best["k"]),
        "silhouette": best["silhouette"],
        "inertia": float(best["inertia"]),
        "candidate_scores": candidate_scores,
    }


def cluster_vote_metrics(
    y_true: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, Any]:
    unique = sorted(set(int(x) for x in y_true.tolist()))
    if unique != [0, 1]:
        return {
            "prevalence": float(np.mean(y_true)) if len(y_true) else None,
            "accuracy": None,
            "balanced_accuracy": None,
            "adjusted_mutual_info": None,
            "adjusted_rand": None,
            "label_silhouette_2d": None,
        }

    cluster_to_label: dict[int, int] = {}
    for cluster_id in sorted(set(int(x) for x in cluster_ids.tolist())):
        mask = cluster_ids == cluster_id
        positives = int(np.sum(y_true[mask]))
        negatives = int(np.sum(mask)) - positives
        cluster_to_label[cluster_id] = int(positives > negatives)

    predictions = np.array([cluster_to_label[int(cid)] for cid in cluster_ids], dtype=int)
    return {
        "prevalence": float(np.mean(y_true)),
        "accuracy": float(np.mean(predictions == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y_true, cluster_ids)),
        "adjusted_rand": float(adjusted_rand_score(y_true, cluster_ids)),
    }


def label_silhouette(coords: np.ndarray, labels: np.ndarray) -> float | None:
    unique = sorted(set(int(x) for x in labels.tolist()))
    if unique != [0, 1]:
        return None
    counts = [int(np.sum(labels == value)) for value in unique]
    if min(counts) < 2:
        return None
    return float(silhouette_score(coords, labels))


def continuous_cluster_r2(values: np.ndarray, cluster_ids: np.ndarray) -> float | None:
    if len(values) < 2 or np.allclose(values, values[0]):
        return None
    grand_mean = float(np.mean(values))
    sst = float(np.sum((values - grand_mean) ** 2))
    if sst <= 0.0:
        return None
    sse = 0.0
    for cluster_id in sorted(set(int(x) for x in cluster_ids.tolist())):
        mask = cluster_ids == cluster_id
        cluster_values = values[mask]
        cluster_mean = float(np.mean(cluster_values))
        sse += float(np.sum((cluster_values - cluster_mean) ** 2))
    return float(max(0.0, 1.0 - (sse / sst)))


def continuous_projection_metrics(values: np.ndarray, coords: np.ndarray) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "mean": None,
            "std": None,
            "linear_r2_2d": None,
            "spearman_pc1": None,
            "spearman_pc2": None,
            "max_abs_spearman_pc": None,
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    if np.allclose(values, values[0]):
        return {
            "mean": mean,
            "std": std,
            "linear_r2_2d": None,
            "spearman_pc1": None,
            "spearman_pc2": None,
            "max_abs_spearman_pc": None,
        }
    linear_r2 = float(LinearRegression().fit(coords, values).score(coords, values))
    spearman_pc1 = spearman_corr(coords[:, 0], values)
    spearman_pc2 = spearman_corr(coords[:, 1], values)
    max_abs = None
    if spearman_pc1 is not None or spearman_pc2 is not None:
        candidates = [
            abs(value)
            for value in (spearman_pc1, spearman_pc2)
            if value is not None
        ]
        max_abs = float(max(candidates))
    return {
        "mean": mean,
        "std": std,
        "linear_r2_2d": linear_r2,
        "spearman_pc1": spearman_pc1,
        "spearman_pc2": spearman_pc2,
        "max_abs_spearman_pc": max_abs,
    }


def detect_dataset_name(manifest: dict[str, Any]) -> str:
    spec = manifest.get("train_spec") or manifest.get("test_spec") or {}
    dataset = spec.get("dataset")
    config = spec.get("config")
    if dataset and config:
        dataset_text = str(dataset)
        dataset_path = Path(dataset_text)
        if dataset_path.suffix:
            dataset_label = dataset_path.stem
        else:
            dataset_label = dataset_path.name or dataset_text
        return f"{dataset_label}:{config}"
    if dataset:
        dataset_text = str(dataset)
        dataset_path = Path(dataset_text)
        if dataset_path.suffix:
            return dataset_path.stem
        return dataset_path.name or dataset_text
    return str(manifest.get("task_kind", "unknown"))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(data_dir / "manifest.json")

    thresholds = sorted({float(x) for x in args.tail_thresholds})
    if not thresholds:
        raise SystemExit("At least one tail threshold is required.")

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
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    cluster_ids, cluster_summary = choose_clusters(
        coords,
        max_clusters=int(args.max_clusters),
    )

    for row, coord, cluster_id in zip(feature_rows, coords, cluster_ids):
        row["pc1"] = float(coord[0])
        row["pc2"] = float(coord[1])
        row["cluster_id"] = int(cluster_id)
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

    sample_id_set = {int(row["sample_id"]) for row in feature_rows}
    correctness_lookup, task_kind = build_correctness_lookup(
        manifest,
        sample_ids=sample_id_set,
        args=args,
    )

    prompt_rows: list[dict[str, Any]] = []
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

        sample_meta = correctness_lookup.get(int(feature_row["sample_id"]))
        rollouts = list(archive_row["rollouts"])
        num_rollouts = len(rollouts)
        relative_lengths = np.array(
            [float(rollout["relative_length"]) for rollout in rollouts],
            dtype=float,
        )
        cap_hits = np.array([int(rollout["cap_hit"]) for rollout in rollouts], dtype=int)
        loop_flags = np.array([int(rollout["loop_flag"]) for rollout in rollouts], dtype=int)
        finish_reasons = [str(rollout["finish_reason"]) for rollout in rollouts]

        correct_flags: list[int] = []
        for rollout in rollouts:
            correct_flag = grade_rollout(
                task_kind,
                sample_meta,
                str(rollout["completion_text"]),
            )
            if correct_flag is not None:
                correct_flags.append(int(correct_flag))

        row: dict[str, Any] = {
            "split": str(feature_row["split"]),
            "sample_id": int(feature_row["sample_id"]),
            "pc1": float(feature_row["pc1"]),
            "pc2": float(feature_row["pc2"]),
            "cluster_id": int(feature_row["cluster_id"]),
            "feature_norm": float(feature_row["feature_norm"]),
            "target_value": float(feature_row["target_value"]),
            "prompt_token_count": int(profile_row["prompt_token_count"]),
            "effective_max_tokens": int(profile_row["effective_max_tokens"]),
            "mean_length": float(profile_row["mean_length"]),
            "mean_relative_length": float(profile_row["mean_relative_length"]),
            "mu_log_rel": float(profile_row["mu_log_rel"]),
            "p_cap": float(profile_row["p_cap"]),
            "p_loop": float(profile_row["p_loop"]),
            "majority_cap": int(np.sum(cap_hits) > (num_rollouts / 2.0)),
            "majority_loop": int(np.sum(loop_flags) > (num_rollouts / 2.0)),
            "any_cap": int(np.any(cap_hits)),
            "any_loop": int(np.any(loop_flags)),
            "rollout_count": int(profile_row["num_rollouts"]),
            "prompt_preview": prompt_preview(str(archive_row["prompt"])),
            "finish_reason_length_rate": float(
                np.mean([int(reason == "length") for reason in finish_reasons])
            ),
            "majority_finish_reason_length": int(
                np.sum([int(reason == "length") for reason in finish_reasons])
                > (num_rollouts / 2.0)
            ),
        }

        if correct_flags:
            correct_values = np.array(correct_flags, dtype=int)
            row["correct_rate"] = float(np.mean(correct_values))
            row["majority_correct"] = int(np.sum(correct_values) > (len(correct_values) / 2.0))
        else:
            row["correct_rate"] = ""
            row["majority_correct"] = ""

        for threshold in thresholds:
            threshold_text = format_threshold(threshold)
            tail_hits = (relative_lengths >= float(threshold)).astype(int)
            row[f"s_{threshold_text}"] = float(np.mean(tail_hits))
            row[f"majority_s_{threshold_text}"] = int(np.sum(tail_hits) > (num_rollouts / 2.0))

        prompt_rows.append(row)

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

    threshold_fields = []
    for threshold in thresholds:
        threshold_text = format_threshold(threshold)
        threshold_fields.extend([f"s_{threshold_text}", f"majority_s_{threshold_text}"])

    prompt_fieldnames = [
        "split",
        "sample_id",
        "pc1",
        "pc2",
        "cluster_id",
        "feature_norm",
        "target_value",
        "prompt_token_count",
        "effective_max_tokens",
        "mean_length",
        "mean_relative_length",
        "mu_log_rel",
        "p_cap",
        "p_loop",
        "majority_cap",
        "majority_loop",
        "any_cap",
        "any_loop",
        "finish_reason_length_rate",
        "majority_finish_reason_length",
        "correct_rate",
        "majority_correct",
        *threshold_fields,
        "rollout_count",
        "prompt_preview",
    ]
    write_csv(out_dir / "prompt_projection.csv", prompt_fieldnames, prompt_rows)

    coord_array = np.stack(
        [[float(row["pc1"]), float(row["pc2"])] for row in prompt_rows],
        axis=0,
    )
    cluster_array = np.array([int(row["cluster_id"]) for row in prompt_rows], dtype=int)

    binary_metrics: dict[str, Any] = {}
    binary_labels = [
        "majority_cap",
        "majority_loop",
        "majority_finish_reason_length",
        "majority_correct",
        *[f"majority_s_{format_threshold(threshold)}" for threshold in thresholds],
    ]
    for label_name in binary_labels:
        raw_values = [row.get(label_name, "") for row in prompt_rows]
        if any(value == "" for value in raw_values):
            continue
        values = np.array([int(value) for value in raw_values], dtype=int)
        metrics = cluster_vote_metrics(values, cluster_array)
        metrics["label_silhouette_2d"] = label_silhouette(coord_array, values)
        binary_metrics[label_name] = metrics

    continuous_metrics: dict[str, Any] = {}
    continuous_labels = [
        "p_cap",
        "p_loop",
        "finish_reason_length_rate",
        "mean_relative_length",
        "correct_rate",
        *[f"s_{format_threshold(threshold)}" for threshold in thresholds],
    ]
    for label_name in continuous_labels:
        raw_values = [row.get(label_name, "") for row in prompt_rows]
        if any(value == "" for value in raw_values):
            continue
        values = np.array([float(value) for value in raw_values], dtype=float)
        metrics = continuous_projection_metrics(values, coord_array)
        metrics["cluster_r2"] = continuous_cluster_r2(values, cluster_array)
        continuous_metrics[label_name] = metrics

    summary = {
        "data_dir": str(data_dir),
        "dataset_name": detect_dataset_name(manifest),
        "projection_view": args.projection_view,
        "task_kind": task_kind,
        "manifest_default_feature_key": manifest.get("default_feature_key"),
        "num_prompts": len(prompt_rows),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "split_counts": {
            "prompts": {
                "train": sum(1 for row in prompt_rows if row["split"] == "train"),
                "test": sum(1 for row in prompt_rows if row["split"] == "test"),
            }
        },
        "tail_thresholds": thresholds,
        "cluster_summary": {
            **cluster_summary,
            "num_clusters": int(len(set(cluster_array.tolist()))),
            "cluster_sizes": {
                str(cluster_id): int(np.sum(cluster_array == cluster_id))
                for cluster_id in sorted(set(cluster_array.tolist()))
            },
        },
        "prompt_stats": {
            "mean_p_cap": float(np.mean([float(row["p_cap"]) for row in prompt_rows])),
            "mean_p_loop": float(np.mean([float(row["p_loop"]) for row in prompt_rows])),
            "mean_mean_relative_length": float(
                np.mean([float(row["mean_relative_length"]) for row in prompt_rows])
            ),
            "mean_correct_rate": (
                None
                if any(row["correct_rate"] == "" for row in prompt_rows)
                else float(np.mean([float(row["correct_rate"]) for row in prompt_rows]))
            ),
        },
        "binary_label_alignment": binary_metrics,
        "continuous_signal": continuous_metrics,
    }
    write_json(out_dir / "projection_summary.json", summary)

    print(
        json.dumps(
            {
                "prompt_csv": str(out_dir / "prompt_projection.csv"),
                "summary_json": str(out_dir / "projection_summary.json"),
                "explained_variance_ratio": summary["explained_variance_ratio"],
                "cluster_k": summary["cluster_summary"]["k"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
