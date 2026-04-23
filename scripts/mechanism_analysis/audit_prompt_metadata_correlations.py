#!/usr/bin/env python3
"""Audit cheap prompt metadata correlations for the April prompt-profile surfaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_ORDER = ("gpqa", "aime", "math500", "mmlu_pro", "livecodebench")
DISPLAY_NAMES = {
    "gpqa": "GPQA",
    "aime": "AIME",
    "math500": "MATH-500",
    "mmlu_pro": "MMLU-Pro",
    "livecodebench": "LiveCodeBench",
}
FEATURE_ORDER = (
    "prompt_token_count",
    "char_length",
    "newline_count",
    "digit_count",
    "dollar_count",
    "choice_count",
)


@dataclass(frozen=True)
class ArchiveRow:
    sample_id: int
    split: str
    prompt: str
    prompt_token_count: float
    mean_relative_length: float
    majority_s_0_5: int | None
    p_loop: float | None
    p_cap: float | None

    @property
    def char_length(self) -> float:
        return float(len(self.prompt))

    @property
    def newline_count(self) -> float:
        return float(self.prompt.count("\n"))

    @property
    def digit_count(self) -> float:
        return float(sum(ch.isdigit() for ch in self.prompt))

    @property
    def dollar_count(self) -> float:
        return float(self.prompt.count("$"))

    @property
    def choice_count(self) -> float:
        return float(self.prompt.count("\nA.") + self.prompt.count("\nB.") + self.prompt.count("\nC.") + self.prompt.count("\nD."))

    def feature_value(self, name: str) -> float:
        if name == "prompt_token_count":
            return self.prompt_token_count
        if name == "char_length":
            return self.char_length
        if name == "newline_count":
            return self.newline_count
        if name == "digit_count":
            return self.digit_count
        if name == "dollar_count":
            return self.dollar_count
        if name == "choice_count":
            return self.choice_count
        raise KeyError(f"Unknown feature '{name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", required=True)
    parser.add_argument("--binary-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def archive_path(*, root: Path, dataset: str, target: str) -> Path:
    if target == "regression":
        return root / dataset / "shared_archive" / "diagnostics" / "prompt_rollout_archive.jsonl"
    if target == "binary":
        return root / dataset / "majority_s_0.5" / "data" / "diagnostics" / "prompt_rollout_archive.jsonl"
    raise KeyError(target)


def read_archive_rows(path: Path) -> list[ArchiveRow]:
    rows: list[ArchiveRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_num}") from exc
            prompt = payload.get("prompt")
            split = payload.get("split")
            sample_id = payload.get("sample_id")
            if not isinstance(prompt, str) or not isinstance(split, str) or not isinstance(sample_id, int):
                raise SystemExit(f"Missing prompt/split/sample_id at {path}:{line_num}")
            rows.append(
                ArchiveRow(
                    sample_id=sample_id,
                    split=split,
                    prompt=prompt,
                    prompt_token_count=float(payload["prompt_token_count"]),
                    mean_relative_length=float(payload["mean_relative_length"]),
                    majority_s_0_5=(
                        None if payload.get("majority_s_0.5") is None else int(payload["majority_s_0.5"])
                    ),
                    p_loop=None if payload.get("p_loop") is None else float(payload["p_loop"]),
                    p_cap=None if payload.get("p_cap") is None else float(payload["p_cap"]),
                )
            )
    return rows


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        avg_rank = (start + 1 + stop) / 2.0
        for pos in range(start, stop):
            ranks[order[pos]] = avg_rank
        start = stop
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    sum_x = sum(value * value for value in dx)
    sum_y = sum(value * value for value in dy)
    if sum_x <= 0.0 or sum_y <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(sum_x * sum_y)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def top_fraction_indices(scores: list[float], fraction: float) -> list[int]:
    keep = max(1, math.ceil(len(scores) * fraction))
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    return order[:keep]


def top_capture_fraction(target: list[float], scores: list[float], fraction: float) -> float | None:
    if not target or len(target) != len(scores):
        return None
    total = sum(target)
    if total <= 0.0:
        return None
    chosen = top_fraction_indices(scores, fraction)
    return sum(target[idx] for idx in chosen) / total


def roc_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for value in y_true if value == 1)
    negatives = len(y_true) - positives
    if positives < 1 or negatives < 1:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    positive_ranks = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        avg_rank = (start + 1 + stop) / 2.0
        group_pos = sum(1 for pos in range(start, stop) if y_true[order[pos]] == 1)
        positive_ranks += group_pos * avg_rank
        start = stop
    numerator = positive_ranks - positives * (positives + 1) / 2.0
    denominator = positives * negatives
    return numerator / denominator if denominator > 0 else None


def pr_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for value in y_true if value == 1)
    if positives < 1:
        return None
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        group_true = [y_true[order[pos]] for pos in range(start, stop)]
        tp += sum(1 for value in group_true if value == 1)
        fp += sum(1 for value in group_true if value == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / positives
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        start = stop
    return ap


def positive_rate_at_fraction(y_true: list[int], scores: list[float], fraction: float) -> float | None:
    if not y_true or len(y_true) != len(scores):
        return None
    chosen = top_fraction_indices(scores, fraction)
    if not chosen:
        return None
    return sum(y_true[idx] for idx in chosen) / len(chosen)


def quantile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    edges = [ordered[0]]
    for bucket in range(1, bins):
        position = bucket * (len(ordered) - 1) / bins
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            edge = ordered[lower]
        else:
            frac = position - lower
            edge = ordered[lower] * (1.0 - frac) + ordered[upper] * frac
        edges.append(edge)
    edges.append(ordered[-1])
    return edges


def assign_quantile_bucket(value: float, edges: list[float]) -> int:
    if not edges:
        return 0
    for idx in range(1, len(edges)):
        upper = edges[idx]
        if idx == len(edges) - 1 or value <= upper:
            return idx - 1
    return len(edges) - 2


def summarize_length_bins(rows: list[ArchiveRow], *, bins: int = 4) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row.split == "test"]
    lengths = [row.prompt_token_count for row in test_rows]
    edges = quantile_edges(lengths, bins)
    grouped: list[list[ArchiveRow]] = [[] for _ in range(max(1, bins))]
    for row in test_rows:
        bucket = assign_quantile_bucket(row.prompt_token_count, edges)
        grouped[bucket].append(row)

    summaries: list[dict[str, Any]] = []
    for bucket, group in enumerate(grouped):
        if not group:
            continue
        summaries.append(
            {
                "bucket": bucket + 1,
                "num_rows": len(group),
                "prompt_token_count_mean": sum(row.prompt_token_count for row in group) / len(group),
                "char_length_mean": sum(row.char_length for row in group) / len(group),
                "mean_relative_length_mean": sum(row.mean_relative_length for row in group) / len(group),
                "p_loop_mean": sum((row.p_loop or 0.0) for row in group) / len(group),
                "p_cap_mean": sum((row.p_cap or 0.0) for row in group) / len(group),
                "majority_s_0.5_rate": (
                    None
                    if any(row.majority_s_0_5 is None for row in group)
                    else sum(int(row.majority_s_0_5 or 0) for row in group) / len(group)
                ),
                "bucket_prompt_token_min": min(row.prompt_token_count for row in group),
                "bucket_prompt_token_max": max(row.prompt_token_count for row in group),
            }
        )
    return summaries


def regression_feature_metrics(rows: list[ArchiveRow]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row.split == "test"]
    targets = [row.mean_relative_length for row in test_rows]
    out: list[dict[str, Any]] = []
    for feature in FEATURE_ORDER:
        scores = [row.feature_value(feature) for row in test_rows]
        out.append(
            {
                "feature": feature,
                "spearman": spearman(scores, targets),
                "top_10p_capture": top_capture_fraction(targets, scores, 0.10),
                "top_20p_capture": top_capture_fraction(targets, scores, 0.20),
            }
        )
    return out


def binary_feature_metrics(rows: list[ArchiveRow]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row.split == "test"]
    usable = [row for row in test_rows if row.majority_s_0_5 is not None]
    labels = [int(row.majority_s_0_5 or 0) for row in usable]
    prevalence = sum(labels) / len(labels) if labels else None
    out: list[dict[str, Any]] = []
    for feature in FEATURE_ORDER:
        scores = [row.feature_value(feature) for row in usable]
        out.append(
            {
                "feature": feature,
                "prevalence": prevalence,
                "roc_auc": roc_auc(labels, scores),
                "pr_auc": pr_auc(labels, scores),
                "positive_rate_top_10p": positive_rate_at_fraction(labels, scores, 0.10),
                "positive_rate_top_20p": positive_rate_at_fraction(labels, scores, 0.20),
            }
        )
    return out


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


def main() -> None:
    args = parse_args()
    regression_root = Path(args.regression_root)
    binary_root = Path(args.binary_root)
    out_dir = Path(args.out_dir)

    summary: dict[str, Any] = {"datasets": {}}
    regression_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []

    for dataset in DATASET_ORDER:
        regression_archive = read_archive_rows(archive_path(root=regression_root, dataset=dataset, target="regression"))
        binary_archive = read_archive_rows(archive_path(root=binary_root, dataset=dataset, target="binary"))

        reg_metrics = regression_feature_metrics(regression_archive)
        bin_metrics = binary_feature_metrics(binary_archive)
        length_bins = summarize_length_bins(binary_archive)

        summary["datasets"][dataset] = {
            "display_name": DISPLAY_NAMES[dataset],
            "regression_archive": str(archive_path(root=regression_root, dataset=dataset, target="regression")),
            "binary_archive": str(archive_path(root=binary_root, dataset=dataset, target="binary")),
            "regression_feature_metrics": reg_metrics,
            "binary_feature_metrics": bin_metrics,
            "binary_prompt_length_bins": length_bins,
        }

        for row in reg_metrics:
            regression_rows.append({"dataset": dataset, "dataset_name": DISPLAY_NAMES[dataset], **row})
        for row in bin_metrics:
            binary_rows.append({"dataset": dataset, "dataset_name": DISPLAY_NAMES[dataset], **row})
        for row in length_bins:
            length_rows.append({"dataset": dataset, "dataset_name": DISPLAY_NAMES[dataset], **row})

    write_json(out_dir / "metadata_correlation_summary.json", summary)
    write_csv(
        out_dir / "regression_feature_metrics.csv",
        ["dataset", "dataset_name", "feature", "spearman", "top_10p_capture", "top_20p_capture"],
        regression_rows,
    )
    write_csv(
        out_dir / "binary_feature_metrics.csv",
        [
            "dataset",
            "dataset_name",
            "feature",
            "prevalence",
            "roc_auc",
            "pr_auc",
            "positive_rate_top_10p",
            "positive_rate_top_20p",
        ],
        binary_rows,
    )
    write_csv(
        out_dir / "binary_prompt_length_bins.csv",
        [
            "dataset",
            "dataset_name",
            "bucket",
            "num_rows",
            "bucket_prompt_token_min",
            "bucket_prompt_token_max",
            "prompt_token_count_mean",
            "char_length_mean",
            "mean_relative_length_mean",
            "p_loop_mean",
            "p_cap_mean",
            "majority_s_0.5_rate",
        ],
        length_rows,
    )


if __name__ == "__main__":
    main()
