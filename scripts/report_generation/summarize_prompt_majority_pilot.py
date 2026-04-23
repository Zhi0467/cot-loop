#!/usr/bin/env python3
"""Summarize prompt-majority pilot outputs across datasets and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path


PROJECTION_RE = re.compile(
    r"^prompt_profile_projection_(?P<dataset>.+?)_(?P<head>majority\d+)"
    r"(?:_seed(?P<seed>\d+))?_(?P<date>\d{8})$"
)
PROBE_RE = re.compile(
    r"^prompt_majority\d+_(?P<dataset>.+?)_(?P<arm>last_layer|ensemble)"
    r"_seed(?P<seed>\d+)_(?P<date>\d{8})$"
)
DATASET_ORDER = {
    "gpqa": 0,
    "aime": 1,
    "math500": 2,
    "mmlu": 3,
    "livecodebench": 4,
}
DATASET_ALIASES = {
    "gpqa": "gpqa",
    "aime": "aime",
    "math": "math500",
    "math500": "math500",
    "mmlu": "mmlu",
    "livecodebench": "livecodebench",
}
PROBE_METRICS = (
    "accuracy",
    "macro_f1",
    "positive_f1",
    "positive_precision",
    "positive_recall",
    "pr_auc",
    "prevalence",
    "roc_auc",
)
SUMMARY_COLUMNS = (
    "dataset",
    "seed",
    "status",
    "dataset_name",
    "task_kind",
    "target_name",
    "train_prompts",
    "test_prompts",
    "num_prompts",
    "projection_dir",
    "projection_date_tag",
    "projection_ready",
    "baseline_prompt_length_direction",
    "baseline_prompt_length_threshold",
    "baseline_prompt_length_accuracy",
    "baseline_prompt_length_pr_auc",
    "baseline_prompt_length_roc_auc",
    "baseline_prompt_length_macro_f1",
    "baseline_prompt_length_positive_f1",
    "baseline_prompt_length_positive_precision",
    "baseline_prompt_length_positive_recall",
    "baseline_prompt_length_prevalence",
    "baseline_effective_budget_direction",
    "baseline_effective_budget_threshold",
    "baseline_effective_budget_accuracy",
    "baseline_effective_budget_pr_auc",
    "baseline_effective_budget_roc_auc",
    "baseline_effective_budget_macro_f1",
    "baseline_effective_budget_positive_f1",
    "baseline_effective_budget_positive_precision",
    "baseline_effective_budget_positive_recall",
    "baseline_effective_budget_prevalence",
    "last_layer_dir",
    "last_layer_date_tag",
    "last_layer_selection_kind",
    "last_layer_selection_rule",
    "last_layer_ready",
    "last_layer_epoch",
    "last_layer_step",
    "last_layer_accuracy",
    "last_layer_macro_f1",
    "last_layer_positive_f1",
    "last_layer_positive_precision",
    "last_layer_positive_recall",
    "last_layer_pr_auc",
    "last_layer_prevalence",
    "last_layer_roc_auc",
    "ensemble_dir",
    "ensemble_date_tag",
    "ensemble_selection_kind",
    "ensemble_selection_rule",
    "ensemble_ready",
    "ensemble_epoch",
    "ensemble_step",
    "ensemble_accuracy",
    "ensemble_macro_f1",
    "ensemble_positive_f1",
    "ensemble_positive_precision",
    "ensemble_positive_recall",
    "ensemble_pr_auc",
    "ensemble_prevalence",
    "ensemble_roc_auc",
)
AGGREGATE_COLUMNS = (
    "dataset",
    "num_seeds",
    "seed_list",
    "projection_dates",
    "last_layer_dates",
    "ensemble_dates",
    "train_prompts_mean",
    "test_prompts_mean",
    "num_prompts_mean",
    "baseline_prompt_length_accuracy_mean",
    "baseline_prompt_length_accuracy_std",
    "baseline_prompt_length_pr_auc_mean",
    "baseline_prompt_length_pr_auc_std",
    "baseline_prompt_length_roc_auc_mean",
    "baseline_prompt_length_roc_auc_std",
    "baseline_prompt_length_macro_f1_mean",
    "baseline_prompt_length_macro_f1_std",
    "baseline_prompt_length_positive_precision_mean",
    "baseline_prompt_length_positive_precision_std",
    "baseline_prompt_length_positive_recall_mean",
    "baseline_prompt_length_positive_recall_std",
    "baseline_prompt_length_prevalence_mean",
    "baseline_prompt_length_prevalence_std",
    "last_layer_accuracy_mean",
    "last_layer_accuracy_std",
    "last_layer_pr_auc_mean",
    "last_layer_pr_auc_std",
    "last_layer_roc_auc_mean",
    "last_layer_roc_auc_std",
    "last_layer_macro_f1_mean",
    "last_layer_macro_f1_std",
    "last_layer_positive_precision_mean",
    "last_layer_positive_precision_std",
    "last_layer_positive_recall_mean",
    "last_layer_positive_recall_std",
    "last_layer_prevalence_mean",
    "last_layer_prevalence_std",
    "ensemble_accuracy_mean",
    "ensemble_accuracy_std",
    "ensemble_pr_auc_mean",
    "ensemble_pr_auc_std",
    "ensemble_roc_auc_mean",
    "ensemble_roc_auc_std",
    "ensemble_macro_f1_mean",
    "ensemble_macro_f1_std",
    "ensemble_positive_precision_mean",
    "ensemble_positive_precision_std",
    "ensemble_positive_recall_mean",
    "ensemble_positive_recall_std",
    "ensemble_prevalence_mean",
    "ensemble_prevalence_std",
)
AGGREGATE_METRICS = (
    "train_prompts",
    "test_prompts",
    "num_prompts",
    "baseline_prompt_length_accuracy",
    "baseline_prompt_length_pr_auc",
    "baseline_prompt_length_roc_auc",
    "baseline_prompt_length_macro_f1",
    "baseline_prompt_length_positive_precision",
    "baseline_prompt_length_positive_recall",
    "baseline_prompt_length_prevalence",
    "last_layer_accuracy",
    "last_layer_pr_auc",
    "last_layer_roc_auc",
    "last_layer_macro_f1",
    "last_layer_positive_precision",
    "last_layer_positive_recall",
    "last_layer_prevalence",
    "ensemble_accuracy",
    "ensemble_pr_auc",
    "ensemble_roc_auc",
    "ensemble_macro_f1",
    "ensemble_positive_precision",
    "ensemble_positive_recall",
    "ensemble_prevalence",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root output directory containing prompt-majority projection/probe folders.",
    )
    parser.add_argument(
        "--date-tag",
        default="",
        help="Optional YYYYMMDD filter (for example 20260322).",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV summary path.",
    )
    parser.add_argument(
        "--out-aggregate-json",
        default="",
        help="Optional dataset-aggregated JSON summary path.",
    )
    parser.add_argument(
        "--out-aggregate-csv",
        default="",
        help="Optional dataset-aggregated CSV summary path.",
    )
    parser.add_argument(
        "--prefer-best-rank",
        action="store_true",
        help="Use best_rank_metrics.json when present; otherwise fall back to best_metrics.json.",
    )
    return parser.parse_args()


def _json_load(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_scalar(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write_json(path: str, payload: object) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _coerce_scalar(row.get(key, "")) for key in SUMMARY_COLUMNS})


def _write_aggregate_csv(path: str, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _coerce_scalar(row.get(key, "")) for key in AGGREGATE_COLUMNS})


def _default_row(dataset: str, seed: int) -> dict[str, object]:
    row = {key: "" for key in SUMMARY_COLUMNS}
    row.update(
        {
            "dataset": dataset,
            "seed": seed,
            "projection_ready": False,
            "last_layer_ready": False,
            "ensemble_ready": False,
        }
    )
    return row


def _canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _baseline_prefix(name: str) -> str | None:
    if name == "prompt_token_count":
        return "baseline_prompt_length"
    if name == "effective_max_tokens":
        return "baseline_effective_budget"
    return None


def _apply_metric_bundle(row: dict[str, object], prefix: str, metrics: dict[str, object]) -> None:
    for metric in PROBE_METRICS:
        row[f"{prefix}_{metric}"] = metrics.get(metric, "")


def _apply_projection(
    row: dict[str, object],
    path: Path,
    payload: dict[str, object],
    date_tag: str,
) -> None:
    row["projection_dir"] = str(path)
    row["projection_date_tag"] = date_tag
    row["projection_ready"] = True
    row["dataset_name"] = payload.get("dataset_name", "")
    row["task_kind"] = payload.get("task_kind", "")
    row["num_prompts"] = payload.get("num_prompts", "")

    split_counts = payload.get("split_counts", {})
    if isinstance(split_counts, dict):
        prompt_counts = split_counts.get("prompts", {})
        if isinstance(prompt_counts, dict):
            row["train_prompts"] = prompt_counts.get("train", "")
            row["test_prompts"] = prompt_counts.get("test", "")

    leakage = payload.get("leakage_baselines", {})
    if not isinstance(leakage, dict):
        return
    for name, info in leakage.items():
        prefix = _baseline_prefix(str(name))
        if prefix is None or not isinstance(info, dict):
            continue
        row[f"{prefix}_direction"] = info.get(
            "direction",
            "constant" if "constant_prediction" in info else "",
        )
        row[f"{prefix}_threshold"] = info.get("threshold", "")
        test_metrics = info.get("test", {})
        if not isinstance(test_metrics, dict):
            continue
        _apply_metric_bundle(row, prefix, test_metrics)


def _apply_probe(
    row: dict[str, object],
    arm: str,
    path: Path,
    payload: dict[str, object],
    date_tag: str,
) -> None:
    prefix = "last_layer" if arm == "last_layer" else "ensemble"
    row[f"{prefix}_dir"] = str(path)
    row[f"{prefix}_date_tag"] = date_tag
    row[f"{prefix}_selection_kind"] = payload.get("selection_kind", "best_metrics")
    row[f"{prefix}_selection_rule"] = payload.get("selection_rule", "")
    row[f"{prefix}_ready"] = True
    row[f"{prefix}_epoch"] = payload.get("epoch", "")
    row[f"{prefix}_step"] = payload.get("step", "")
    if not row.get("target_name"):
        row["target_name"] = payload.get("target_name", "")
    _apply_metric_bundle(
        row,
        prefix,
        {metric: payload.get(f"eval_{metric}", "") for metric in PROBE_METRICS},
    )


def _status(row: dict[str, object]) -> str:
    ready = (
        bool(row.get("projection_ready")),
        bool(row.get("last_layer_ready")),
        bool(row.get("ensemble_ready")),
    )
    if ready == (True, True, True):
        return "complete"
    if ready[0] and any(ready[1:]):
        return "partial_probe"
    if ready[0]:
        return "projection_only"
    if any(ready[1:]):
        return "probe_only"
    return "missing"


def _sorted_rows(rows: dict[tuple[str, int, str], dict[str, object]]) -> list[dict[str, object]]:
    items = list(rows.values())
    for row in items:
        row["status"] = _status(row)
    return sorted(
        items,
        key=lambda row: (
            DATASET_ORDER.get(str(row["dataset"]), 999),
            str(row["dataset"]),
            int(row["seed"]),
        ),
    )


def _latest_date(previous: object, current: str) -> bool:
    if not previous:
        return True
    return str(previous) <= current


def _numeric_value(value: object) -> float | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean_value = _mean(values)
    assert mean_value is not None
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for dataset, dataset_rows in sorted(
        grouped.items(),
        key=lambda item: DATASET_ORDER.get(item[0], 999),
    ):
        aggregate: dict[str, object] = {key: "" for key in AGGREGATE_COLUMNS}
        aggregate["dataset"] = dataset
        aggregate["num_seeds"] = len(dataset_rows)
        aggregate["seed_list"] = ",".join(str(int(row["seed"])) for row in dataset_rows)
        aggregate["projection_dates"] = ",".join(
            sorted({str(row["projection_date_tag"]) for row in dataset_rows if row["projection_date_tag"]})
        )
        aggregate["last_layer_dates"] = ",".join(
            sorted({str(row["last_layer_date_tag"]) for row in dataset_rows if row["last_layer_date_tag"]})
        )
        aggregate["ensemble_dates"] = ",".join(
            sorted({str(row["ensemble_date_tag"]) for row in dataset_rows if row["ensemble_date_tag"]})
        )
        for metric in AGGREGATE_METRICS:
            values = [
                numeric
                for row in dataset_rows
                if (numeric := _numeric_value(row.get(metric))) is not None
            ]
            aggregate[f"{metric}_mean"] = _mean(values)
            aggregate[f"{metric}_std"] = _std(values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def main() -> None:
    args = _parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.is_dir():
        raise SystemExit(f"--root-dir does not exist or is not a directory: {root_dir}")

    rows: dict[tuple[str, int], dict[str, object]] = {}

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue

        projection_match = PROJECTION_RE.match(child.name)
        if projection_match is not None:
            date_tag = projection_match.group("date")
            if args.date_tag and date_tag != args.date_tag:
                continue
            dataset = _canonical_dataset_name(projection_match.group("dataset"))
            seed = int(projection_match.group("seed") or "0")
            summary_path = child / "export" / "projection_summary.json"
            key = (dataset, seed)
            row = rows.setdefault(key, _default_row(dataset, seed))
            if summary_path.exists() and _latest_date(row.get("projection_date_tag"), date_tag):
                _apply_projection(row, child, _json_load(summary_path), date_tag)
            continue

        probe_match = PROBE_RE.match(child.name)
        if probe_match is None:
            continue
        date_tag = probe_match.group("date")
        if args.date_tag and date_tag != args.date_tag:
            continue
        dataset = _canonical_dataset_name(probe_match.group("dataset"))
        seed = int(probe_match.group("seed"))
        arm = probe_match.group("arm")
        metrics_path = child / "best_metrics.json"
        if args.prefer_best_rank and (child / "best_rank_metrics.json").exists():
            metrics_path = child / "best_rank_metrics.json"
        key = (dataset, seed)
        row = rows.setdefault(key, _default_row(dataset, seed))
        current_date = row.get(f"{'last_layer' if arm == 'last_layer' else 'ensemble'}_date_tag")
        if metrics_path.exists() and _latest_date(current_date, date_tag):
            _apply_probe(row, arm, child, _json_load(metrics_path), date_tag)

    summary_rows = _sorted_rows(rows)
    aggregate_rows = _aggregate_rows(summary_rows)
    payload = {
        "root_dir": str(root_dir),
        "prefer_best_rank": args.prefer_best_rank,
        "rows": summary_rows,
        "aggregate_rows": aggregate_rows,
    }

    if args.out_json:
        _write_json(args.out_json, payload)
    if args.out_csv:
        _write_csv(args.out_csv, summary_rows)
    if args.out_aggregate_json:
        _write_json(
            args.out_aggregate_json,
            {
                "root_dir": str(root_dir),
                "prefer_best_rank": args.prefer_best_rank,
                "aggregate_rows": aggregate_rows,
            },
        )
    if args.out_aggregate_csv:
        _write_aggregate_csv(args.out_aggregate_csv, aggregate_rows)

    writer = csv.DictWriter(
        os.sys.stdout,
        fieldnames=(
            "dataset",
            "seed",
            "status",
            "train_prompts",
            "test_prompts",
            "baseline_prompt_length_pr_auc",
            "baseline_prompt_length_positive_precision",
            "baseline_prompt_length_positive_recall",
            "baseline_effective_budget_pr_auc",
            "last_layer_pr_auc",
            "last_layer_positive_precision",
            "last_layer_positive_recall",
            "last_layer_macro_f1",
            "ensemble_pr_auc",
            "ensemble_positive_precision",
            "ensemble_positive_recall",
            "ensemble_macro_f1",
        ),
    )
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({key: _coerce_scalar(row.get(key, "")) for key in writer.fieldnames})


if __name__ == "__main__":
    main()
