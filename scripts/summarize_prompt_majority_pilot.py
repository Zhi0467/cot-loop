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
    "date_tag",
    "status",
    "dataset_name",
    "task_kind",
    "target_name",
    "train_prompts",
    "test_prompts",
    "num_prompts",
    "projection_dir",
    "projection_ready",
    "baseline_prompt_length_direction",
    "baseline_prompt_length_pr_auc",
    "baseline_prompt_length_roc_auc",
    "baseline_prompt_length_macro_f1",
    "baseline_effective_budget_direction",
    "baseline_effective_budget_pr_auc",
    "baseline_effective_budget_roc_auc",
    "baseline_effective_budget_macro_f1",
    "last_layer_dir",
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


def _default_row(dataset: str, seed: int, date_tag: str) -> dict[str, object]:
    row = {key: "" for key in SUMMARY_COLUMNS}
    row.update(
        {
            "dataset": dataset,
            "seed": seed,
            "date_tag": date_tag,
            "projection_ready": False,
            "last_layer_ready": False,
            "ensemble_ready": False,
        }
    )
    return row


def _baseline_prefix(name: str) -> str | None:
    if name == "prompt_token_count":
        return "baseline_prompt_length"
    if name == "effective_max_tokens":
        return "baseline_effective_budget"
    return None


def _apply_projection(row: dict[str, object], path: Path, payload: dict[str, object]) -> None:
    row["projection_dir"] = str(path)
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
        test_metrics = info.get("test", {})
        if not isinstance(test_metrics, dict):
            continue
        row[f"{prefix}_pr_auc"] = test_metrics.get("pr_auc", "")
        row[f"{prefix}_roc_auc"] = test_metrics.get("roc_auc", "")
        row[f"{prefix}_macro_f1"] = test_metrics.get("macro_f1", "")


def _apply_probe(
    row: dict[str, object],
    arm: str,
    path: Path,
    payload: dict[str, object],
) -> None:
    prefix = "last_layer" if arm == "last_layer" else "ensemble"
    row[f"{prefix}_dir"] = str(path)
    row[f"{prefix}_ready"] = True
    row[f"{prefix}_epoch"] = payload.get("epoch", "")
    row[f"{prefix}_step"] = payload.get("step", "")
    if not row.get("target_name"):
        row["target_name"] = payload.get("target_name", "")
    for metric in PROBE_METRICS:
        row[f"{prefix}_{metric}"] = payload.get(f"eval_{metric}", "")


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
            str(row["date_tag"]),
        ),
    )


def main() -> None:
    args = _parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.is_dir():
        raise SystemExit(f"--root-dir does not exist or is not a directory: {root_dir}")

    rows: dict[tuple[str, int, str], dict[str, object]] = {}

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue

        projection_match = PROJECTION_RE.match(child.name)
        if projection_match is not None:
            date_tag = projection_match.group("date")
            if args.date_tag and date_tag != args.date_tag:
                continue
            dataset = projection_match.group("dataset")
            seed = int(projection_match.group("seed") or "0")
            summary_path = child / "export" / "projection_summary.json"
            key = (dataset, seed, date_tag)
            row = rows.setdefault(key, _default_row(dataset, seed, date_tag))
            if summary_path.exists():
                _apply_projection(row, child, _json_load(summary_path))
            continue

        probe_match = PROBE_RE.match(child.name)
        if probe_match is None:
            continue
        date_tag = probe_match.group("date")
        if args.date_tag and date_tag != args.date_tag:
            continue
        dataset = probe_match.group("dataset")
        seed = int(probe_match.group("seed"))
        arm = probe_match.group("arm")
        metrics_path = child / "best_metrics.json"
        key = (dataset, seed, date_tag)
        row = rows.setdefault(key, _default_row(dataset, seed, date_tag))
        if metrics_path.exists():
            _apply_probe(row, arm, child, _json_load(metrics_path))

    summary_rows = _sorted_rows(rows)
    payload = {"root_dir": str(root_dir), "rows": summary_rows}

    if args.out_json:
        _write_json(args.out_json, payload)
    if args.out_csv:
        _write_csv(args.out_csv, summary_rows)

    writer = csv.DictWriter(
        os.sys.stdout,
        fieldnames=(
            "dataset",
            "seed",
            "status",
            "train_prompts",
            "test_prompts",
            "baseline_prompt_length_pr_auc",
            "baseline_effective_budget_pr_auc",
            "last_layer_pr_auc",
            "last_layer_macro_f1",
            "ensemble_pr_auc",
            "ensemble_macro_f1",
        ),
    )
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({key: _coerce_scalar(row.get(key, "")) for key in writer.fieldnames})


if __name__ == "__main__":
    main()
