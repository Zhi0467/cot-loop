#!/usr/bin/env python3
"""Aggregate probe metrics across multiple seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics


BASE_SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "prevalence",
)
EXTRA_SUMMARY_METRICS = (
    "brier",
    "mse",
    "mae",
    "rmse",
    "spearman",
    "top_10p_capture",
    "top_20p_capture",
    "target_mean",
    "pred_mean",
)
DEFAULT_SELECTION = ("roc_auc", "macro_f1")
SELECTION_METRIC_CHOICES = BASE_SUMMARY_METRICS + EXTRA_SUMMARY_METRICS
LOWER_IS_BETTER_METRICS = {"brier", "mse", "mae", "rmse"}
NON_METRIC_FIELDS = {
    "run_dir",
    "seed",
    "epoch",
    "step",
    "lr",
    "train_loss",
    "classifier_layer",
    "resolved_classifier_layer",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default="roc_auc",
        help="Metric used to select best checkpoint row per run.",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=SELECTION_METRIC_CHOICES,
        default="macro_f1",
        help="Secondary metric used when selection metric ties.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional JSON output path for per-run best rows and aggregate stats.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path for aggregate mean/std.",
    )
    return parser.parse_args()


def _rank_value(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(v):
        return float("-inf")
    return v


def _metric_value(row: dict[str, object], metric: str) -> object:
    if metric in row:
        return row.get(metric)
    return row.get(f"eval_{metric}")


def _selection_rank_value(row: dict[str, object], metric: str) -> float:
    value = _rank_value(_metric_value(row, metric))
    if value == float("-inf"):
        return value
    if metric in LOWER_IS_BETTER_METRICS:
        return -value
    return value


def _as_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _infer_selection_pair(row: dict[str, object]) -> tuple[str, str] | None:
    selection_metric = row.get("selection_metric")
    tie_breaker = row.get("tie_breaker")
    if isinstance(selection_metric, str) and isinstance(tie_breaker, str):
        return selection_metric, tie_breaker

    selection_rule = row.get("selection_rule")
    if not isinstance(selection_rule, str):
        return None

    parts = re.findall(r"max\(([^)]+)\)", selection_rule)
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def _best_row_from_jsonl(
    path: str,
    *,
    selection_metric: str,
    tie_breaker: str,
) -> dict[str, object]:
    best_row: dict[str, object] | None = None
    best_key = (float("-inf"), float("-inf"))

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_num}") from exc

            rank_key = (
                _selection_rank_value(row, selection_metric),
                _selection_rank_value(row, tie_breaker),
            )
            if rank_key > best_key:
                best_key = rank_key
                best_row = row

    if best_row is None:
        raise SystemExit(f"No metric rows found in {path}")
    return best_row


def _infer_seed(run_dir: str, row: dict[str, object]) -> int | None:
    if "seed" in row:
        try:
            return int(row["seed"])
        except Exception:
            pass

    base = os.path.basename(os.path.normpath(run_dir))
    if base.startswith("seed_"):
        suffix = base[len("seed_") :]
        if suffix.lstrip("-").isdigit():
            return int(suffix)
    return None


def _has_metric_value(row: dict[str, object], metric: str) -> bool:
    value = _metric_value(row, metric)
    try:
        metric_float = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(metric_float)


def _iter_numeric_metric_names(row: dict[str, object]) -> list[str]:
    metric_names: list[str] = []
    seen: set[str] = set()
    for key, value in row.items():
        if key.startswith("train_"):
            continue
        name = key[len("eval_") :] if key.startswith("eval_") else key
        if name in NON_METRIC_FIELDS or name in seen:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        metric_names.append(name)
        seen.add(name)
    return metric_names


def _summary_metrics(rows: list[dict[str, object]]) -> list[str]:
    metrics: list[str] = []
    seen: set[str] = set()
    for metric in BASE_SUMMARY_METRICS + EXTRA_SUMMARY_METRICS:
        if any(_has_metric_value(row, metric) for row in rows):
            metrics.append(metric)
            seen.add(metric)
    for row in rows:
        for metric in _iter_numeric_metric_names(row):
            if metric in seen:
                continue
            metrics.append(metric)
            seen.add(metric)
    return metrics


def _format_row(
    run_dir: str,
    row: dict[str, object],
    *,
    summary_metrics: list[str],
) -> dict[str, object]:
    payload = {
        "run_dir": run_dir,
        "seed": _infer_seed(run_dir, row),
        "epoch": row.get("epoch"),
        "step": row.get("step"),
    }
    for metric in summary_metrics:
        payload[metric] = _as_float_or_nan(_metric_value(row, metric))
    return payload


def _supports_selection_pair(
    row: dict[str, object],
    *,
    selection_metric: str,
    tie_breaker: str,
) -> bool:
    return _has_metric_value(row, selection_metric) and _has_metric_value(row, tie_breaker)


def _load_best_row(
    run_dir: str,
    *,
    selection_metric: str,
    tie_breaker: str,
) -> dict[str, object]:
    best_metrics_path = os.path.join(run_dir, "best_metrics.json")
    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    best_row: dict[str, object] | None = None
    best_selection: tuple[str, str] | None = None

    if os.path.exists(best_metrics_path):
        with open(best_metrics_path, "r", encoding="utf-8") as f:
            best_row = json.load(f)
        best_selection = _infer_selection_pair(best_row)

    if os.path.exists(metrics_jsonl):
        if best_row is not None:
            if best_selection == (selection_metric, tie_breaker):
                return best_row
            if not _supports_selection_pair(
                best_row,
                selection_metric=selection_metric,
                tie_breaker=tie_breaker,
            ):
                return best_row
        return _best_row_from_jsonl(
            metrics_jsonl,
            selection_metric=selection_metric,
            tie_breaker=tie_breaker,
        )

    if best_row is not None:
        return best_row

    raise SystemExit(f"Missing both best_metrics.json and metrics.jsonl under {run_dir}")


def _aggregate(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    vals = [
        float(row[metric])
        for row in rows
        if not math.isnan(float(row[metric]))
    ]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _sanitize_json(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def _write_summary_csv(
    path: str,
    summary: dict[str, dict[str, object]],
    *,
    summary_metrics: list[str],
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n"])
        writer.writeheader()
        for metric in summary_metrics:
            stats = summary[metric]
            writer.writerow(
                {
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n": stats["n"],
                }
            )


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def main() -> None:
    args = _parse_args()

    raw_rows = [
        _load_best_row(
            run_dir,
            selection_metric=args.selection_metric,
            tie_breaker=args.tie_breaker,
        )
        for run_dir in args.run_dirs
    ]
    summary_metrics = _summary_metrics(raw_rows)
    rows = [
        _format_row(
            run_dir,
            row,
            summary_metrics=summary_metrics,
        )
        for run_dir, row in zip(args.run_dirs, raw_rows, strict=True)
    ]

    summary = {
        metric: _aggregate(rows, metric)
        for metric in summary_metrics
    }
    payload = {
        "selection_metric": args.selection_metric,
        "tie_breaker": args.tie_breaker,
        "num_runs": len(rows),
        "summary_metrics": summary_metrics,
        "runs": rows,
        "aggregate": summary,
    }

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(_sanitize_json(payload), f, indent=2, sort_keys=True)
            f.write("\n")

    if args.out_csv:
        _write_summary_csv(
            args.out_csv,
            summary,
            summary_metrics=summary_metrics,
        )

    print(
        f"Aggregated {len(rows)} run(s) using selection={args.selection_metric} "
        f"tie_breaker={args.tie_breaker}",
        flush=True,
    )
    for metric in summary_metrics:
        stats = summary[metric]
        mean_val = float(stats["mean"])
        std_val = float(stats["std"])
        n = int(stats["n"])
        print(
            f"{metric}: mean={_format_metric(mean_val)} std={_format_metric(std_val)} n={n}",
            flush=True,
        )


if __name__ == "__main__":
    main()
