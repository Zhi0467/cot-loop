#!/usr/bin/env python3
"""Summarize the locked prompt-profile full-train run and its metadata controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_OUT_ROOT = ROOT / "outputs" / "full_train"
DATASET_ORDER = ("gpqa", "aime", "math500", "mmlu_pro", "livecodebench")
REGRESSION_METRICS = (
    "mse",
    "mae",
    "rmse",
    "target_mean",
    "pred_mean",
    "spearman",
    "top_10p_capture",
    "top_20p_capture",
)
BINARY_METRICS = (
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "prevalence",
    "roc_auc",
    "pr_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Defaults to <out-root>/summary when omitted.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        help="Dataset key(s) to summarize. Defaults to all known datasets.",
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


def manifest_path(data_dir: Path) -> Path:
    return data_dir / "manifest.json"


def feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for name in feature_names:
            if name == "prompt_token_count":
                values.append(float(row["prompt_token_count"]))
            elif name == "effective_max_tokens":
                values.append(float(row["effective_max_tokens"]))
            else:
                raise SystemExit(f"Unsupported metadata feature '{name}'.")
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def safe_spearman(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(scores, scores[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        return float("nan")
    corr = spearmanr(y_true, scores).correlation
    if corr is None or not math.isfinite(float(corr)):
        return float("nan")
    return float(corr)


def top_capture_fraction(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fraction: float,
) -> float:
    if y_true.size == 0:
        return float("nan")
    total_mass = float(np.sum(y_true))
    if total_mass <= 0.0:
        return float("nan")
    keep = max(1, int(math.ceil(float(y_true.size) * fraction)))
    order = np.argsort(-scores, kind="stable")
    captured = float(np.sum(y_true[order[:keep]]))
    return captured / total_mass


def regression_metrics_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    errors = scores - y_true
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(scores)),
        "spearman": safe_spearman(y_true, scores),
        "top_10p_capture": top_capture_fraction(y_true, scores, fraction=0.10),
        "top_20p_capture": top_capture_fraction(y_true, scores, fraction=0.20),
    }


def binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives < 1 or negatives < 1:
        return None
    if np.isnan(scores).any():
        return None
    sorted_order = np.argsort(scores.astype(float), kind="mergesort")
    sorted_scores = scores[sorted_order].astype(float)
    sorted_true = y_true[sorted_order].astype(int)
    positive_ranks = 0.0
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = float((start + 1 + stop) / 2.0)
        positive_ranks += float(np.sum(sorted_true[start:stop] == 1)) * average_rank
        start = stop
    numerator = positive_ranks - (positives * (positives + 1) / 2.0)
    denominator = float(positives * negatives)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def binary_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    if positives < 1:
        return None
    if np.isnan(scores).any():
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order].astype(float)
    ranked_true = y_true[order].astype(int)
    tp = 0
    fp = 0
    average_precision = 0.0
    previous_recall = 0.0
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        group_true = ranked_true[start:stop]
        tp += int(np.sum(group_true == 1))
        fp += int(group_true.size - np.sum(group_true == 1))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(average_precision)


def binary_threshold_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y_true_i = y_true.astype(int)
    pred_i = predictions.astype(int)
    tp = int(np.sum((y_true_i == 1) & (pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (pred_i == 0)))
    fp = int(np.sum((y_true_i == 0) & (pred_i == 1)))
    fn = int(np.sum((y_true_i == 1) & (pred_i == 0)))
    acc = float(np.mean(y_true_i == pred_i))

    def safe_div(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    pos_precision = safe_div(tp, tp + fp)
    pos_recall = safe_div(tp, tp + fn)
    pos_f1 = (
        0.0
        if (pos_precision + pos_recall) <= 0.0
        else float(2.0 * pos_precision * pos_recall / (pos_precision + pos_recall))
    )
    neg_precision = safe_div(tn, tn + fn)
    neg_recall = safe_div(tn, tn + fp)
    neg_f1 = (
        0.0
        if (neg_precision + neg_recall) <= 0.0
        else float(2.0 * neg_precision * neg_recall / (neg_precision + neg_recall))
    )
    macro_f1 = float((pos_f1 + neg_f1) / 2.0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "positive_precision": float(pos_precision),
        "positive_recall": float(pos_recall),
        "positive_f1": float(pos_f1),
        "prevalence": float(np.mean(y_true_i == 1)),
    }


def select_orientation_and_threshold(
    y_train: np.ndarray,
    raw_scores: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    orientations = [
        ("higher_is_more_positive", raw_scores.astype(float)),
        ("lower_is_more_positive", (-raw_scores).astype(float)),
    ]
    best_orientation: tuple[np.ndarray, str] | None = None
    best_key: tuple[float, float] | None = None
    for direction, scores in orientations:
        pr_auc = binary_pr_auc(y_train, scores)
        roc_auc = binary_roc_auc(y_train, scores)
        key = (
            -math.inf if pr_auc is None else float(pr_auc),
            -math.inf if roc_auc is None else float(roc_auc),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_orientation = (scores, direction)
    if best_orientation is None:
        raise RuntimeError("Failed to choose a binary metadata orientation.")

    oriented_scores, direction = best_orientation
    thresholds = [float(value) for value in np.unique(oriented_scores).tolist()]
    thresholds.append(float(np.max(oriented_scores)) + 1.0)
    best_threshold = thresholds[0]
    best_threshold_key: tuple[float, float, float] | None = None
    for threshold in thresholds:
        predictions = (oriented_scores >= threshold).astype(int)
        metrics = binary_threshold_metrics(y_train, predictions)
        key = (
            float(metrics["macro_f1"]),
            float(metrics["positive_f1"]),
            float(metrics["accuracy"]),
        )
        if best_threshold_key is None or key > best_threshold_key:
            best_threshold_key = key
            best_threshold = threshold
    return oriented_scores, direction, best_threshold


def summarize_binary_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_name: str,
) -> dict[str, Any]:
    y_train = np.asarray([int(row["target_value"]) for row in train_rows], dtype=int)
    y_test = np.asarray([int(row["target_value"]) for row in test_rows], dtype=int)
    train_values = np.asarray([float(row[feature_name]) for row in train_rows], dtype=np.float64)
    test_values = np.asarray([float(row[feature_name]) for row in test_rows], dtype=np.float64)

    summary: dict[str, Any] = {
        "feature_names": [feature_name],
        "train_unique_values": int(len(np.unique(train_values))),
        "test_unique_values": int(len(np.unique(test_values))),
        "selection_rule": (
            "orientation=max(train_pr_auc), tie_break=max(train_roc_auc); "
            "threshold=max(train_macro_f1), tie_break=max(train_positive_f1), max(train_accuracy)"
        ),
    }
    if len(np.unique(train_values)) <= 1:
        constant_candidates = []
        for prediction_value in (0, 1):
            predictions = np.full_like(y_train, prediction_value)
            metrics = binary_threshold_metrics(y_train, predictions)
            key = (
                float(metrics["macro_f1"]),
                float(metrics["positive_f1"]),
                float(metrics["accuracy"]),
            )
            constant_candidates.append((key, prediction_value, metrics))
        _best_key, prediction_value, train_metrics = max(constant_candidates, key=lambda item: item[0])
        test_predictions = np.full_like(y_test, prediction_value)
        summary["constant_prediction"] = int(prediction_value)
        summary["train"] = {**train_metrics, "roc_auc": None, "pr_auc": None}
        summary["test"] = {
            **binary_threshold_metrics(y_test, test_predictions),
            "roc_auc": None,
            "pr_auc": None,
        }
        return summary

    train_scores, direction, threshold = select_orientation_and_threshold(y_train, train_values)
    test_scores = test_values if direction == "higher_is_more_positive" else (-test_values)
    train_predictions = (train_scores >= threshold).astype(int)
    test_predictions = (test_scores >= threshold).astype(int)
    summary["direction"] = direction
    summary["threshold"] = float(threshold)
    summary["train"] = {
        **binary_threshold_metrics(y_train, train_predictions),
        "roc_auc": binary_roc_auc(y_train, train_scores),
        "pr_auc": binary_pr_auc(y_train, train_scores),
    }
    summary["test"] = {
        **binary_threshold_metrics(y_test, test_predictions),
        "roc_auc": binary_roc_auc(y_test, test_scores),
        "pr_auc": binary_pr_auc(y_test, test_scores),
    }
    return summary


def summarize_regression_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    x_train = feature_matrix(train_rows, feature_names)
    x_test = feature_matrix(test_rows, feature_names)
    y_train = np.asarray([float(row["target_value"]) for row in train_rows], dtype=np.float64)
    y_test = np.asarray([float(row["target_value"]) for row in test_rows], dtype=np.float64)

    scaler_mean = np.mean(x_train, axis=0)
    scaler_scale = np.std(x_train, axis=0)
    scaler_scale = np.where(scaler_scale <= 0.0, 1.0, scaler_scale)
    x_train_scaled = (x_train - scaler_mean) / scaler_scale
    x_test_scaled = (x_test - scaler_mean) / scaler_scale
    design_train = np.concatenate(
        [np.ones((x_train_scaled.shape[0], 1), dtype=np.float64), x_train_scaled],
        axis=1,
    )
    design_test = np.concatenate(
        [np.ones((x_test_scaled.shape[0], 1), dtype=np.float64), x_test_scaled],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    train_predictions = np.clip(design_train @ coef, 0.0, 1.0)
    test_predictions = np.clip(design_test @ coef, 0.0, 1.0)

    train_metrics = regression_metrics_from_scores(y_train, train_predictions)
    test_metrics = regression_metrics_from_scores(y_test, test_predictions)
    return {
        "feature_names": list(feature_names),
        "model_payload": {
            "scaler_mean": scaler_mean.tolist(),
            "scaler_scale": scaler_scale.tolist(),
            "coef": coef[1:].tolist(),
            "intercept": float(coef[0]),
        },
        "train": train_metrics,
        "test": test_metrics,
    }


def load_prompt_profile_rows(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_file = data_dir / "manifest.json"
    if not manifest_file.exists():
        raise SystemExit(f"Dataset manifest is missing: {data_dir}")
    manifest = read_json(manifest_file)
    prompt_profile_files = manifest.get("prompt_profile_files")
    if not isinstance(prompt_profile_files, dict):
        raise SystemExit(f"Dataset is missing prompt_profile_files: {data_dir}")
    train_rel = prompt_profile_files.get("train")
    test_rel = prompt_profile_files.get("test")
    if not isinstance(train_rel, str) or not isinstance(test_rel, str):
        raise SystemExit(f"Dataset prompt_profile_files is malformed: {data_dir}")
    return read_jsonl(data_dir / train_rel), read_jsonl(data_dir / test_rel)


def aggregate_metric_payloads(
    run_dir: Path,
    *,
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    if not run_dir.exists():
        return {"status": "missing", "run_dir": str(run_dir)}

    seed_dirs = sorted(
        child for child in run_dir.iterdir() if child.is_dir() and child.name.startswith("seed_")
    )
    if not seed_dirs:
        seed_dirs = [run_dir]

    seed_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed_dir in seed_dirs:
        best_loss = seed_dir / "best_loss_metrics.json"
        best_rank = seed_dir / "best_rank_metrics.json"
        if not best_loss.exists() or not best_rank.exists():
            missing.append(str(seed_dir))
            continue
        best_loss_payload = read_json(best_loss)
        best_rank_payload = read_json(best_rank)
        seed_rows.append(
            {
                "seed_dir": str(seed_dir),
                "best_loss": {name: best_loss_payload.get(name) for name in metric_names},
                "best_rank": {name: best_rank_payload.get(name) for name in metric_names},
            }
        )

    if not seed_rows:
        return {
            "status": "missing",
            "run_dir": str(run_dir),
            "missing_seed_dirs": missing,
        }

    def summarize(selection_name: str) -> dict[str, Any]:
        aggregate: dict[str, Any] = {}
        for metric_name in metric_names:
            values = [
                float(row[selection_name][metric_name])
                for row in seed_rows
                if isinstance(row[selection_name].get(metric_name), (int, float))
                and not isinstance(row[selection_name].get(metric_name), bool)
                and not math.isnan(float(row[selection_name][metric_name]))
            ]
            if not values:
                continue
            aggregate[metric_name] = {
                "mean": float(statistics.fmean(values)),
                "std": (float(statistics.stdev(values)) if len(values) > 1 else None),
                "count": len(values),
            }
        return aggregate

    return {
        "status": "complete" if not missing else "partial",
        "run_dir": str(run_dir),
        "seed_count": len(seed_rows),
        "missing_seed_dirs": missing,
        "per_seed": seed_rows,
        "aggregate": {
            "best_loss": summarize("best_loss"),
            "best_rank": summarize("best_rank"),
        },
    }


def flatten_probe_rows(
    *,
    dataset: str,
    target_name: str,
    view_name: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = payload.get("aggregate", {})
    for selection_kind, metrics in aggregate.items():
        row: dict[str, Any] = {
            "dataset": dataset,
            "target_name": target_name,
            "source_type": "probe",
            "source_name": view_name,
            "selection_kind": selection_kind,
            "status": payload.get("status"),
        }
        for metric_name, summary in metrics.items():
            if not isinstance(summary, dict):
                continue
            row[f"{metric_name}_mean"] = summary.get("mean")
            row[f"{metric_name}_std"] = summary.get("std")
        rows.append(row)
    if not rows:
        rows.append(
            {
                "dataset": dataset,
                "target_name": target_name,
                "source_type": "probe",
                "source_name": view_name,
                "selection_kind": "",
                "status": payload.get("status", "missing"),
            }
        )
    return rows


def flatten_metadata_row(
    *,
    dataset: str,
    target_name: str,
    source_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "target_name": target_name,
        "source_type": "metadata_baseline",
        "source_name": source_name,
        "selection_kind": "",
        "status": "complete",
    }
    test_metrics = payload.get("test", {})
    if isinstance(test_metrics, dict):
        for metric_name, value in test_metrics.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isnan(float(value))
            ):
                row[f"{metric_name}_mean"] = float(value)
                row[f"{metric_name}_std"] = None
    if "direction" in payload:
        row["direction"] = payload["direction"]
    if "threshold" in payload:
        row["threshold"] = payload["threshold"]
    return row


def component_status(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "missing"
    status = payload.get("status")
    return status if isinstance(status, str) else "missing"


def dataset_summary(dataset_key: str, out_root: Path) -> dict[str, Any]:
    dataset_root = out_root / dataset_key
    shared_archive = dataset_root / "shared_archive"
    binary_data = dataset_root / "majority_s_0.5" / "data"

    summary: dict[str, Any] = {
        "dataset": dataset_key,
        "dataset_root": str(dataset_root),
        "shared_archive": str(shared_archive),
        "binary_data_dir": str(binary_data),
        "status": "missing",
    }

    if not manifest_path(shared_archive).exists():
        summary["status"] = "missing_shared_archive"
        return summary

    train_reg_rows, test_reg_rows = load_prompt_profile_rows(shared_archive)
    regression_baselines = {
        "prompt_length": summarize_regression_baseline(
            train_reg_rows,
            test_reg_rows,
            feature_names=("prompt_token_count",),
        ),
        "effective_budget": summarize_regression_baseline(
            train_reg_rows,
            test_reg_rows,
            feature_names=("effective_max_tokens",),
        ),
        "prompt_length_plus_effective_budget": summarize_regression_baseline(
            train_reg_rows,
            test_reg_rows,
            feature_names=("prompt_token_count", "effective_max_tokens"),
        ),
    }

    regression_views = {
        "ensemble": aggregate_metric_payloads(
            dataset_root / "mean_relative_length" / "ensemble",
            metric_names=REGRESSION_METRICS,
        ),
        "last_layer": aggregate_metric_payloads(
            dataset_root / "mean_relative_length" / "last_layer",
            metric_names=REGRESSION_METRICS,
        ),
    }

    regression_status = "complete"
    if any(component_status(payload) != "complete" for payload in regression_views.values()):
        regression_status = "partial"

    binary_status = "missing"
    binary_baselines: dict[str, Any] | None = None
    binary_views: dict[str, Any] | None = None
    if manifest_path(binary_data).exists():
        train_bin_rows, test_bin_rows = load_prompt_profile_rows(binary_data)
        binary_baselines = {
            "prompt_length": summarize_binary_baseline(
                train_bin_rows,
                test_bin_rows,
                feature_name="prompt_token_count",
            ),
            "effective_budget": summarize_binary_baseline(
                train_bin_rows,
                test_bin_rows,
                feature_name="effective_max_tokens",
            ),
        }
        binary_views = {
            "ensemble": aggregate_metric_payloads(
                dataset_root / "majority_s_0.5" / "ensemble",
                metric_names=BINARY_METRICS,
            ),
            "last_layer": aggregate_metric_payloads(
                dataset_root / "majority_s_0.5" / "last_layer",
                metric_names=BINARY_METRICS,
            ),
        }
        binary_status = "complete"
        if any(component_status(payload) != "complete" for payload in binary_views.values()):
            binary_status = "partial"
    elif binary_data.exists():
        binary_status = "partial"

    summary["mean_relative_length"] = {
        "status": regression_status,
        "metadata_baselines": regression_baselines,
        "views": regression_views,
    }
    summary["majority_s_0.5"] = {
        "status": binary_status,
        "metadata_baselines": binary_baselines,
        "views": binary_views,
    }
    if binary_status == "missing":
        summary["status"] = "missing_binary"
    elif regression_status != "complete" or binary_status != "complete":
        summary["status"] = "partial"
    else:
        summary["status"] = "complete"
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    summary_dir = (
        Path(args.summary_dir).resolve()
        if args.summary_dir
        else (out_root / "summary").resolve()
    )
    selected = args.dataset or list(DATASET_ORDER)

    all_payload: dict[str, Any] = {
        "out_root": str(out_root),
        "datasets": {},
    }
    flat_rows: list[dict[str, Any]] = []

    for dataset_key in selected:
        payload = dataset_summary(dataset_key, out_root)
        all_payload["datasets"][dataset_key] = payload
        write_json(summary_dir / f"{dataset_key}.json", payload)

        regression = payload.get("mean_relative_length", {})
        regression_baselines = regression.get("metadata_baselines", {})
        if isinstance(regression_baselines, dict):
            for baseline_name, baseline_payload in regression_baselines.items():
                if isinstance(baseline_payload, dict):
                    flat_rows.append(
                        flatten_metadata_row(
                            dataset=dataset_key,
                            target_name="mean_relative_length",
                            source_name=baseline_name,
                            payload=baseline_payload,
                        )
                    )
        regression_views = regression.get("views", {})
        if isinstance(regression_views, dict):
            for view_name, view_payload in regression_views.items():
                if isinstance(view_payload, dict):
                    flat_rows.extend(
                        flatten_probe_rows(
                            dataset=dataset_key,
                            target_name="mean_relative_length",
                            view_name=view_name,
                            payload=view_payload,
                        )
                    )

        binary = payload.get("majority_s_0.5", {})
        binary_baselines = binary.get("metadata_baselines", {})
        if isinstance(binary_baselines, dict):
            for baseline_name, baseline_payload in binary_baselines.items():
                if isinstance(baseline_payload, dict):
                    flat_rows.append(
                        flatten_metadata_row(
                            dataset=dataset_key,
                            target_name="majority_s_0.5",
                            source_name=baseline_name,
                            payload=baseline_payload,
                        )
                    )
        binary_views = binary.get("views", {})
        if isinstance(binary_views, dict):
            for view_name, view_payload in binary_views.items():
                if isinstance(view_payload, dict):
                    flat_rows.extend(
                        flatten_probe_rows(
                            dataset=dataset_key,
                            target_name="majority_s_0.5",
                            view_name=view_name,
                            payload=view_payload,
                        )
                    )

    write_json(summary_dir / "cross_dataset_summary.json", all_payload)
    if flat_rows:
        write_csv(summary_dir / "cross_dataset_summary.csv", flat_rows)
    print(summary_dir / "cross_dataset_summary.json")


if __name__ == "__main__":
    main()
