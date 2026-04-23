#!/usr/bin/env python3
"""Train prompt-only baselines on a materialized prompt-profile stage split."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from loop_probe.labeling import prompt_profile_majority_tail_label
from loop_probe.stage_artifacts import stable_json_sha256
from loop_probe.train_utils import evaluate_binary_metrics_from_scores


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "prompt_length": ("prompt_token_count",),
    "prompt_shape": (
        "prompt_token_count",
        "log_prompt_token_count",
        "char_length",
        "newline_count",
        "digit_count",
        "dollar_count",
        "choice_count",
    ),
}


@dataclass(frozen=True)
class PromptRow:
    sample_id: int
    prompt: str
    prompt_token_count: float
    label: int

    @property
    def log_prompt_token_count(self) -> float:
        return float(math.log1p(self.prompt_token_count))

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
        return float(
            self.prompt.count("\nA.")
            + self.prompt.count("\nB.")
            + self.prompt.count("\nC.")
            + self.prompt.count("\nD.")
        )

    def feature_value(self, name: str) -> float:
        if name == "prompt_token_count":
            return float(self.prompt_token_count)
        if name == "log_prompt_token_count":
            return self.log_prompt_token_count
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
        raise SystemExit(f"Unsupported prompt feature '{name}'.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("prompt_length", "prompt_shape_linear", "prompt_shape_tree"),
        default=[
            "prompt_length",
            "prompt_shape_linear",
            "prompt_shape_tree",
        ],
    )
    parser.add_argument(
        "--linear-c-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--tree-max-depths",
        nargs="+",
        default=["2", "3", "4", "5", "none"],
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_prompt_rows(path: Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            prompt = payload.get("prompt")
            sample_id = payload.get("sample_id")
            prompt_token_count = payload.get("prompt_token_count")
            if not isinstance(prompt, str) or not isinstance(sample_id, int):
                raise SystemExit(f"Malformed prompt row at {path}:{line_num}.")
            if not isinstance(prompt_token_count, (int, float)):
                raise SystemExit(
                    f"Prompt row at {path}:{line_num} is missing prompt_token_count."
                )
            stage_label = payload.get("stage_majority_tail")
            if isinstance(stage_label, (int, float)):
                label = int(stage_label)
            else:
                tail_threshold = payload.get("stage_tail_threshold", 0.5)
                label = int(prompt_profile_majority_tail_label(payload, threshold=float(tail_threshold)))
            rows.append(
                PromptRow(
                    sample_id=sample_id,
                    prompt=prompt,
                    prompt_token_count=float(prompt_token_count),
                    label=label,
                )
            )
    return rows

def _feature_matrix(rows: list[PromptRow], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [
            [row.feature_value(feature_name) for feature_name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )


def _labels(rows: list[PromptRow]) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=np.int64)


def _metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["pr_auc"]),
        float(metrics["roc_auc"]),
        float(metrics["macro_f1"]),
    )


def _score_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    scores_t = torch.tensor(scores, dtype=torch.float32)
    predictions_t = (scores_t >= float(threshold)).to(dtype=torch.int64)
    labels_t = torch.tensor(y_true, dtype=torch.int64)
    return evaluate_binary_metrics_from_scores(labels_t, scores_t, predictions_t)


def _metric_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    value_f = float(value)
    if math.isnan(value_f):
        return default
    return value_f


def _select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds = sorted(float(value) for value in np.unique(scores).tolist())
    if thresholds:
        thresholds.append(float(max(thresholds) + 1.0))
    else:
        thresholds = [0.5]
    best_threshold = thresholds[0]
    best_metrics = _score_metrics(y_true, scores, threshold=best_threshold)
    best_key = (
        _metric_float(best_metrics.get("macro_f1"), default=float("-inf")),
        _metric_float(best_metrics.get("positive_f1"), default=float("-inf")),
        _metric_float(best_metrics.get("accuracy"), default=float("-inf")),
        -best_threshold,
    )
    for threshold in thresholds[1:]:
        metrics = _score_metrics(y_true, scores, threshold=threshold)
        key = (
            _metric_float(metrics.get("macro_f1"), default=float("-inf")),
            _metric_float(metrics.get("positive_f1"), default=float("-inf")),
            _metric_float(metrics.get("accuracy"), default=float("-inf")),
            -threshold,
        )
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def _logistic_scores(
    *,
    train_rows: list[PromptRow],
    eval_rows: list[PromptRow],
    feature_names: tuple[str, ...],
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(_feature_matrix(train_rows, feature_names))
    train_y = _labels(train_rows)
    eval_x = scaler.transform(_feature_matrix(eval_rows, feature_names))
    model = LogisticRegression(
        C=float(c_value),
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(eval_x)[:, 1]
    payload = {
        "kind": "logistic_regression",
        "feature_names": list(feature_names),
        "C": float(c_value),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    return scores, payload


def _tree_scores(
    *,
    train_rows: list[PromptRow],
    eval_rows: list[PromptRow],
    feature_names: tuple[str, ...],
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x = _feature_matrix(train_rows, feature_names)
    train_y = _labels(train_rows)
    eval_x = _feature_matrix(eval_rows, feature_names)
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(eval_x)[:, 1]
    payload = {
        "kind": "decision_tree_classifier",
        "feature_names": list(feature_names),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "feature_importances": model.feature_importances_.tolist(),
    }
    return scores, payload


def _evaluate_model(
    *,
    model_name: str,
    train_rows: list[PromptRow],
    val_rows: list[PromptRow],
    test_rows: list[PromptRow],
    split_manifest: dict[str, Any],
    seed: int,
    linear_c_grid: list[float],
    tree_max_depths: list[str],
    tree_min_samples_leaf: list[int],
) -> dict[str, Any]:
    if model_name == "prompt_length":
        feature_names = FEATURE_SETS["prompt_length"]
        candidates = [
            {
                "selector": {"C": float(c_value)},
                "feature_set_name": "prompt_length",
                "feature_names": feature_names,
                "scorer": lambda rows, c=c_value: _logistic_scores(
                    train_rows=train_rows,
                    eval_rows=rows,
                    feature_names=feature_names,
                    c_value=c,
                    seed=seed,
                ),
            }
            for c_value in linear_c_grid
        ]
    elif model_name == "prompt_shape_linear":
        feature_names = FEATURE_SETS["prompt_shape"]
        candidates = [
            {
                "selector": {"C": float(c_value)},
                "feature_set_name": "prompt_shape",
                "feature_names": feature_names,
                "scorer": lambda rows, c=c_value: _logistic_scores(
                    train_rows=train_rows,
                    eval_rows=rows,
                    feature_names=feature_names,
                    c_value=c,
                    seed=seed,
                ),
            }
            for c_value in linear_c_grid
        ]
    elif model_name == "prompt_shape_tree":
        feature_names = FEATURE_SETS["prompt_shape"]
        candidates = []
        for raw_depth in tree_max_depths:
            max_depth = None if raw_depth.lower() == "none" else int(raw_depth)
            for min_samples_leaf in tree_min_samples_leaf:
                candidates.append(
                    {
                        "selector": {
                            "max_depth": max_depth,
                            "min_samples_leaf": int(min_samples_leaf),
                        },
                        "feature_set_name": "prompt_shape",
                        "feature_names": feature_names,
                        "scorer": lambda rows, depth=max_depth, leaf=min_samples_leaf: _tree_scores(
                            train_rows=train_rows,
                            eval_rows=rows,
                            feature_names=feature_names,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            seed=seed,
                        ),
                    }
                )
    else:
        raise SystemExit(f"Unsupported model '{model_name}'.")

    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    selection_rows: list[dict[str, Any]] = []
    train_labels = _labels(train_rows)
    val_labels = _labels(val_rows)
    test_labels = _labels(test_rows)
    for candidate in candidates:
        train_scores, _ = candidate["scorer"](train_rows)
        decision_threshold, train_metrics = _select_threshold(train_labels, train_scores)
        val_scores, model_payload = candidate["scorer"](val_rows)
        val_metrics = _score_metrics(val_labels, val_scores, threshold=decision_threshold)
        selection_row = {
            "selector": candidate["selector"],
            "decision_threshold": float(decision_threshold),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        selection_rows.append(selection_row)
        val_key = _metric_key(val_metrics)
        if best_key is None or val_key > best_key:
            test_scores, _ = candidate["scorer"](test_rows)
            best_result = {
                "model_name": model_name,
                "feature_set_name": candidate["feature_set_name"],
                "feature_names": list(candidate["feature_names"]),
                "hyperparameters": candidate["selector"],
                "model_payload": model_payload,
                "decision_threshold": float(decision_threshold),
                "threshold_selection_rule": "train_macro_f1_then_positive_f1_then_accuracy",
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": _score_metrics(
                    test_labels,
                    test_scores,
                    threshold=decision_threshold,
                ),
                "selection_rows": selection_rows,
                "prompt_ids": split_manifest["prompt_ids"],
                "prompt_id_hashes": split_manifest.get("prompt_id_hashes"),
                "prompt_text_hashes": split_manifest.get("prompt_text_hashes"),
                "train_label_prevalence": float(np.mean(_labels(train_rows))),
                "val_label_prevalence": float(np.mean(val_labels)),
                "test_label_prevalence": float(np.mean(test_labels)),
                "sample_ids_sha256": {
                    "train": stable_json_sha256([row.sample_id for row in train_rows]),
                    "val": stable_json_sha256([row.sample_id for row in val_rows]),
                    "test": stable_json_sha256([row.sample_id for row in test_rows]),
                },
            }
            best_key = val_key
    if best_result is None:
        raise SystemExit(f"No candidate results produced for {model_name}.")
    return best_result


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    split_manifest = _read_json(data_dir / "split_manifest.json")
    train_rows = _read_prompt_rows(data_dir / "diagnostics" / "train_prompt_rows.jsonl")
    val_rows = _read_prompt_rows(data_dir / "diagnostics" / "val_prompt_rows.jsonl")
    test_rows = _read_prompt_rows(data_dir / "diagnostics" / "test_prompt_rows.jsonl")

    summary_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        result = _evaluate_model(
            model_name=model_name,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            split_manifest=split_manifest,
            seed=args.seed,
            linear_c_grid=list(args.linear_c_grid),
            tree_max_depths=list(args.tree_max_depths),
            tree_min_samples_leaf=list(args.tree_min_samples_leaf),
        )
        _write_json(out_dir / f"{model_name}.json", result)
        summary_rows.append(
            {
                "model_name": model_name,
                "feature_set_name": result["feature_set_name"],
                "hyperparameters": result["hyperparameters"],
                "val_pr_auc": result["val_metrics"]["pr_auc"],
                "val_roc_auc": result["val_metrics"]["roc_auc"],
                "test_pr_auc": result["test_metrics"]["pr_auc"],
                "test_roc_auc": result["test_metrics"]["roc_auc"],
                "test_macro_f1": result["test_metrics"]["macro_f1"],
                "test_positive_f1": result["test_metrics"]["positive_f1"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_prevalence": result["test_metrics"]["prevalence"],
            }
        )
    _write_json(out_dir / "summary.json", summary_rows)


if __name__ == "__main__":
    main()
