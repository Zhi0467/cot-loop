#!/usr/bin/env python3
"""Train metadata controls and compare top-risk prompt buckets on saved bundles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_probe.configs import ProbeConfig, build_probe_model
from loop_probe.dataloader import (
    ActivationDataset,
    read_manifest,
    resolve_input_dim,
    resolve_sample_shape,
    resolve_split_info,
)
from loop_probe.train_utils import (
    choose_device,
    probe_scores_and_predictions,
    resolve_classifier_layer,
)


DATASET_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "gpqa",
        "display_name": "GPQA",
        "projection_prefix": "gpqa",
        "majority_prefix": "gpqa",
        "continuous_prefix": "gpqa",
    },
    {
        "key": "aime",
        "display_name": "AIME",
        "projection_prefix": "aime",
        "majority_prefix": "aime",
        "continuous_prefix": "aime",
    },
    {
        "key": "math500",
        "display_name": "MATH-500",
        "projection_prefix": "math500",
        "majority_prefix": "math",
        "continuous_prefix": "math",
    },
    {
        "key": "mmlu_pro",
        "display_name": "MMLU-Pro",
        "projection_prefix": "mmlu",
        "majority_prefix": "mmlu",
        "continuous_prefix": "mmlu",
    },
    {
        "key": "livecodebench",
        "display_name": "LiveCodeBench",
        "projection_prefix": "livecodebench",
        "majority_prefix": "livecodebench",
        "continuous_prefix": "livecodebench",
    },
)

FEATURE_SET_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("prompt_length", "prompt_length", ("prompt_token_count",)),
    ("effective_budget", "effective_budget", ("effective_max_tokens",)),
    (
        "prompt_length_plus_effective_budget",
        "prompt_length + effective_budget",
        ("prompt_token_count", "effective_max_tokens"),
    ),
)


@dataclass(frozen=True)
class PromptRow:
    sample_id: int
    split: str
    prompt_token_count: float
    effective_max_tokens: float
    p_loop: float
    p_cap: float
    mean_relative_length: float
    correct_rate: float | None
    rollout_count: int
    majority_s_0_5: int


@dataclass(frozen=True)
class DatasetBundle:
    key: str
    display_name: str
    projection_root: Path
    majority_last_dir: Path
    majority_ensemble_dir: Path
    p_loop_data_dir: Path
    p_loop_last_dir: Path
    p_loop_ensemble_dir: Path
    mean_data_dir: Path
    mean_last_dir: Path
    mean_ensemble_dir: Path


@dataclass(frozen=True)
class MetadataFitResult:
    feature_set_key: str
    feature_set_label: str
    feature_names: tuple[str, ...]
    target_name: str
    metrics: dict[str, float]
    score_by_sample_id: dict[int, float]
    model_payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", required=True, help="Saved cot-loop output root.")
    parser.add_argument("--out-dir", required=True, help="Where to write analysis artifacts.")
    parser.add_argument("--bucket-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latest_path(paths: list[Path]) -> Path:
    if not paths:
        raise SystemExit("Expected at least one candidate path.")
    return sorted(paths)[-1]


def find_projection_candidates(outputs_root: Path, prefix: str) -> list[Path]:
    pattern = f"prompt_profile_projection_{prefix}_majority05*"
    return sorted(
        path
        for path in outputs_root.glob(pattern)
        if (path / "export" / "prompt_projection.csv").exists()
    )


def find_majority_run_dirs(outputs_root: Path, prefix: str, variant: str) -> list[Path]:
    pattern = f"prompt_majority05_{prefix}_{variant}*"
    return sorted(path for path in outputs_root.glob(pattern) if (path / "best.pt").exists())


def resolve_dataset_data_dir(path: Path) -> Path | None:
    if (path / "manifest.json").exists():
        return path
    if (path / "data" / "manifest.json").exists():
        return path / "data"
    return None


def find_continuous_data_root(outputs_root: Path, prefix: str, target_stem: str) -> Path:
    pattern = f"{prefix}_{target_stem}_from_archive_*"
    candidates = sorted(
        path
        for path in outputs_root.glob(pattern)
        if resolve_dataset_data_dir(path) is not None
    )
    return latest_path(candidates)


def find_continuous_run_dir(
    outputs_root: Path,
    prefix: str,
    target_stem: str,
    variant: str,
) -> Path:
    pattern = f"{prefix}_{target_stem}_from_archive_*_{variant}"
    candidates = sorted(path for path in outputs_root.glob(pattern) if (path / "best.pt").exists())
    return latest_path(candidates)


def load_prompt_projection_rows(path: Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            correct_rate_text = (raw.get("correct_rate") or "").strip()
            rows.append(
                PromptRow(
                    sample_id=int(raw["sample_id"]),
                    split=str(raw["split"]),
                    prompt_token_count=float(raw["prompt_token_count"]),
                    effective_max_tokens=float(raw["effective_max_tokens"]),
                    p_loop=float(raw["p_loop"]),
                    p_cap=float(raw["p_cap"]),
                    mean_relative_length=float(raw["mean_relative_length"]),
                    correct_rate=float(correct_rate_text) if correct_rate_text else None,
                    rollout_count=int(raw["rollout_count"]),
                    majority_s_0_5=int(raw["majority_s_0.5"]),
                )
            )
    return rows


def rows_by_split(rows: list[PromptRow], split: str) -> list[PromptRow]:
    return [row for row in rows if row.split == split]


def shard_sample_ids(data_dir: Path, split: str) -> list[int]:
    manifest = read_manifest(str(data_dir))
    split_info, feature_key = resolve_split_info(manifest, split=split, feature_key=None)
    del feature_key
    sample_ids: list[int] = []
    for rel_path in split_info["shards"]:
        shard = torch.load(data_dir / rel_path, map_location="cpu")
        sample_ids.extend(int(x) for x in shard["sample_ids"].tolist())
    return sample_ids


def projection_sample_ids(path: Path, split: str) -> list[int]:
    rows = load_prompt_projection_rows(path / "export" / "prompt_projection.csv")
    return sorted(row.sample_id for row in rows if row.split == split)


def resolve_projection_root(
    candidates: list[Path],
    *,
    reference_train_ids: list[int],
    reference_test_ids: list[int],
) -> Path:
    matches: list[Path] = []
    for candidate in candidates:
        train_ids = projection_sample_ids(candidate, "train")
        test_ids = projection_sample_ids(candidate, "test")
        if train_ids == sorted(reference_train_ids) and test_ids == sorted(reference_test_ids):
            matches.append(candidate)
    if not matches:
        raise SystemExit(
            "Failed to match a prompt-profile projection root to the continuous data split."
        )
    return latest_path(matches)


def seed_tag_from_path(path: Path) -> str | None:
    name = path.name
    for token in name.split("_"):
        if token.startswith("seed") and token[4:].isdigit():
            return token
    return None


def resolve_majority_run_dir(candidates: list[Path], *, seed_tag: str | None) -> Path:
    filtered = candidates
    if seed_tag is not None:
        filtered = [path for path in candidates if seed_tag in path.name]
    return latest_path(filtered)


def discover_dataset_bundle(outputs_root: Path, spec: dict[str, str]) -> DatasetBundle:
    p_loop_data_root = find_continuous_data_root(outputs_root, spec["continuous_prefix"], "p_loop")
    mean_data_root = find_continuous_data_root(outputs_root, spec["continuous_prefix"], "mean_relative")
    p_loop_data_dir = resolve_dataset_data_dir(p_loop_data_root)
    mean_data_dir = resolve_dataset_data_dir(mean_data_root)
    if p_loop_data_dir is None or mean_data_dir is None:
        raise SystemExit(f"{spec['display_name']}: failed to resolve data-dir layout.")

    p_loop_train_ids = shard_sample_ids(p_loop_data_dir, "train")
    p_loop_test_ids = shard_sample_ids(p_loop_data_dir, "test")
    mean_train_ids = shard_sample_ids(mean_data_dir, "train")
    mean_test_ids = shard_sample_ids(mean_data_dir, "test")
    if sorted(p_loop_train_ids) != sorted(mean_train_ids) or sorted(p_loop_test_ids) != sorted(mean_test_ids):
        raise SystemExit(
            f"{spec['display_name']}: p_loop and mean_relative_length data roots do not share the same split."
        )

    projection_root = resolve_projection_root(
        find_projection_candidates(outputs_root, spec["projection_prefix"]),
        reference_train_ids=p_loop_train_ids,
        reference_test_ids=p_loop_test_ids,
    )
    seed_tag = seed_tag_from_path(projection_root)

    majority_last_dir = resolve_majority_run_dir(
        find_majority_run_dirs(outputs_root, spec["majority_prefix"], "last_layer"),
        seed_tag=seed_tag,
    )
    majority_ensemble_dir = resolve_majority_run_dir(
        find_majority_run_dirs(outputs_root, spec["majority_prefix"], "ensemble"),
        seed_tag=seed_tag,
    )

    return DatasetBundle(
        key=spec["key"],
        display_name=spec["display_name"],
        projection_root=projection_root,
        majority_last_dir=majority_last_dir,
        majority_ensemble_dir=majority_ensemble_dir,
        p_loop_data_dir=p_loop_data_dir,
        p_loop_last_dir=find_continuous_run_dir(outputs_root, spec["continuous_prefix"], "p_loop", "last_layer"),
        p_loop_ensemble_dir=find_continuous_run_dir(outputs_root, spec["continuous_prefix"], "p_loop", "ensemble"),
        mean_data_dir=mean_data_dir,
        mean_last_dir=find_continuous_run_dir(
            outputs_root,
            spec["continuous_prefix"],
            "mean_relative",
            "last_layer",
        ),
        mean_ensemble_dir=find_continuous_run_dir(
            outputs_root,
            spec["continuous_prefix"],
            "mean_relative",
            "ensemble",
        ),
    )


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[ordered[k][0]] = avg
        i = j
    return out


def safe_spearman(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(scores, scores[0]):
        return float("nan")
    rx = np.asarray(ranks(y_true.tolist()), dtype=np.float64)
    ry = np.asarray(ranks(scores.tolist()), dtype=np.float64)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def top_capture_fraction(y_true: np.ndarray, scores: np.ndarray, *, fraction: float) -> float:
    if y_true.size == 0:
        return float("nan")
    total_mass = float(np.sum(y_true))
    if total_mass <= 0.0:
        return float("nan")
    keep = max(1, int(math.ceil(float(y_true.size) * fraction)))
    order = np.argsort(-scores, kind="stable")
    captured = float(np.sum(y_true[order[:keep]]))
    return captured / total_mass


def probability_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    errors = scores - y_true
    return {
        "brier": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(scores)),
        "spearman": safe_spearman(y_true, scores),
        "top_10p_capture": top_capture_fraction(y_true, scores, fraction=0.10),
        "top_20p_capture": top_capture_fraction(y_true, scores, fraction=0.20),
    }


def regression_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    errors = scores - y_true
    return {
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(scores)),
        "spearman": safe_spearman(y_true, scores),
        "top_10p_capture": top_capture_fraction(y_true, scores, fraction=0.10),
        "top_20p_capture": top_capture_fraction(y_true, scores, fraction=0.20),
    }


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(y_true.astype(int)).size < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true.astype(int), scores))
        roc_auc = float(roc_auc_score(y_true.astype(int), scores))
    return {
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(scores)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


class ProbabilityMetadataModel:
    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model: LogisticRegression | None = None
        self.constant_probability: float | None = None

    def fit(self, rows: list[PromptRow]) -> None:
        x = feature_matrix(rows, self.feature_names)
        x_scaled = self.scaler.fit_transform(x)
        expanded_x: list[np.ndarray] = []
        expanded_y: list[int] = []
        for idx, row in enumerate(rows):
            rollout_count = max(1, int(row.rollout_count))
            loop_count = int(round(float(row.p_loop) * rollout_count))
            loop_count = max(0, min(rollout_count, loop_count))
            expanded_x.extend([x_scaled[idx]] * rollout_count)
            expanded_y.extend([1] * loop_count + [0] * (rollout_count - loop_count))
        y = np.asarray(expanded_y, dtype=np.int64)
        if y.size == 0:
            raise SystemExit("Cannot fit a metadata probability model on an empty split.")
        if np.unique(y).size < 2:
            self.constant_probability = float(np.mean(y))
            self.model = None
            return
        self.constant_probability = None
        self.model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=0,
        )
        self.model.fit(np.asarray(expanded_x, dtype=np.float64), y)

    def predict(self, rows: list[PromptRow]) -> np.ndarray:
        x = feature_matrix(rows, self.feature_names)
        if self.constant_probability is not None:
            return np.full(x.shape[0], self.constant_probability, dtype=np.float64)
        if self.model is None:
            raise SystemExit("ProbabilityMetadataModel used before fit().")
        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)[:, 1]

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "constant_probability": self.constant_probability,
        }
        if self.model is not None:
            payload["coef"] = self.model.coef_.tolist()
            payload["intercept"] = self.model.intercept_.tolist()
        return payload


class RegressionMetadataModel:
    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model: LinearRegression | None = None
        self.constant_value: float | None = None

    def fit(self, rows: list[PromptRow]) -> None:
        x = feature_matrix(rows, self.feature_names)
        x_scaled = self.scaler.fit_transform(x)
        y = np.asarray([row.mean_relative_length for row in rows], dtype=np.float64)
        if y.size == 0:
            raise SystemExit("Cannot fit a metadata regression model on an empty split.")
        if np.allclose(y, y[0]):
            self.constant_value = float(y[0])
            self.model = None
            return
        self.constant_value = None
        self.model = LinearRegression()
        self.model.fit(x_scaled, y)

    def predict(self, rows: list[PromptRow]) -> np.ndarray:
        x = feature_matrix(rows, self.feature_names)
        if self.constant_value is not None:
            return np.full(x.shape[0], self.constant_value, dtype=np.float64)
        if self.model is None:
            raise SystemExit("RegressionMetadataModel used before fit().")
        x_scaled = self.scaler.transform(x)
        predictions = self.model.predict(x_scaled)
        return np.clip(predictions, 0.0, 1.0)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "constant_value": self.constant_value,
        }
        if self.model is not None:
            payload["coef"] = self.model.coef_.tolist()
            payload["intercept"] = float(self.model.intercept_)
        return payload


def feature_matrix(rows: list[PromptRow], feature_names: tuple[str, ...]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        feature_row: list[float] = []
        for name in feature_names:
            if name == "prompt_token_count":
                feature_row.append(float(row.prompt_token_count))
            elif name == "effective_max_tokens":
                feature_row.append(float(row.effective_max_tokens))
            else:
                raise SystemExit(f"Unsupported metadata feature '{name}'.")
        values.append(feature_row)
    return np.asarray(values, dtype=np.float64)


def fit_metadata_baselines(
    train_rows: list[PromptRow],
    test_rows: list[PromptRow],
    *,
    target_name: str,
) -> list[MetadataFitResult]:
    results: list[MetadataFitResult] = []
    if target_name not in {"p_loop", "mean_relative_length"}:
        raise SystemExit(f"Unsupported metadata target '{target_name}'.")
    for feature_set_key, feature_set_label, feature_names in FEATURE_SET_SPECS:
        if target_name == "p_loop":
            model = ProbabilityMetadataModel(feature_names)
            model.fit(train_rows)
            predictions = model.predict(test_rows)
            y_true = np.asarray([row.p_loop for row in test_rows], dtype=np.float64)
            metrics = probability_metrics(y_true, predictions)
        else:
            model = RegressionMetadataModel(feature_names)
            model.fit(train_rows)
            predictions = model.predict(test_rows)
            y_true = np.asarray([row.mean_relative_length for row in test_rows], dtype=np.float64)
            metrics = regression_metrics(y_true, predictions)
        results.append(
            MetadataFitResult(
                feature_set_key=feature_set_key,
                feature_set_label=feature_set_label,
                feature_names=feature_names,
                target_name=target_name,
                metrics=metrics,
                score_by_sample_id={
                    row.sample_id: float(score) for row, score in zip(test_rows, predictions)
                },
                model_payload=model.to_json(),
            )
        )
    return results


def probe_cfg_from_checkpoint(payload: dict[str, Any]) -> ProbeConfig:
    probe_cfg_raw = payload.get("probe_config")
    if not isinstance(probe_cfg_raw, dict):
        raise SystemExit("Checkpoint is missing probe_config.")
    return ProbeConfig(
        probe_type=str(probe_cfg_raw.get("probe_type", "linear")),
        hidden_dim=int(probe_cfg_raw.get("hidden_dim", 128)),
        dropout=float(probe_cfg_raw.get("dropout", 0.0)),
        depth=int(probe_cfg_raw.get("depth", probe_cfg_raw.get("mlp_depth", 1))),
        classifier_mode=str(probe_cfg_raw.get("classifier_mode", "last_layer")),
        classifier_layer=int(probe_cfg_raw.get("classifier_layer", -1)),
        vote_rule=str(probe_cfg_raw.get("vote_rule", "majority")),
        score_rule=str(probe_cfg_raw.get("score_rule", "vote_fraction")),
    )


def prepare_model_inputs(
    x: torch.Tensor,
    *,
    classifier_mode: str,
    resolved_classifier_layer: int | None,
) -> torch.Tensor:
    if classifier_mode == "last_layer":
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            if resolved_classifier_layer is None:
                raise SystemExit("Missing resolved classifier layer for stacked last-layer inputs.")
            return x[:, resolved_classifier_layer, :]
        raise SystemExit(f"Unsupported last_layer input shape {tuple(x.shape)}.")
    if classifier_mode == "ensemble":
        if x.ndim != 3:
            raise SystemExit(f"ensemble mode expects stacked inputs, got {tuple(x.shape)}.")
        return x
    raise SystemExit(f"Unsupported classifier_mode '{classifier_mode}'.")


def score_checkpoint(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[int, float]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"Checkpoint payload missing state_dict: {checkpoint_path}")

    manifest = read_manifest(str(data_dir))
    checkpoint_feature_key = payload.get("feature_key")
    resolved_feature_key_arg = None
    if isinstance(checkpoint_feature_key, str) and checkpoint_feature_key:
        if isinstance(manifest.get("feature_views"), dict):
            resolved_feature_key_arg = checkpoint_feature_key

    split_info, resolved_feature_key = resolve_split_info(
        manifest,
        split=split,
        feature_key=resolved_feature_key_arg,
    )
    if int(split_info.get("num_rows", split_info.get("num_samples", 0))) < 1:
        raise SystemExit(f"Split '{split}' in {data_dir} has no rows.")

    probe_cfg = probe_cfg_from_checkpoint(payload)
    input_dim = resolve_input_dim(manifest, resolved_feature_key)
    sample_shape = resolve_sample_shape(manifest, resolved_feature_key)
    resolved_classifier_layer: int | None = None
    if probe_cfg.classifier_mode == "last_layer" and len(sample_shape) == 2:
        resolved_classifier_layer = resolve_classifier_layer(int(sample_shape[0]), probe_cfg.classifier_layer)

    model = build_probe_model(
        input_dim=input_dim,
        probe_cfg=probe_cfg,
        sample_shape=sample_shape,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    dataset = ActivationDataset(str(data_dir), split=split, feature_key=resolved_feature_key)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for x, _y in dataloader:
            x = prepare_model_inputs(
                x.to(device),
                classifier_mode=probe_cfg.classifier_mode,
                resolved_classifier_layer=resolved_classifier_layer,
            )
            logits = model(x)
            all_logits.append(logits.detach().cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    scores, _predictions = probe_scores_and_predictions(
        logits_cat,
        classifier_mode=probe_cfg.classifier_mode,
        score_rule=probe_cfg.score_rule,
    )
    sample_ids = dataset.sample_ids.tolist()
    if len(sample_ids) != int(scores.numel()):
        raise SystemExit(f"Score/sample_id length mismatch for {checkpoint_path}.")
    return {int(sample_id): float(score) for sample_id, score in zip(sample_ids, scores.tolist())}


def overall_prompt_rates(rows: list[PromptRow]) -> dict[str, float | None]:
    total_rollouts = float(sum(row.rollout_count for row in rows))
    if total_rollouts <= 0.0:
        raise SystemExit("Cannot summarize prompt rates with zero rollouts.")
    accuracy_numer = 0.0
    has_accuracy = False
    for row in rows:
        if row.correct_rate is not None:
            has_accuracy = True
            accuracy_numer += float(row.correct_rate) * float(row.rollout_count)
    return {
        "prompt_count": float(len(rows)),
        "rollout_count": total_rollouts,
        "loop_rate": float(
            sum(float(row.p_loop) * float(row.rollout_count) for row in rows) / total_rollouts
        ),
        "cap_rate": float(
            sum(float(row.p_cap) * float(row.rollout_count) for row in rows) / total_rollouts
        ),
        "accuracy": float(accuracy_numer / total_rollouts) if has_accuracy else None,
    }


def bucket_stats(
    rows: list[PromptRow],
    *,
    score_by_sample_id: dict[int, float],
    bucket_fraction: float,
    overall_rates: dict[str, float | None],
) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.sample_id in score_by_sample_id]
    if len(scored_rows) != len(rows):
        missing = sorted(row.sample_id for row in rows if row.sample_id not in score_by_sample_id)
        raise SystemExit(f"Missing scores for sample_ids: {missing[:10]}")
    keep = max(1, int(math.ceil(len(scored_rows) * bucket_fraction)))
    ordered = sorted(
        scored_rows,
        key=lambda row: (-float(score_by_sample_id[row.sample_id]), row.sample_id),
    )
    bucket_rows = ordered[:keep]
    total_rollouts = float(sum(row.rollout_count for row in bucket_rows))
    if total_rollouts <= 0.0:
        raise SystemExit("Selected bucket has zero rollouts.")
    loop_rate = float(
        sum(float(row.p_loop) * float(row.rollout_count) for row in bucket_rows) / total_rollouts
    )
    cap_rate = float(
        sum(float(row.p_cap) * float(row.rollout_count) for row in bucket_rows) / total_rollouts
    )
    accuracy_numer = 0.0
    has_accuracy = False
    for row in bucket_rows:
        if row.correct_rate is not None:
            has_accuracy = True
            accuracy_numer += float(row.correct_rate) * float(row.rollout_count)
    accuracy = float(accuracy_numer / total_rollouts) if has_accuracy else None
    overall_loop_rate = float(overall_rates["loop_rate"])
    overall_cap_rate = float(overall_rates["cap_rate"])
    overall_accuracy = overall_rates["accuracy"]
    return {
        "prompt_count": len(bucket_rows),
        "rollout_count": total_rollouts,
        "bucket_fraction": float(bucket_fraction),
        "min_score": float(score_by_sample_id[bucket_rows[-1].sample_id]),
        "max_score": float(score_by_sample_id[bucket_rows[0].sample_id]),
        "mean_score": float(np.mean([score_by_sample_id[row.sample_id] for row in bucket_rows])),
        "loop_rate": loop_rate,
        "cap_rate": cap_rate,
        "accuracy": accuracy,
        "loop_rate_enrichment": (
            float(loop_rate / overall_loop_rate) if overall_loop_rate > 0.0 else float("nan")
        ),
        "cap_rate_enrichment": (
            float(cap_rate / overall_cap_rate) if overall_cap_rate > 0.0 else float("nan")
        ),
        "accuracy_delta": (
            float(accuracy - float(overall_accuracy))
            if accuracy is not None and overall_accuracy is not None
            else None
        ),
        "sample_ids": [row.sample_id for row in bucket_rows],
    }


def load_saved_metrics(run_dir: Path, stem: str) -> dict[str, Any]:
    path = run_dir / f"{stem}.json"
    if not path.exists():
        raise SystemExit(f"Missing metrics file: {path}")
    return read_json(path)


def select_target_variant(variants: dict[str, dict[str, Any]], *, target_name: str) -> str:
    if target_name == "majority_s_0.5":
        return max(
            variants.items(),
            key=lambda item: (
                float(item[1]["derived_test_metrics"].get("pr_auc", float("-inf"))),
                float(item[1]["derived_test_metrics"].get("roc_auc", float("-inf"))),
            ),
        )[0]
    if target_name == "p_loop":
        return max(
            variants.items(),
            key=lambda item: (
                float(item[1]["derived_test_metrics"].get("top_20p_capture", float("-inf"))),
                float(item[1]["derived_test_metrics"].get("spearman", float("-inf"))),
                -float(item[1]["derived_test_metrics"].get("brier", float("inf"))),
            ),
        )[0]
    if target_name == "mean_relative_length":
        return max(
            variants.items(),
            key=lambda item: (
                float(item[1]["derived_test_metrics"].get("spearman", float("-inf"))),
                float(item[1]["derived_test_metrics"].get("top_20p_capture", float("-inf"))),
                -float(item[1]["derived_test_metrics"].get("mse", float("inf"))),
            ),
        )[0]
    raise SystemExit(f"Unsupported target '{target_name}'.")


def select_metadata_variant(results: list[MetadataFitResult], *, target_name: str) -> MetadataFitResult:
    if target_name == "p_loop":
        return max(
            results,
            key=lambda item: (
                float(item.metrics["top_20p_capture"]),
                float(item.metrics["spearman"]),
                -float(item.metrics["brier"]),
            ),
        )
    if target_name == "mean_relative_length":
        return max(
            results,
            key=lambda item: (
                float(item.metrics["spearman"]),
                float(item.metrics["top_20p_capture"]),
                -float(item.metrics["mse"]),
            ),
        )
    raise SystemExit(f"Unsupported target '{target_name}'.")


def select_overall_metadata_baseline(
    metadata_bucket_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return max(
        metadata_bucket_rows,
        key=lambda row: (
            float(row["bucket_loop_rate"]),
            float(row["bucket_cap_rate"]),
            -float(row["bucket_accuracy"]) if row["bucket_accuracy"] is not None else float("-inf"),
        ),
    )


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    bundles = [discover_dataset_bundle(outputs_root, spec) for spec in DATASET_SPECS]
    discovery_payload = {
        bundle.key: {
            "display_name": bundle.display_name,
            "projection_root": str(bundle.projection_root),
            "majority_last_dir": str(bundle.majority_last_dir),
            "majority_ensemble_dir": str(bundle.majority_ensemble_dir),
            "p_loop_data_dir": str(bundle.p_loop_data_dir),
            "p_loop_last_dir": str(bundle.p_loop_last_dir),
            "p_loop_ensemble_dir": str(bundle.p_loop_ensemble_dir),
            "mean_data_dir": str(bundle.mean_data_dir),
            "mean_last_dir": str(bundle.mean_last_dir),
            "mean_ensemble_dir": str(bundle.mean_ensemble_dir),
        }
        for bundle in bundles
    }
    write_json(out_dir / "discovery.json", discovery_payload)

    all_bucket_rows: list[dict[str, Any]] = []
    all_metadata_metric_rows: list[dict[str, Any]] = []
    dataset_payloads: dict[str, Any] = {}

    for bundle in bundles:
        prompt_rows = load_prompt_projection_rows(bundle.projection_root / "export" / "prompt_projection.csv")
        train_rows = rows_by_split(prompt_rows, "train")
        test_rows = rows_by_split(prompt_rows, "test")
        overall_rates = overall_prompt_rates(test_rows)

        metadata_p_loop = fit_metadata_baselines(train_rows, test_rows, target_name="p_loop")
        metadata_mean = fit_metadata_baselines(train_rows, test_rows, target_name="mean_relative_length")
        selected_p_loop_metadata = select_metadata_variant(metadata_p_loop, target_name="p_loop")
        selected_mean_metadata = select_metadata_variant(
            metadata_mean,
            target_name="mean_relative_length",
        )

        majority_variants: dict[str, dict[str, Any]] = {}
        for variant_name, run_dir in (
            ("last_layer", bundle.majority_last_dir),
            ("ensemble", bundle.majority_ensemble_dir),
        ):
            score_by_sample_id = score_checkpoint(
                checkpoint_path=run_dir / "best.pt",
                data_dir=bundle.projection_root / "data",
                split="test",
                batch_size=args.batch_size,
                device=device,
            )
            bucket = bucket_stats(
                test_rows,
                score_by_sample_id=score_by_sample_id,
                bucket_fraction=args.bucket_fraction,
                overall_rates=overall_rates,
            )
            saved_metrics = load_saved_metrics(run_dir, "best_metrics")
            majority_variants[variant_name] = {
                "checkpoint_path": str(run_dir / "best.pt"),
                "saved_metrics": saved_metrics,
                "score_by_sample_id": score_by_sample_id,
                "bucket": bucket,
                "derived_test_metrics": binary_metrics(
                    np.asarray([row.majority_s_0_5 for row in test_rows], dtype=np.float64),
                    np.asarray([score_by_sample_id[row.sample_id] for row in test_rows], dtype=np.float64),
                ),
            }

        p_loop_variants: dict[str, dict[str, Any]] = {}
        for variant_name, run_dir in (
            ("last_layer", bundle.p_loop_last_dir),
            ("ensemble", bundle.p_loop_ensemble_dir),
        ):
            score_by_sample_id = score_checkpoint(
                checkpoint_path=run_dir / "best_rank.pt",
                data_dir=bundle.p_loop_data_dir,
                split="test",
                batch_size=args.batch_size,
                device=device,
            )
            bucket = bucket_stats(
                test_rows,
                score_by_sample_id=score_by_sample_id,
                bucket_fraction=args.bucket_fraction,
                overall_rates=overall_rates,
            )
            saved_metrics = load_saved_metrics(run_dir, "best_rank_metrics")
            p_loop_variants[variant_name] = {
                "checkpoint_path": str(run_dir / "best_rank.pt"),
                "saved_metrics": saved_metrics,
                "score_by_sample_id": score_by_sample_id,
                "bucket": bucket,
                "derived_test_metrics": probability_metrics(
                    np.asarray([row.p_loop for row in test_rows], dtype=np.float64),
                    np.asarray([score_by_sample_id[row.sample_id] for row in test_rows], dtype=np.float64),
                ),
            }

        mean_variants: dict[str, dict[str, Any]] = {}
        for variant_name, run_dir in (
            ("last_layer", bundle.mean_last_dir),
            ("ensemble", bundle.mean_ensemble_dir),
        ):
            score_by_sample_id = score_checkpoint(
                checkpoint_path=run_dir / "best_rank.pt",
                data_dir=bundle.mean_data_dir,
                split="test",
                batch_size=args.batch_size,
                device=device,
            )
            bucket = bucket_stats(
                test_rows,
                score_by_sample_id=score_by_sample_id,
                bucket_fraction=args.bucket_fraction,
                overall_rates=overall_rates,
            )
            saved_metrics = load_saved_metrics(run_dir, "best_rank_metrics")
            mean_variants[variant_name] = {
                "checkpoint_path": str(run_dir / "best_rank.pt"),
                "saved_metrics": saved_metrics,
                "score_by_sample_id": score_by_sample_id,
                "bucket": bucket,
                "derived_test_metrics": regression_metrics(
                    np.asarray([row.mean_relative_length for row in test_rows], dtype=np.float64),
                    np.asarray([score_by_sample_id[row.sample_id] for row in test_rows], dtype=np.float64),
                ),
            }

        selected_majority_variant = select_target_variant(
            majority_variants,
            target_name="majority_s_0.5",
        )
        selected_p_loop_variant = select_target_variant(p_loop_variants, target_name="p_loop")
        selected_mean_variant = select_target_variant(
            mean_variants,
            target_name="mean_relative_length",
        )

        metadata_bucket_rows: list[dict[str, Any]] = []
        for target_name, result in [
            ("p_loop", fit_result) for fit_result in metadata_p_loop
        ] + [
            ("mean_relative_length", fit_result) for fit_result in metadata_mean
        ]:
            bucket = bucket_stats(
                test_rows,
                score_by_sample_id=result.score_by_sample_id,
                bucket_fraction=args.bucket_fraction,
                overall_rates=overall_rates,
            )
            row = {
                "dataset_key": bundle.key,
                "dataset_name": bundle.display_name,
                "target_name": target_name,
                "feature_set_key": result.feature_set_key,
                "feature_set_label": result.feature_set_label,
                "metrics": result.metrics,
                "bucket": bucket,
                "model_payload": result.model_payload,
                "bucket_loop_rate": float(bucket["loop_rate"]),
                "bucket_cap_rate": float(bucket["cap_rate"]),
                "bucket_accuracy": bucket["accuracy"],
            }
            metadata_bucket_rows.append(row)
            all_metadata_metric_rows.append(
                {
                    "dataset_key": bundle.key,
                    "dataset_name": bundle.display_name,
                    "target_name": target_name,
                    "feature_set_key": result.feature_set_key,
                    "feature_set_label": result.feature_set_label,
                    **result.metrics,
                }
            )

        selected_metadata_bucket = select_overall_metadata_baseline(metadata_bucket_rows)

        dataset_payload = {
            "display_name": bundle.display_name,
            "bundle_paths": discovery_payload[bundle.key],
            "test_overall_rates": overall_rates,
            "selected_variants": {
                "majority_s_0.5": selected_majority_variant,
                "p_loop": selected_p_loop_variant,
                "mean_relative_length": selected_mean_variant,
                "metadata_baseline": {
                    "target_name": selected_metadata_bucket["target_name"],
                    "feature_set_key": selected_metadata_bucket["feature_set_key"],
                    "feature_set_label": selected_metadata_bucket["feature_set_label"],
                },
            },
            "majority_s_0.5": majority_variants,
            "p_loop": p_loop_variants,
            "mean_relative_length": mean_variants,
            "metadata_baselines": {
                "p_loop": {
                    "selected_feature_set": selected_p_loop_metadata.feature_set_key,
                    "feature_sets": {
                        item.feature_set_key: {
                            "feature_set_label": item.feature_set_label,
                            "feature_names": list(item.feature_names),
                            "metrics": item.metrics,
                            "model_payload": item.model_payload,
                            "bucket": next(
                                row["bucket"]
                                for row in metadata_bucket_rows
                                if row["target_name"] == "p_loop"
                                and row["feature_set_key"] == item.feature_set_key
                            ),
                        }
                        for item in metadata_p_loop
                    },
                },
                "mean_relative_length": {
                    "selected_feature_set": selected_mean_metadata.feature_set_key,
                    "feature_sets": {
                        item.feature_set_key: {
                            "feature_set_label": item.feature_set_label,
                            "feature_names": list(item.feature_names),
                            "metrics": item.metrics,
                            "model_payload": item.model_payload,
                            "bucket": next(
                                row["bucket"]
                                for row in metadata_bucket_rows
                                if row["target_name"] == "mean_relative_length"
                                and row["feature_set_key"] == item.feature_set_key
                            ),
                        }
                        for item in metadata_mean
                    },
                },
                "selected_overall": selected_metadata_bucket,
            },
        }
        dataset_payloads[bundle.key] = dataset_payload

        comparison_rows = [
            {
                "dataset_key": bundle.key,
                "dataset_name": bundle.display_name,
                "score_name": "majority_s_0.5",
                "score_variant": selected_majority_variant,
                "target_name": "majority_s_0.5",
                "feature_set_key": "",
                "feature_set_label": "",
                **{
                    f"bucket_{key}": value
                    for key, value in majority_variants[selected_majority_variant]["bucket"].items()
                    if key != "sample_ids"
                },
            },
            {
                "dataset_key": bundle.key,
                "dataset_name": bundle.display_name,
                "score_name": "p_loop",
                "score_variant": selected_p_loop_variant,
                "target_name": "p_loop",
                "feature_set_key": "",
                "feature_set_label": "",
                **{
                    f"bucket_{key}": value
                    for key, value in p_loop_variants[selected_p_loop_variant]["bucket"].items()
                    if key != "sample_ids"
                },
            },
            {
                "dataset_key": bundle.key,
                "dataset_name": bundle.display_name,
                "score_name": "mean_relative_length",
                "score_variant": selected_mean_variant,
                "target_name": "mean_relative_length",
                "feature_set_key": "",
                "feature_set_label": "",
                **{
                    f"bucket_{key}": value
                    for key, value in mean_variants[selected_mean_variant]["bucket"].items()
                    if key != "sample_ids"
                },
            },
            {
                "dataset_key": bundle.key,
                "dataset_name": bundle.display_name,
                "score_name": "metadata_baseline",
                "score_variant": selected_metadata_bucket["target_name"],
                "target_name": selected_metadata_bucket["target_name"],
                "feature_set_key": selected_metadata_bucket["feature_set_key"],
                "feature_set_label": selected_metadata_bucket["feature_set_label"],
                **{
                    f"bucket_{key}": value
                    for key, value in selected_metadata_bucket["bucket"].items()
                    if key != "sample_ids"
                },
            },
        ]
        all_bucket_rows.extend(comparison_rows)

        write_json(out_dir / bundle.key / "summary.json", dataset_payload)

        joined_score_rows: list[dict[str, Any]] = []
        for row in test_rows:
            joined_score_rows.append(
                {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "prompt_token_count": row.prompt_token_count,
                    "effective_max_tokens": row.effective_max_tokens,
                    "p_loop": row.p_loop,
                    "p_cap": row.p_cap,
                    "mean_relative_length": row.mean_relative_length,
                    "correct_rate": row.correct_rate,
                    "rollout_count": row.rollout_count,
                    "majority_s_0.5": row.majority_s_0_5,
                    "majority_last_layer_score": majority_variants["last_layer"]["score_by_sample_id"][row.sample_id],
                    "majority_ensemble_score": majority_variants["ensemble"]["score_by_sample_id"][row.sample_id],
                    "p_loop_last_layer_score": p_loop_variants["last_layer"]["score_by_sample_id"][row.sample_id],
                    "p_loop_ensemble_score": p_loop_variants["ensemble"]["score_by_sample_id"][row.sample_id],
                    "mean_relative_last_layer_score": mean_variants["last_layer"]["score_by_sample_id"][row.sample_id],
                    "mean_relative_ensemble_score": mean_variants["ensemble"]["score_by_sample_id"][row.sample_id],
                    **{
                        f"metadata_{target}_{result.feature_set_key}_score": result.score_by_sample_id[row.sample_id]
                        for target, result in [
                            ("p_loop", item) for item in metadata_p_loop
                        ] + [
                            ("mean_relative_length", item) for item in metadata_mean
                        ]
                    },
                }
            )
        write_csv(
            out_dir / bundle.key / "test_prompt_scores.csv",
            joined_score_rows[0].keys(),
            joined_score_rows,
        )

    write_csv(
        out_dir / "cross_dataset_bucket_summary.csv",
        all_bucket_rows[0].keys(),
        all_bucket_rows,
    )
    write_csv(
        out_dir / "metadata_baseline_metrics.csv",
        sorted({key for row in all_metadata_metric_rows for key in row}),
        all_metadata_metric_rows,
    )

    win_counts = {
        "loop_rate": {},
        "cap_rate": {},
        "lowest_accuracy": {},
    }
    grouped_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in all_bucket_rows:
        grouped_by_dataset.setdefault(str(row["dataset_key"]), []).append(row)
    for dataset_key, rows in grouped_by_dataset.items():
        best_loop = max(rows, key=lambda row: float(row["bucket_loop_rate"]))
        best_cap = max(rows, key=lambda row: float(row["bucket_cap_rate"]))
        accuracy_rows = [row for row in rows if row["bucket_accuracy"] is not None]
        if accuracy_rows:
            best_low_accuracy = min(accuracy_rows, key=lambda row: float(row["bucket_accuracy"]))
            win_counts["lowest_accuracy"][dataset_key] = str(best_low_accuracy["score_name"])
        win_counts["loop_rate"][dataset_key] = str(best_loop["score_name"])
        win_counts["cap_rate"][dataset_key] = str(best_cap["score_name"])

    cross_dataset_payload = {
        "bucket_fraction": float(args.bucket_fraction),
        "datasets": dataset_payloads,
        "bucket_summary_csv": str(out_dir / "cross_dataset_bucket_summary.csv"),
        "metadata_metrics_csv": str(out_dir / "metadata_baseline_metrics.csv"),
        "win_counts": win_counts,
    }
    write_json(out_dir / "cross_dataset_summary.json", cross_dataset_payload)
    print(out_dir)


if __name__ == "__main__":
    main()
