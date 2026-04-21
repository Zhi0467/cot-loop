#!/usr/bin/env python3
"""Train native layerwise Laplace RFMs on frozen prompt-profile archives."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_probe.dataloader import ActivationDataset, read_manifest, resolve_sample_shape
from loop_probe.labeling import prompt_profile_majority_tail_label
from loop_probe.prompt_profile_rfm_stage_registry import (
    active_stage_datasets,
    get_stage_dataset,
    validate_stage_dataset,
)
from loop_probe.rfm import LaplaceRFM, LaplaceRFMConfig
from loop_probe.stage_artifacts import (
    build_rfm_detector_run_record,
    current_git_commit,
    stable_json_sha256,
    write_stage_artifact_record,
)
from loop_probe.train_utils import evaluate_binary_metrics_from_scores, set_seed


DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_STAGE_LABEL = "majority_s_0.5"
DEFAULT_SIGN_CONVENTION = "positive score means higher predicted majority_s_0.5 risk"


@dataclass(frozen=True)
class SplitData:
    x: torch.Tensor
    y: torch.Tensor
    sample_ids: list[int]
    prompts: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(dataset.key for dataset in active_stage_datasets()),
    )
    parser.add_argument("--archive-source-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--bandwidths",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0],
    )
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--balance-train",
        choices=("none", "downsample"),
        default="downsample",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver", choices=("solve", "cholesky"), default="solve")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--centering", action="store_true")
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--center-batch-size", type=int, default=None)
    parser.add_argument("--max-agop-samples", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer indices to train. Defaults to all source layers.",
    )
    return parser.parse_args()


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_ids_sha256(sample_ids: list[int]) -> str:
    body = json.dumps(sample_ids, separators=(",", ":"), ensure_ascii=True)
    return stable_json_sha256(json.loads(body))


def _prompt_text_sha256(prompts: list[str]) -> str:
    body = json.dumps(prompts, separators=(",", ":"), ensure_ascii=True)
    return stable_json_sha256(json.loads(body))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_archive_rows_by_key(data_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = read_manifest(str(data_dir))
    archive_relpath = manifest.get("prompt_rollout_archive_file")
    if not isinstance(archive_relpath, str) or not archive_relpath:
        raise SystemExit(f"Manifest is missing prompt_rollout_archive_file: {data_dir}")
    rows = _read_jsonl_rows(data_dir / archive_relpath)
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        split = row.get("split")
        sample_id = row.get("sample_id")
        if not isinstance(split, str) or not isinstance(sample_id, int):
            raise SystemExit(f"Malformed prompt rollout archive row in {data_dir}.")
        rows_by_key[(split, sample_id)] = row
    return rows_by_key


def _load_binary_split(
    *,
    data_dir: Path,
    split: str,
    feature_key: str,
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
    tail_threshold: float,
) -> SplitData:
    dataset = ActivationDataset(
        data_dir=str(data_dir),
        split=split,
        feature_key=feature_key,
    )
    sample_ids = [int(value) for value in dataset.sample_ids.tolist()]
    prompts: list[str] = []
    labels: list[float] = []
    for sample_id in sample_ids:
        row = rows_by_key.get((split, sample_id))
        if row is None:
            raise SystemExit(f"Missing prompt rollout archive row for {split}:{sample_id}.")
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise SystemExit(f"Missing prompt text for {split}:{sample_id}.")
        prompts.append(prompt)
        labels.append(float(prompt_profile_majority_tail_label(row, threshold=tail_threshold)))
    return SplitData(
        x=dataset.x.detach().to(dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.float32),
        sample_ids=sample_ids,
        prompts=prompts,
    )


def _subset_split(data: SplitData, indices: list[int]) -> SplitData:
    return SplitData(
        x=data.x[indices].clone(),
        y=data.y[indices].clone(),
        sample_ids=[data.sample_ids[index] for index in indices],
        prompts=[data.prompts[index] for index in indices],
    )


def _split_indices_by_label(labels: torch.Tensor) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(labels.tolist()):
        label_i = int(round(float(label)))
        if label_i not in (0, 1):
            raise SystemExit("Expected binary labels for stage-1 RFM training.")
        groups[label_i].append(index)
    return groups


def _train_val_indices(labels: torch.Tensor, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in (0, 1).")
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    groups = _split_indices_by_label(labels)
    for label_value, indices in groups.items():
        if len(indices) < 2:
            raise SystemExit(
                "Need at least two examples per class in the natural train split "
                f"to create a val split; label={label_value} count={len(indices)}."
            )
        shuffled = list(indices)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])
    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def _balanced_train_indices(labels: torch.Tensor, *, seed: int) -> list[int]:
    groups = _split_indices_by_label(labels)
    pos = groups[1]
    neg = groups[0]
    if not pos or not neg:
        raise SystemExit("Balanced training requires both classes in the train subset.")
    keep_per_class = min(len(pos), len(neg))
    rng = random.Random(seed)
    pos = list(pos)
    neg = list(neg)
    rng.shuffle(pos)
    rng.shuffle(neg)
    kept = sorted(pos[:keep_per_class] + neg[:keep_per_class])
    return kept


def _metric_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        value_f = float(value)
        if np.isnan(value_f):
            return default
        return value_f
    return default


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _metric_float(row.get("val_pr_auc"), default=float("-inf")),
        _metric_float(row.get("val_roc_auc"), default=float("-inf")),
        _metric_float(row.get("val_macro_f1"), default=float("-inf")),
    )


def _metrics_from_scores(
    labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores >= threshold).to(dtype=torch.int64)
    return evaluate_binary_metrics_from_scores(
        labels.to(dtype=torch.int64),
        scores.to(dtype=torch.float32),
        predictions,
    )


def _select_score_sign(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor]:
    candidates = []
    for sign in (+1.0, -1.0):
        signed_scores = scores * float(sign)
        metrics = _metrics_from_scores(
            labels,
            signed_scores,
            threshold=0.0,
        )
        candidates.append((metrics["pr_auc"], metrics["roc_auc"], float(sign), signed_scores))
    candidates.sort(
        key=lambda item: (
            _metric_float(item[0], default=float("-inf")),
            _metric_float(item[1], default=float("-inf")),
            item[2],
        ),
        reverse=True,
    )
    _, _, best_sign, signed_scores = candidates[0]
    return best_sign, signed_scores


def _select_threshold(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, dict[str, float]]:
    unique_scores = torch.unique(scores.detach().cpu()).tolist()
    thresholds = sorted(float(value) for value in unique_scores)
    if thresholds:
        thresholds.append(float(max(thresholds) + 1.0))
    else:
        thresholds = [0.0]
    best_threshold = thresholds[0]
    best_metrics = _metrics_from_scores(labels, scores, threshold=best_threshold)
    best_key = (
        _metric_float(best_metrics.get("macro_f1"), default=float("-inf")),
        _metric_float(best_metrics.get("positive_f1"), default=float("-inf")),
        _metric_float(best_metrics.get("accuracy"), default=float("-inf")),
        -best_threshold,
    )
    for threshold in thresholds[1:]:
        metrics = _metrics_from_scores(labels, scores, threshold=threshold)
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
    return float(best_threshold), best_metrics


def _bootstrap_summary(
    *,
    labels: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    num_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if num_samples < 1:
        return {}
    labels_cpu = labels.detach().cpu()
    scores_cpu = scores.detach().cpu()
    n = len(labels_cpu)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(num_samples):
        sampled = torch.tensor(rng.integers(0, n, size=n), dtype=torch.int64)
        sampled_labels = labels_cpu[sampled]
        sampled_scores = scores_cpu[sampled]
        rows.append(_metrics_from_scores(sampled_labels, sampled_scores, threshold=threshold))
    summary: dict[str, dict[str, float]] = {}
    metric_names = (
        "pr_auc",
        "roc_auc",
        "accuracy",
        "macro_f1",
        "positive_precision",
        "positive_recall",
        "positive_f1",
    )
    for metric_name in metric_names:
        values = np.array(
            [
                _metric_float(row.get(metric_name), default=np.nan)
                for row in rows
            ],
            dtype=np.float64,
        )
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
    return summary


def _scores_for_split(model: LaplaceRFM, split_x: torch.Tensor, *, layer: int) -> torch.Tensor:
    return model.predict(split_x[:, layer, :]).detach().cpu()


def _evaluate_iteration(
    *,
    model: LaplaceRFM,
    layer: int,
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    bandwidth: float,
    iteration: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw_train_scores = _scores_for_split(model, train_data.x, layer=layer)
    score_sign, signed_train_scores = _select_score_sign(raw_train_scores, train_data.y)
    threshold, train_metrics = _select_threshold(signed_train_scores, train_data.y)
    signed_val_scores = _scores_for_split(model, val_data.x, layer=layer) * score_sign
    signed_test_scores = _scores_for_split(model, test_data.x, layer=layer) * score_sign
    val_metrics = _metrics_from_scores(val_data.y, signed_val_scores, threshold=threshold)
    test_metrics = _metrics_from_scores(test_data.y, signed_test_scores, threshold=threshold)

    row: dict[str, Any] = {
        "layer": layer,
        "bandwidth": float(bandwidth),
        "iteration": int(iteration),
        "score_sign": float(score_sign),
        "decision_threshold": float(threshold),
        "train_num_samples": len(train_data.sample_ids),
        "train_num_positive": int(train_data.y.sum().item()),
        "val_num_samples": len(val_data.sample_ids),
        "val_num_positive": int(val_data.y.sum().item()),
        "test_num_samples": len(test_data.sample_ids),
        "test_num_positive": int(test_data.y.sum().item()),
    }
    row.update({f"train_{key}": float(value) for key, value in train_metrics.items()})
    row.update({f"val_{key}": float(value) for key, value in val_metrics.items()})
    row.update({f"test_{key}": float(value) for key, value in test_metrics.items()})
    return row, {
        "train_scores": signed_train_scores,
        "val_scores": signed_val_scores,
        "test_scores": signed_test_scores,
    }


def _fit_one_bandwidth(
    *,
    layer: int,
    bandwidth: float,
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, object], dict[str, torch.Tensor]]:
    model = LaplaceRFM(
        LaplaceRFMConfig(
            bandwidth=bandwidth,
            reg=args.reg,
            diag=args.diag,
            centering=args.centering,
            sample_batch_size=args.sample_batch_size,
            center_batch_size=args.center_batch_size,
            max_agop_samples=args.max_agop_samples,
            device=args.device,
            solver=args.solver,
        )
    )
    history: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, object] | None = None
    best_outputs: dict[str, torch.Tensor] | None = None
    train_targets = train_data.y.reshape(-1, 1)

    for iteration in range(args.iters):
        model.fit_predictor(train_data.x[:, layer, :], train_targets)
        row, outputs = _evaluate_iteration(
            model=model,
            layer=layer,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            bandwidth=bandwidth,
            iteration=iteration,
        )
        history.append(row)
        if best_row is None or _selection_key(row) > _selection_key(best_row):
            best_row = dict(row)
            best_state = model.export_state()
            best_outputs = outputs
        model.fit_M(train_data.x[:, layer, :])

    model.fit_predictor(train_data.x[:, layer, :], train_targets)
    row, outputs = _evaluate_iteration(
        model=model,
        layer=layer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        bandwidth=bandwidth,
        iteration=args.iters,
    )
    history.append(row)
    if best_row is None or _selection_key(row) > _selection_key(best_row):
        best_row = dict(row)
        best_state = model.export_state()
        best_outputs = outputs

    if best_row is None or best_state is None or best_outputs is None:
        raise RuntimeError("Expected a best RFM row for each bandwidth.")
    return history, best_row, best_state, best_outputs


def _resolve_layers(sample_shape: tuple[int, ...], requested: list[int] | None) -> list[int]:
    if len(sample_shape) != 2:
        raise SystemExit(f"Expected stacked [layer, hidden] features, got {sample_shape}.")
    num_layers = int(sample_shape[0])
    if requested is None:
        return list(range(num_layers))
    resolved: list[int] = []
    for layer in requested:
        value = layer if layer >= 0 else num_layers + layer
        if value < 0 or value >= num_layers:
            raise SystemExit(
                f"--layers contains out-of-range layer {layer}; valid=[-{num_layers}, {num_layers - 1}]"
            )
        resolved.append(value)
    deduped = sorted(set(resolved))
    return deduped


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    if args.iters < 1:
        raise SystemExit("--iters must be >= 1.")
    if args.reg < 0.0:
        raise SystemExit("--reg must be >= 0.")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be >= 0.")

    benchmark = get_stage_dataset(args.benchmark)
    validation = validate_stage_dataset(
        benchmark,
        archive_source_root=args.archive_source_root,
    )
    data_dir = Path(validation.archive_data_dir)
    manifest = read_manifest(str(data_dir))
    feature_key = benchmark.feature_key
    sample_shape = resolve_sample_shape(manifest, feature_key)
    rows_by_key = _load_archive_rows_by_key(data_dir)

    natural_train = _load_binary_split(
        data_dir=data_dir,
        split="train",
        feature_key=feature_key,
        rows_by_key=rows_by_key,
        tail_threshold=0.5,
    )
    natural_test = _load_binary_split(
        data_dir=data_dir,
        split="test",
        feature_key=feature_key,
        rows_by_key=rows_by_key,
        tail_threshold=0.5,
    )
    train_indices, val_indices = _train_val_indices(
        natural_train.y,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    fit_train = _subset_split(natural_train, train_indices)
    val_split = _subset_split(natural_train, val_indices)
    if args.balance_train == "downsample":
        balanced_indices = _balanced_train_indices(fit_train.y, seed=args.seed)
        fit_train = _subset_split(fit_train, balanced_indices)

    layers = _resolve_layers(sample_shape, args.layers)
    out_dir = Path(args.out_dir)
    checkpoints_dir = out_dir / "checkpoints"
    records_dir = out_dir / "artifacts"
    history_jsonl = out_dir / "layer_metrics.jsonl"
    if history_jsonl.exists():
        history_jsonl.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = {
        "source_data_dir": str(data_dir),
        "source_manifest_path": str(validation.manifest_path),
        "source_target_name": benchmark.source_target_name,
        "stage_label_name": DEFAULT_STAGE_LABEL,
        "feature_key": feature_key,
        "sample_shape": list(sample_shape),
        "scaler": {"kind": "identity"},
        "train_val_split": {
            "val_fraction": float(args.val_fraction),
            "split_seed": int(args.seed),
            "balance_train": args.balance_train,
            "source_train_prompt_ids": natural_train.sample_ids,
            "source_train_prompt_ids_sha256": _sample_ids_sha256(natural_train.sample_ids),
            "source_train_prompt_text_sha256": _prompt_text_sha256(natural_train.prompts),
            "fit_train_prompt_ids": fit_train.sample_ids,
            "fit_train_prompt_ids_sha256": _sample_ids_sha256(fit_train.sample_ids),
            "fit_train_prompt_text_sha256": _prompt_text_sha256(fit_train.prompts),
            "val_prompt_ids": val_split.sample_ids,
            "val_prompt_ids_sha256": _sample_ids_sha256(val_split.sample_ids),
            "val_prompt_text_sha256": _prompt_text_sha256(val_split.prompts),
            "test_prompt_ids": natural_test.sample_ids,
            "test_prompt_ids_sha256": _sample_ids_sha256(natural_test.sample_ids),
            "test_prompt_text_sha256": _prompt_text_sha256(natural_test.prompts),
        },
    }
    git_commit = current_git_commit(ROOT)

    run_config = {
        "benchmark": benchmark.key,
        "display_name": benchmark.display_name,
        "archive_source_root": args.archive_source_root,
        "archive_data_dir": str(data_dir),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "bandwidths": [float(value) for value in args.bandwidths],
        "reg": float(args.reg),
        "iters": int(args.iters),
        "diag": bool(args.diag),
        "centering": bool(args.centering),
        "sample_batch_size": args.sample_batch_size,
        "center_batch_size": args.center_batch_size,
        "max_agop_samples": args.max_agop_samples,
        "balance_train": args.balance_train,
        "val_fraction": float(args.val_fraction),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "layers": layers,
        "git_commit": git_commit,
    }
    _write_json(out_dir / "run_config.json", run_config)
    _write_json(
        out_dir / "split_manifest.json",
        {
            "benchmark": benchmark.key,
            "prompt_ids": {
                "train": fit_train.sample_ids,
                "val": val_split.sample_ids,
                "test": natural_test.sample_ids,
            },
            "source_train_prompt_ids": natural_train.sample_ids,
            "prompt_id_hashes": {
                "train": _sample_ids_sha256(fit_train.sample_ids),
                "val": _sample_ids_sha256(val_split.sample_ids),
                "test": _sample_ids_sha256(natural_test.sample_ids),
            },
            "prompt_text_hashes": {
                "train": _prompt_text_sha256(fit_train.prompts),
                "val": _prompt_text_sha256(val_split.prompts),
                "test": _prompt_text_sha256(natural_test.prompts),
            },
            "train_num_positive": int(fit_train.y.sum().item()),
            "val_num_positive": int(val_split.y.sum().item()),
            "test_num_positive": int(natural_test.y.sum().item()),
            "preprocessing": preprocessing,
        },
    )

    best_layers: list[dict[str, Any]] = []
    for layer in layers:
        layer_best_row: dict[str, Any] | None = None
        layer_best_state: dict[str, object] | None = None
        layer_best_outputs: dict[str, torch.Tensor] | None = None
        for bandwidth in args.bandwidths:
            history_rows, best_row, best_state, best_outputs = _fit_one_bandwidth(
                layer=layer,
                bandwidth=float(bandwidth),
                train_data=fit_train,
                val_data=val_split,
                test_data=natural_test,
                args=args,
            )
            for row in history_rows:
                row["benchmark"] = benchmark.key
                row["git_commit"] = git_commit
                row["model_id"] = args.model_id
                _write_jsonl(history_jsonl, row)
            if layer_best_row is None or _selection_key(best_row) > _selection_key(layer_best_row):
                layer_best_row = dict(best_row)
                layer_best_state = best_state
                layer_best_outputs = best_outputs

        if layer_best_row is None or layer_best_state is None or layer_best_outputs is None:
            raise RuntimeError(f"Layer {layer} did not produce a best RFM row.")

        checkpoint_path = checkpoints_dir / f"layer_{layer:02d}_best.pt"
        checkpoint_payload = {
            "schema_name": "prompt_profile_rfm_detector_checkpoint.v1",
            "benchmark": benchmark.key,
            "layer": int(layer),
            "feature_key": feature_key,
            "prompt_ids": {
                "train": fit_train.sample_ids,
                "val": val_split.sample_ids,
                "test": natural_test.sample_ids,
            },
            "preprocessing": preprocessing,
            "rfm_hyperparameters": {
                "kernel": "laplace",
                "bandwidth": float(layer_best_row["bandwidth"]),
                "reg": float(args.reg),
                "iters": int(args.iters),
                "diag": bool(args.diag),
                "centering": bool(args.centering),
                "sample_batch_size": args.sample_batch_size,
                "center_batch_size": args.center_batch_size,
                "max_agop_samples": args.max_agop_samples,
                "solver": args.solver,
            },
            "selection": {
                "selection_split": "val",
                "selection_metric": "pr_auc",
                "tie_breakers": ["roc_auc", "macro_f1"],
                "best_iteration": int(layer_best_row["iteration"]),
            },
            "score_sign": float(layer_best_row["score_sign"]),
            "decision_threshold": float(layer_best_row["decision_threshold"]),
            "git_commit": git_commit,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "random_seed": int(args.seed),
            "state": layer_best_state,
        }
        torch.save(checkpoint_payload, checkpoint_path)

        test_bootstrap = _bootstrap_summary(
            labels=natural_test.y,
            scores=layer_best_outputs["test_scores"],
            threshold=float(layer_best_row["decision_threshold"]),
            num_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + layer,
        )

        detector_record = build_rfm_detector_run_record(
            benchmark=benchmark.key,
            layer=int(layer),
            train_prompt_ids=fit_train.sample_ids,
            val_prompt_ids=val_split.sample_ids,
            test_prompt_ids=natural_test.sample_ids,
            feature_key=feature_key,
            preprocessing=preprocessing,
            rfm_hyperparameters={
                "kernel": "laplace",
                "bandwidth": float(layer_best_row["bandwidth"]),
                "reg": float(args.reg),
                "iters": int(args.iters),
                "diag": bool(args.diag),
                "centering": bool(args.centering),
                "sample_batch_size": args.sample_batch_size,
                "center_batch_size": args.center_batch_size,
                "max_agop_samples": args.max_agop_samples,
                "solver": args.solver,
            },
            selection={
                "selection_split": "val",
                "selection_metric": "pr_auc",
                "tie_breakers": ["roc_auc", "macro_f1"],
                "best_iteration": int(layer_best_row["iteration"]),
                "test_bootstrap": test_bootstrap,
            },
            sign_convention=DEFAULT_SIGN_CONVENTION,
            score_sign=float(layer_best_row["score_sign"]),
            decision_threshold=float(layer_best_row["decision_threshold"]),
            train_metrics={
                key[len("train_") :]: value
                for key, value in layer_best_row.items()
                if key.startswith("train_") and isinstance(value, (int, float))
            },
            val_metrics={
                key[len("val_") :]: value
                for key, value in layer_best_row.items()
                if key.startswith("val_") and isinstance(value, (int, float))
            },
            test_metrics={
                key[len("test_") :]: value
                for key, value in layer_best_row.items()
                if key.startswith("test_") and isinstance(value, (int, float))
            },
            git_commit=git_commit,
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            random_seed=int(args.seed),
            output_path=str(out_dir),
            checkpoint_path=str(checkpoint_path),
        )
        record_path = records_dir / f"layer_{layer:02d}_detector_record.json"
        write_stage_artifact_record(record_path, detector_record)

        layer_summary = dict(layer_best_row)
        layer_summary["artifact_record_path"] = str(record_path)
        layer_summary["checkpoint_path"] = str(checkpoint_path)
        layer_summary["test_bootstrap"] = test_bootstrap
        best_layers.append(layer_summary)

    best_layers.sort(key=_selection_key, reverse=True)
    _write_json(
        out_dir / "best_layers.json",
        {
            "benchmark": benchmark.key,
            "feature_key": feature_key,
            "selection_rule": {
                "selection_split": "val",
                "selection_metric": "pr_auc",
                "tie_breakers": ["roc_auc", "macro_f1"],
            },
            "best_layers": best_layers,
        },
    )
    print(
        json.dumps(
            {
                "benchmark": benchmark.key,
                "num_layers": len(best_layers),
                "best_layer": best_layers[0]["layer"] if best_layers else None,
                "best_val_pr_auc": best_layers[0]["val_pr_auc"] if best_layers else None,
                "out_dir": str(out_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
