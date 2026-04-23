#!/usr/bin/env python3
"""Export signed layerwise RFM vectors and first direction diagnostics."""

from __future__ import annotations

import argparse
import json
import math
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

from loop_probe.dataloader import ActivationDataset, read_manifest
from loop_probe.labeling import prompt_profile_majority_tail_label
from loop_probe.rfm import LaplaceRFM, LaplaceRFMConfig
from loop_probe.stage_artifacts import (
    build_rfm_vector_direction_bootstrap_record,
    build_rfm_vector_bundle_record,
    current_git_commit,
    stable_json_sha256,
    tensor_checksum_hex,
    write_stage_artifact_record,
)
from loop_probe.train_utils import evaluate_binary_metrics_from_scores


DEFAULT_SIGN_CONVENTION = "positive projection means higher predicted majority_s_0.5 risk"
DEFAULT_EXTRACTION_FORMULA = "metric_top_eigenvector_rank1.v1"


@dataclass(frozen=True)
class SplitData:
    x: torch.Tensor
    y: torch.Tensor
    sample_ids: list[int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rfm-run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument(
        "--vector-scale",
        choices=("sqrt_eigenvalue", "eigenvalue", "unit"),
        default="sqrt_eigenvalue",
    )
    parser.add_argument("--projection-bootstrap-samples", type=int, default=0)
    parser.add_argument("--projection-bootstrap-seed", type=int, default=0)
    parser.add_argument("--direction-bootstrap-samples", type=int, default=0)
    parser.add_argument("--direction-bootstrap-seed", type=int, default=0)
    parser.add_argument("--direction-bootstrap-device", default="auto")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _metric_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        value_f = float(value)
        if np.isnan(value_f):
            return default
        return value_f
    return default


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
            [_metric_float(row.get(metric_name), default=np.nan) for row in rows],
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


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "low": float(np.percentile(array, 2.5)),
        "median": float(np.percentile(array, 50.0)),
        "high": float(np.percentile(array, 97.5)),
        "fraction_negative": float(np.mean(array < 0.0)),
        "fraction_below_0_5": float(np.mean(array < 0.5)),
    }


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    source_split: str,
    feature_key: str,
    sample_ids: list[int],
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
    tail_threshold: float,
) -> SplitData:
    dataset = ActivationDataset(
        data_dir=str(data_dir),
        split=source_split,
        feature_key=feature_key,
    )
    id_to_index = {
        int(value): index
        for index, value in enumerate(dataset.sample_ids.tolist())
    }
    selected_x: list[torch.Tensor] = []
    labels: list[float] = []
    resolved_sample_ids: list[int] = []
    for sample_id in sample_ids:
        index = id_to_index.get(int(sample_id))
        if index is None:
            raise SystemExit(f"Missing sample_id={sample_id} in split='{source_split}'.")
        row = rows_by_key.get((source_split, int(sample_id)))
        if row is None:
            raise SystemExit(f"Missing prompt rollout archive row for {source_split}:{sample_id}.")
        selected_x.append(dataset.x[index].detach().to(dtype=torch.float32))
        labels.append(float(prompt_profile_majority_tail_label(row, threshold=tail_threshold)))
        resolved_sample_ids.append(int(sample_id))
    return SplitData(
        x=torch.stack(selected_x, dim=0),
        y=torch.tensor(labels, dtype=torch.float32),
        sample_ids=resolved_sample_ids,
    )


def _resolve_layers(best_layers_payload: dict[str, Any], requested: list[int] | None) -> list[int]:
    available_layers = [
        int(row["layer"])
        for row in best_layers_payload.get("best_layers", [])
        if isinstance(row, dict) and "layer" in row
    ]
    if requested is None:
        return available_layers
    requested_set = {int(layer) for layer in requested}
    missing = sorted(requested_set.difference(available_layers))
    if missing:
        raise SystemExit(f"Requested layers not present in best_layers.json: {missing}")
    return sorted(requested_set)


def _resolve_record_path(path_text: str, *, rfm_run_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _extract_signed_vector(
    state: dict[str, Any],
    *,
    vector_scale: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    metric = state["M"].detach().to(dtype=torch.float32, device="cpu")
    if metric.ndim == 1:
        diag_metric = metric.clamp_min(0.0)
        normalized_vector = torch.zeros_like(diag_metric)
        max_index = int(torch.argmax(diag_metric).item())
        normalized_vector[max_index] = 1.0
        raw_scale = 1.0
        if vector_scale == "sqrt_eigenvalue":
            raw_scale = float(torch.sqrt(diag_metric[max_index]).item())
        elif vector_scale == "eigenvalue":
            raw_scale = float(diag_metric[max_index].item())
        raw_vector = normalized_vector * raw_scale
        extraction = {
            "formula_version": DEFAULT_EXTRACTION_FORMULA,
            "metric_kind": "diagonal",
            "selected_index": max_index,
            "selected_eigenvalue": float(diag_metric[max_index].item()),
            "vector_scale": vector_scale,
        }
        return raw_vector, normalized_vector, extraction

    metric = 0.5 * (metric + metric.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    top_eigenvalue = float(eigenvalues[-1].item())
    top_eigenvector = eigenvectors[:, -1].to(dtype=torch.float32)
    if vector_scale == "sqrt_eigenvalue":
        raw_scale = math.sqrt(max(top_eigenvalue, 0.0))
    elif vector_scale == "eigenvalue":
        raw_scale = top_eigenvalue
    else:
        raw_scale = 1.0
    raw_vector = top_eigenvector * float(raw_scale)
    normalized_vector = top_eigenvector / top_eigenvector.norm().clamp_min(1e-12)
    positive_trace = float(eigenvalues.clamp_min(0.0).sum().item())
    extraction = {
        "formula_version": DEFAULT_EXTRACTION_FORMULA,
        "metric_kind": "full",
        "metric_symmetrized": True,
        "selected_rank": 1,
        "selected_eigenvalue": top_eigenvalue,
        "selected_eigenvalue_share": (
            top_eigenvalue / positive_trace if positive_trace > 0.0 else None
        ),
        "second_eigenvalue": float(eigenvalues[-2].item()) if eigenvalues.numel() > 1 else None,
        "vector_scale": vector_scale,
    }
    return raw_vector, normalized_vector, extraction


def _projection_summary(
    *,
    normalized_vector: torch.Tensor,
    layer: int,
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    train_scores = (train_data.x[:, layer, :] @ normalized_vector).detach().cpu()
    score_sign, signed_train_scores = _select_score_sign(train_scores, train_data.y)
    threshold, train_metrics = _select_threshold(signed_train_scores, train_data.y)

    split_rows: dict[str, dict[str, Any]] = {}
    split_scores: dict[str, torch.Tensor] = {}
    for index, (split_name, split_data) in enumerate(
        (
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        )
    ):
        scores = (split_data.x[:, layer, :] @ normalized_vector * score_sign).detach().cpu()
        split_scores[split_name] = scores
        metrics = _metrics_from_scores(split_data.y, scores, threshold=threshold)
        positive_scores = scores[split_data.y == 1]
        negative_scores = scores[split_data.y == 0]
        row: dict[str, Any] = {
            "metrics": metrics,
            "score_mean": float(scores.mean().item()),
            "score_std": float(scores.std(unbiased=False).item()) if len(scores) > 1 else 0.0,
            "positive_mean": (
                float(positive_scores.mean().item()) if positive_scores.numel() else None
            ),
            "negative_mean": (
                float(negative_scores.mean().item()) if negative_scores.numel() else None
            ),
            "positive_minus_negative_mean": (
                float((positive_scores.mean() - negative_scores.mean()).item())
                if positive_scores.numel() and negative_scores.numel()
                else None
            ),
        }
        if split_name == "test":
            row["bootstrap"] = _bootstrap_summary(
                labels=split_data.y,
                scores=scores,
                threshold=threshold,
                num_samples=bootstrap_samples,
                seed=bootstrap_seed + index + layer,
            )
        split_rows[split_name] = row

    projection_summary = {
        "score_sign": float(score_sign),
        "decision_threshold": float(threshold),
        "sign_selection_rule": "train_pr_auc_then_roc_auc",
        "threshold_selection_rule": "train_macro_f1_then_positive_f1_then_accuracy",
        "splits": split_rows,
    }
    return split_scores["test"], float(score_sign), projection_summary


def _cross_layer_cosines(vectors_by_layer: dict[int, torch.Tensor]) -> dict[str, Any]:
    layers = sorted(vectors_by_layer)
    if not layers:
        return {"layers": [], "matrix": []}
    matrix: list[list[float]] = []
    for layer_i in layers:
        row: list[float] = []
        vector_i = vectors_by_layer[layer_i]
        for layer_j in layers:
            vector_j = vectors_by_layer[layer_j]
            value = torch.nn.functional.cosine_similarity(
                vector_i.unsqueeze(0),
                vector_j.unsqueeze(0),
            ).item()
            row.append(float(value))
        matrix.append(row)
    return {
        "layers": layers,
        "matrix": matrix,
    }


def _fit_state_for_selected_iteration(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    rfm_hyperparameters: dict[str, Any],
    selected_iteration: int,
    device: str,
) -> dict[str, Any]:
    total_iters = int(rfm_hyperparameters["iters"])
    if selected_iteration < 0 or selected_iteration > total_iters:
        raise SystemExit(
            "Bootstrap replay needs selected_iteration in "
            f"[0, {total_iters}], got {selected_iteration}."
        )
    model = LaplaceRFM(
        LaplaceRFMConfig(
            bandwidth=float(rfm_hyperparameters["bandwidth"]),
            reg=float(rfm_hyperparameters["reg"]),
            diag=bool(rfm_hyperparameters["diag"]),
            centering=bool(rfm_hyperparameters["centering"]),
            sample_batch_size=rfm_hyperparameters["sample_batch_size"],
            center_batch_size=rfm_hyperparameters["center_batch_size"],
            max_agop_samples=rfm_hyperparameters["max_agop_samples"],
            device=device,
            solver=str(rfm_hyperparameters["solver"]),
        )
    )
    train_targets = train_y.reshape(-1, 1).to(dtype=torch.float32)
    train_x = train_x.to(dtype=torch.float32)
    for iteration in range(total_iters):
        model.fit_predictor(train_x, train_targets)
        model.fit_M(train_x)
        if iteration == selected_iteration:
            return model.export_state()
    model.fit_predictor(train_x, train_targets)
    if selected_iteration == total_iters:
        return model.export_state()
    raise RuntimeError("Direction bootstrap replay missed the selected iteration.")


def _direction_bootstrap_summary(
    *,
    benchmark: str,
    layer: int,
    feature_key: str,
    preprocessing: dict[str, Any],
    rfm_hyperparameters: dict[str, Any],
    selected_iteration: int,
    vector_scale: str,
    reference_vector: torch.Tensor,
    reference_vector_checksum: str,
    train_data: SplitData,
    val_prompt_ids: list[int],
    test_prompt_ids: list[int],
    sign_convention: str,
    num_samples: int,
    seed: int,
    device: str,
    git_commit: str,
    model_id: str | None,
    model_revision: str | None,
    tokenizer_revision: str | None,
    random_seed: int,
    output_path: str,
) -> dict[str, Any] | None:
    if num_samples < 1:
        return None

    rng = np.random.default_rng(seed + layer)
    train_x = train_data.x[:, layer, :].detach().cpu()
    train_y = train_data.y.detach().cpu()
    num_examples = len(train_data.sample_ids)
    sample_rows: list[dict[str, Any]] = []
    cosine_values: list[float] = []

    for bootstrap_index in range(num_samples):
        sampled_indices = torch.tensor(
            rng.integers(0, num_examples, size=num_examples),
            dtype=torch.int64,
        )
        bootstrap_state = _fit_state_for_selected_iteration(
            train_x=train_x[sampled_indices],
            train_y=train_y[sampled_indices],
            rfm_hyperparameters=rfm_hyperparameters,
            selected_iteration=selected_iteration,
            device=device,
        )
        raw_vector, normalized_vector, extraction = _extract_signed_vector(
            state=bootstrap_state,
            vector_scale=vector_scale,
        )
        bootstrap_scores = (train_x @ normalized_vector).detach().cpu()
        score_sign, _ = _select_score_sign(bootstrap_scores, train_y)
        signed_vector = (normalized_vector * score_sign).detach().cpu()
        cosine = torch.nn.functional.cosine_similarity(
            signed_vector.unsqueeze(0),
            reference_vector.unsqueeze(0),
        ).item()
        cosine_values.append(float(cosine))
        sample_rows.append(
            {
                "bootstrap_index": int(bootstrap_index),
                "score_sign": float(score_sign),
                "cosine_to_reference": float(cosine),
                "raw_vector_norm": float((raw_vector * score_sign).norm().item()),
                "normalized_vector_checksum": tensor_checksum_hex(signed_vector),
                "selected_eigenvalue": extraction.get("selected_eigenvalue"),
            }
        )

    cosine_summary = _summary_stats(cosine_values)
    if cosine_summary is None:
        return None

    bootstrap_record = build_rfm_vector_direction_bootstrap_record(
        benchmark=benchmark,
        layer=layer,
        train_prompt_ids=[int(value) for value in train_data.sample_ids],
        val_prompt_ids=val_prompt_ids,
        test_prompt_ids=test_prompt_ids,
        feature_key=feature_key,
        preprocessing=preprocessing,
        rfm_hyperparameters=rfm_hyperparameters,
        vector_extraction_formula=DEFAULT_EXTRACTION_FORMULA,
        sign_convention=sign_convention,
        reference_vector_checksum=reference_vector_checksum,
        bootstrap={
            "sampling": "fit_train_with_replacement",
            "num_requested": int(num_samples),
            "num_completed": len(sample_rows),
            "seed": int(seed + layer),
            "selection_iteration": int(selected_iteration),
            "score_sign_selection_rule": "fit_train_pr_auc_then_roc_auc",
            "vector_scale": vector_scale,
        },
        cosine_to_reference=cosine_summary,
        git_commit=git_commit,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        random_seed=random_seed,
        output_path=output_path,
        samples=sample_rows,
    )
    return bootstrap_record


def main() -> None:
    args = _parse_args()
    rfm_run_dir = Path(args.rfm_run_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else rfm_run_dir / "vector_exports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    records_dir = out_dir / "artifacts"
    records_dir.mkdir(parents=True, exist_ok=True)

    best_layers_payload = _read_json(rfm_run_dir / "best_layers.json")
    run_config = _read_json(rfm_run_dir / "run_config.json")
    layers = _resolve_layers(best_layers_payload, args.layers)
    git_commit = current_git_commit(ROOT)

    summary_rows: list[dict[str, Any]] = []
    vectors_by_layer: dict[int, torch.Tensor] = {}

    for layer in layers:
        detector_record_path = rfm_run_dir / "artifacts" / f"layer_{layer:02d}_detector_record.json"
        detector_record = _read_json(detector_record_path)
        checkpoint_path = _resolve_record_path(
            str(detector_record["checkpoint_path"]),
            rfm_run_dir=rfm_run_dir,
        )
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint_payload["state"]
        feature_key = str(detector_record["feature_key"])
        preprocessing = dict(detector_record["preprocessing"])
        source_data_dir = Path(preprocessing["source_data_dir"])
        rows_by_key = _load_archive_rows_by_key(source_data_dir)

        prompt_ids = detector_record["prompt_ids"]
        train_data = _load_binary_split(
            data_dir=source_data_dir,
            source_split="train",
            feature_key=feature_key,
            sample_ids=[int(value) for value in prompt_ids["train"]],
            rows_by_key=rows_by_key,
            tail_threshold=0.5,
        )
        val_data = _load_binary_split(
            data_dir=source_data_dir,
            source_split="train",
            feature_key=feature_key,
            sample_ids=[int(value) for value in prompt_ids["val"]],
            rows_by_key=rows_by_key,
            tail_threshold=0.5,
        )
        test_data = _load_binary_split(
            data_dir=source_data_dir,
            source_split="test",
            feature_key=feature_key,
            sample_ids=[int(value) for value in prompt_ids["test"]],
            rows_by_key=rows_by_key,
            tail_threshold=0.5,
        )

        raw_vector, normalized_vector, extraction = _extract_signed_vector(
            state=state,
            vector_scale=args.vector_scale,
        )
        _, score_sign, projection_summary = _projection_summary(
            normalized_vector=normalized_vector,
            layer=layer,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            bootstrap_samples=args.projection_bootstrap_samples,
            bootstrap_seed=args.projection_bootstrap_seed,
        )
        signed_raw_vector = raw_vector * score_sign
        signed_normalized_vector = normalized_vector * score_sign
        vectors_by_layer[layer] = signed_normalized_vector

        vector_payload = {
            "schema_name": "prompt_profile_rfm_vector_payload.v1",
            "benchmark": detector_record["benchmark"],
            "layer": layer,
            "vector_extraction": extraction,
            "sign_convention": DEFAULT_SIGN_CONVENTION,
            "score_sign": float(score_sign),
            "raw_vector": signed_raw_vector,
            "normalized_vector": signed_normalized_vector,
            "source_checkpoint_path": str(checkpoint_path),
            "source_detector_artifact_hash": detector_record["artifact_sha256"],
        }
        vector_path = vectors_dir / f"layer_{layer:02d}_vector.pt"
        torch.save(vector_payload, vector_path)

        prompt_text_hashes = {
            "train": preprocessing["train_val_split"]["fit_train_prompt_text_sha256"],
            "val": preprocessing["train_val_split"]["val_prompt_text_sha256"],
            "test": preprocessing["train_val_split"]["test_prompt_text_sha256"],
        }
        projection_rule = {
            "score_sign": float(score_sign),
            "sign_selection_rule": projection_summary["sign_selection_rule"],
            "decision_threshold": projection_summary["decision_threshold"],
            "threshold_selection_rule": projection_summary["threshold_selection_rule"],
        }
        direction_bootstrap_summary = None
        if args.direction_bootstrap_samples > 0:
            bootstrap_record = _direction_bootstrap_summary(
                benchmark=str(detector_record["benchmark"]),
                layer=layer,
                feature_key=feature_key,
                preprocessing=preprocessing,
                rfm_hyperparameters=dict(detector_record["rfm_hyperparameters"]),
                selected_iteration=int(detector_record["selection"]["best_iteration"]),
                vector_scale=args.vector_scale,
                reference_vector=signed_normalized_vector,
                reference_vector_checksum=tensor_checksum_hex(signed_normalized_vector),
                train_data=train_data,
                val_prompt_ids=[int(value) for value in prompt_ids["val"]],
                test_prompt_ids=[int(value) for value in prompt_ids["test"]],
                sign_convention=DEFAULT_SIGN_CONVENTION,
                num_samples=args.direction_bootstrap_samples,
                seed=args.direction_bootstrap_seed,
                device=args.direction_bootstrap_device,
                git_commit=git_commit,
                model_id=detector_record.get("model_id"),
                model_revision=detector_record.get("model_revision"),
                tokenizer_revision=detector_record.get("tokenizer_revision"),
                random_seed=int(detector_record["random_seed"]),
                output_path=str(out_dir),
            )
            if bootstrap_record is not None:
                bootstrap_record_path = (
                    records_dir / f"layer_{layer:02d}_direction_bootstrap_record.json"
                )
                written_bootstrap_record = write_stage_artifact_record(
                    bootstrap_record_path,
                    bootstrap_record,
                )
                direction_bootstrap_summary = {
                    "record_path": str(bootstrap_record_path),
                    "sampling": written_bootstrap_record["bootstrap"]["sampling"],
                    "num_requested": written_bootstrap_record["bootstrap"]["num_requested"],
                    "num_completed": written_bootstrap_record["bootstrap"]["num_completed"],
                    "seed": written_bootstrap_record["bootstrap"]["seed"],
                    "selection_iteration": written_bootstrap_record["bootstrap"][
                        "selection_iteration"
                    ],
                    "cosine_to_reference": written_bootstrap_record["cosine_to_reference"],
                }
        bundle_record = build_rfm_vector_bundle_record(
            benchmark=str(detector_record["benchmark"]),
            layer=layer,
            train_prompt_ids=[int(value) for value in prompt_ids["train"]],
            val_prompt_ids=[int(value) for value in prompt_ids["val"]],
            test_prompt_ids=[int(value) for value in prompt_ids["test"]],
            feature_key=feature_key,
            preprocessing=preprocessing,
            rfm_hyperparameters=dict(detector_record["rfm_hyperparameters"]),
            vector_extraction=extraction,
            sign_convention=DEFAULT_SIGN_CONVENTION,
            raw_vector_norm=float(signed_raw_vector.norm().item()),
            raw_vector=signed_raw_vector,
            normalized_vector=signed_normalized_vector,
            git_commit=git_commit,
            model_id=detector_record.get("model_id"),
            model_revision=detector_record.get("model_revision"),
            tokenizer_revision=detector_record.get("tokenizer_revision"),
            random_seed=int(detector_record["random_seed"]),
            prompt_text_hashes=prompt_text_hashes,
            projection_rule=projection_rule,
            projection_metrics=projection_summary["splits"],
            direction_bootstrap=direction_bootstrap_summary,
            output_path=str(out_dir),
            source_checkpoint_path=str(checkpoint_path),
            source_detector_artifact_hash=detector_record["artifact_sha256"],
        )
        record_path = records_dir / f"layer_{layer:02d}_vector_record.json"
        written_record = write_stage_artifact_record(record_path, bundle_record)

        summary_rows.append(
            {
                "layer": layer,
                "selected_eigenvalue": extraction.get("selected_eigenvalue"),
                "selected_eigenvalue_share": extraction.get("selected_eigenvalue_share"),
                "raw_vector_norm": float(signed_raw_vector.norm().item()),
                "score_sign": float(score_sign),
                "decision_threshold": projection_summary["decision_threshold"],
                "train_pr_auc": projection_summary["splits"]["train"]["metrics"]["pr_auc"],
                "val_pr_auc": projection_summary["splits"]["val"]["metrics"]["pr_auc"],
                "test_pr_auc": projection_summary["splits"]["test"]["metrics"]["pr_auc"],
                "test_roc_auc": projection_summary["splits"]["test"]["metrics"]["roc_auc"],
                "test_positive_minus_negative_mean": projection_summary["splits"]["test"][
                    "positive_minus_negative_mean"
                ],
                "vector_checksum": written_record["normalized_vector_checksum"],
                "direction_bootstrap_mean_cosine": (
                    None
                    if direction_bootstrap_summary is None
                    else direction_bootstrap_summary["cosine_to_reference"]["mean"]
                ),
                "direction_bootstrap_low_cosine": (
                    None
                    if direction_bootstrap_summary is None
                    else direction_bootstrap_summary["cosine_to_reference"]["low"]
                ),
                "direction_bootstrap_high_cosine": (
                    None
                    if direction_bootstrap_summary is None
                    else direction_bootstrap_summary["cosine_to_reference"]["high"]
                ),
                "direction_bootstrap_num_completed": (
                    0
                    if direction_bootstrap_summary is None
                    else direction_bootstrap_summary["num_completed"]
                ),
                "record_path": str(record_path),
                "vector_path": str(vector_path),
            }
        )

    cross_layer = _cross_layer_cosines(vectors_by_layer)
    summary_rows.sort(
        key=lambda row: (
            _metric_float(row["val_pr_auc"], default=float("-inf")),
            _metric_float(row["test_pr_auc"], default=float("-inf")),
        ),
        reverse=True,
    )
    summary_payload = {
        "schema_name": "prompt_profile_rfm_vector_export_summary.v1",
        "benchmark": best_layers_payload["benchmark"],
        "rfm_run_dir": str(rfm_run_dir),
        "vector_export_dir": str(out_dir),
        "vector_extraction_formula": DEFAULT_EXTRACTION_FORMULA,
        "vector_scale": args.vector_scale,
        "projection_bootstrap_samples": int(args.projection_bootstrap_samples),
        "projection_bootstrap_seed": int(args.projection_bootstrap_seed),
        "direction_bootstrap_samples": int(args.direction_bootstrap_samples),
        "direction_bootstrap_seed": int(args.direction_bootstrap_seed),
        "direction_bootstrap_device": args.direction_bootstrap_device,
        "feature_key": best_layers_payload["feature_key"],
        "git_commit": git_commit,
        "source_git_commit": run_config.get("git_commit"),
        "cross_layer_cosine": cross_layer,
        "layers": summary_rows,
    }
    _write_json(out_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
