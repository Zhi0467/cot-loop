"""Artifact-record helpers for the prompt-profile RFM steering stage."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_sha256(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def tensor_checksum_hex(vector: Any) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is available in project envs
        raise SystemExit("torch is required to checksum vector artifacts.") from exc

    if isinstance(vector, torch.Tensor):
        tensor = vector.detach().to(dtype=torch.float32, device="cpu").contiguous()
    else:
        tensor = torch.tensor(vector, dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def current_git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def artifact_body_sha256(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key != "artifact_sha256"
    }
    return stable_json_sha256(body)


def write_stage_artifact_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("created_at", utc_now_iso())
    body["artifact_sha256"] = artifact_body_sha256(body)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return body


def build_rfm_vector_bundle_record(
    *,
    benchmark: str,
    layer: int,
    train_prompt_ids: list[int],
    val_prompt_ids: list[int],
    test_prompt_ids: list[int],
    feature_key: str,
    preprocessing: dict[str, Any],
    rfm_hyperparameters: dict[str, Any],
    vector_extraction: dict[str, Any],
    sign_convention: str,
    raw_vector_norm: float,
    raw_vector: Any,
    normalized_vector: Any,
    git_commit: str,
    model_id: str | None,
    model_revision: str | None,
    tokenizer_revision: str | None,
    random_seed: int,
    prompt_text_hashes: dict[str, str] | None = None,
    projection_rule: dict[str, Any] | None = None,
    projection_metrics: dict[str, Any] | None = None,
    output_path: str | None = None,
    source_checkpoint_path: str | None = None,
    source_detector_artifact_hash: str | None = None,
) -> dict[str, Any]:
    prompt_ids = {
        "train": train_prompt_ids,
        "val": val_prompt_ids,
        "test": test_prompt_ids,
    }
    prompt_id_hashes = {
        split_name: stable_json_sha256(prompt_ids[split_name])
        for split_name in prompt_ids
    }
    record = {
        "schema_name": "prompt_profile_rfm_vector_bundle.v1",
        "benchmark": benchmark,
        "layer": layer,
        "prompt_ids": prompt_ids,
        "prompt_id_hashes": prompt_id_hashes,
        "feature_key": feature_key,
        "preprocessing": preprocessing,
        "rfm_hyperparameters": rfm_hyperparameters,
        "vector_extraction": vector_extraction,
        "sign_convention": sign_convention,
        "raw_vector_norm": raw_vector_norm,
        "raw_vector_checksum": tensor_checksum_hex(raw_vector),
        "normalized_vector_checksum": tensor_checksum_hex(normalized_vector),
        "git_commit": git_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "random_seed": random_seed,
    }
    if prompt_text_hashes is not None:
        record["prompt_text_hashes"] = prompt_text_hashes
    if projection_rule is not None:
        record["projection_rule"] = projection_rule
    if projection_metrics is not None:
        record["projection_metrics"] = projection_metrics
    if output_path is not None:
        record["output_path"] = output_path
    if source_checkpoint_path is not None:
        record["source_checkpoint_path"] = source_checkpoint_path
    if source_detector_artifact_hash is not None:
        record["source_detector_artifact_hash"] = source_detector_artifact_hash
    return record


def build_rfm_detector_run_record(
    *,
    benchmark: str,
    layer: int,
    train_prompt_ids: list[int],
    val_prompt_ids: list[int],
    test_prompt_ids: list[int],
    feature_key: str,
    preprocessing: dict[str, Any],
    rfm_hyperparameters: dict[str, Any],
    selection: dict[str, Any],
    sign_convention: str,
    score_sign: float,
    decision_threshold: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    git_commit: str,
    model_id: str | None,
    model_revision: str | None,
    tokenizer_revision: str | None,
    random_seed: int,
    output_path: str,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    prompt_ids = {
        "train": train_prompt_ids,
        "val": val_prompt_ids,
        "test": test_prompt_ids,
    }
    prompt_id_hashes = {
        split_name: stable_json_sha256(prompt_ids[split_name])
        for split_name in prompt_ids
    }
    record = {
        "schema_name": "prompt_profile_rfm_detector_run.v1",
        "benchmark": benchmark,
        "layer": layer,
        "prompt_ids": prompt_ids,
        "prompt_id_hashes": prompt_id_hashes,
        "feature_key": feature_key,
        "preprocessing": preprocessing,
        "rfm_hyperparameters": rfm_hyperparameters,
        "selection": selection,
        "sign_convention": sign_convention,
        "score_sign": float(score_sign),
        "decision_threshold": float(decision_threshold),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "git_commit": git_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "random_seed": random_seed,
        "output_path": output_path,
        "checkpoint_path": checkpoint_path,
    }
    return record


def build_steering_run_record(
    *,
    condition_name: str,
    vector_artifact_hash: str,
    hook_site: str,
    t: float,
    seeds: list[int],
    prompt_ids: list[int],
    generation_config: dict[str, Any],
    grader_version: str,
    output_path: str,
) -> dict[str, Any]:
    record = {
        "schema_name": "prompt_profile_rfm_steering_run.v1",
        "condition_name": condition_name,
        "vector_artifact_hash": vector_artifact_hash,
        "hook_site": hook_site,
        "t": t,
        "seeds": seeds,
        "prompt_ids": prompt_ids,
        "generation_config": generation_config,
        "grader_version": grader_version,
        "output_path": output_path,
    }
    return record
