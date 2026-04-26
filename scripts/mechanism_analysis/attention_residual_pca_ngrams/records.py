from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BUCKET_CORRECT_IN_BUDGET = "loop_correct_in_budget"
BUCKET_WRONG_MAX_LENGTH = "loop_wrong_max_length"
BUCKETS = (BUCKET_CORRECT_IN_BUDGET, BUCKET_WRONG_MAX_LENGTH)

VECTOR_ATTENTION_WRITE = "attention_write"
VECTOR_POST_ATTENTION_RESIDUAL = "post_attention_residual"
VECTOR_FINAL_HIDDEN = "final_hidden"
VECTOR_LABELS = {
    VECTOR_ATTENTION_WRITE: "A",
    VECTOR_POST_ATTENTION_RESIDUAL: "P",
    VECTOR_FINAL_HIDDEN: "H",
}


@dataclass(frozen=True)
class BundleSpec:
    bundle_path: Path
    sidecar_path: Path | None
    dataset: str
    dataset_key: str
    thinking_mode: str
    task_kind: str
    model_id: str | None
    loop_n: int | None
    loop_k: int | None


@dataclass(frozen=True)
class SelectedRollout:
    selection_id: str
    bundle_index: int
    source_bundle: str
    source_sidecar: str | None
    dataset: str
    dataset_key: str
    thinking_mode: str
    task_kind: str
    split: str
    sample_id: str
    record_id: str
    rollout_index: int
    bucket: str
    correct: bool
    max_length_hit: bool
    finish_reason: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    prompt_token_count: int
    completion_token_count: int
    replay_token_count: int
    loop_n: int
    loop_k: int
    ngram_token_ids: list[int]
    saved_ngram_start_positions: list[int]
    rescan_ngram_start_positions: list[int]
    boundary_positions: list[int]
    first_loop_prefix_length: int | None
    grading: dict[str, Any]


@dataclass(frozen=True)
class CapturedVectors:
    vectors: dict[str, np.ndarray]
    repeat_probabilities: list[float]
    repeat_logit_margins: list[float]
    repeat_token_logits: list[float]
    top_token_ids: list[int]
    top_token_logits: list[float]


@dataclass(frozen=True)
class PcaResult:
    coords: np.ndarray
    explained_variance_ratio: tuple[float, float]
    total_variance: float
    n_points: int
    n_features: int
