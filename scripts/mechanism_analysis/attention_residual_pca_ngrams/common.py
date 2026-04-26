from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True, default=json_default)
            handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def sharded_path(
    out_dir: Path,
    filename: str,
    *,
    num_shards: int,
    shard_index: int,
) -> Path:
    if num_shards == 1:
        return out_dir / filename
    stem, suffix = os.path.splitext(filename)
    if filename.endswith(".jsonl"):
        stem = filename[: -len(".jsonl")]
        suffix = ".jsonl"
    return out_dir / f"{stem}.shard_{shard_index:03d}{suffix}"


def selection_summary_fieldnames() -> list[str]:
    return [
        "dataset",
        "dataset_key",
        "thinking_mode",
        "task_kind",
        "source_bundle",
        "bucket",
        "eligible_rollouts",
        "selected_rollouts",
        "requested_rollouts",
        "underfilled",
        "total_rollouts",
        "loop_rollouts",
        "other_loop_rollouts",
        "missing_grading",
        "missing_completion_token_ids",
        "missing_prompt_token_ids",
        "missing_loop_trigger",
        "invalid_loop_trigger",
        "ngram_rescan_miss",
        "boundary_mapping_invalid",
    ]


def pca_summary_fieldnames() -> list[str]:
    return [
        "pca_scope",
        "vector_name",
        "vector_label",
        "dataset",
        "dataset_key",
        "thinking_mode",
        "bucket",
        "selection_id",
        "sample_id",
        "rollout_index",
        "n_points",
        "n_features",
        "explained_variance_pc1",
        "explained_variance_pc2",
        "explained_variance_top2",
        "total_variance",
        "figure_path",
    ]
