#!/usr/bin/env python3
"""Materialize the March prompt-profile stage object as a binary dataset on disk."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_probe.dataloader import ActivationDataset, read_manifest, resolve_sample_shape
from loop_probe.prompt_profile_rfm_stage_registry import (
    active_stage_datasets,
    get_stage_dataset,
    validate_stage_dataset,
)
from loop_probe.stage_artifacts import stable_json_sha256


DEFAULT_STAGE_LABEL = "majority_s_0.5"


@dataclass(frozen=True)
class SplitData:
    x: torch.Tensor
    y: torch.Tensor
    sample_ids: list[int]
    prompts: list[str]
    rows: list[dict[str, Any]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(dataset.key for dataset in active_stage_datasets()),
    )
    parser.add_argument("--archive-source-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--balance-train",
        choices=("none", "downsample"),
        default="downsample",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
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


def _majority_tail_label(row: dict[str, Any], *, threshold: float) -> float:
    num_rollouts = int(row.get("num_rollouts", 0))
    if num_rollouts < 1:
        raise SystemExit("Prompt rollout archive row is missing num_rollouts.")
    tail_hits = row.get("tail_hit_count")
    if isinstance(tail_hits, int):
        return float(int(tail_hits > (num_rollouts / 2.0)))
    majority_tail = row.get("majority_tail")
    if isinstance(majority_tail, (int, float)):
        return float(int(majority_tail))
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise SystemExit("Prompt rollout archive row is missing rollout entries.")
    hits = 0
    for rollout in rollouts:
        relative_length = rollout.get("relative_length")
        if not isinstance(relative_length, (int, float)):
            raise SystemExit("Rollout row is missing relative_length.")
        if float(relative_length) >= threshold:
            hits += 1
    return float(int(hits > (len(rollouts) / 2.0)))


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
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        row = rows_by_key.get((split, sample_id))
        if row is None:
            raise SystemExit(f"Missing prompt rollout archive row for {split}:{sample_id}.")
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise SystemExit(f"Missing prompt text for {split}:{sample_id}.")
        prompts.append(prompt)
        labels.append(_majority_tail_label(row, threshold=tail_threshold))
        rows.append(dict(row))
    return SplitData(
        x=dataset.x.detach().to(dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.float32),
        sample_ids=sample_ids,
        prompts=prompts,
        rows=rows,
    )


def _subset_split(data: SplitData, indices: list[int]) -> SplitData:
    return SplitData(
        x=data.x[indices].clone(),
        y=data.y[indices].clone(),
        sample_ids=[data.sample_ids[index] for index in indices],
        prompts=[data.prompts[index] for index in indices],
        rows=[dict(data.rows[index]) for index in indices],
    )


def _split_indices_by_label(labels: torch.Tensor) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(labels.tolist()):
        label_i = int(round(float(label)))
        if label_i not in (0, 1):
            raise SystemExit("Expected binary labels for stage materialization.")
        groups[label_i].append(index)
    return groups


def _train_val_indices(
    labels: torch.Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
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
    pos = list(groups[1])
    neg = list(groups[0])
    if not pos or not neg:
        raise SystemExit("Balanced training requires both classes in the train subset.")
    keep_per_class = min(len(pos), len(neg))
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    selected = pos[:keep_per_class] + neg[:keep_per_class]
    selected.sort()
    return selected


def _split_payload(split_name: str, shard_relpath: str, data: SplitData) -> dict[str, Any]:
    num_positive = int(data.y.sum().item())
    return {
        "name": split_name,
        "num_samples": len(data.sample_ids),
        "num_positive": num_positive,
        "num_negative": len(data.sample_ids) - num_positive,
        "shards": [shard_relpath],
    }


def _sample_ids_sha256(sample_ids: list[int]) -> str:
    return stable_json_sha256(sample_ids)


def _prompt_text_sha256(prompts: list[str]) -> str:
    return stable_json_sha256(prompts)


def _write_split_shard(out_dir: Path, split_name: str, data: SplitData) -> str:
    relpath = Path("data") / split_name / "shard-00000.pt"
    fullpath = out_dir / relpath
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": data.x.to(dtype=torch.float32),
            "y": data.y.to(dtype=torch.float32),
            "sample_ids": torch.tensor(data.sample_ids, dtype=torch.int64),
        },
        fullpath,
    )
    return relpath.as_posix()


def main() -> None:
    args = _parse_args()
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": fit_train,
        "val": val_split,
        "test": natural_test,
    }
    feature_splits: dict[str, Any] = {}
    for split_name, split_data in split_map.items():
        shard_relpath = _write_split_shard(out_dir, split_name, split_data)
        feature_splits[split_name] = _split_payload(split_name, shard_relpath, split_data)
        _write_jsonl(
            out_dir / "diagnostics" / f"{split_name}_prompt_rows.jsonl",
            split_data.rows,
        )

    materialization = {
        "benchmark": benchmark.key,
        "display_name": benchmark.display_name,
        "archive_source_root": args.archive_source_root,
        "archive_data_dir": str(data_dir),
        "prompt_rollout_archive_file": manifest.get("prompt_rollout_archive_file"),
        "feature_key": feature_key,
        "sample_shape": list(sample_shape),
        "stage_label_name": DEFAULT_STAGE_LABEL,
        "source_target_name": benchmark.source_target_name,
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "tokenizer_name_or_path": manifest.get("tokenizer_name_or_path"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "val_fraction": float(args.val_fraction),
        "balance_train": args.balance_train,
        "seed": int(args.seed),
        "source_train_count": len(natural_train.sample_ids),
        "source_test_count": len(natural_test.sample_ids),
    }
    _write_json(
        out_dir / "manifest.json",
        {
            "default_feature_key": feature_key,
            "feature_views": {
                feature_key: {
                    "input_dim": int(sample_shape[-1]),
                    "sample_shape": list(sample_shape),
                    **feature_splits,
                }
            },
            "target_spec": {
                "kind": "binary",
                "name": DEFAULT_STAGE_LABEL,
                "horizon": None,
            },
            "stage_materialization": materialization,
        },
    )
    _write_json(
        out_dir / "split_manifest.json",
        {
            "benchmark": benchmark.key,
            "prompt_ids": {
                split_name: split_data.sample_ids
                for split_name, split_data in split_map.items()
            },
            "prompt_id_hashes": {
                split_name: _sample_ids_sha256(split_data.sample_ids)
                for split_name, split_data in split_map.items()
            },
            "prompt_text_hashes": {
                split_name: _prompt_text_sha256(split_data.prompts)
                for split_name, split_data in split_map.items()
            },
            "num_positive": {
                split_name: int(split_data.y.sum().item())
                for split_name, split_data in split_map.items()
            },
            "materialization": materialization,
        },
    )

    print(
        json.dumps(
            {
                "benchmark": benchmark.key,
                "out_dir": str(out_dir),
                "train_n": len(fit_train.sample_ids),
                "train_pos": int(fit_train.y.sum().item()),
                "val_n": len(val_split.sample_ids),
                "val_pos": int(val_split.y.sum().item()),
                "test_n": len(natural_test.sample_ids),
                "test_pos": int(natural_test.y.sum().item()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
