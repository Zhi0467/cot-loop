import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone

import torch


def save_split_shards(
    out_dir: str,
    split: str,
    features: torch.Tensor,
    labels: Sequence[int],
    sample_ids: Sequence[int],
    *,
    shard_size: int,
) -> dict[str, object]:
    if shard_size < 1:
        raise SystemExit("--shard-size must be >= 1.")

    if features.ndim != 2:
        raise SystemExit(f"Expected 2D features tensor, got shape {tuple(features.shape)}")

    num_samples = features.size(0)
    if num_samples != len(labels) or num_samples != len(sample_ids):
        raise SystemExit("Feature/label/sample_id length mismatch.")

    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    x = features.to(dtype=torch.float16).cpu()
    y = torch.tensor(labels, dtype=torch.uint8)
    sid = torch.tensor(sample_ids, dtype=torch.int64)

    shard_files: list[str] = []
    for start in range(0, num_samples, shard_size):
        end = min(start + shard_size, num_samples)
        shard_name = f"shard-{start // shard_size:05d}.pt"
        shard_path = os.path.join(split_dir, shard_name)
        torch.save(
            {
                "x": x[start:end],
                "y": y[start:end],
                "sample_ids": sid[start:end],
            },
            shard_path,
        )
        shard_files.append(os.path.relpath(shard_path, out_dir))

    return {
        "num_samples": num_samples,
        "num_positive": int(sum(labels)),
        "num_negative": int(num_samples - sum(labels)),
        "shards": shard_files,
    }


def write_manifest(out_dir: str, payload: dict[str, object]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    body = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, sort_keys=True)
        f.write("\n")
