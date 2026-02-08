import json
import os

import torch
from torch.utils.data import DataLoader, Dataset


class ActivationDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise SystemExit(f"Split '{split}' not found in manifest: {manifest_path}")

        split_info = manifest[split]
        shard_paths = split_info.get("shards", [])
        if not shard_paths:
            raise SystemExit(f"No shard files listed for split '{split}'.")

        xs = []
        ys = []
        sample_ids = []
        for rel_path in shard_paths:
            full_path = os.path.join(data_dir, rel_path)
            shard = torch.load(full_path, map_location="cpu")
            xs.append(shard["x"].to(dtype=torch.float32))
            ys.append(shard["y"].to(dtype=torch.float32))
            sample_ids.append(shard["sample_ids"].to(dtype=torch.int64))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)
        self.sample_ids = torch.cat(sample_ids, dim=0)
        if self.x.size(0) != self.y.size(0):
            raise SystemExit(
                f"Split '{split}' has mismatched x/y lengths: "
                f"{self.x.size(0)} vs {self.y.size(0)}"
            )

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_dataloader(
    data_dir: str,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = ActivationDataset(data_dir=data_dir, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def read_manifest(data_dir: str) -> dict[str, object]:
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
