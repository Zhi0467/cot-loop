from __future__ import annotations

import csv
import json


def resolve_sample_id(value: object, default: int) -> int:
    try:
        sample_id = int(value)
    except Exception:
        sample_id = default
    if sample_id < 0:
        return default
    return sample_id


def load_local_rows(path: str) -> list[dict[str, object]]:
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
