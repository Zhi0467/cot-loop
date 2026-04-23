#!/usr/bin/env python3
"""Summarize one prompt-profile follow-up bundle into a compact JSON artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-profile-root",
        required=True,
        help="Prompt-profile projection root containing export/ and data/ artifacts.",
    )
    parser.add_argument(
        "--summary-path",
        required=True,
        help="Where to write the compact JSON summary.",
    )
    parser.add_argument(
        "--majority-last",
        required=True,
        help="Last-layer majority_s_0.5 training output directory.",
    )
    parser.add_argument(
        "--majority-ensemble",
        required=True,
        help="Ensemble majority_s_0.5 training output directory.",
    )
    parser.add_argument(
        "--direct-head",
        action="append",
        nargs=3,
        metavar=("TARGET_NAME", "LAST_DIR", "ENSEMBLE_DIR"),
        default=[],
        help="Direct-head outputs to summarize.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k][0]] = avg
        i = j
    return out


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx = _ranks(xs)
    ry = _ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _read_test_rows(prompt_profile_root: Path) -> list[dict[str, str]]:
    prompt_projection_csv = prompt_profile_root / "export" / "prompt_projection.csv"
    with prompt_projection_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get("split") == "test"]


def _load_metrics(run_dir: Path, stem: str) -> dict[str, Any] | None:
    path = run_dir / f"{stem}.json"
    if path.exists():
        return _load_json(path)
    return None


def _run_summary(run_dir: Path) -> dict[str, Any]:
    best_loss = _load_metrics(run_dir, "best_loss_metrics")
    best_rank = _load_metrics(run_dir, "best_rank_metrics")
    if best_loss is None:
        best_loss = _load_metrics(run_dir, "best_metrics")
    return {
        "path": str(run_dir),
        "best_loss": best_loss,
        "best_rank": best_rank,
    }


def _target_spearman(rows: list[dict[str, str]], target_name: str) -> float | None:
    if not rows or target_name not in rows[0]:
        return None
    prompt_lengths = [float(row["prompt_token_count"]) for row in rows]
    target_values = [float(row[target_name]) for row in rows]
    return _spearman(prompt_lengths, target_values)


def main() -> None:
    args = _parse_args()
    prompt_profile_root = Path(args.prompt_profile_root)
    summary_path = Path(args.summary_path)
    rows = _read_test_rows(prompt_profile_root)
    projection_summary = _load_json(prompt_profile_root / "export" / "projection_summary.json")

    payload: dict[str, Any] = {
        "source_prompt_profile_root": str(prompt_profile_root),
        "projection": {
            "prompt_length_baseline_test": projection_summary["leakage_baselines"]["prompt_token_count"]["test"],
            "effective_budget_baseline_test": projection_summary["leakage_baselines"]["effective_max_tokens"]["test"],
        },
        "majority_s_0.5": {
            "prompt_length_spearman_test": _target_spearman(rows, "majority_s_0.5"),
            "last_layer": _run_summary(Path(args.majority_last)),
            "ensemble": _run_summary(Path(args.majority_ensemble)),
        },
        "direct_heads": {},
    }

    for target_name, last_dir, ensemble_dir in args.direct_head:
        payload["direct_heads"][target_name] = {
            "prompt_length_spearman_test": _target_spearman(rows, target_name),
            "last_layer": _run_summary(Path(last_dir)),
            "ensemble": _run_summary(Path(ensemble_dir)),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(summary_path)


if __name__ == "__main__":
    main()
