#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


LAYER_METRICS = [
    "mean_prev_loop_mass",
    "mean_last_prev_loop_mass",
    "mean_prompt_mass",
    "mean_current_trigger_mass",
    "mean_recent_nonloop_mass",
    "mean_other_completion_mass",
    "top1_fraction_previous_loop",
    "top1_fraction_last_previous_loop",
    "top1_fraction_prompt",
    "top1_fraction_current_trigger",
    "top1_fraction_recent_nonloop",
    "top1_fraction_other_completion",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def _sample_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["dataset"]),
        int(row["sample_id"]),
        int(row["rollout_index"]),
    )


def _normalized_analysis_config(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("shard_index", None)
    return normalized


def _weighted_merge_layer_means(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, float | int | str]] = {}
    for shard_dir in shard_dirs:
        layer_means_path = shard_dir / "attention_layer_means.csv"
        if not layer_means_path.is_file():
            continue
        with layer_means_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset = str(row["dataset"])
                layer = int(row["layer"])
                num_rows = int(row["num_rows"])
                key = (dataset, layer)
                bucket = merged.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "layer": layer,
                        "num_rows": 0,
                        "query_position_mode": str(row["query_position_mode"]),
                    },
                )
                if str(bucket["query_position_mode"]) != str(row["query_position_mode"]):
                    raise SystemExit(
                        f"query_position_mode mismatch for dataset={dataset}, "
                        f"layer={layer}: {bucket['query_position_mode']} vs "
                        f"{row['query_position_mode']}."
                    )
                bucket["num_rows"] = int(bucket["num_rows"]) + num_rows
                for metric in LAYER_METRICS:
                    bucket[metric] = float(bucket.get(metric, 0.0)) + (
                        float(row[metric]) * num_rows
                    )

    combined_rows: list[dict[str, Any]] = []
    for (dataset, layer), bucket in sorted(merged.items()):
        num_rows = int(bucket["num_rows"])
        combined_rows.append(
            {
                "dataset": dataset,
                "layer": layer,
                "num_rows": num_rows,
                "query_position_mode": str(bucket["query_position_mode"]),
                **{
                    metric: (float(bucket[metric]) / num_rows if num_rows else 0.0)
                    for metric in LAYER_METRICS
                },
            }
        )
    return combined_rows


def _weighted_merge_attention_summary(shard_dirs: list[Path]) -> dict[str, Any]:
    merged: dict[str, dict[str, float | int]] = {}
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "attention_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        for dataset, payload in summary.items():
            num_rows = int(payload["num_selected_rows"])
            bucket = merged.setdefault(
                dataset,
                {
                    "num_selected_rows": 0,
                    "summary_layer": int(payload["summary_layer"]),
                    "query_position_mode": str(payload["query_position_mode"]),
                },
            )
            if int(bucket["summary_layer"]) != int(payload["summary_layer"]):
                raise SystemExit(
                    f"Summary layer mismatch for dataset {dataset}: "
                    f"{bucket['summary_layer']} vs {payload['summary_layer']}."
                )
            if str(bucket["query_position_mode"]) != str(payload["query_position_mode"]):
                raise SystemExit(
                    f"Summary query_position_mode mismatch for dataset {dataset}: "
                    f"{bucket['query_position_mode']} vs {payload['query_position_mode']}."
                )
            bucket["num_selected_rows"] = int(bucket["num_selected_rows"]) + num_rows
            for metric in LAYER_METRICS:
                bucket[metric] = float(bucket.get(metric, 0.0)) + (
                    float(payload[metric]) * num_rows
                )

    combined: dict[str, Any] = {}
    for dataset, bucket in sorted(merged.items()):
        num_rows = int(bucket["num_selected_rows"])
        combined[dataset] = {
            "num_selected_rows": num_rows,
            "summary_layer": int(bucket["summary_layer"]),
            "query_position_mode": str(bucket["query_position_mode"]),
            **{
                metric: (float(bucket[metric]) / num_rows if num_rows else 0.0)
                for metric in LAYER_METRICS
            },
        }
    return combined


def main() -> None:
    args = _parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_root = Path(args.shard_root)
    shard_dirs = [shard_root / f"shard_{idx}" for idx in range(args.num_shards)]
    missing_shards = [str(path) for path in shard_dirs if not path.is_dir()]
    if missing_shards:
        raise SystemExit(f"Missing shard output directories: {', '.join(missing_shards)}")

    reference_reconstruction = _read_json(shard_dirs[0] / "reconstruction_summary.json")
    reference_analysis_config = _normalized_analysis_config(
        _read_json(shard_dirs[0] / "analysis_config.json")
    )
    for shard_dir in shard_dirs[1:]:
        shard_reconstruction = _read_json(shard_dir / "reconstruction_summary.json")
        if shard_reconstruction != reference_reconstruction:
            raise SystemExit(
                f"Reconstruction summary mismatch in {shard_dir}; shard outputs are inconsistent."
            )
        shard_analysis_config = _normalized_analysis_config(
            _read_json(shard_dir / "analysis_config.json")
        )
        if shard_analysis_config != reference_analysis_config:
            raise SystemExit(
                f"Analysis config mismatch in {shard_dir}; shard outputs are inconsistent."
            )

    selected_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        shard_manifests.append(_read_json(shard_dir / "shard_manifest.json"))
        selected_rows.extend(list(_iter_jsonl(shard_dir / "selected_rows.jsonl")))
        attention_path = shard_dir / "attention_per_sample.json"
        if attention_path.is_file():
            attention_rows.extend(_read_json(attention_path))

    selected_rows.sort(key=_sample_sort_key)
    attention_rows.sort(key=_sample_sort_key)

    _write_json(out_dir / "reconstruction_summary.json", reference_reconstruction)
    _write_json(out_dir / "analysis_config.json", reference_analysis_config)
    _write_json(out_dir / "shard_manifest.json", shard_manifests)
    _write_jsonl(out_dir / "selected_rows.jsonl", selected_rows)
    _write_json(out_dir / "attention_per_sample.json", attention_rows)

    layer_means = _weighted_merge_layer_means(shard_dirs)
    if layer_means:
        with (out_dir / "attention_layer_means.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_means[0].keys()))
            writer.writeheader()
            writer.writerows(layer_means)

    _write_json(
        out_dir / "attention_summary.json",
        _weighted_merge_attention_summary(shard_dirs),
    )


if __name__ == "__main__":
    main()
