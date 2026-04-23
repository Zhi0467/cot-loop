#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BUCKETS = ("prompt", "previous_loop", "current_trigger", "other_completion")
MASS_KEYS = {
    "prompt": "mean_prompt_mass",
    "previous_loop": "mean_prev_loop_mass",
    "current_trigger": "mean_current_trigger_mass",
    "other_completion": "mean_other_completion_mass",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input in the form label=PATH, where PATH is a trigger-attention bundle dir.",
    )
    parser.add_argument("--out-overall-csv", required=True)
    parser.add_argument("--out-dataset-csv", required=True)
    parser.add_argument("--out-layer-csv", required=True)
    return parser.parse_args()


def _parse_label_and_path(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"Expected label=PATH, got {item!r}.")
    label, path_str = item.split("=", 1)
    path = Path(path_str)
    if not path.is_dir():
        raise SystemExit(f"Missing bundle directory: {path}")
    return label, path


def _load_selected_rows(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["dataset"]), int(row["sample_id"]), int(row["rollout_index"]))
            rows[key] = row
    return rows


def _region_counts(row: dict[str, Any], query_position_mode: str) -> dict[str, int]:
    prompt_len = int(row["prompt_token_count"])
    loop_trigger = row["loop_trigger"]
    ngram_len = len(loop_trigger["ngram_token_ids"])
    current_start = prompt_len + int(loop_trigger["trigger_start"])
    current_end = prompt_len + int(loop_trigger["trigger_end"])

    previous_positions: set[int] = set()
    for start in loop_trigger["ngram_start_positions"][:-1]:
        global_start = prompt_len + int(start)
        previous_positions.update(range(global_start, global_start + ngram_len))
    current_positions = set(range(current_start, current_end + 1))
    previous_positions -= current_positions

    if query_position_mode == "trigger_end":
        query_position = current_end
    elif query_position_mode == "pre_trigger_start":
        query_position = current_start - 1
    elif query_position_mode == "trigger_start":
        query_position = current_start
    else:
        raise SystemExit(f"Unsupported query position mode: {query_position_mode}")

    accessible_positions = set(range(query_position + 1))
    prompt_positions = set(range(prompt_len))
    previous_positions &= accessible_positions
    current_positions &= accessible_positions
    prompt_positions &= accessible_positions
    other_completion_positions = (
        accessible_positions - prompt_positions - previous_positions - current_positions
    )

    return {
        "prompt": len(prompt_positions),
        "previous_loop": len(previous_positions),
        "current_trigger": len(current_positions),
        "other_completion": len(other_completion_positions),
        "total": len(accessible_positions),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = _parse_args()
    parsed_inputs = [_parse_label_and_path(item) for item in args.input]

    overall_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for label, bundle_dir in parsed_inputs:
        selected_rows = _load_selected_rows(bundle_dir / "selected_rows.jsonl")
        samples = json.loads((bundle_dir / "attention_per_sample.json").read_text(encoding="utf-8"))
        if not samples:
            raise SystemExit(f"No attention samples found under {bundle_dir}.")

        query_position_mode = str(
            samples[0]["layer_summaries"][0].get("query_position_mode")
            or json.loads((bundle_dir / "analysis_config.json").read_text(encoding="utf-8"))[
                "query_position_mode"
            ]
        )

        per_sample_counts: dict[tuple[str, int, int], dict[str, float]] = {}
        for sample in samples:
            key = (str(sample["dataset"]), int(sample["sample_id"]), int(sample["rollout_index"]))
            counts = _region_counts(selected_rows[key], query_position_mode)
            total = float(counts["total"])
            per_sample_counts[key] = {
                bucket: float(counts[bucket]) / total for bucket in BUCKETS
            }

        layer_count = len(samples[0]["layer_summaries"])
        overall_shares = {
            bucket: sum(per_sample_counts[key][bucket] for key in per_sample_counts)
            / float(len(per_sample_counts))
            for bucket in BUCKETS
        }

        for layer_index in range(layer_count):
            layer_masses = {
                bucket: sum(
                    float(sample["layer_summaries"][layer_index][MASS_KEYS[bucket]])
                    for sample in samples
                )
                / float(len(samples))
                for bucket in BUCKETS
            }
            layer_rows.append(
                {
                    "label": label,
                    "layer": layer_index,
                    **{
                        f"{bucket}_token_share": overall_shares[bucket] for bucket in BUCKETS
                    },
                    **{f"{bucket}_mass": layer_masses[bucket] for bucket in BUCKETS},
                    **{
                        f"{bucket}_enrichment": _safe_div(
                            layer_masses[bucket],
                            overall_shares[bucket],
                        )
                        for bucket in BUCKETS
                    },
                }
            )

        final_layer = samples[0]["layer_summaries"][-1]["layer"]
        final_layer_samples = [sample["layer_summaries"][-1] for sample in samples]
        final_masses = {
            bucket: sum(float(layer[MASS_KEYS[bucket]]) for layer in final_layer_samples)
            / float(len(final_layer_samples))
            for bucket in BUCKETS
        }
        overall_rows.append(
            {
                "label": label,
                "query_position_mode": query_position_mode,
                "final_layer": final_layer,
                "rows": len(samples),
                **{f"{bucket}_token_share": overall_shares[bucket] for bucket in BUCKETS},
                **{f"{bucket}_mass": final_masses[bucket] for bucket in BUCKETS},
                **{
                    f"{bucket}_enrichment": _safe_div(
                        final_masses[bucket],
                        overall_shares[bucket],
                    )
                    for bucket in BUCKETS
                },
            }
        )

        dataset_to_layers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            dataset_to_layers[str(sample["dataset"])].append(sample["layer_summaries"][-1])

        dataset_to_keys: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for key in per_sample_counts:
            dataset_to_keys[key[0]].append(key)

        for dataset in sorted(dataset_to_layers):
            dataset_keys = dataset_to_keys[dataset]
            dataset_shares = {
                bucket: sum(per_sample_counts[key][bucket] for key in dataset_keys)
                / float(len(dataset_keys))
                for bucket in BUCKETS
            }
            dataset_masses = {
                bucket: sum(float(layer[MASS_KEYS[bucket]]) for layer in dataset_to_layers[dataset])
                / float(len(dataset_to_layers[dataset]))
                for bucket in BUCKETS
            }
            dataset_rows.append(
                {
                    "label": label,
                    "query_position_mode": query_position_mode,
                    "dataset": dataset,
                    "final_layer": final_layer,
                    "rows": len(dataset_to_layers[dataset]),
                    **{f"{bucket}_token_share": dataset_shares[bucket] for bucket in BUCKETS},
                    **{f"{bucket}_mass": dataset_masses[bucket] for bucket in BUCKETS},
                    **{
                        f"{bucket}_enrichment": _safe_div(
                            dataset_masses[bucket],
                            dataset_shares[bucket],
                        )
                        for bucket in BUCKETS
                    },
                }
            )

    def _write_csv(path_str: str, rows: list[dict[str, Any]]) -> None:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise SystemExit(f"No rows to write for {path}.")
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(args.out_overall_csv, overall_rows)
    _write_csv(args.out_dataset_csv, dataset_rows)
    _write_csv(args.out_layer_csv, layer_rows)


if __name__ == "__main__":
    main()
