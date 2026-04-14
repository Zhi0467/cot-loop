#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_METRICS = [
    ("mean_prompt_mass", "Prompt", "#1f77b4"),
    ("mean_prev_loop_mass", "Previous loop", "#d62728"),
    ("mean_current_trigger_mass", "Current trigger", "#2ca02c"),
    ("mean_recent_nonloop_mass", "Recent non-loop", "#ff7f0e"),
    ("mean_other_completion_mass", "Other completion", "#9467bd"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Plot input in the form label=PATH, where PATH is a bundle dir or CSV.",
    )
    parser.add_argument("--out-figure", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--title",
        default="Qwen3 loop-trigger attention mass by layer",
    )
    return parser.parse_args()


def _resolve_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        return path / "attention_layer_means.csv"
    return path


def _parse_label_and_path(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"Expected label=PATH, got: {item}")
    label, path_str = item.split("=", 1)
    csv_path = _resolve_csv(path_str)
    if not csv_path.is_file():
        raise SystemExit(f"Missing attention-layer CSV: {csv_path}")
    return label, csv_path


def _derive_other_completion(row: dict[str, str]) -> float:
    if "mean_other_completion_mass" in row and row["mean_other_completion_mass"] != "":
        return float(row["mean_other_completion_mass"])
    known_mass = sum(
        float(row[key])
        for key in (
            "mean_prev_loop_mass",
            "mean_prompt_mass",
            "mean_current_trigger_mass",
            "mean_recent_nonloop_mass",
        )
    )
    return max(0.0, 1.0 - known_mass)


def _load_weighted_overall(csv_path: Path) -> list[dict[str, float | int]]:
    by_layer: dict[int, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            layer = int(row["layer"])
            num_rows = int(row["num_rows"])
            bucket = by_layer.setdefault(layer, {"num_rows": 0.0})
            bucket["num_rows"] += num_rows
            for key, _, _ in PLOT_METRICS:
                value = (
                    _derive_other_completion(row)
                    if key == "mean_other_completion_mass"
                    else float(row[key])
                )
                bucket[key] = bucket.get(key, 0.0) + value * num_rows

    overall_rows: list[dict[str, float | int]] = []
    for layer in sorted(by_layer):
        bucket = by_layer[layer]
        num_rows = int(bucket["num_rows"])
        overall_rows.append(
            {
                "layer": layer,
                "num_rows": num_rows,
                **{
                    key: float(bucket[key]) / num_rows
                    for key, _, _ in PLOT_METRICS
                },
            }
        )
    return overall_rows


def _write_combined_csv(
    out_path: Path,
    bundles: list[tuple[str, list[dict[str, float | int]]]],
) -> None:
    fieldnames = ["label", "layer", "num_rows"] + [key for key, _, _ in PLOT_METRICS]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in bundles:
            for row in rows:
                writer.writerow({"label": label, **row})


def main() -> None:
    args = _parse_args()
    parsed_inputs = [_parse_label_and_path(item) for item in args.input]
    bundles = [(label, _load_weighted_overall(csv_path)) for label, csv_path in parsed_inputs]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_combined_csv(out_csv, bundles)

    fig, axes = plt.subplots(
        1,
        len(bundles),
        figsize=(6.0 * len(bundles), 4.5),
        sharey=True,
        constrained_layout=True,
    )
    if len(bundles) == 1:
        axes = [axes]

    for ax, (label, rows) in zip(axes, bundles):
        layers = [int(row["layer"]) for row in rows]
        for key, legend_label, color in PLOT_METRICS:
            values = [float(row[key]) for row in rows]
            if max(values, default=0.0) <= 1e-8:
                continue
            ax.plot(layers, values, label=legend_label, color=color, linewidth=2.0)
        ax.set_title(label)
        ax.set_xlabel("Layer")
        ax.set_xticks(layers[:: max(1, len(layers) // 7)])
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel("Mean attention mass")
    axes[0].set_ylim(0.0, 1.0)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.suptitle(args.title, y=1.02)

    out_figure = Path(args.out_figure)
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
