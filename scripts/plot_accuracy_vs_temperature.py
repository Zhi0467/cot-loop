#!/usr/bin/env python3
"""Plot accuracy vs. temperature using rep1/rep2/rep3 metrics in outputs/.

Example:
uv run python scripts/plot_accuracy_vs_temperature.py \
    --model-id open-thoughts/OpenThinker3-1.5B \
    --out outputs/accuracy_vs_temperature.png
"""

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

REPETITIONS = (1, 2, 3)


def extract_repetition(path: str) -> int | None:
    root = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"\.rep(\d+)$", root)
    if match:
        return int(match.group(1))
    return None


def find_rep_metric_paths(outputs_dir: str) -> List[str]:
    paths: List[str] = []
    for rep in REPETITIONS:
        pattern = os.path.join(outputs_dir, f"*_metrics.rep{rep}.csv")
        paths.extend(sorted(glob.glob(pattern)))
    # Deduplicate while keeping deterministic order.
    return list(dict.fromkeys(paths))


def load_accuracy_by_temp(path: str, model_id: str | None) -> Dict[float, float]:
    per_temp_weighted: Dict[float, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_model_id = row.get("model_id")
            if model_id and row_model_id != model_id:
                continue

            try:
                temp = float(row["temperature"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid temperature in {path}") from exc

            accuracy = None
            weight = 1.0
            if row.get("accuracy"):
                try:
                    accuracy = float(row["accuracy"])
                except ValueError as exc:
                    raise ValueError(f"Invalid accuracy in {path}") from exc
            elif row.get("num_correct") and row.get("num_samples"):
                try:
                    num_correct = float(row["num_correct"])
                    num_samples = float(row["num_samples"])
                    if num_samples > 0:
                        accuracy = num_correct / num_samples
                        weight = num_samples
                except ValueError as exc:
                    raise ValueError(f"Invalid num_correct/num_samples in {path}") from exc

            if accuracy is None:
                continue

            prev_sum, prev_weight = per_temp_weighted.get(temp, (0.0, 0.0))
            per_temp_weighted[temp] = (
                prev_sum + accuracy * weight,
                prev_weight + weight,
            )

    accuracy_by_temp: Dict[float, float] = {}
    for temp, (weighted_sum, total_weight) in per_temp_weighted.items():
        if total_weight > 0:
            accuracy_by_temp[temp] = weighted_sum / total_weight
    return accuracy_by_temp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/accuracy_vs_temperature.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--model-id",
        default="open-thoughts/OpenThinker3-1.5B",
        help="Filter rows by model_id (exact match).",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing *_metrics.rep1/2/3.csv files.",
    )
    args = parser.parse_args()

    metric_paths = find_rep_metric_paths(args.outputs_dir)
    if not metric_paths:
        raise SystemExit(
            f"No repetition metrics files found in '{args.outputs_dir}' "
            "(expected *_metrics.rep1.csv, *_metrics.rep2.csv, *_metrics.rep3.csv)."
        )

    series_by_rep: Dict[int, Tuple[List[float], List[float], str]] = {}
    for path in metric_paths:
        rep = extract_repetition(path)
        if rep not in REPETITIONS:
            continue
        accuracy_by_temp = load_accuracy_by_temp(path, args.model_id)
        if not accuracy_by_temp:
            continue
        temps = sorted(accuracy_by_temp.keys())
        accuracies = [accuracy_by_temp[t] for t in temps]
        if rep in series_by_rep:
            prev_path = series_by_rep[rep][2]
            raise SystemExit(
                f"Multiple rep {rep} files matched model_id '{args.model_id}': "
                f"{prev_path}, {path}"
            )
        series_by_rep[rep] = (temps, accuracies, path)

    missing_reps = [rep for rep in REPETITIONS if rep not in series_by_rep]
    if missing_reps:
        raise SystemExit(
            f"Missing accuracy data for model_id '{args.model_id}' in repetition(s): "
            f"{', '.join(str(rep) for rep in missing_reps)}"
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for rep in REPETITIONS:
        temps, accuracies, _ = series_by_rep[rep]
        ax.plot(temps, accuracies, marker="o", linewidth=2, label=f"rep {rep}")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy")
    if args.model_id:
        ax.set_title(f"{args.model_id}: Accuracy vs. Temperature")
    else:
        ax.set_title("Accuracy vs. Temperature")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
