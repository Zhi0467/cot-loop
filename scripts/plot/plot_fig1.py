#!/usr/bin/env python3
"""Plot Figure 1 (looping fraction, average CoT length, accuracy) from metrics CSVs.

Examples
  # Plot from three per-model metrics files
  python scripts/plot_fig1.py \
    --metrics outputs/qwq32b_metrics.csv \
    --metrics outputs/openthinker3_7b_metrics.csv \
    --metrics outputs/openthinker3_1p5b_metrics.csv \
    --out outputs/fig1.png

  # Glob or directory inputs are also accepted (including rep suffixes)
  python scripts/plot_fig1.py --metrics "outputs/*_metrics*.csv" --out outputs/fig1.png
  python scripts/plot_fig1.py --metrics outputs --out outputs/fig1.png
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def is_rank_shard(path: str) -> bool:
    root = os.path.splitext(os.path.basename(path))[0]
    return re.search(r"\.rank\d+", root) is not None


def extract_repetition(path: str) -> int:
    root = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"\.rep(\d+)$", root)
    if match:
        return int(match.group(1))
    return 1


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def avg_by_temp(stats_by_temp):
    temps = sorted(stats_by_temp.keys())
    loop_fracs = []
    avg_tokens = []
    accuracies = []
    has_accuracy = False
    for temp in temps:
        stats = stats_by_temp[temp]
        count = stats["count"]
        loop = (stats["loop"] / count) if count else 0.0
        avg = (stats["token_sum"] / count) if count else 0.0
        loop_fracs.append(loop)
        avg_tokens.append(avg)
        graded = stats.get("graded", 0)
        if graded:
            accuracies.append(stats.get("correct", 0.0) / graded)
            has_accuracy = True
        else:
            accuracies.append(None)
    return temps, loop_fracs, avg_tokens, accuracies, has_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        required=True,
        action="append",
        help="Metrics CSVs (repeatable, comma-separated, or globbed).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--models",
        default="open-thoughts/OpenThinker3-1.5B,open-thoughts/OpenThinker3-7B,Qwen/QwQ-32B",
    )
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    want_models = set(model_list)

    metric_paths = []
    for entry in args.metrics:
        for part in entry.split(","):
            part = part.strip()
            if not part:
                continue
            if os.path.isdir(part):
                metric_paths.extend(
                    sorted(glob.glob(os.path.join(part, "*_metrics*.csv")))
                )
            elif any(ch in part for ch in "*?[]"):
                metric_paths.extend(sorted(glob.glob(part)))
            else:
                metric_paths.append(part)

    metric_paths = [p for p in metric_paths if p]
    if not metric_paths:
        raise SystemExit("No metrics files found (after expanding inputs).")

    missing = [p for p in metric_paths if not os.path.isfile(p)]
    if missing:
        print(
            f"Warning: skipping missing metrics files: {', '.join(missing)}",
            file=sys.stderr,
        )
    metric_paths = [p for p in metric_paths if os.path.isfile(p)]
    if not metric_paths:
        raise SystemExit("No readable metrics files found (after filtering missing).")

    rank_paths = [p for p in metric_paths if is_rank_shard(p)]
    if rank_paths:
        print(
            f"Warning: skipping rank-sharded metrics files: {', '.join(rank_paths)}",
            file=sys.stderr,
        )
    filtered_paths = [p for p in metric_paths if not is_rank_shard(p)]

    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {
                    "count": 0,
                    "loop": 0.0,
                    "token_sum": 0.0,
                    "correct": 0.0,
                    "graded": 0.0,
                }
            )
        )
    )
    for path in filtered_paths:
        num_rep = extract_repetition(path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = row["model_id"]
                if model_id not in want_models:
                    continue
                temp = float(row["temperature"])
                loop_frac = float(row["loop_fraction"])
                avg_tokens = float(row["avg_tokens"])
                count = int(row["num_samples"])
                correct = None
                graded = None
                if "num_correct" in row and row["num_correct"] != "":
                    try:
                        correct = float(row["num_correct"])
                        graded = count
                    except ValueError:
                        correct = None
                elif "accuracy" in row and row["accuracy"] != "":
                    try:
                        acc = float(row["accuracy"])
                        correct = acc * count
                        graded = count
                    except ValueError:
                        correct = None
                stats = data[model_id][num_rep][temp]
                stats["count"] += count
                stats["loop"] += loop_frac * count
                stats["token_sum"] += avg_tokens * count
                if correct is not None and graded is not None:
                    stats["correct"] += correct
                    stats["graded"] += graded

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plotted_models = False
    plotted_accuracy = False

    for model_id in model_list:
        reps = data.get(model_id)
        if not reps or 1 not in reps:
            continue
        temps, loop_fracs, avg_tokens, accuracies, has_accuracy = avg_by_temp(reps[1])
        if not temps:
            continue
        label = f"{model_id} (rep 1)"
        axes[0].plot(temps, loop_fracs, marker="o", label=label)
        axes[1].plot(temps, avg_tokens, marker="o", label=label)
        if has_accuracy:
            axes[2].plot(temps, accuracies, marker="o", label=label)
            plotted_accuracy = True
        plotted_models = True

    if not plotted_models:
        axes[0].text(0.5, 0.5, "No rep 1 data", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No rep 1 data", ha="center", va="center")
    if not plotted_accuracy:
        axes[2].text(0.5, 0.5, "No accuracy data", ha="center", va="center")

    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Looping Fraction")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Average CoT Length (tokens)")
    axes[2].set_xlabel("Temperature")
    axes[2].set_ylabel("Accuracy")

    if plotted_models:
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    if plotted_accuracy:
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200)

    base_root, base_ext = os.path.splitext(args.out)
    for model_id in model_list:
        reps = data.get(model_id, {})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plotted = False
        plotted_acc = False
        for rep in (1, 2, 3):
            if rep not in reps:
                continue
            temps, loop_fracs, avg_tokens, accuracies, has_accuracy = avg_by_temp(
                reps[rep]
            )
            if not temps:
                continue
            axes[0].plot(temps, loop_fracs, marker="o", label=f"rep {rep}")
            axes[1].plot(temps, avg_tokens, marker="o", label=f"rep {rep}")
            if has_accuracy:
                axes[2].plot(temps, accuracies, marker="o", label=f"rep {rep}")
                plotted_acc = True
            plotted = True

        if not plotted:
            axes[0].text(0.5, 0.5, "No rep 1-3 data", ha="center", va="center")
            axes[1].text(0.5, 0.5, "No rep 1-3 data", ha="center", va="center")
        if not plotted_acc:
            axes[2].text(0.5, 0.5, "No accuracy data", ha="center", va="center")

        axes[0].set_xlabel("Temperature")
        axes[0].set_ylabel("Looping Fraction")
        axes[1].set_xlabel("Temperature")
        axes[1].set_ylabel("Average CoT Length (tokens)")
        axes[2].set_xlabel("Temperature")
        axes[2].set_ylabel("Accuracy")
        if plotted:
            axes[0].legend(fontsize=8)
            axes[1].legend(fontsize=8)
        if plotted_acc:
            axes[2].legend(fontsize=8)
        fig.suptitle(f"{model_id}: num_repetition sweep", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        model_out = f"{base_root}.{model_slug(model_id)}.repetition{base_ext}"
        fig.savefig(model_out, dpi=200)


if __name__ == "__main__":
    main()
