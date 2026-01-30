#!/usr/bin/env python3
"""Plot Figure 1 (looping fraction and average CoT length) from metrics CSVs.

Examples
  # Plot from three per-model metrics files
  python scripts/plot_fig1.py \
    --metrics outputs/qwq32b_metrics.csv \
    --metrics outputs/openthinker3_7b_metrics.csv \
    --metrics outputs/openthinker3_1p5b_metrics.csv \
    --out outputs/fig1.png

  # Glob or directory inputs are also accepted
  python scripts/plot_fig1.py --metrics "outputs/*_metrics.csv" --out outputs/fig1.png
  python scripts/plot_fig1.py --metrics outputs --out outputs/fig1.png
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt


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

    want_models = set(m.strip() for m in args.models.split(",") if m.strip())

    metric_paths = []
    for entry in args.metrics:
        for part in entry.split(","):
            part = part.strip()
            if not part:
                continue
            if os.path.isdir(part):
                metric_paths.extend(sorted(glob.glob(os.path.join(part, "*_metrics.csv"))))
            elif any(ch in part for ch in "*?[]"):
                metric_paths.extend(sorted(glob.glob(part)))
            else:
                metric_paths.append(part)

    metric_paths = [p for p in metric_paths if p]
    if not metric_paths:
        raise SystemExit("No metrics files found (after expanding inputs).")

    missing = [p for p in metric_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing metrics files: {', '.join(missing)}")

    data = defaultdict(list)
    for path in metric_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = row["model_id"]
                if model_id not in want_models:
                    continue
                temp = float(row["temperature"])
                loop_frac = float(row["loop_fraction"])
                avg_tokens = float(row["avg_tokens"])
                data[model_id].append((temp, loop_frac, avg_tokens))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for model_id, rows in data.items():
        rows = sorted(rows, key=lambda x: x[0])
        temps = [r[0] for r in rows]
        loop_fracs = [r[1] for r in rows]
        avg_tokens = [r[2] for r in rows]
        axes[0].plot(temps, loop_fracs, marker="o", label=model_id)
        axes[1].plot(temps, avg_tokens, marker="o", label=model_id)

    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Looping Fraction")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Average CoT Length (tokens)")

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
