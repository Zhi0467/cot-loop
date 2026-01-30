#!/usr/bin/env python3
"""Plot Figure 1 (looping fraction and average CoT length) from metrics CSV."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--models",
        default="open-thoughts/OpenThinker3-1.5B,open-thoughts/OpenThinker3-7B,Qwen/QwQ-32B",
    )
    args = parser.parse_args()

    want_models = set(m.strip() for m in args.models.split(",") if m.strip())

    data = defaultdict(list)
    with open(args.metrics, "r", encoding="utf-8") as f:
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
