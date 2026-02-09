#!/usr/bin/env python3
"""Plot train/eval probe curves aggregated across seed runs.

Expected multi-seed layout:
  <run_dir>/seed_0/metrics.jsonl
  <run_dir>/seed_1/metrics.jsonl
  ...

Each metrics JSONL row is produced by scripts/train_probe.py and includes:
  epoch, train_loss, train_accuracy, train_macro_f1, train_roc_auc,
  accuracy, macro_f1, roc_auc
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


MetricSeries = Dict[str, List[dict]]


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_metrics_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            epoch = _safe_int(row.get("epoch"))
            if epoch is None:
                continue
            row["epoch"] = epoch
            rows.append(row)
    rows.sort(key=lambda row: int(row["epoch"]))
    return rows


def _discover_seed_metrics(run_dir: str) -> MetricSeries:
    seed_dirs = sorted(
        path
        for path in glob.glob(os.path.join(run_dir, "seed_*"))
        if os.path.isdir(path)
    )

    by_seed: MetricSeries = {}
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        metrics_path = os.path.join(seed_dir, "metrics.jsonl")
        if not os.path.isfile(metrics_path):
            continue
        rows = _load_metrics_jsonl(metrics_path)
        if rows:
            by_seed[seed_name] = rows

    if by_seed:
        return by_seed

    single_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.isfile(single_path):
        rows = _load_metrics_jsonl(single_path)
        if rows:
            run_name = os.path.basename(os.path.normpath(run_dir)) or "run"
            return {run_name: rows}

    raise SystemExit(
        f"No metrics found under '{run_dir}'. Expected seed_*/metrics.jsonl "
        "or run_dir/metrics.jsonl."
    )


def _seed_curve(rows: List[dict], key: str) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    values: List[float] = []
    for row in rows:
        epoch = _safe_int(row.get("epoch"))
        value = _safe_float(row.get(key))
        if epoch is None or value is None:
            continue
        epochs.append(epoch)
        values.append(value)
    return epochs, values


def _aggregate_by_epoch(by_seed: MetricSeries, key: str) -> Tuple[List[int], List[float], List[float]]:
    values_by_epoch: Dict[int, List[float]] = defaultdict(list)
    for rows in by_seed.values():
        for row in rows:
            epoch = _safe_int(row.get("epoch"))
            value = _safe_float(row.get(key))
            if epoch is None or value is None:
                continue
            values_by_epoch[epoch].append(value)

    epochs = sorted(values_by_epoch.keys())
    means: List[float] = []
    stds: List[float] = []
    for epoch in epochs:
        vals = values_by_epoch[epoch]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return epochs, means, stds


def _plot_metric(
    ax,
    by_seed: MetricSeries,
    title: str,
    train_key: str,
    eval_key: str | None,
) -> None:
    train_color = "#1f77b4"
    eval_color = "#ff7f0e"
    plotted_any = False

    for rows in by_seed.values():
        x_train, y_train = _seed_curve(rows, train_key)
        if x_train:
            ax.plot(
                x_train,
                y_train,
                color=train_color,
                linewidth=1.0,
                alpha=0.20,
                label="_nolegend_",
            )
            plotted_any = True
        if eval_key:
            x_eval, y_eval = _seed_curve(rows, eval_key)
            if x_eval:
                ax.plot(
                    x_eval,
                    y_eval,
                    color=eval_color,
                    linewidth=1.0,
                    alpha=0.20,
                    linestyle="--",
                    label="_nolegend_",
                )
                plotted_any = True

    train_epochs, train_mean, train_std = _aggregate_by_epoch(by_seed, train_key)
    if train_epochs:
        train_mean_arr = np.array(train_mean, dtype=float)
        train_std_arr = np.array(train_std, dtype=float)
        train_low = train_mean_arr - train_std_arr
        train_high = train_mean_arr + train_std_arr
        ax.plot(
            train_epochs,
            train_mean,
            color=train_color,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="train mean",
        )
        ax.fill_between(train_epochs, train_low, train_high, color=train_color, alpha=0.15)
        plotted_any = True

    if eval_key:
        eval_epochs, eval_mean, eval_std = _aggregate_by_epoch(by_seed, eval_key)
        if eval_epochs:
            eval_mean_arr = np.array(eval_mean, dtype=float)
            eval_std_arr = np.array(eval_std, dtype=float)
            eval_low = eval_mean_arr - eval_std_arr
            eval_high = eval_mean_arr + eval_std_arr
            ax.plot(
                eval_epochs,
                eval_mean,
                color=eval_color,
                linewidth=2.2,
                marker="o",
                markersize=4,
                linestyle="--",
                label="eval mean",
            )
            ax.fill_between(eval_epochs, eval_low, eval_high, color=eval_color, alpha=0.15)
            plotted_any = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    if plotted_any:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory that contains seed_*/metrics.jsonl.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output figure path (default: <run_dir>/probe_multiseed_curves.png).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    by_seed = _discover_seed_metrics(args.run_dir)
    num_seeds = len(by_seed)

    out_path = args.out or os.path.join(args.run_dir, "probe_multiseed_curves.png")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_metric(axes[0, 0], by_seed, "Accuracy", "train_accuracy", "accuracy")
    _plot_metric(axes[0, 1], by_seed, "Macro-F1", "train_macro_f1", "macro_f1")
    _plot_metric(axes[1, 0], by_seed, "ROC-AUC", "train_roc_auc", "roc_auc")
    _plot_metric(axes[1, 1], by_seed, "Loss", "train_loss", None)

    run_name = os.path.basename(os.path.normpath(args.run_dir)) or args.run_dir
    fig.suptitle(f"{run_name}: probe curves across {num_seeds} seed(s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
