#!/usr/bin/env python3
"""Render prompt-level prompt-profile activation projection panels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render prompt-level activation projections into cluster/binary/"
            "continuous panels and a separability summary table."
        )
    )
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prompt-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--figure-label",
        default="",
        help="Optional short override to prepend in figure titles.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    return float(text)


def int_or_none(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    return int(text)


def prompt_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    rows = load_csv(path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed_row: dict[str, Any] = {
            **row,
            "sample_id": int(row["sample_id"]),
            "pc1": float(row["pc1"]),
            "pc2": float(row["pc2"]),
            "cluster_id": int(row["cluster_id"]),
            "feature_norm": float(row["feature_norm"]),
            "target_value": float(row["target_value"]),
            "prompt_token_count": int(row["prompt_token_count"]),
            "effective_max_tokens": int(row["effective_max_tokens"]),
            "mean_length": float(row["mean_length"]),
            "mean_relative_length": float(row["mean_relative_length"]),
            "mu_log_rel": float(row["mu_log_rel"]),
            "p_cap": float(row["p_cap"]),
            "p_loop": float(row["p_loop"]),
            "majority_cap": int(row["majority_cap"]),
            "majority_loop": int(row["majority_loop"]),
            "any_cap": int(row["any_cap"]),
            "any_loop": int(row["any_loop"]),
            "finish_reason_length_rate": float(row["finish_reason_length_rate"]),
            "majority_finish_reason_length": int(row["majority_finish_reason_length"]),
            "correct_rate": float_or_none(row["correct_rate"]),
            "majority_correct": int_or_none(row["majority_correct"]),
            "rollout_count": int(row["rollout_count"]),
        }
        for key, value in row.items():
            if key.startswith("s_"):
                parsed_row[key] = float(value)
            elif key.startswith("majority_s_"):
                parsed_row[key] = int(value)
        parsed.append(parsed_row)
    return parsed


def compute_limits(rows: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.array([row["pc1"] for row in rows], dtype=float)
    ys = np.array([row["pc2"] for row in rows], dtype=float)
    x_pad = max((xs.max() - xs.min()) * 0.08, 0.15)
    y_pad = max((ys.max() - ys.min()) * 0.08, 0.15)
    return (xs.min() - x_pad, xs.max() + x_pad), (ys.min() - y_pad, ys.max() + y_pad)


def style_axes(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    x_label: str,
    y_label: str,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.18, linewidth=0.6)


def pc_axis_labels(summary: dict[str, Any]) -> tuple[str, str]:
    explained = summary.get("explained_variance_ratio")
    if not isinstance(explained, list):
        explained = []

    def _label(name: str, idx: int) -> str:
        if idx >= len(explained):
            return name
        try:
            return f"{name} ({float(explained[idx]) * 100:.1f}% var)"
        except (TypeError, ValueError):
            return name

    return _label("PC1", 0), _label("PC2", 1)


def figure_grid(num_panels: int, *, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, num_panels))
    rows = math.ceil(num_panels / cols)
    return rows, cols


def metric_title(metric_name: str) -> str:
    custom = {
        "majority_cap": "Majority Max-Length Hit",
        "majority_loop": "Majority Loop",
        "majority_correct": "Majority Correct",
        "majority_finish_reason_length": "Majority Finish=length",
        "p_cap": "Prompt p(max length)",
        "p_loop": "Prompt p(loop)",
        "correct_rate": "Prompt Correct Rate",
        "finish_reason_length_rate": "Prompt Finish=length Rate",
        "mean_relative_length": "Mean Relative Length",
    }
    if metric_name in custom:
        return custom[metric_name]
    if metric_name.startswith("majority_s_"):
        return f"Majority s_{metric_name.removeprefix('majority_s_')}"
    if metric_name.startswith("s_"):
        return metric_name.replace("_", " ", 1).replace("s ", "s_")
    return metric_name.replace("_", " ").title()


def render_cluster_panel(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    cluster_ids = sorted({int(row["cluster_id"]) for row in rows})
    palette = list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors)
    for cluster_id in cluster_ids:
        subset = [row for row in rows if row["cluster_id"] == cluster_id]
        ax.scatter(
            [row["pc1"] for row in subset],
            [row["pc2"] for row in subset],
            s=48,
            c=[palette[cluster_id % len(palette)]],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.25,
            label=f"C{cluster_id}",
        )
    ax.set_title("Unsupervised Clusters")
    ax.legend(loc="best", fontsize=8, frameon=False)


def render_binary_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    title: str,
    positive_color: str,
    negative_color: str = "#b8bcc8",
    unknown_color: str = "#f0c987",
) -> None:
    negative = [row for row in rows if row[label_key] == 0]
    positive = [row for row in rows if row[label_key] == 1]
    unknown = [row for row in rows if row[label_key] is None]

    if negative:
        ax.scatter(
            [row["pc1"] for row in negative],
            [row["pc2"] for row in negative],
            s=40,
            c=negative_color,
            alpha=0.5,
            linewidths=0,
            label="0",
        )
    if positive:
        ax.scatter(
            [row["pc1"] for row in positive],
            [row["pc2"] for row in positive],
            s=46,
            c=positive_color,
            alpha=0.86,
            edgecolors="black",
            linewidths=0.18,
            label="1",
        )
    if unknown:
        ax.scatter(
            [row["pc1"] for row in unknown],
            [row["pc2"] for row in unknown],
            s=44,
            c=unknown_color,
            alpha=0.72,
            linewidths=0,
            label="NA",
        )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)


def render_binary_panels(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: Path,
    *,
    figure_label: str,
) -> None:
    threshold_labels = [
        f"majority_s_{format(float(threshold), 'g')}"
        for threshold in summary["tail_thresholds"]
    ]
    metrics = [
        "cluster_id",
        "majority_cap",
        "majority_loop",
        "majority_finish_reason_length",
        *threshold_labels,
    ]
    if any(row.get("majority_correct") is not None for row in rows):
        metrics.append("majority_correct")

    rows_n, cols_n = figure_grid(len(metrics))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(4.6 * cols_n, 4.3 * rows_n))
    axes_arr = np.array(axes, dtype=object).reshape(rows_n, cols_n)

    x_label, y_label = pc_axis_labels(summary)
    xlim, ylim = compute_limits(rows)

    color_cycle = {
        "majority_cap": "#d64933",
        "majority_loop": "#6a4c93",
        "majority_finish_reason_length": "#cc5a71",
        "majority_correct": "#2d8f5f",
    }

    flat_axes = list(axes_arr.flat)
    for ax, metric in zip(flat_axes, metrics):
        if metric == "cluster_id":
            render_cluster_panel(ax, rows)
        else:
            render_binary_panel(
                ax,
                rows,
                label_key=metric,
                title=metric_title(metric),
                positive_color=color_cycle.get(metric, "#2b6cb0"),
            )
        style_axes(ax, xlim, ylim, x_label=x_label, y_label=y_label)

    for ax in flat_axes[len(metrics) :]:
        ax.axis("off")

    title_prefix = f"{figure_label}: " if figure_label else ""
    fig.suptitle(
        f"{title_prefix}{summary['dataset_name']} Prompt-Level Binary Views",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "prompt_binary_panels.png", dpi=220)
    fig.savefig(out_dir / "prompt_binary_panels.pdf")
    plt.close(fig)


def render_continuous_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    cmap: str,
) -> None:
    values = np.array(
        [np.nan if row[metric] is None else float(row[metric]) for row in rows],
        dtype=float,
    )
    scatter = ax.scatter(
        [row["pc1"] for row in rows],
        [row["pc2"] for row in rows],
        c=values,
        cmap=cmap,
        s=50,
        alpha=0.96,
        edgecolors="black",
        linewidths=0.24,
    )
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def render_continuous_panels(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: Path,
    *,
    figure_label: str,
) -> None:
    metrics: list[tuple[str, str, str]] = [
        ("p_cap", metric_title("p_cap"), "magma"),
        ("p_loop", metric_title("p_loop"), "cividis"),
        ("finish_reason_length_rate", metric_title("finish_reason_length_rate"), "inferno"),
        ("mean_relative_length", metric_title("mean_relative_length"), "viridis"),
    ]
    metrics.extend(
        (
            f"s_{format(float(threshold), 'g')}",
            metric_title(f"s_{format(float(threshold), 'g')}"),
            "plasma",
        )
        for threshold in summary["tail_thresholds"]
    )
    if any(row.get("correct_rate") is not None for row in rows):
        metrics.append(("correct_rate", metric_title("correct_rate"), "Greens"))

    rows_n, cols_n = figure_grid(len(metrics))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(4.9 * cols_n, 4.3 * rows_n))
    axes_arr = np.array(axes, dtype=object).reshape(rows_n, cols_n)

    x_label, y_label = pc_axis_labels(summary)
    xlim, ylim = compute_limits(rows)

    flat_axes = list(axes_arr.flat)
    for ax, (metric, title, cmap) in zip(flat_axes, metrics):
        render_continuous_panel(ax, rows, metric=metric, title=title, cmap=cmap)
        style_axes(ax, xlim, ylim, x_label=x_label, y_label=y_label)

    for ax in flat_axes[len(metrics) :]:
        ax.axis("off")

    title_prefix = f"{figure_label}: " if figure_label else ""
    fig.suptitle(
        f"{title_prefix}{summary['dataset_name']} Prompt-Level Derived Statistics",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "prompt_continuous_panels.png", dpi=220)
    fig.savefig(out_dir / "prompt_continuous_panels.pdf")
    plt.close(fig)


def format_metric(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def render_summary_table(
    summary: dict[str, Any],
    out_dir: Path,
    *,
    figure_label: str,
) -> None:
    binary_rows = []
    for label_name, metrics in summary["binary_label_alignment"].items():
        binary_rows.append(
            [
                metric_title(label_name),
                format_metric(metrics.get("prevalence")),
                format_metric(metrics.get("balanced_accuracy")),
                format_metric(metrics.get("adjusted_mutual_info")),
                format_metric(metrics.get("label_silhouette_2d")),
            ]
        )

    continuous_rows = []
    for label_name, metrics in summary["continuous_signal"].items():
        continuous_rows.append(
            [
                metric_title(label_name),
                format_metric(metrics.get("cluster_r2")),
                format_metric(metrics.get("linear_r2_2d")),
                format_metric(metrics.get("max_abs_spearman_pc")),
                format_metric(metrics.get("mean")),
            ]
        )

    fig = plt.figure(figsize=(12, 7.5))
    ax_top = fig.add_axes([0.03, 0.54, 0.94, 0.4])
    ax_bottom = fig.add_axes([0.03, 0.07, 0.94, 0.36])
    for ax in (ax_top, ax_bottom):
        ax.axis("off")

    title_prefix = f"{figure_label}: " if figure_label else ""
    fig.suptitle(
        f"{title_prefix}{summary['dataset_name']} Quantitative Separability",
        fontsize=14,
    )

    cluster_title = (
        f"Unsupervised clustering picked k={summary['cluster_summary']['k']} "
        f"(silhouette={format_metric(summary['cluster_summary']['silhouette'])})"
    )
    ax_top.text(0.0, 1.08, cluster_title, fontsize=11, transform=ax_top.transAxes)

    binary_table = ax_top.table(
        cellText=binary_rows or [["No binary labels", "", "", "", ""]],
        colLabels=["Binary label", "Prev", "Cluster bal acc", "AMI", "Label silhouette"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    binary_table.auto_set_font_size(False)
    binary_table.set_fontsize(9)
    binary_table.scale(1.0, 1.35)

    ax_bottom.text(
        0.0,
        1.08,
        "Continuous labels: variance explained by cluster means and by the 2D plane.",
        fontsize=11,
        transform=ax_bottom.transAxes,
    )
    continuous_table = ax_bottom.table(
        cellText=continuous_rows or [["No continuous labels", "", "", "", ""]],
        colLabels=["Continuous stat", "Cluster R2", "2D linear R2", "Max |Spearman|", "Mean"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    continuous_table.auto_set_font_size(False)
    continuous_table.set_fontsize(9)
    continuous_table.scale(1.0, 1.35)

    fig.savefig(out_dir / "prompt_separability_summary.png", dpi=220)
    fig.savefig(out_dir / "prompt_separability_summary.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    prompt_rows = prompt_rows_from_csv(Path(args.prompt_csv))

    render_binary_panels(
        prompt_rows,
        summary,
        out_dir,
        figure_label=args.figure_label.strip(),
    )
    render_continuous_panels(
        prompt_rows,
        summary,
        out_dir,
        figure_label=args.figure_label.strip(),
    )
    render_summary_table(
        summary,
        out_dir,
        figure_label=args.figure_label.strip(),
    )


if __name__ == "__main__":
    main()
