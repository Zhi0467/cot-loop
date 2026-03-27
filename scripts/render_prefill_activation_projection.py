#!/usr/bin/env python3
"""Render prompt- and rollout-level projection panels from exported CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render saved prefill projection tables into prompt/rollout panels."
    )
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prompt-csv", required=True)
    parser.add_argument("--rollout-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--jitter-scale",
        type=float,
        default=0.03,
        help="Rollout jitter as a fraction of the smaller axis span.",
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
        parsed.append(
            {
                **row,
                "sample_id": int(row["sample_id"]),
                "pc1": float(row["pc1"]),
                "pc2": float(row["pc2"]),
                "feature_norm": float(row["feature_norm"]),
                "target_value": float(row["target_value"]),
                "prompt_token_count": int(row["prompt_token_count"]),
                "effective_max_tokens": int(row["effective_max_tokens"]),
                "mean_length": float(row["mean_length"]),
                "mean_relative_length": float(row["mean_relative_length"]),
                "mu_log_rel": float(row["mu_log_rel"]),
                "p_cap": float(row["p_cap"]),
                "p_loop": float(row["p_loop"]),
                "correct_rate": float_or_none(row["correct_rate"]),
                "rollout_count": int(row["rollout_count"]),
            }
        )
    return parsed


def rollout_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    rows = load_csv(path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "sample_id": int(row["sample_id"]),
                "rollout_index": int(row["rollout_index"]),
                "pc1": float(row["pc1"]),
                "pc2": float(row["pc2"]),
                "cap_hit": int(row["cap_hit"]),
                "loop_flag": int(row["loop_flag"]),
                "correct": int_or_none(row["correct"]),
                "length": int(row["length"]),
                "relative_length": float(row["relative_length"]),
                "first_loop_prefix_length": int_or_none(row["first_loop_prefix_length"]),
            }
        )
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


def render_prompt_panels(
    prompt_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: Path,
) -> None:
    metrics: list[tuple[str, str, str]] = [
        ("mean_relative_length", "Mean Relative Length", "viridis"),
        ("p_cap", "Prompt p(max length)", "magma"),
        ("p_loop", "Prompt p(loop)", "cividis"),
    ]
    if any(row["correct_rate"] is not None for row in prompt_rows):
        metrics.append(("correct_rate", "Prompt Correct Rate", "plasma"))
    else:
        metrics.append(("target_value", "Target Value", "plasma"))

    x_label = f"PC1 ({summary['explained_variance_ratio'][0] * 100:.1f}% var)"
    y_label = f"PC2 ({summary['explained_variance_ratio'][1] * 100:.1f}% var)"
    xlim, ylim = compute_limits(prompt_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
    for ax, (metric, title, cmap) in zip(axes.flat, metrics):
        values = np.array(
            [
                np.nan if row[metric] is None else float(row[metric])
                for row in prompt_rows
            ],
            dtype=float,
        )
        scatter = ax.scatter(
            [row["pc1"] for row in prompt_rows],
            [row["pc2"] for row in prompt_rows],
            c=values,
            cmap=cmap,
            s=68,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.set_title(title)
        style_axes(ax, xlim, ylim, x_label=x_label, y_label=y_label)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "GPQA Prefill Activation Projection: Prompt-Level Propensity",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "prompt_projection_panels.png", dpi=220)
    fig.savefig(out_dir / "prompt_projection_panels.pdf")
    plt.close(fig)


def deterministic_jitter(
    sample_id: int,
    rollout_index: int,
    *,
    scale: float,
) -> tuple[float, float]:
    angle_deg = (sample_id * 104_729 + rollout_index * 13_007) % 360
    radius_bucket = (sample_id * 31_337 + rollout_index * 911) % 1000
    radius = 0.45 + 0.55 * (radius_bucket / 999.0)
    angle = math.radians(angle_deg)
    return (
        scale * radius * math.cos(angle),
        scale * radius * math.sin(angle),
    )


def add_rollout_jitter(
    rollout_rows: list[dict[str, Any]],
    *,
    scale: float,
) -> list[dict[str, Any]]:
    jittered: list[dict[str, Any]] = []
    for row in rollout_rows:
        dx, dy = deterministic_jitter(
            int(row["sample_id"]),
            int(row["rollout_index"]),
            scale=scale,
        )
        jittered.append(
            {
                **row,
                "jx": float(row["pc1"]) + dx,
                "jy": float(row["pc2"]) + dy,
            }
        )
    return jittered


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
            [row["jx"] for row in negative],
            [row["jy"] for row in negative],
            s=18,
            c=negative_color,
            alpha=0.45,
            linewidths=0,
            label="0",
        )
    if positive:
        ax.scatter(
            [row["jx"] for row in positive],
            [row["jy"] for row in positive],
            s=22,
            c=positive_color,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.15,
            label="1",
        )
    if unknown:
        ax.scatter(
            [row["jx"] for row in unknown],
            [row["jy"] for row in unknown],
            s=20,
            c=unknown_color,
            alpha=0.7,
            linewidths=0,
            label="NA",
        )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)


def render_finish_reason_panel(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    palette = {
        "stop": "#3b6fb6",
        "length": "#d64933",
    }
    fallback = "#9d9d9d"
    categories = sorted({str(row["finish_reason"]) for row in rows})
    for category in categories:
        subset = [row for row in rows if row["finish_reason"] == category]
        ax.scatter(
            [row["jx"] for row in subset],
            [row["jy"] for row in subset],
            s=20,
            c=palette.get(category, fallback),
            alpha=0.75,
            linewidths=0,
            label=category,
        )
    ax.set_title("Finish Reason")
    ax.legend(loc="best", fontsize=8, frameon=False)


def render_rollout_panels(
    rollout_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: Path,
    *,
    jitter_scale_fraction: float,
) -> None:
    x_label = f"PC1 ({summary['explained_variance_ratio'][0] * 100:.1f}% var)"
    y_label = f"PC2 ({summary['explained_variance_ratio'][1] * 100:.1f}% var)"
    xlim, ylim = compute_limits(rollout_rows)
    base_scale = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * jitter_scale_fraction
    jittered_rows = add_rollout_jitter(rollout_rows, scale=base_scale)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
    render_binary_panel(
        axes[0, 0],
        jittered_rows,
        label_key="cap_hit",
        title="Max Length Hit",
        positive_color="#d64933",
    )
    render_binary_panel(
        axes[0, 1],
        jittered_rows,
        label_key="loop_flag",
        title="Loop Flag",
        positive_color="#6a4c93",
    )
    render_binary_panel(
        axes[1, 0],
        jittered_rows,
        label_key="correct",
        title="Correct",
        positive_color="#2d8f5f",
    )
    render_finish_reason_panel(axes[1, 1], jittered_rows)

    for ax in axes.flat:
        style_axes(ax, xlim, ylim, x_label=x_label, y_label=y_label)

    fig.suptitle(
        "GPQA Prefill Activation Projection: Repeated Rollout Labels",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "rollout_projection_panels.png", dpi=220)
    fig.savefig(out_dir / "rollout_projection_panels.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    prompt_rows = prompt_rows_from_csv(Path(args.prompt_csv))
    rollout_rows = rollout_rows_from_csv(Path(args.rollout_csv))

    render_prompt_panels(prompt_rows, summary, out_dir)
    render_rollout_panels(
        rollout_rows,
        summary,
        out_dir,
        jitter_scale_fraction=float(args.jitter_scale),
    )


if __name__ == "__main__":
    main()
