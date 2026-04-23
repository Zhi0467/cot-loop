#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MATPLOTLIB_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    matplotlib = None
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "weeks" / "2026-W15" / "prompt_profile_unified_report_20260409"
DOC_OUT = ROOT / "docs" / "weeks" / "2026-W15" / "prompt-profile-unified-report-2026-04-09.md"

REGRESSION_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W14"
    / "prompt_profile_natural_regression_rerun_20260405"
    / "remote_summary"
    / "cross_dataset_summary.json"
)
BINARY_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W14"
    / "prompt_profile_binary_capacity_controls_20260404"
    / "capacity_comparison_summary.json"
)
METADATA_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W14"
    / "prompt_profile_metadata_audit_20260405"
    / "metadata_correlation_summary.json"
)
MECHANISM_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W15"
    / "prompt_profile_length_mechanism_20260409"
    / "mechanism_summary.json"
)
METADATA_MECHANISM_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W15"
    / "prompt_profile_metadata_mechanism_20260406"
    / "metadata_mechanism_summary.json"
)
REGRESSION_MODELS_CSV = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W15"
    / "prompt_profile_length_mechanism_20260409"
    / "regression_model_metrics.csv"
)
PROMPT_BINS_CSV = (
    ROOT
    / "outputs"
    / "weeks"
    / "2026-W15"
    / "prompt_profile_length_mechanism_20260409"
    / "prompt_length_bins.csv"
)

DATASETS = [
    ("gpqa", "GPQA"),
    ("aime", "AIME"),
    ("math500", "MATH-500"),
    ("mmlu_pro", "MMLU-Pro"),
    ("livecodebench", "LiveCodeBench"),
]

REGRESSION_MODEL_ORDER = [
    ("prompt_length", "Prompt length"),
    ("shape_linear", "Prompt-shape linear"),
    ("shape_tree", "Prompt-shape tree"),
    ("last_layer", "Last layer"),
    ("ensemble", "Ensemble"),
]

BINARY_MODEL_ORDER = [
    ("prompt_length", "Prompt length"),
    ("best_prompt_only", "Best cheap prompt feature"),
    ("last_layer", "Last layer h256 d2"),
    ("ensemble", "Ensemble h256 d1"),
]

FEATURE_LABELS = {
    "prompt_token_count": "Prompt length",
    "char_length": "Character length",
    "newline_count": "Newline count",
    "digit_count": "Digit count",
    "dollar_count": "Dollar count",
    "choice_count": "Choice count",
    "log_token_length": "Log token length",
}

COLORS = {
    "prompt_length": "#355c7d",
    "shape_linear": "#6c8ebf",
    "shape_tree": "#59a14f",
    "last_layer": "#e17c05",
    "ensemble": "#c0392b",
    "best_prompt_only": "#8c564b",
}


@dataclass
class MetricPoint:
    mean: float
    std: float | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    figures_dir = path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)


def shifted_positions(count: int, width: float, offset_index: int) -> list[float]:
    center = (count - 1) / 2.0
    return [idx + (offset_index - center) * width for idx in range(len(DATASETS))]


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_mean_std(point: MetricPoint, digits: int = 3) -> str:
    if point.std is None:
        return fmt(point.mean, digits)
    return f"{fmt(point.mean, digits)} +/- {fmt(point.std, digits)}"


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def dataset_counts(
    metadata_mechanism_summary: dict[str, Any], dataset_key: str
) -> tuple[int, int]:
    counts = metadata_mechanism_summary["datasets"][dataset_key]["counts"]
    return int(counts["regression_train"]), int(counts["regression_test"])


def get_regression_activation_point(
    regression_summary: dict[str, Any],
    dataset_key: str,
    view: str,
    metric: str,
) -> MetricPoint:
    node = (
        regression_summary["datasets"][dataset_key]["mean_relative_length"]["views"][view][
            "aggregate"
        ]["best_loss"][metric]
    )
    return MetricPoint(float(node["mean"]), float(node["std"]))


def get_regression_prompt_length_point(
    regression_summary: dict[str, Any],
    dataset_key: str,
    metric: str,
) -> MetricPoint:
    node = (
        regression_summary["datasets"][dataset_key]["mean_relative_length"]["metadata_baselines"][
            "prompt_length"
        ]["test"][metric]
    )
    return MetricPoint(float(node), None)


def build_regression_prompt_models(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        dataset = row["dataset"]
        model = row["model_name"]
        out.setdefault(dataset, {})[model] = {
            "top_20p_capture": float(row["top_20p_capture"]),
            "rmse": float(row["rmse"]),
            "spearman": float(row["spearman"]),
        }
    return out


def build_binary_best_prompt_only(
    metadata_audit: dict[str, Any],
) -> dict[str, dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for dataset, payload in metadata_audit["datasets"].items():
        best = max(
            payload["binary_feature_metrics"],
            key=lambda item: float(item["pr_auc"]),
        )
        out[dataset] = {
            "feature": str(best["feature"]),
            "label": FEATURE_LABELS.get(str(best["feature"]), str(best["feature"])),
            "pr_auc": float(best["pr_auc"]),
        }
    return out


def build_prompt_bins(rows: list[dict[str, str]]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(
            {
                "bucket": float(row["bucket"]),
                "prompt_token_mean": float(row["prompt_token_mean"]),
                "mean_relative_length_mean": float(row["mean_relative_length_mean"]),
            }
        )
    for dataset_rows in grouped.values():
        dataset_rows.sort(key=lambda item: item["bucket"])
    return grouped


def plot_regression_views(
    out_path: Path,
    regression_summary: dict[str, Any],
    prompt_models: dict[str, dict[str, dict[str, float]]],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate regression figures")
    dataset_labels = [display for _, display in DATASETS]
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    panels = [
        ("top_20p_capture", "Top-20% Capture", axes[0]),
        ("rmse", "RMSE", axes[1]),
    ]
    for panel_metric, panel_title, ax in panels:
        for idx, (model_key, label) in enumerate(REGRESSION_MODEL_ORDER):
            xpos = shifted_positions(len(REGRESSION_MODEL_ORDER), width, idx)
            values: list[float] = []
            yerr: list[float] = []
            use_err = model_key in {"last_layer", "ensemble"}
            for dataset_key, _ in DATASETS:
                if model_key == "prompt_length":
                    point = get_regression_prompt_length_point(
                        regression_summary, dataset_key, panel_metric
                    )
                elif model_key in {"last_layer", "ensemble"}:
                    point = get_regression_activation_point(
                        regression_summary, dataset_key, model_key, panel_metric
                    )
                else:
                    point = MetricPoint(
                        prompt_models[dataset_key][model_key][panel_metric], None
                    )
                values.append(point.mean)
                yerr.append(point.std or 0.0)
            ax.bar(
                xpos,
                values,
                width=width,
                color=COLORS[model_key],
                label=label,
                yerr=yerr if use_err else None,
                capsize=3 if use_err else 0,
                linewidth=0,
            )
        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=3, fontsize=9, frameon=False, loc="upper left")
    axes[1].set_xticks(range(len(DATASETS)))
    axes[1].set_xticklabels(dataset_labels, rotation=0)
    fig.suptitle("Regression views on the natural split", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_binary_views(
    out_path: Path,
    binary_summary: dict[str, Any],
    best_prompt_only: dict[str, dict[str, float | str]],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate classification figures")
    dataset_labels = [display for _, display in DATASETS]
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for idx, (model_key, label) in enumerate(BINARY_MODEL_ORDER):
        xpos = shifted_positions(len(BINARY_MODEL_ORDER), width, idx)
        values: list[float] = []
        for dataset_key, _ in DATASETS:
            row = binary_summary["datasets"][dataset_key]
            if model_key == "prompt_length":
                values.append(float(row["prompt_length_baseline"]["pr_auc"]))
            elif model_key == "best_prompt_only":
                values.append(float(best_prompt_only[dataset_key]["pr_auc"]))
            elif model_key == "last_layer":
                values.append(
                    float(
                        row["views"]["last_layer"]["width_depth_h256d2_best_loss"][
                            "pr_auc"
                        ]
                    )
                )
            else:
                values.append(
                    float(
                        row["views"]["ensemble"]["width_only_h256d1_best_loss"][
                            "pr_auc"
                        ]
                    )
                )
        ax.bar(
            xpos,
            values,
            width=width,
            color=COLORS[model_key],
            label=label,
            linewidth=0,
        )
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(dataset_labels)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Classification views on the balanced-train / natural-test split")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_quartiles(
    out_path: Path,
    prompt_bins: dict[str, list[dict[str, float]]],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate quartile figures")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.8))
    axes_flat = axes.flatten()
    for idx, (dataset_key, display) in enumerate(DATASETS):
        ax = axes_flat[idx]
        rows = prompt_bins[dataset_key]
        xs = [row["prompt_token_mean"] for row in rows]
        ys = [row["mean_relative_length_mean"] for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2.2, color="#355c7d")
        ax.set_title(display)
        ax.set_xlabel("Mean prompt tokens")
        ax.set_ylabel("Mean relative length")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    axes_flat[-1].axis("off")
    fig.suptitle(
        "Prompt-length quartiles on the natural regression surface", fontsize=14, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_markdown(
    generated_utc: str,
    out_dir: Path,
    regression_summary: dict[str, Any],
    binary_summary: dict[str, Any],
    best_prompt_only: dict[str, dict[str, float | str]],
) -> str:
    out_rel = out_dir.relative_to(ROOT).as_posix()
    lines = [
        "# Unified Prompt-Profile Report",
        "",
        f"Last updated: {generated_utc}",
        "",
        "## Executive Summary",
        "",
        "- This is the current single prompt-profile report surface for the thread.",
        "  - Regression stays on the natural prompt-disjoint split with natural sampling.",
        "  - Classification stays on balanced train with natural test.",
        "  - The same report now carries setup, predictor views, metrics, results, and the plain-English metadata interpretation.",
        "- Regression headline:",
        "  - `mean_relative_length` is useful as a screening score, not as a clean calibrated regressor.",
        "  - On `top_20p_capture`, ensemble beats prompt length on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`.",
        "- Classification headline:",
        "  - `majority_s_0.5` is still the cleaner deployment-facing head.",
        "  - The best single global activation surface is still `ensemble h256 d1`.",
        "- Metadata headline:",
        "  - The old reported metadata baseline is only a train-fit prompt-length baseline on this fixed-budget run.",
        "  - Prompt-only cues are strong mostly because prompt surface structure already carries a lot of the same long-completion signal.",
        "",
        "## Main Artifact",
        "",
        f"- Unified PDF: `{out_rel}/prompt_profile_unified_report_20260409.pdf`",
        f"- Unified TeX: `{out_rel}/prompt_profile_unified_report_20260409.tex`",
        f"- Figures: `{out_rel}/figures/`",
        "",
        "## Current Read",
        "",
        "- Natural regression:",
    ]
    for dataset_key, display in DATASETS:
        top20_prompt = get_regression_prompt_length_point(
            regression_summary, dataset_key, "top_20p_capture"
        )
        top20_ensemble = get_regression_activation_point(
            regression_summary, dataset_key, "ensemble", "top_20p_capture"
        )
        lines.append(
            f"  - `{display}`: prompt length `{fmt(top20_prompt.mean)}`, ensemble `{fmt_mean_std(top20_ensemble)}`"
        )
    lines.extend(
        [
            "- Balanced classification:",
        ]
    )
    for dataset_key, display in DATASETS:
        row = binary_summary["datasets"][dataset_key]
        lines.append(
            "  - "
            + f"`{display}`: prompt length `"
            + fmt(float(row["prompt_length_baseline"]["pr_auc"]))
            + "`, best cheap prompt feature `"
            + tex_escape(str(best_prompt_only[dataset_key]["label"]))
            + " = "
            + fmt(float(best_prompt_only[dataset_key]["pr_auc"]))
            + "`, ensemble `"
            + fmt(float(row["views"]["ensemble"]["width_only_h256d1_best_loss"]["pr_auc"]))
            + "`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prompt length predicts completion length here mostly because, on these datasets, longer prompts usually mean the prompt already contains more work.",
            "- That story is strongest on `AIME`, `MATH-500`, and much of `LiveCodeBench`.",
            "- `GPQA` is the important exception: raw length is weak there, and prompt structure matters more than raw length.",
            "- For classification, some of the same prompt cues carry over because the positive label is still largely asking whether rollouts cross about half the fixed budget.",
            "- So the present results support lift over raw prompt length on some surfaces, but not yet a general lift over strong prompt-only controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_tex(
    generated_utc: str,
    regression_summary: dict[str, Any],
    metadata_mechanism_summary: dict[str, Any],
    prompt_models: dict[str, dict[str, dict[str, float]]],
    binary_summary: dict[str, Any],
    best_prompt_only: dict[str, dict[str, float | str]],
    prompt_bins: dict[str, list[dict[str, float]]],
) -> str:
    dataset_rows = []
    for dataset_key, display in DATASETS:
        train_count, test_count = dataset_counts(
            metadata_mechanism_summary, dataset_key
        )
        dataset_rows.append(
            f"\\texttt{{{tex_escape(display)}}} & {train_count} & {test_count} \\\\"
        )

    predictor_rows = [
        r"\texttt{prompt length} & train-fit linear scorer on \texttt{prompt\_token\_count}; on this run \texttt{effective\_max\_tokens} is fixed so the metadata baseline reduces to prompt length \\",
        r"\texttt{prompt-shape linear} & linear model on \texttt{prompt\_token\_count}, \texttt{log\_token\_length}, \texttt{char\_length}, \texttt{newline\_count}, \texttt{digit\_count}, \texttt{dollar\_count}, \texttt{choice\_count} \\",
        r"\texttt{prompt-shape tree} & nonlinear tree on the same seven prompt-surface features \\",
        r"\texttt{last\_layer} & one MLP on the final prompt-prefill layer; regression keeps the locked \texttt{h128 d1}, binary control uses \texttt{h256 d2} \\",
        r"\texttt{ensemble} & one MLP per saved layer over the full \texttt{28 x 2048} prompt-prefill surface; regression keeps locked \texttt{h128 d1} and aggregates by \texttt{mean\_prob}; classification recommendation is \texttt{h256 d1} and aggregates by \texttt{vote\_fraction} \\",
    ]

    regression_table_rows = []
    for dataset_key, display in DATASETS:
        prompt_top20 = get_regression_prompt_length_point(
            regression_summary, dataset_key, "top_20p_capture"
        )
        last_top20 = get_regression_activation_point(
            regression_summary, dataset_key, "last_layer", "top_20p_capture"
        )
        ensemble_top20 = get_regression_activation_point(
            regression_summary, dataset_key, "ensemble", "top_20p_capture"
        )
        prompt_rmse = get_regression_prompt_length_point(
            regression_summary, dataset_key, "rmse"
        )
        last_rmse = get_regression_activation_point(
            regression_summary, dataset_key, "last_layer", "rmse"
        )
        ensemble_rmse = get_regression_activation_point(
            regression_summary, dataset_key, "ensemble", "rmse"
        )
        regression_table_rows.append(
            " & ".join(
                [
                    rf"\texttt{{{tex_escape(display)}}}",
                    fmt(prompt_top20.mean),
                    fmt_mean_std(last_top20),
                    fmt_mean_std(ensemble_top20),
                    fmt(prompt_rmse.mean),
                    fmt_mean_std(last_rmse),
                    fmt_mean_std(ensemble_rmse),
                ]
            )
            + r" \\"
        )

    binary_table_rows = []
    for dataset_key, display in DATASETS:
        row = binary_summary["datasets"][dataset_key]
        ensemble = row["views"]["ensemble"]["width_only_h256d1_best_loss"]
        binary_table_rows.append(
            " & ".join(
                [
                    rf"\texttt{{{tex_escape(display)}}}",
                    fmt(float(row["test_prevalence"]), 4),
                    fmt(float(row["prompt_length_baseline"]["pr_auc"])),
                    fmt(float(best_prompt_only[dataset_key]["pr_auc"])),
                    fmt(float(ensemble["pr_auc"])),
                    fmt(float(ensemble["positive_precision"])),
                    fmt(float(ensemble["positive_recall"])),
                    fmt(float(ensemble["positive_f1"])),
                    fmt(float(ensemble["macro_f1"])),
                ]
            )
            + r" \\"
        )

    prompt_only_rows = []
    for dataset_key, display in DATASETS:
        regression_best = max(
            (
                (
                    feature["feature"],
                    float(feature["top_20p_capture"]),
                )
                for feature in load_json(METADATA_AUDIT_PATH)["datasets"][dataset_key][
                    "regression_feature_metrics"
                ]
            ),
            key=lambda item: item[1],
        )
        prompt_only_rows.append(
            " & ".join(
                [
                    rf"\texttt{{{tex_escape(display)}}}",
                    rf"{tex_escape(FEATURE_LABELS.get(str(regression_best[0]), str(regression_best[0])))} {fmt(float(regression_best[1]))}",
                    rf"{tex_escape(str(best_prompt_only[dataset_key]['label']))} {fmt(float(best_prompt_only[dataset_key]['pr_auc']))}",
                ]
            )
            + r" \\"
        )

    quartile_lines = []
    for dataset_key, display in DATASETS:
        quartiles = ", ".join(
            f"{int(round(row['prompt_token_mean']))} -> {fmt(row['mean_relative_length_mean'])}"
            for row in prompt_bins[dataset_key]
        )
        quartile_lines.append(
            rf"\item \texttt{{{tex_escape(display)}}}: {tex_escape(quartiles)}"
        )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{longtable}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.6em}}

\title{{Unified Prompt-Profile Report:\\Setup, Views, Results, and Mechanism}}
\author{{}}
\date{{{tex_escape(generated_utc)}}}

\begin{{document}}
\maketitle

\section*{{Executive Summary}}
\begin{{itemize}}[leftmargin=1.5em]
\item This is the single current prompt-profile report surface.
  \begin{{itemize}}[leftmargin=1.5em]
  \item Regression stays on the natural prompt-disjoint split with natural sampling.
  \item Classification stays on balanced train with natural test.
  \item The same note now carries setup, predictor views, metrics, results, and the plain-English metadata explanation in one place.
  \end{{itemize}}
\item Regression read:
  \begin{{itemize}}[leftmargin=1.5em]
  \item \texttt{{mean\_relative\_length}} is useful as a screening score, not as a clean calibrated regressor.
  \item On \texttt{{top\_20p\_capture}}, ensemble beats prompt length on \texttt{{GPQA}}, \texttt{{MMLU-Pro}}, and \texttt{{LiveCodeBench}}.
  \end{{itemize}}
\item Classification read:
  \begin{{itemize}}[leftmargin=1.5em]
  \item \texttt{{majority\_s\_0.5}} is still the cleaner deployment-facing head.
  \item The best single global activation surface is still \texttt{{ensemble h256 d1}}.
  \end{{itemize}}
\item Metadata read:
  \begin{{itemize}}[leftmargin=1.5em]
  \item The old reported metadata baseline is only prompt length on this fixed-budget run.
  \item Prompt-only cues are strong mostly because prompt surface structure already carries much of the same long-completion signal.
  \item \texttt{{GPQA}} is the main exception where raw length is weak and prompt structure matters more.
  \end{{itemize}}
\end{{itemize}}

\section*{{Setup}}
\subsection*{{Common rollout contract}}
\begin{{itemize}}[leftmargin=1.5em]
\item model: \texttt{{Qwen/Qwen3-1.7B}}
\item decode policy: \texttt{{temperature=0.2}}, \texttt{{num\_generations=4}}, \texttt{{max\_tokens=30000}}
\item features: prompt-prefill last-token activations over all saved layers (\texttt{{28 x 2048}})
\item frozen checkpoint rule for headline tables: \texttt{{best\_loss}}
\item compute contract for the latest mechanism audit: GPU node, \texttt{{2}} GPUs
\end{{itemize}}

\subsection*{{Dataset surface}}
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
Dataset & Regression train prompts & Regression test prompts \\
\midrule
{chr(10).join(dataset_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Split contract}}
\begin{{itemize}}[leftmargin=1.5em]
\item Regression: natural prompt-disjoint train/test split with natural sampling.
\item Classification: train balanced by downsampling, test kept at natural prevalence.
\item This split is now part of the settled object, not an open choice:
  \begin{{itemize}}[leftmargin=1.5em]
  \item regression should not be balanced because the target is continuous;
  \item classification stays balanced at train time because the target is binary and rare on several datasets.
  \end{{itemize}}
\end{{itemize}}

\subsection*{{Predictor views}}
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}p{{1.5in}}p{{4.7in}}@{{}}}}
\toprule
View & Definition \\
\midrule
{chr(10).join(predictor_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Hyperparameters}}
\begin{{itemize}}[leftmargin=1.5em]
\item Regression activation probes keep the locked full-train object: \texttt{{h128 d1}}, \texttt{{dropout=0.1}}, seeds \texttt{{0,1,2}}, \texttt{{epochs=10}}, \texttt{{batch\_size=256}}, \texttt{{lr=1e-4}}, \texttt{{weight\_decay=0.1}}.
\item Classification capacity sweep compared \texttt{{h128 d1}}, \texttt{{h256 d1}}, and \texttt{{h256 d2}} with the same optimizer surface: \texttt{{epochs=15}}, \texttt{{batch\_size=256}}, \texttt{{lr=1e-4}}, \texttt{{weight\_decay=0.05}}, \texttt{{dropout=0.1}}.
\item The current recommended binary activation surface is \texttt{{ensemble h256 d1}}. The current last-layer control is \texttt{{h256 d2}}.
\end{{itemize}}

\section*{{Metrics In Plain English}}
\begin{{itemize}}[leftmargin=1.5em]
\item \texttt{{top\_20p\_capture}}: sort held-out prompts by score, keep the top \texttt{{20\%}}, and ask how much of the actual long-completion mass those prompts contain.
\item \texttt{{RMSE}}: ordinary pointwise prediction error on aligned prompt pairs. Lower is better.
\item \texttt{{PR-AUC}}: ranking quality for the positive class. This is the main binary metric because several datasets are highly imbalanced.
\item Positive precision / recall / \texttt{{F1}}: threshold metrics for how the chosen operating point behaves on the positive class.
\item Macro \texttt{{F1}}: average \texttt{{F1}} across both classes, so it penalizes collapsing to one class.
\item \texttt{{Spearman}}: monotone-ordering diagnostic only. It is not the task definition.
\end{{itemize}}

\section*{{Regression Results}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/regression_views.png}}
\end{{figure}}

\begin{{center}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
Dataset & Prompt \texttt{{top\_20}} & Last-layer \texttt{{top\_20}} & Ensemble \texttt{{top\_20}} & Prompt \texttt{{RMSE}} & Last-layer \texttt{{RMSE}} & Ensemble \texttt{{RMSE}} \\
\midrule
{chr(10).join(regression_table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{center}}

Plain read:
\begin{{itemize}}[leftmargin=1.5em]
\item Screening is the right primary read for this lane.
\item Ensemble is strongest on \texttt{{GPQA}}, \texttt{{MMLU-Pro}}, and \texttt{{LiveCodeBench}}.
\item Prompt length still wins on calibration for \texttt{{AIME}}, \texttt{{MATH-500}}, and \texttt{{MMLU-Pro}}.
\item So \texttt{{mean\_relative\_length}} is useful as a long-completion screening score, but still mixed as a calibrated regression target.
\end{{itemize}}

\section*{{Classification Results}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/binary_views.png}}
\end{{figure}}

\begin{{center}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
Dataset & Prev. & Prompt \texttt{{PR-AUC}} & Best prompt-only & Ensemble \texttt{{PR-AUC}} & Prec. & Rec. & Pos. \texttt{{F1}} & Macro \texttt{{F1}} \\
\midrule
{chr(10).join(binary_table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{center}}

Plain read:
\begin{{itemize}}[leftmargin=1.5em]
\item \texttt{{majority\_s\_0.5}} remains the cleaner finished head.
\item \texttt{{ensemble h256 d1}} is still the best single global activation surface by \texttt{{PR-AUC}}.
\item Threshold metrics stay recall-heavy on rare-positive datasets because train balance and test prevalence differ.
\item That is why \texttt{{PR-AUC}} remains the headline metric and precision / recall / \texttt{{F1}} stay diagnostic.
\end{{itemize}}

\section*{{Why The Metadata Baseline Is Strong}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/prompt_length_quartiles.png}}
\end{{figure}}

\subsection*{{Prompt-only cues by dataset}}
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}lp{{2.2in}}p{{2.2in}}@{{}}}}
\toprule
Dataset & Best regression prompt-only cue & Best classification prompt-only cue \\
\midrule
{chr(10).join(prompt_only_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Plain-English interpretation}}
\begin{{itemize}}[leftmargin=1.5em]
\item The old ``metadata baseline'' is narrow: on this run it is really just prompt length, because \texttt{{effective\_max\_tokens}} is fixed at \texttt{{30000}}.
\item Prompt length predicts completion length mostly because, on these datasets, longer prompts usually mean the prompt itself contains more work.
\item That story is strongest on \texttt{{AIME}}, \texttt{{MATH-500}}, and much of \texttt{{LiveCodeBench}}.
\item \texttt{{MMLU-Pro}} has the same effect, but more weakly.
\item \texttt{{GPQA}} is the honest exception: the longest prompts are not the riskiest prompts there, which is why prompt-shape models help more than raw length.
\item Classification inherits some of the same prompt-only signal because \texttt{{majority\_s\_0.5}} is still largely asking whether rollouts cross about half the fixed budget.
\item That is why prompt-only features can be strong on both heads without implying a hidden implementation bug.
\end{{itemize}}

\subsection*{{Quartile read}}
\begin{{itemize}}[leftmargin=1.5em]
{chr(10).join(quartile_lines)}
\end{{itemize}}

\section*{{Bottom Line}}
\begin{{itemize}}[leftmargin=1.5em]
\item Regression answer: keep the natural split / natural sampler object and read it screening-first.
\item Classification answer: keep balanced train / natural test and use \texttt{{ensemble h256 d1}} as the current default activation surface.
\item Metadata answer: prompt-only predictors are strong mainly because prompt surface structure already acts as a workload proxy on these fixed-budget targets.
\item Activation answer: the current surface shows lift over raw prompt length on some slices, but it still does not prove general lift over strong prompt-only controls.
\item Next honest check: a stronger prompt-shape baseline plus residualized activation lift on the natural regression head.
\end{{itemize}}

\section*{{Artifacts}}
\begin{{itemize}}[leftmargin=1.5em]
\item unified bundle: \texttt{{outputs/weeks/2026-W15/prompt\_profile\_unified\_report\_20260409/}}
\item source notes behind this synthesis:
  \begin{{itemize}}[leftmargin=1.5em]
  \item \texttt{{docs/weeks/2026-W14/prompt-profile-combined-audit-2026-04-05.md}}
  \item \texttt{{docs/weeks/2026-W14/prompt-profile-binary-capacity-controls-2026-04-04.md}}
  \item \texttt{{docs/weeks/2026-W14/prompt-profile-natural-regression-rerun-2026-04-05.md}}
  \item \texttt{{docs/weeks/2026-W15/prompt-profile-length-mechanism-2026-04-09.md}}
  \end{{itemize}}
\end{{itemize}}

\end{{document}}
"""
    return textwrap.dedent(tex).strip() + "\n"


def compile_pdf(tex_path: Path) -> None:
    for _ in range(2):
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"pdflatex failed while building {tex_path.name}.\n{exc.stdout}"
            ) from exc
    for suffix in [".aux", ".log", ".out"]:
        aux_path = tex_path.with_suffix(suffix)
        if aux_path.exists():
            aux_path.unlink()


def write_doc(path: Path, text: str) -> None:
    path.write_text(text)


def maybe_reuse_existing_bundle(out_dir: Path, doc_out: Path) -> bool:
    if MATPLOTLIB_IMPORT_ERROR is None:
        return False

    required_paths = [
        out_dir / "prompt_profile_unified_report_20260409.pdf",
        out_dir / "prompt_profile_unified_report_20260409.tex",
        out_dir / "figures" / "regression_views.png",
        out_dir / "figures" / "binary_views.png",
        out_dir / "figures" / "prompt_length_quartiles.png",
        doc_out,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "matplotlib is unavailable and the reusable committed bundle is incomplete "
            f"({missing_text}). Re-run inside the prompt-profile environment or install "
            "matplotlib first."
        )
    print(
        "matplotlib is unavailable; reusing the existing committed report bundle in "
        f"{out_dir}",
        flush=True,
    )
    return True


def ensure_figures(
    figures_dir: Path,
    regression_summary: dict[str, Any],
    prompt_models: dict[str, dict[str, dict[str, float]]],
    binary_summary: dict[str, Any],
    best_prompt_only: dict[str, dict[str, float | str]],
    prompt_bins: dict[str, list[dict[str, float]]],
) -> None:
    figure_paths = [
        figures_dir / "regression_views.png",
        figures_dir / "binary_views.png",
        figures_dir / "prompt_length_quartiles.png",
    ]
    if MATPLOTLIB_IMPORT_ERROR is not None:
        missing = [path.name for path in figure_paths if not path.exists()]
        if missing:
            missing_text = ", ".join(missing)
            raise SystemExit(
                "matplotlib is required to regenerate missing figures "
                f"({missing_text}). Re-run inside the prompt-profile environment "
                "or install matplotlib first."
            )
        print(
            "matplotlib is unavailable; reusing the existing committed figures in "
            f"{figures_dir}",
            flush=True,
        )
        return

    plot_regression_views(figure_paths[0], regression_summary, prompt_models)
    plot_binary_views(figure_paths[1], binary_summary, best_prompt_only)
    plot_quartiles(figure_paths[2], prompt_bins)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-out", type=Path, default=DOC_OUT)
    args = parser.parse_args()

    ensure_clean_dir(args.out_dir)
    if maybe_reuse_existing_bundle(args.out_dir, args.doc_out):
        return

    regression_summary = load_json(REGRESSION_SUMMARY_PATH)
    binary_summary = load_json(BINARY_SUMMARY_PATH)
    metadata_audit = load_json(METADATA_AUDIT_PATH)
    _mechanism_summary = load_json(MECHANISM_SUMMARY_PATH)
    metadata_mechanism_summary = load_json(METADATA_MECHANISM_SUMMARY_PATH)
    prompt_model_rows = read_csv_rows(REGRESSION_MODELS_CSV)
    prompt_bins_rows = read_csv_rows(PROMPT_BINS_CSV)

    prompt_models = build_regression_prompt_models(prompt_model_rows)
    best_prompt_only = build_binary_best_prompt_only(metadata_audit)
    prompt_bins = build_prompt_bins(prompt_bins_rows)

    figures_dir = args.out_dir / "figures"
    ensure_figures(
        figures_dir,
        regression_summary,
        prompt_models,
        binary_summary,
        best_prompt_only,
        prompt_bins,
    )

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tex_text = build_tex(
        generated_utc,
        regression_summary,
        metadata_mechanism_summary,
        prompt_models,
        binary_summary,
        best_prompt_only,
        prompt_bins,
    )
    tex_path = args.out_dir / "prompt_profile_unified_report_20260409.tex"
    tex_path.write_text(tex_text)
    compile_pdf(tex_path)

    markdown_text = build_markdown(
        generated_utc, args.out_dir, regression_summary, binary_summary, best_prompt_only
    )
    args.doc_out.write_text(markdown_text)


if __name__ == "__main__":
    main()
