#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


FIGURES_DIRNAME = "figures"


METRIC_GROUPS: tuple[tuple[str, str, float], ...] = (
    ("success_fraction", "Rollout success (%)", 100.0),
    ("loop_fraction", "Looped (%)", 100.0),
    ("max_length_hit_fraction", "Max-length hit (%)", 100.0),
)

OVERLAP_GROUPS: tuple[tuple[str, str, float], ...] = (
    ("loop_max_length_hit_fraction", "Looped -> max length (%)", 100.0),
    ("max_length_hit_loop_fraction", "Max length -> looped (%)", 100.0),
    ("loop_success_fraction", "Looped -> correct (%)", 100.0),
)

LENGTH_GROUPS: tuple[tuple[str, str, float], ...] = (
    ("avg_generation_length", "Average generation (k tok)", 0.001),
    ("avg_loop_generation_length", "Average looped generation (k tok)", 0.001),
    ("avg_wrong_generation_length", "Average wrong generation (k tok)", 0.001),
)


OVERLAP_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("correct_no_loop_no_max", "Correct / no loop / no max", "#2a9d8f"),
    ("correct_no_loop_max", "Correct / no loop / max", "#52b788"),
    ("correct_loop_no_max", "Correct / loop / no max", "#84a98c"),
    ("correct_loop_max", "Correct / loop / max", "#f4a261"),
    ("wrong_no_loop_no_max", "Wrong / no loop / no max", "#577590"),
    ("wrong_no_loop_max", "Wrong / no loop / max", "#277da1"),
    ("wrong_loop_no_max", "Wrong / loop / no max", "#b56576"),
    ("wrong_loop_max", "Wrong / loop / max", "#e76f51"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help=(
            "Stage specification in the form label=/path/to/cross_dataset_rollout_summary.json "
            "or label=/path/to/report_dir."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--report-stem",
        default="rollout_progression_report",
        help="Base filename for generated summary files and report artifacts.",
    )
    parser.add_argument(
        "--title",
        default="Rollout Degeneration Progression Report",
        help="Report title.",
    )
    parser.add_argument(
        "--pdflatex",
        default="pdflatex",
        help="pdflatex executable to use for report compilation.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write summary files and TeX but do not compile the PDF.",
    )
    return parser.parse_args()


def _latex_escape(text: Any) -> str:
    raw = "" if text is None else str(text)
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
    return "".join(replacements.get(char, char) for char in raw)


def _parse_stage_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid --stage {raw!r}; expected label=/path/to/summary.json")
    label, path_str = raw.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip())
    if not label:
        raise SystemExit(f"Invalid --stage {raw!r}; stage label is empty.")
    if path.is_dir():
        path = path / "cross_dataset_rollout_summary.json"
    if not path.is_file():
        raise SystemExit(f"Missing stage summary JSON: {path}")
    return label, path.resolve()


def _load_stage_payloads(stage_specs: list[str]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for raw in stage_specs:
        label, path = _parse_stage_spec(raw)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
            raise SystemExit(f"Stage summary payload is malformed: {path}")
        stages.append(
            {
                "label": label,
                "path": path,
                "payload": payload,
            }
        )
    return stages


def _validate_progression_contract(stages: list[dict[str, Any]]) -> None:
    baseline = stages[0]["payload"]
    baseline_rows = baseline["datasets"]
    baseline_keys = [row["key"] for row in baseline_rows]
    baseline_generation = baseline.get("common_generation_config", {})
    baseline_statistics = sorted(baseline.get("tracked_statistics", []))
    mismatches: list[str] = []

    for stage in stages[1:]:
        payload = stage["payload"]
        keys = [row["key"] for row in payload["datasets"]]
        if keys != baseline_keys:
            mismatches.append(
                f"{stage['label']}: dataset keys {keys!r} do not match baseline {baseline_keys!r}"
            )
        stats = sorted(payload.get("tracked_statistics", []))
        if stats != baseline_statistics:
            mismatches.append(
                f"{stage['label']}: tracked statistics {stats!r} do not match baseline {baseline_statistics!r}"
            )
        generation = payload.get("common_generation_config", {})
        for key in (
            "temperature",
            "num_generations",
            "max_tokens",
            "max_model_len",
            "dtype",
            "trust_remote_code",
            "tp",
        ):
            if generation.get(key) != baseline_generation.get(key):
                mismatches.append(
                    f"{stage['label']}: generation_config.{key}={generation.get(key)!r} "
                    f"!= baseline {baseline_generation.get(key)!r}"
                )

    if mismatches:
        raise SystemExit(
            "Progression report requires a shared measurement contract.\n"
            + "\n".join(f"- {line}" for line in mismatches)
        )


def _rows_by_stage_and_key(stages: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    mapping: dict[str, dict[str, dict[str, Any]]] = {}
    for stage in stages:
        mapping[stage["label"]] = {
            row["key"]: row for row in stage["payload"]["datasets"]
        }
    return mapping


def _write_summary_files(stages: list[dict[str, Any]], out_dir: Path) -> None:
    summary_json = out_dir / "rollout_progression_summary.json"
    summary_csv = out_dir / "rollout_progression_summary.csv"
    payload = {
        "stages": [
            {
                "label": stage["label"],
                "source": str(stage["path"]),
                "model_id": stage["payload"].get("model_id"),
                "datasets": stage["payload"]["datasets"],
                "common_generation_config": stage["payload"].get("common_generation_config"),
                "tracked_statistics": stage["payload"].get("tracked_statistics"),
            }
            for stage in stages
        ]
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    fieldnames = [
        "stage",
        "model_id",
        "dataset_key",
        "display_name",
        "prompt_format_requested",
        "prompt_format_resolved",
        "samples",
        "generated",
        "graded",
        "correct",
        "wrong",
        "looped",
        "max_length_hits",
        "loop_and_max_hits",
        "correct_and_looped",
        "correct_and_max_hits",
        "correct_and_loop_and_max_hits",
    ] + [name for name, _, _ in METRIC_GROUPS + OVERLAP_GROUPS + LENGTH_GROUPS]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stage in stages:
            for row in stage["payload"]["datasets"]:
                writer.writerow(
                    {
                        "stage": stage["label"],
                        "model_id": row["model_id"],
                        "dataset_key": row["key"],
                        "display_name": row["display_name"],
                        "prompt_format_requested": row.get("prompt_format_requested"),
                        "prompt_format_resolved": row.get("prompt_format_resolved"),
                        "samples": row.get("samples"),
                        "generated": row.get("generated"),
                        "graded": row.get("graded"),
                        "correct": row.get("correct"),
                        "wrong": row.get("wrong"),
                        "looped": row.get("looped"),
                        "max_length_hits": row.get("max_length_hits"),
                        "loop_and_max_hits": row.get("loop_and_max_hits"),
                        "correct_and_looped": row.get("correct_and_looped"),
                        "correct_and_max_hits": row.get("correct_and_max_hits"),
                        "correct_and_loop_and_max_hits": row.get("correct_and_loop_and_max_hits"),
                        **{
                            metric_name: row.get(metric_name)
                            for metric_name, _, _ in METRIC_GROUPS + OVERLAP_GROUPS + LENGTH_GROUPS
                        },
                    }
                )


def _plot_metric_group(
    stages: list[dict[str, Any]],
    metric_group: tuple[tuple[str, str, float], ...],
    out_path: Path,
    title: str,
) -> None:
    rows_by_stage = _rows_by_stage_and_key(stages)
    stage_labels = [stage["label"] for stage in stages]
    dataset_rows = stages[0]["payload"]["datasets"]
    colors = ["#264653", "#2a9d8f", "#e76f51", "#6d597a", "#f4a261"]

    fig, axes = plt.subplots(1, len(metric_group), figsize=(15.0, 4.8), sharex=True)
    if len(metric_group) == 1:
        axes = [axes]

    for ax, (metric_name, ylabel, scale) in zip(axes, metric_group):
        for color, dataset_row in zip(colors, dataset_rows):
            key = dataset_row["key"]
            values = []
            for stage_label in stage_labels:
                value = rows_by_stage[stage_label][key].get(metric_name)
                values.append(math.nan if value is None else float(value) * scale)
            ax.plot(
                stage_labels,
                values,
                marker="o",
                linewidth=2.0,
                markersize=6.0,
                label=dataset_row["short_name"],
                color=color,
            )
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _derive_overlap_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = sum(int(row.get("generated") or 0) for row in rows)
    correct = sum(int(row.get("correct") or 0) for row in rows)
    looped = sum(int(row.get("looped") or 0) for row in rows)
    max_hits = sum(int(row.get("max_length_hits") or 0) for row in rows)
    loop_and_max = sum(int(row.get("loop_and_max_hits") or 0) for row in rows)
    correct_and_looped = sum(int(row.get("correct_and_looped") or 0) for row in rows)
    correct_and_max = sum(int(row.get("correct_and_max_hits") or 0) for row in rows)
    triple_values = [
        row.get("correct_and_loop_and_max_hits")
        for row in rows
    ]
    if all(value is not None for value in triple_values):
        correct_and_loop_and_max = sum(int(value or 0) for value in triple_values)
    else:
        lower_bound = max(
            0,
            correct_and_looped + correct_and_max - correct,
            correct_and_looped + loop_and_max - looped,
            correct_and_max + loop_and_max - max_hits,
        )
        upper_bound = min(
            correct_and_looped,
            correct_and_max,
            loop_and_max,
            total - correct - looped - max_hits + correct_and_looped + correct_and_max + loop_and_max,
        )
        if lower_bound > upper_bound:
            raise SystemExit(
                "Could not infer a feasible triple-overlap count from the saved "
                f"pairwise marginals: lower={lower_bound}, upper={upper_bound}."
            )
        # Legacy bundles did not save the triple-overlap count. Use the minimal
        # feasible value so the composition plot stays conservative.
        correct_and_loop_and_max = lower_bound

    wrong = total - correct
    correct_loop = correct_and_looped
    correct_no_loop = correct - correct_loop
    wrong_loop = looped - correct_loop
    wrong_no_loop = wrong - wrong_loop

    correct_loop_max = correct_and_loop_and_max
    correct_loop_no_max = correct_loop - correct_loop_max
    correct_no_loop_max = correct_and_max - correct_loop_max
    correct_no_loop_no_max = correct_no_loop - correct_no_loop_max

    wrong_max = max_hits - correct_and_max
    wrong_loop_max = loop_and_max - correct_loop_max
    wrong_loop_no_max = wrong_loop - wrong_loop_max
    wrong_no_loop_max = wrong_max - wrong_loop_max
    wrong_no_loop_no_max = wrong_no_loop - wrong_no_loop_max

    breakdown = {
        "correct_no_loop_no_max": correct_no_loop_no_max,
        "correct_no_loop_max": correct_no_loop_max,
        "correct_loop_no_max": correct_loop_no_max,
        "correct_loop_max": correct_loop_max,
        "wrong_no_loop_no_max": wrong_no_loop_no_max,
        "wrong_no_loop_max": wrong_no_loop_max,
        "wrong_loop_no_max": wrong_loop_no_max,
        "wrong_loop_max": wrong_loop_max,
    }
    if any(value < 0 for value in breakdown.values()):
        raise SystemExit(
            f"Derived a negative overlap bucket from saved counts: {breakdown}"
        )
    return breakdown


def _build_overlap_composition_plot(stages: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bottoms = [0.0] * len(stages)
    stage_labels = [stage["label"] for stage in stages]
    stage_breakdowns = [
        _derive_overlap_breakdown(stage["payload"]["datasets"]) for stage in stages
    ]

    for key, label, color in OVERLAP_CATEGORIES:
        values = []
        for breakdown in stage_breakdowns:
            total = sum(breakdown.values())
            value = 100.0 * breakdown[key] / total if total else 0.0
            values.append(value)
        ax.bar(stage_labels, values, bottom=bottoms, label=label, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_ylabel("Percent of generated rollouts")
    ax.set_title("Within-stage overlap composition across all datasets")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _figure_paths(out_dir: Path) -> dict[str, Path]:
    figures_dir = out_dir / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "rates": figures_dir / "progression_rates.png",
        "overlap": figures_dir / "progression_overlap.png",
        "lengths": figures_dir / "progression_lengths.png",
        "composition": figures_dir / "stage_overlap_composition.png",
    }


def _bundle_timestamp(stages: list[dict[str, Any]]) -> str:
    parsed: list[datetime] = []
    for stage in stages:
        for row in stage["payload"]["datasets"]:
            value = row.get("timestamp")
            if value:
                parsed.append(datetime.fromisoformat(value))
    if not parsed:
        return "Generated from stage summaries"
    latest = max(parsed)
    return latest.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _average_metric(stage: dict[str, Any], metric_name: str) -> float:
    values = [
        float(row[metric_name])
        for row in stage["payload"]["datasets"]
        if row.get(metric_name) is not None
    ]
    return sum(values) / len(values) if values else math.nan


def _build_findings(stages: list[dict[str, Any]]) -> str:
    base, *rest = stages
    lines = []
    if rest:
        base_loop = _average_metric(base, "loop_fraction")
        base_cap = _average_metric(base, "max_length_hit_fraction")
        last = rest[-1]
        last_loop = _average_metric(last, "loop_fraction")
        last_cap = _average_metric(last, "max_length_hit_fraction")
        if not math.isnan(base_loop) and not math.isnan(last_loop):
            lines.append(
                f"Average loop fraction rises from {100.0 * base_loop:.1f}\\% on {base['label']} "
                f"to {100.0 * last_loop:.1f}\\% on {last['label']}."
            )
        if not math.isnan(base_cap) and not math.isnan(last_cap):
            lines.append(
                f"Average max-length-hit fraction rises from {100.0 * base_cap:.1f}\\% on {base['label']} "
                f"to {100.0 * last_cap:.1f}\\% on {last['label']}."
            )

    for dataset_row in stages[0]["payload"]["datasets"]:
        key = dataset_row["key"]
        stage_values = [
            (
                stage["label"],
                stage["payload"]["datasets"][idx]["loop_fraction"],
                stage["payload"]["datasets"][idx]["max_length_hit_fraction"],
            )
            for stage in stages
            for idx, row in enumerate(stage["payload"]["datasets"])
            if row["key"] == key
        ]
        if len(stage_values) != len(stages):
            continue
        base_label, base_loop, base_cap = stage_values[0]
        last_label, last_loop, last_cap = stage_values[-1]
        if base_loop is None or last_loop is None or base_cap is None or last_cap is None:
            continue
        lines.append(
            f"{dataset_row['display_name']}: loop fraction moves from {100.0 * float(base_loop):.1f}\\% "
            f"({base_label}) to {100.0 * float(last_loop):.1f}\\% ({last_label}); "
            f"max-length-hit fraction moves from {100.0 * float(base_cap):.1f}\\% "
            f"to {100.0 * float(last_cap):.1f}\\%."
        )
    return "\n\n".join(lines[:6])


def _build_tex(stages: list[dict[str, Any]], out_dir: Path, report_stem: str, title: str) -> Path:
    figures = _figure_paths(out_dir)
    stage_map = "\n".join(
        [
            rf"\item \textbf{{{_latex_escape(stage['label'])}}}: "
            rf"\texttt{{{_latex_escape(stage['payload'].get('model_id'))}}}"
            for stage in stages
        ]
    )
    findings = _build_findings(stages)
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.6em}}

\title{{{_latex_escape(title)}}}
\author{{}}
\date{{{_latex_escape(_bundle_timestamp(stages))}}}

\begin{{document}}
\maketitle

\section*{{Stage Map}}
\begin{{itemize}}[leftmargin=1.5em]
{stage_map}
\end{{itemize}}

\section*{{Shared Contract}}
This progression bundle compares multiple checkpoints under one rollout-statistics contract. The stage summaries were validated to share temperature, number of generations, max token budget, max model length, dtype, trust-remote-code flag, loop detector, and tracked statistics before plotting the progression.

\section*{{Headline Read}}
{findings}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{FIGURES_DIRNAME}/progression_rates.png}}
\caption{{Progression of rollout success, loop rate, and max-length-hit rate across stages, broken out by dataset.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{FIGURES_DIRNAME}/progression_overlap.png}}
\caption{{Progression of overlap conditionals across stages, broken out by dataset.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{FIGURES_DIRNAME}/progression_lengths.png}}
\caption{{Progression of generation-length statistics across stages, broken out by dataset.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{FIGURES_DIRNAME}/stage_overlap_composition.png}}
\caption{{Within-stage overlap composition aggregated across all datasets. This uses the saved raw counts to decompose generated rollouts by correctness, loop status, and max-length-hit status.}}
\end{{figure}}

\section*{{Interpretation}}
The progression figures answer the exact object of the note: where the degenerate-rollout regime enters the model family under a fixed rollout contract. The line plots are the stage comparison surface; the overlap-composition panel makes the within-stage structure visible without collapsing everything to one scalar rate. If the loop and max-length-hit curves are near-zero in the base stage and then rise sharply after SFT or RLVR, that is direct evidence for a stage-linked introduction of degenerate rollouts. If they are already substantial in the base model, the hypothesis weakens.

\end{{document}}
"""
    tex_path = out_dir / f"{report_stem}.tex"
    tex_path.write_text(tex)
    return tex_path


def _compile_pdf(tex_path: Path, pdflatex: str) -> Path:
    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        subprocess.run(
            cmd,
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    args = parse_args()
    stages = _load_stage_payloads(args.stage)
    _validate_progression_contract(stages)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_files(stages, out_dir)

    figures = _figure_paths(out_dir)
    _plot_metric_group(stages, METRIC_GROUPS, figures["rates"], "Stage progression: success, looping, and max-length hits")
    _plot_metric_group(stages, OVERLAP_GROUPS, figures["overlap"], "Stage progression: overlap conditionals")
    _plot_metric_group(stages, LENGTH_GROUPS, figures["lengths"], "Stage progression: generation lengths")
    _build_overlap_composition_plot(stages, figures["composition"])

    tex_path = _build_tex(stages, out_dir, args.report_stem, args.title)
    if not args.skip_pdf:
        _compile_pdf(tex_path, args.pdflatex)


if __name__ == "__main__":
    main()
