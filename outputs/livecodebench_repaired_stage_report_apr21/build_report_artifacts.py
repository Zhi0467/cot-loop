from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
REPORT_STEM = "livecodebench_repaired_stage_report_apr21"


@dataclass(frozen=True)
class MetricRow:
    label: str
    group: str
    pr_auc: float
    roc_auc: float


def load_json(name: str):
    return json.loads((SOURCE / name).read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def short_hash(value: str, n: int = 12) -> str:
    return value[:n]


def latex_escape(text: str) -> str:
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
    return "".join(replacements.get(ch, ch) for ch in text)


def build_summary() -> dict:
    best_layers = load_json("best_layers.json")
    rfm_config = load_json("rfm_run_config.json")
    split_manifest = load_json("rfm_split_manifest.json")
    vector_summary = load_json("vector_exports_summary.json")
    prompt_summary = load_json("prompt_baselines_summary.json")
    activation_summary = load_json("activation_mean_summary.json")
    steering_summary = load_json("steering_summary.json")

    selected_detector = best_layers["best_layers"][0]
    source_train = split_manifest["preprocessing"]["train_val_split"]["source_train_prompt_ids"]
    fit_train = split_manifest["preprocessing"]["train_val_split"]["fit_train_prompt_ids"]
    val_prompts = split_manifest["preprocessing"]["train_val_split"]["val_prompt_ids"]
    test_prompts = split_manifest["prompt_ids"]["test"]

    activation_rows = {
        (row["model"], row["checkpoint_rule"]): row for row in activation_summary
    }
    prompt_rows = {row["model_name"]: row for row in prompt_summary}
    vector_rows = sorted(vector_summary["layers"], key=lambda row: row["layer"])
    steering_rows = steering_summary["conditions"]
    no_steer_record = load_json("no_steer_steering_run_record.json")

    stability_sorted = sorted(
        vector_rows,
        key=lambda row: row["direction_bootstrap_mean_cosine"],
        reverse=True,
    )
    selected_vector = next(
        row for row in vector_rows if row["layer"] == selected_detector["layer"]
    )

    return {
        "generated_at_utc": utc_now(),
        "dataset": {
            "benchmark": split_manifest["benchmark"],
            "display_name": rfm_config["display_name"],
            "model_id": rfm_config["model_id"],
            "model_revision": rfm_config["model_revision"],
            "tokenizer_revision": rfm_config["tokenizer_revision"],
            "feature_key": split_manifest["preprocessing"]["feature_key"],
            "label_name": split_manifest["preprocessing"]["stage_label_name"],
            "label_definition": "Prompt-level strict majority of 4 saved rollouts with relative_length > 0.5.",
            "source_archive_dir": split_manifest["preprocessing"]["source_data_dir"],
            "archive_tail_threshold": split_manifest["preprocessing"]["archive_tail_threshold"],
            "stage_tail_threshold": split_manifest["preprocessing"]["stage_tail_threshold"],
            "stage_label_requires_rollout_recompute": split_manifest["preprocessing"][
                "stage_label_requires_rollout_recompute"
            ],
            "sample_shape": split_manifest["preprocessing"]["sample_shape"],
            "scaler": split_manifest["preprocessing"]["scaler"],
            "source_train_prompt_count": len(source_train),
            "fit_train_prompt_count": len(fit_train),
            "val_prompt_count": len(val_prompts),
            "test_prompt_count": len(test_prompts),
            "fit_train_positive_count": split_manifest["train_num_positive"],
            "val_positive_count": split_manifest["val_num_positive"],
            "test_positive_count": split_manifest["test_num_positive"],
            "fit_train_balance_policy": split_manifest["preprocessing"]["train_val_split"][
                "balance_train"
            ],
            "fit_train_prompt_ids": fit_train,
            "val_prompt_ids": val_prompts,
            "test_prompt_ids": test_prompts,
            "fit_train_prompt_ids_sha256": split_manifest["preprocessing"]["train_val_split"][
                "fit_train_prompt_ids_sha256"
            ],
            "val_prompt_ids_sha256": split_manifest["preprocessing"]["train_val_split"][
                "val_prompt_ids_sha256"
            ],
            "test_prompt_ids_sha256": split_manifest["preprocessing"]["train_val_split"][
                "test_prompt_ids_sha256"
            ],
        },
        "detector": {
            "git_commit": rfm_config["git_commit"],
            "seed": rfm_config["seed"],
            "bootstrap_seed": rfm_config["bootstrap_seed"],
            "bootstrap_samples": rfm_config["bootstrap_samples"],
            "iters": rfm_config["iters"],
            "reg": rfm_config["reg"],
            "bandwidth_grid": rfm_config["bandwidths"],
            "selected_layer": selected_detector["layer"],
            "selected_bandwidth": selected_detector["bandwidth"],
            "score_sign": selected_detector["score_sign"],
            "decision_threshold": selected_detector["decision_threshold"],
            "val_pr_auc": selected_detector["val_pr_auc"],
            "val_roc_auc": selected_detector["val_roc_auc"],
            "test_pr_auc": selected_detector["test_pr_auc"],
            "test_roc_auc": selected_detector["test_roc_auc"],
            "test_accuracy": selected_detector["test_accuracy"],
            "test_positive_f1": selected_detector["test_positive_f1"],
            "test_bootstrap": selected_detector["test_bootstrap"],
            "artifact_record_path": selected_detector["artifact_record_path"],
            "checkpoint_path": selected_detector["checkpoint_path"],
        },
        "detector_comparison": {
            "prompt_only": prompt_rows,
            "activation_mean_summary": activation_summary,
        },
        "direction_quality": {
            "vector_export_git_commit": vector_summary["git_commit"],
            "source_git_commit": vector_summary["source_git_commit"],
            "schema_name": vector_summary["schema_name"],
            "vector_extraction_formula": vector_summary["vector_extraction_formula"],
            "vector_scale": vector_summary["vector_scale"],
            "direction_bootstrap_samples": vector_summary["direction_bootstrap_samples"],
            "direction_bootstrap_seed": vector_summary["direction_bootstrap_seed"],
            "selected_layer_projection": {
                "layer": selected_vector["layer"],
                "test_pr_auc": selected_vector["test_pr_auc"],
                "test_roc_auc": selected_vector["test_roc_auc"],
                "val_pr_auc": selected_vector["val_pr_auc"],
                "raw_vector_norm": selected_vector["raw_vector_norm"],
                "vector_checksum": selected_vector["vector_checksum"],
                "direction_bootstrap_mean_cosine": selected_vector[
                    "direction_bootstrap_mean_cosine"
                ],
                "direction_bootstrap_low_cosine": selected_vector[
                    "direction_bootstrap_low_cosine"
                ],
                "direction_bootstrap_high_cosine": selected_vector[
                    "direction_bootstrap_high_cosine"
                ],
            },
            "weakest_layer_low_cosine": min(
                row["direction_bootstrap_low_cosine"] for row in vector_rows
            ),
            "all_layers": vector_rows,
            "top_stable_layers": [
                {
                    "layer": row["layer"],
                    "mean_cosine": row["direction_bootstrap_mean_cosine"],
                    "low_cosine": row["direction_bootstrap_low_cosine"],
                    "high_cosine": row["direction_bootstrap_high_cosine"],
                }
                for row in stability_sorted[:5]
            ],
        },
        "steering": {
            "summary_schema_name": steering_summary["schema_name"],
            "vector_bundle_hash": steering_summary["vector_bundle_hash"],
            "config": steering_summary["config"],
            "conditions": steering_rows,
            "prompt_ids": no_steer_record["prompt_ids"],
            "prompt_text_hash": no_steer_record["prompt_text_hash"],
            "hook_site": no_steer_record["hook_site"],
            "t": no_steer_record["t"],
            "generation_config": no_steer_record["generation_config"],
            "grader_version": no_steer_record["grader_version"],
            "run_git_commit": no_steer_record["git_commit"],
            "records": {
                "no_steer": load_json("no_steer_steering_run_record.json"),
                "minus_v_spherical": load_json("minus_v_spherical_steering_run_record.json"),
                "plus_v_spherical": load_json("plus_v_spherical_steering_run_record.json"),
                "random_spherical": load_json("random_spherical_steering_run_record.json"),
            },
        },
        "narrative": {
            "detector_read": (
                "On the repaired LiveCodeBench object, RFM beats the prompt-only and "
                "activation-linear baselines, ties activation MLP last-layer under "
                "best-rank selection, and trails the same MLP row slightly under the "
                "best-loss checkpoint rule."
            ),
            "direction_read": (
                "The exported vector bundle is directionally stable enough to treat as "
                "a real stage-2 object: all 28 layers clear mean bootstrap cosine >= 0.781, "
                "with the strongest coherence in late layers 23-26."
            ),
            "steering_read": (
                "The first larger 32-prompt spherical steering table is negative: all "
                "four conditions stay at 0/32 pass@1, and every steered condition is "
                "worse than baseline on loop fraction."
            ),
        },
    }


def detector_rows(summary: dict) -> list[MetricRow]:
    prompt_rows = summary["detector_comparison"]["prompt_only"]
    activation_rows = {
        (row["model"], row["checkpoint_rule"]): row
        for row in summary["detector_comparison"]["activation_mean_summary"]
    }
    det = summary["detector"]
    return [
        MetricRow("Prompt length", "Prompt-only", prompt_rows["prompt_length"]["test_pr_auc"], prompt_rows["prompt_length"]["test_roc_auc"]),
        MetricRow("Prompt shape linear", "Prompt-only", prompt_rows["prompt_shape_linear"]["test_pr_auc"], prompt_rows["prompt_shape_linear"]["test_roc_auc"]),
        MetricRow("Prompt shape tree", "Prompt-only", prompt_rows["prompt_shape_tree"]["test_pr_auc"], prompt_rows["prompt_shape_tree"]["test_roc_auc"]),
        MetricRow("Activation linear (last)", "Activation linear", activation_rows[("linear_last_layer", "best_rank")]["mean_pr_auc"], activation_rows[("linear_last_layer", "best_rank")]["mean_roc_auc"]),
        MetricRow("Activation linear (ensemble)", "Activation linear", activation_rows[("linear_ensemble", "best_rank")]["mean_pr_auc"], activation_rows[("linear_ensemble", "best_rank")]["mean_roc_auc"]),
        MetricRow("Activation MLP (ensemble)", "Activation MLP", activation_rows[("mlp256d1_ensemble", "best_rank")]["mean_pr_auc"], activation_rows[("mlp256d1_ensemble", "best_rank")]["mean_roc_auc"]),
        MetricRow("Activation MLP (last, best-rank)", "Activation MLP", activation_rows[("mlp256d1_last_layer", "best_rank")]["mean_pr_auc"], activation_rows[("mlp256d1_last_layer", "best_rank")]["mean_roc_auc"]),
        MetricRow("Activation MLP (last, best-loss)", "Activation MLP", activation_rows[("mlp256d1_last_layer", "best_loss")]["mean_pr_auc"], activation_rows[("mlp256d1_last_layer", "best_loss")]["mean_roc_auc"]),
        MetricRow(f"RFM layer {det['selected_layer']}", "RFM", det["test_pr_auc"], det["test_roc_auc"]),
    ]


def render_detector_plot(summary: dict) -> None:
    rows = detector_rows(summary)
    labels = [row.label for row in rows]
    pr = [row.pr_auc for row in rows]
    roc = [row.roc_auc for row in rows]
    groups = [row.group for row in rows]
    palette = {
        "Prompt-only": "#7f7f7f",
        "Activation linear": "#4c78a8",
        "Activation MLP": "#f58518",
        "RFM": "#e45756",
    }
    colors = [palette[group] for group in groups]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True)
    y = np.arange(len(rows))

    for ax, values, title in zip(axes, [pr, roc], ["Test PR-AUC", "Test ROC-AUC"]):
        ax.barh(y, values, color=colors)
        ax.set_xlim(0, 1.0)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        for yi, value in enumerate(values):
            ax.text(min(value + 0.01, 0.97), yi, f"{value:.3f}", va="center", fontsize=9)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=group)
        for group, color in palette.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("LiveCodeBench repaired detector comparison")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(ROOT / "detector_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_direction_plot(summary: dict) -> None:
    rows = summary["direction_quality"]["all_layers"]
    x = np.array([row["layer"] for row in rows])
    mean = np.array([row["direction_bootstrap_mean_cosine"] for row in rows])
    low = np.array([row["direction_bootstrap_low_cosine"] for row in rows])
    high = np.array([row["direction_bootstrap_high_cosine"] for row in rows])
    selected = summary["detector"]["selected_layer"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(x, low, high, color="#72b7b2", alpha=0.25, label="95% interval")
    ax.plot(x, mean, color="#1f77b4", marker="o", linewidth=2, label="Mean cosine")
    ax.axvline(selected, color="#e45756", linestyle="--", linewidth=1.5, label=f"Selected detector layer {selected}")
    ax.set_ylim(0.65, 0.95)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Bootstrap cosine to exported direction")
    ax.set_title("Layerwise direction stability")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / "direction_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_steering_plot(summary: dict) -> None:
    conditions = ["no_steer", "minus_v_spherical", "plus_v_spherical", "random_spherical"]
    labels = ["No steer", "-v spherical", "+v spherical", "Random spherical"]
    loop = [
        summary["steering"]["conditions"][cond]["aggregate"]["loop_fraction"]["mean"]
        for cond in conditions
    ]
    avg_len = [
        summary["steering"]["conditions"][cond]["aggregate"]["avg_generation_length"]["mean"]
        for cond in conditions
    ]
    colors = ["#4c78a8", "#e45756", "#f58518", "#72b7b2"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(labels, loop, color=colors)
    axes[0].set_ylim(0, 0.4)
    axes[0].set_ylabel("Loop fraction")
    axes[0].set_title("32-prompt steering controls")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    for i, value in enumerate(loop):
        axes[0].text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)

    axes[1].bar(labels, avg_len, color=colors)
    axes[1].set_ylim(0, 1100)
    axes[1].set_ylabel("Average generation length")
    axes[1].set_title("Same prompts, same decode config")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    for i, value in enumerate(avg_len):
        axes[1].text(i, value + 20, f"{value:.1f}", ha="center", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)

    fig.suptitle("All four conditions stayed at 0/32 pass@1")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "steering_controls.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(summary: dict) -> None:
    (ROOT / "livecodebench_stage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def detector_table_tex(summary: dict) -> str:
    rows = detector_rows(summary)
    lines = []
    for row in rows:
        lines.append(
            f"{latex_escape(row.label)} & {latex_escape(row.group)} & {row.pr_auc:.3f} & {row.roc_auc:.3f} \\\\"
        )
    return "\n".join(lines)


def write_report_tex(summary: dict) -> None:
    dataset = summary["dataset"]
    detector = summary["detector"]
    direction = summary["direction_quality"]
    steering = summary["steering"]
    top_layers = ", ".join(
        f"{row['layer']} ({row['mean_cosine']:.3f})" for row in direction["top_stable_layers"][:4]
    )
    template = dedent(
        r"""
        \documentclass[11pt]{article}
        \usepackage[margin=1in]{geometry}
        \usepackage{graphicx}
        \usepackage{booktabs}
        \usepackage{array}
        \usepackage{tabularx}
        \usepackage{longtable}
        \usepackage{float}
        \usepackage{hyperref}
        \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{0.6em}
        \begin{document}

        {\Large \textbf{LiveCodeBench repaired RFM stage report}}\\
        Generated: @@GENERATED_AT@@

        \section*{Object}
        This report freezes the current \textbf{LiveCodeBench-only} stage on the same prompt-level label family used by the other probe surfaces: \texttt{majority\_s\_0.5}, defined here as a prompt being positive when a strict majority of its four saved rollouts have relative length greater than \texttt{0.5}. The saved March archive still carries an archive-time tail bit at \texttt{0.9}, so the stage label is recomputed from saved rollout lengths rather than read from the archive bit directly.

        The frozen source train pool has @@SOURCE_TRAIN_COUNT@@ prompts. The detector fit-train split uses the same stage-materialization contract as the other probe lanes, including \texttt{balance\_train=@@BALANCE_TRAIN@@}, which yields @@FIT_TRAIN_COUNT@@ fit-train prompts, @@VAL_COUNT@@ validation prompts, and @@TEST_COUNT@@ test prompts. Positive counts are @@FIT_TRAIN_POS@@, @@VAL_POS@@, and @@TEST_POS@@ respectively.

        \begin{table}[H]
        \centering
        \begin{tabular}{lrr}
        \toprule
        Split & Prompt count & Positive count \\
        \midrule
        Source train pool & @@SOURCE_TRAIN_COUNT@@ & -- \\
        Fit-train & @@FIT_TRAIN_COUNT@@ & @@FIT_TRAIN_POS@@ \\
        Validation & @@VAL_COUNT@@ & @@VAL_POS@@ \\
        Test & @@TEST_COUNT@@ & @@TEST_POS@@ \\
        \bottomrule
        \end{tabular}
        \caption{LiveCodeBench repaired prompt object.}
        \end{table}

        \section*{Detector comparison}
        The selected RFM detector row is layer @@DETECTOR_LAYER@@ with bandwidth @@DETECTOR_BANDWIDTH@@, validation PR-AUC @@DETECTOR_VAL_PR@@, and test PR-AUC / ROC-AUC @@DETECTOR_TEST_PR@@ / @@DETECTOR_TEST_ROC@@. This is a real improvement over the prompt-only and activation-linear baselines on the repaired object, but it is not a clean activation-MLP win: activation MLP last-layer under the \emph{best-rank} checkpoint rule is essentially tied on PR-AUC, and under \emph{best-loss} it is slightly higher.

        \begin{figure}[H]
        \centering
        \includegraphics[width=0.98\textwidth]{detector_comparison.png}
        \caption{Test-set detector comparison on the repaired LiveCodeBench object. Prompt-only rows are single-model fits; activation rows are three-seed means; RFM is the selected single-seed layerwise detector.}
        \end{figure}

        \begin{table}[H]
        \centering
        \begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.39\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}rr}
        \toprule
        Method & Group & Test PR-AUC & Test ROC-AUC \\
        \midrule
        @@DETECTOR_TABLE@@
        \bottomrule
        \end{tabularx}
        \caption{Exact metrics used in the detector plot.}
        \end{table}

        \section*{Direction quality}
        The stage-2 object is no longer hypothetical. The exported vector bundle carries @@BOOTSTRAP_SAMPLES@@ bootstrap replay fits per layer under the fixed selected hyperparameters. All 28 layers have mean cosine at least 0.781 to the exported signed direction; the weakest 95\% lower bound across layers is @@WEAKEST_LOW@@. The most stable layers are @@TOP_LAYERS@@. The selected detector layer @@DETECTOR_LAYER@@ is stable enough to be usable but is not the most coherent layer: its mean cosine is @@SELECTED_MEAN@@ with lower bound @@SELECTED_LOW@@. Its 1D projection score also remains predictive on held-out data (test PR-AUC @@PROJECTION_TEST_PR@@).

        \begin{figure}[H]
        \centering
        \includegraphics[width=0.92\textwidth]{direction_stability.png}
        \caption{Bootstrap cosine stability for the exported per-layer LiveCodeBench directions.}
        \end{figure}

        \section*{Benchmark-local spherical steering}
        The first larger repaired control table is now complete on 32 held-out LiveCodeBench prompts with fixed spherical strength \texttt{t = @@STEER_T@@} at the hook site \texttt{@@HOOK_SITE@@}. All four conditions stay at 0/32 \texttt{pass@1}. The loop-fraction read is negative rather than merely inconclusive: baseline \texttt{no\_steer} is 0.03125, while \texttt{minus\_v\_spherical} is 0.28125, \texttt{plus\_v\_spherical} is 0.125, and \texttt{random\_spherical} is 0.34375.

        \begin{figure}[H]
        \centering
        \includegraphics[width=0.98\textwidth]{steering_controls.png}
        \caption{Finished 32-prompt spherical steering controls. All conditions stay at 0/32 pass@1, and every steered arm is worse than baseline on loop fraction.}
        \end{figure}

        \section*{Provenance}
        Detector training commit: \texttt{@@DETECTOR_COMMIT@@}. The steering table was generated from commit \texttt{@@STEER_COMMIT@@} with vector bundle hash \texttt{@@VECTOR_BUNDLE_HASH@@}. The grader version is \texttt{@@GRADER_VERSION@@}.

        The full machine-readable bundle for this report includes exact prompt IDs, prompt-ID hashes, vector checksums, per-condition artifact hashes, and the copied source receipts in \texttt{source\_data/} alongside \texttt{livecodebench\_stage\_summary.json}.

        \section*{Bottom line}
        The LiveCodeBench repaired stage is finally concrete enough to hand over as a report rather than a thread summary. The honest read is:
        \begin{itemize}
        \item detector side: RFM is real and competitive, but not a clean activation-MLP win;
        \item direction side: the exported bundle is stable enough to study causally;
        \item steering side: the current benchmark-local spherical protocol is negative on the first larger held-out table.
        \end{itemize}

        That means ``finish LiveCodeBench'' at this stage does \emph{not} mean ``claim steering works.'' It means the repaired LiveCodeBench object, detector comparison, vector-quality read, and first benchmark-local steering control table are now all frozen into one deliverable surface.

        \end{document}
        """
    ).strip() + "\n"
    replacements = {
        "@@GENERATED_AT@@": latex_escape(summary["generated_at_utc"]),
        "@@SOURCE_TRAIN_COUNT@@": str(dataset["source_train_prompt_count"]),
        "@@BALANCE_TRAIN@@": latex_escape(dataset["fit_train_balance_policy"]),
        "@@FIT_TRAIN_COUNT@@": str(dataset["fit_train_prompt_count"]),
        "@@VAL_COUNT@@": str(dataset["val_prompt_count"]),
        "@@TEST_COUNT@@": str(dataset["test_prompt_count"]),
        "@@FIT_TRAIN_POS@@": str(dataset["fit_train_positive_count"]),
        "@@VAL_POS@@": str(dataset["val_positive_count"]),
        "@@TEST_POS@@": str(dataset["test_positive_count"]),
        "@@DETECTOR_LAYER@@": str(detector["selected_layer"]),
        "@@DETECTOR_BANDWIDTH@@": f"{detector['selected_bandwidth']:.0f}",
        "@@DETECTOR_VAL_PR@@": f"{detector['val_pr_auc']:.4f}",
        "@@DETECTOR_TEST_PR@@": f"{detector['test_pr_auc']:.4f}",
        "@@DETECTOR_TEST_ROC@@": f"{detector['test_roc_auc']:.4f}",
        "@@DETECTOR_TABLE@@": detector_table_tex(summary),
        "@@BOOTSTRAP_SAMPLES@@": str(direction["direction_bootstrap_samples"]),
        "@@WEAKEST_LOW@@": f"{direction['weakest_layer_low_cosine']:.3f}",
        "@@TOP_LAYERS@@": latex_escape(top_layers),
        "@@SELECTED_MEAN@@": f"{direction['selected_layer_projection']['direction_bootstrap_mean_cosine']:.3f}",
        "@@SELECTED_LOW@@": f"{direction['selected_layer_projection']['direction_bootstrap_low_cosine']:.3f}",
        "@@PROJECTION_TEST_PR@@": f"{direction['selected_layer_projection']['test_pr_auc']:.4f}",
        "@@STEER_T@@": str(steering["t"]),
        "@@HOOK_SITE@@": latex_escape(steering["hook_site"]),
        "@@DETECTOR_COMMIT@@": short_hash(detector["git_commit"]),
        "@@STEER_COMMIT@@": short_hash(steering["run_git_commit"]),
        "@@VECTOR_BUNDLE_HASH@@": short_hash(steering["vector_bundle_hash"]),
        "@@GRADER_VERSION@@": latex_escape(short_hash(steering["grader_version"], 24)),
    }
    tex = template
    for old, new in replacements.items():
        tex = tex.replace(old, new)
    (ROOT / f"{REPORT_STEM}.tex").write_text(tex)


def main() -> None:
    summary = build_summary()
    write_summary_json(summary)
    render_detector_plot(summary)
    render_direction_plot(summary)
    render_steering_plot(summary)
    write_report_tex(summary)


if __name__ == "__main__":
    main()
