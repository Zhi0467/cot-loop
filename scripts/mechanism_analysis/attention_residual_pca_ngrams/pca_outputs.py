from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import slug
from .records import (
    BUCKET_CORRECT_IN_BUDGET,
    BUCKET_WRONG_MAX_LENGTH,
    BUCKETS,
    VECTOR_LABELS,
    CapturedVectors,
    PcaResult,
    SelectedRollout,
)

PooledInputs = dict[
    tuple[str, str, str],
    list[tuple[SelectedRollout, np.ndarray, CapturedVectors]],
]


def fit_pca(matrix: np.ndarray) -> PcaResult:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"PCA matrix must be 2D, got shape {matrix.shape}.")
    n_points, n_features = matrix.shape
    if n_points == 0:
        return _empty_pca(n_points=0, n_features=n_features)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    total_ss = float(np.sum(centered * centered))
    if n_points < 2 or total_ss <= 0.0:
        return _empty_pca(n_points=n_points, n_features=n_features)

    u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    available = min(2, s.shape[0])
    coords = np.zeros((n_points, 2), dtype=np.float64)
    coords[:, :available] = u[:, :available] * s[:available]
    variance = (s * s) / float(n_points - 1)
    total_variance = float(np.sum(variance))
    ratios = [0.0, 0.0]
    if total_variance > 0.0:
        for idx in range(available):
            ratios[idx] = float(variance[idx] / total_variance)
    return PcaResult(
        coords=coords,
        explained_variance_ratio=(ratios[0], ratios[1]),
        total_variance=total_variance,
        n_points=n_points,
        n_features=n_features,
    )


def load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def extend_local_outputs(
    *,
    row: SelectedRollout,
    capture: CapturedVectors,
    out_dir: Path,
    plt,
    pca_point_rows: list[dict],
    pca_summary_rows: list[dict],
    pooled_inputs: PooledInputs,
) -> None:
    for vector_name, vectors in capture.vectors.items():
        pca = fit_pca(vectors)
        figure_path: str | None = None
        if plt is not None:
            figure_path_obj = (
                out_dir
                / "figures"
                / "local"
                / row.dataset_key
                / slug(row.thinking_mode)
                / slug(row.bucket)
                / f"{row.selection_id}__{VECTOR_LABELS[vector_name]}.png"
            )
            _plot_local_pca(
                plt=plt,
                coords=pca.coords,
                row=row,
                vector_name=vector_name,
                pca=pca,
                out_path=figure_path_obj,
            )
            figure_path = str(figure_path_obj)
        points, summary = _local_pca_rows(
            row=row,
            vector_name=vector_name,
            vectors=vectors,
            capture=capture,
            pca=pca,
            figure_path=figure_path,
        )
        pca_point_rows.extend(points)
        pca_summary_rows.append(summary)
        pooled_inputs[(row.dataset_key, row.thinking_mode, vector_name)].append(
            (row, vectors, capture)
        )


def extend_pooled_outputs(
    *,
    pooled_inputs: PooledInputs,
    out_dir: Path,
    plt,
    pca_point_rows: list[dict],
    pca_summary_rows: list[dict],
) -> None:
    for (dataset_key, thinking_mode, vector_name), items in sorted(
        pooled_inputs.items()
    ):
        if not items:
            continue
        matrix = np.concatenate([vectors for _row, vectors, _capture in items], axis=0)
        pca = fit_pca(matrix)
        point_rows_for_plot = _pooled_point_rows(items=items, pca=pca, vector_name=vector_name)
        pca_point_rows.extend(point_rows_for_plot)

        figure_path = ""
        if plt is not None:
            figure_path_obj = (
                out_dir
                / "figures"
                / "pooled"
                / dataset_key
                / slug(thinking_mode)
                / f"{dataset_key}__{slug(thinking_mode)}__{VECTOR_LABELS[vector_name]}.png"
            )
            _plot_pooled_pca(
                plt=plt,
                coords=pca.coords,
                point_rows=point_rows_for_plot,
                vector_name=vector_name,
                pca=pca,
                dataset_key=dataset_key,
                thinking_mode=thinking_mode,
                out_path=figure_path_obj,
            )
            figure_path = str(figure_path_obj)
        pca_summary_rows.append(
            _pca_summary(
                scope="pooled",
                vector_name=vector_name,
                dataset_key=dataset_key,
                thinking_mode=thinking_mode,
                pca=pca,
                figure_path=figure_path,
            )
        )


def _empty_pca(*, n_points: int, n_features: int) -> PcaResult:
    return PcaResult(
        coords=np.zeros((n_points, 2), dtype=np.float64),
        explained_variance_ratio=(0.0, 0.0),
        total_variance=0.0,
        n_points=n_points,
        n_features=n_features,
    )


def _plot_local_pca(
    *,
    plt,
    coords: np.ndarray,
    row: SelectedRollout,
    vector_name: str,
    pca: PcaResult,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    occurrences = np.arange(1, coords.shape[0] + 1)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=occurrences,
        cmap="viridis",
        s=42,
        edgecolor="black",
        linewidth=0.3,
        zorder=3,
    )
    for idx in range(coords.shape[0] - 1):
        ax.annotate(
            "",
            xy=(coords[idx + 1, 0], coords[idx + 1, 1]),
            xytext=(coords[idx, 0], coords[idx, 1]),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555555"},
            zorder=2,
        )
    ax.set_title(
        f"{VECTOR_LABELS[vector_name]} local PCA: {row.dataset_key} {row.thinking_mode}"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio[1]:.1%})")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="occurrence")
    fig.text(
        0.01,
        0.01,
        (
            f"{row.bucket} | sample={row.sample_id} rollout={row.rollout_index} "
            f"| points={coords.shape[0]}"
        ),
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pooled_pca(
    *,
    plt,
    coords: np.ndarray,
    point_rows: list[dict],
    vector_name: str,
    pca: PcaResult,
    dataset_key: str,
    thinking_mode: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    styles = {
        BUCKET_CORRECT_IN_BUDGET: {"color": "#2f7f5f", "marker": "o"},
        BUCKET_WRONG_MAX_LENGTH: {"color": "#b54a3c", "marker": "s"},
    }
    for bucket in BUCKETS:
        indices = [idx for idx, item in enumerate(point_rows) if item["bucket"] == bucket]
        if not indices:
            continue
        style = styles[bucket]
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            label=bucket,
            c=style["color"],
            marker=style["marker"],
            s=36,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.25,
        )

    for indices in _grouped_point_indices(point_rows).values():
        indices.sort(key=lambda idx: int(point_rows[idx]["occurrence_index"]))
        ax.plot(
            coords[indices, 0],
            coords[indices, 1],
            color="#777777",
            alpha=0.28,
            linewidth=0.8,
            zorder=1,
        )

    ax.set_title(f"{VECTOR_LABELS[vector_name]} pooled PCA: {dataset_key} {thinking_mode}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio[1]:.1%})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _local_pca_rows(
    *,
    row: SelectedRollout,
    vector_name: str,
    vectors: np.ndarray,
    capture: CapturedVectors,
    pca: PcaResult,
    figure_path: str | None,
) -> tuple[list[dict], dict]:
    points: list[dict] = []
    norms = np.linalg.norm(vectors, axis=1)
    for idx, start in enumerate(row.rescan_ngram_start_positions):
        points.append(
            _point_row(
                pca_scope="local",
                row=row,
                vector_name=vector_name,
                occurrence_idx=idx,
                ngram_start=int(start),
                vector_norm=float(norms[idx]),
                pc1=float(pca.coords[idx, 0]),
                pc2=float(pca.coords[idx, 1]),
                capture=capture,
            )
        )
    return points, _pca_summary(
        scope="local",
        vector_name=vector_name,
        row=row,
        pca=pca,
        figure_path=figure_path or "",
    )


def _pooled_point_rows(
    *,
    items: list[tuple[SelectedRollout, np.ndarray, CapturedVectors]],
    pca: PcaResult,
    vector_name: str,
) -> list[dict]:
    point_rows: list[dict] = []
    cursor = 0
    for row, vectors, capture in items:
        norms = np.linalg.norm(vectors, axis=1)
        for idx, start in enumerate(row.rescan_ngram_start_positions):
            coords_idx = cursor + idx
            point_rows.append(
                _point_row(
                    pca_scope="pooled",
                    row=row,
                    vector_name=vector_name,
                    occurrence_idx=idx,
                    ngram_start=int(start),
                    vector_norm=float(norms[idx]),
                    pc1=float(pca.coords[coords_idx, 0]),
                    pc2=float(pca.coords[coords_idx, 1]),
                    capture=capture,
                )
            )
        cursor += vectors.shape[0]
    return point_rows


def _point_row(
    *,
    pca_scope: str,
    row: SelectedRollout,
    vector_name: str,
    occurrence_idx: int,
    ngram_start: int,
    vector_norm: float,
    pc1: float,
    pc2: float,
    capture: CapturedVectors,
) -> dict:
    return {
        "pca_scope": pca_scope,
        "vector_name": vector_name,
        "vector_label": VECTOR_LABELS[vector_name],
        "selection_id": row.selection_id,
        "dataset": row.dataset,
        "dataset_key": row.dataset_key,
        "thinking_mode": row.thinking_mode,
        "task_kind": row.task_kind,
        "source_bundle": row.source_bundle,
        "bucket": row.bucket,
        "sample_id": row.sample_id,
        "record_id": row.record_id,
        "rollout_index": row.rollout_index,
        "occurrence_index": occurrence_idx + 1,
        "ngram_start": ngram_start,
        "boundary_position": int(row.boundary_positions[occurrence_idx]),
        "vector_norm": vector_norm,
        "pc1": pc1,
        "pc2": pc2,
        "repeat_probability": capture.repeat_probabilities[occurrence_idx],
        "repeat_logit_margin": capture.repeat_logit_margins[occurrence_idx],
        "repeat_token_logit": capture.repeat_token_logits[occurrence_idx],
        "top_token_id": capture.top_token_ids[occurrence_idx],
        "top_token_logit": capture.top_token_logits[occurrence_idx],
    }


def _pca_summary(
    *,
    scope: str,
    vector_name: str,
    pca: PcaResult,
    figure_path: str,
    row: SelectedRollout | None = None,
    dataset_key: str = "",
    thinking_mode: str = "",
) -> dict:
    return {
        "pca_scope": scope,
        "vector_name": vector_name,
        "vector_label": VECTOR_LABELS[vector_name],
        "dataset": row.dataset if row is not None else "",
        "dataset_key": row.dataset_key if row is not None else dataset_key,
        "thinking_mode": row.thinking_mode if row is not None else thinking_mode,
        "bucket": row.bucket if row is not None else "both",
        "selection_id": row.selection_id if row is not None else "",
        "sample_id": row.sample_id if row is not None else "",
        "rollout_index": row.rollout_index if row is not None else "",
        "n_points": pca.n_points,
        "n_features": pca.n_features,
        "explained_variance_pc1": pca.explained_variance_ratio[0],
        "explained_variance_pc2": pca.explained_variance_ratio[1],
        "explained_variance_top2": (
            pca.explained_variance_ratio[0] + pca.explained_variance_ratio[1]
        ),
        "total_variance": pca.total_variance,
        "figure_path": figure_path,
    }


def _grouped_point_indices(point_rows: list[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(point_rows):
        grouped[str(item["selection_id"])].append(idx)
    return grouped
