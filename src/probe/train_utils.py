import random
import math

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_classifier_layer(num_layers: int, classifier_layer: int) -> int:
    if num_layers < 1:
        raise SystemExit("Expected stacked features with at least one layer.")
    if classifier_layer >= 0:
        resolved = classifier_layer
    else:
        resolved = num_layers + classifier_layer
    if resolved < 0 or resolved >= num_layers:
        raise SystemExit(
            "--classifier-layer is out of range for the stacked feature tensor. "
            f"got {classifier_layer}, valid=[-{num_layers}, {num_layers - 1}]"
        )
    return resolved


def _load_sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except Exception as exc:
        raise SystemExit(
            "scikit-learn is required to compute evaluation metrics. "
            "Install dependencies with `uv sync`."
        ) from exc
    return (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )


def evaluate_binary_metrics_from_scores(
    y_true: torch.Tensor,
    scores: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, float]:
    (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    ) = _load_sklearn_metrics()

    y_true_np = y_true.detach().cpu().numpy().astype(int)
    scores_np = scores.detach().cpu().numpy().astype(float)
    pred_np = predictions.detach().cpu().numpy().astype(int)
    prevalence = float(np.mean(y_true_np == 1))

    acc = float(accuracy_score(y_true_np, pred_np))
    f1 = float(f1_score(y_true_np, pred_np, average="macro", zero_division=0))
    pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        y_true_np,
        pred_np,
        labels=[1],
        average=None,
        zero_division=0,
    )

    unique_labels = np.unique(y_true_np)
    if unique_labels.size < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_np, scores_np))
        pr_auc = float(average_precision_score(y_true_np, scores_np))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "positive_precision": float(pos_precision[0]),
        "positive_recall": float(pos_recall[0]),
        "positive_f1": float(pos_f1[0]),
        "prevalence": prevalence,
        "roc_auc": auc,
        "pr_auc": pr_auc,
    }


def evaluate_binary_metrics(y_true: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    scores = torch.sigmoid(logits)
    predictions = (scores >= 0.5).to(dtype=torch.int64)
    return evaluate_binary_metrics_from_scores(y_true, scores, predictions)


def _safe_spearman(y_true_np: np.ndarray, scores_np: np.ndarray) -> float:
    if y_true_np.size < 2:
        return float("nan")
    if np.allclose(y_true_np, y_true_np[0]) or np.allclose(scores_np, scores_np[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        return float("nan")
    corr = spearmanr(y_true_np, scores_np).correlation
    if corr is None or not math.isfinite(float(corr)):
        return float("nan")
    return float(corr)


def _top_capture_fraction(
    y_true_np: np.ndarray,
    scores_np: np.ndarray,
    *,
    fraction: float,
) -> float:
    if y_true_np.size == 0:
        return float("nan")
    total_mass = float(np.sum(y_true_np))
    if total_mass <= 0.0:
        return float("nan")
    keep = max(1, int(math.ceil(float(y_true_np.size) * fraction)))
    order = np.argsort(-scores_np, kind="stable")
    captured = float(np.sum(y_true_np[order[:keep]]))
    return captured / total_mass


def evaluate_probability_metrics_from_scores(
    y_true: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, float]:
    y_true_np = y_true.detach().cpu().numpy().astype(float)
    scores_np = scores.detach().cpu().numpy().astype(float)
    errors = scores_np - y_true_np
    brier = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {
        "brier": brier,
        "mae": mae,
        "rmse": rmse,
        "target_mean": float(np.mean(y_true_np)),
        "pred_mean": float(np.mean(scores_np)),
        "spearman": _safe_spearman(y_true_np, scores_np),
        "top_10p_capture": _top_capture_fraction(
            y_true_np,
            scores_np,
            fraction=0.10,
        ),
        "top_20p_capture": _top_capture_fraction(
            y_true_np,
            scores_np,
            fraction=0.20,
        ),
    }


def evaluate_regression_metrics_from_scores(
    y_true: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, float]:
    y_true_np = y_true.detach().cpu().numpy().astype(float)
    scores_np = scores.detach().cpu().numpy().astype(float)
    errors = scores_np - y_true_np
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "target_mean": float(np.mean(y_true_np)),
        "pred_mean": float(np.mean(scores_np)),
        "spearman": _safe_spearman(y_true_np, scores_np),
        "top_10p_capture": _top_capture_fraction(
            y_true_np,
            scores_np,
            fraction=0.10,
        ),
        "top_20p_capture": _top_capture_fraction(
            y_true_np,
            scores_np,
            fraction=0.20,
        ),
    }


def probe_scores_and_predictions(
    logits: torch.Tensor,
    *,
    classifier_mode: str,
    score_rule: str = "vote_fraction",
) -> tuple[torch.Tensor, torch.Tensor]:
    if classifier_mode == "last_layer":
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.ndim != 1:
            raise SystemExit(
                "last_layer metrics expect 1D logits, "
                f"got shape {tuple(logits.shape)}"
            )
        scores = torch.sigmoid(logits)
        predictions = (scores >= 0.5).to(dtype=torch.int64)
        return scores, predictions

    if classifier_mode == "ensemble":
        if logits.ndim != 2:
            raise SystemExit(
                "ensemble metrics expect [batch, layer] logits, "
                f"got shape {tuple(logits.shape)}"
            )
        layer_probs = torch.sigmoid(logits)
        if score_rule == "vote_fraction":
            layer_predictions = (layer_probs >= 0.5).to(dtype=torch.float32)
            scores = layer_predictions.mean(dim=1)
            predictions = (scores > 0.5).to(dtype=torch.int64)
        elif score_rule == "mean_prob":
            scores = layer_probs.mean(dim=1)
            predictions = (scores >= 0.5).to(dtype=torch.int64)
        else:
            raise SystemExit(f"Unsupported score_rule '{score_rule}'.")
        return scores, predictions

    raise SystemExit(f"Unsupported classifier_mode '{classifier_mode}'.")


def evaluate_probe_outputs(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    *,
    classifier_mode: str,
    target_kind: str = "binary",
    score_rule: str = "vote_fraction",
) -> dict[str, float]:
    scores, predictions = probe_scores_and_predictions(
        logits,
        classifier_mode=classifier_mode,
        score_rule=score_rule,
    )
    if target_kind == "binary":
        return evaluate_binary_metrics_from_scores(y_true, scores, predictions)
    if target_kind == "probability":
        return evaluate_probability_metrics_from_scores(y_true, scores)
    if target_kind == "regression":
        return evaluate_regression_metrics_from_scores(y_true, scores)
    raise SystemExit(f"Unsupported target_kind '{target_kind}'.")
