import random

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


def probe_scores_and_predictions(
    logits: torch.Tensor,
    *,
    classifier_mode: str,
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
        layer_predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
        vote_fraction = layer_predictions.mean(dim=1)
        predictions = (vote_fraction > 0.5).to(dtype=torch.int64)
        return vote_fraction, predictions

    raise SystemExit(f"Unsupported classifier_mode '{classifier_mode}'.")


def evaluate_probe_outputs(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    *,
    classifier_mode: str,
) -> dict[str, float]:
    scores, predictions = probe_scores_and_predictions(
        logits,
        classifier_mode=classifier_mode,
    )
    return evaluate_binary_metrics_from_scores(y_true, scores, predictions)
