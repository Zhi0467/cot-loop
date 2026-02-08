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


def evaluate_binary_metrics(y_true: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    except Exception as exc:
        raise SystemExit(
            "scikit-learn is required to compute evaluation metrics. "
            "Install dependencies with `uv sync`."
        ) from exc

    y_true_np = y_true.detach().cpu().numpy().astype(int)
    probs_np = torch.sigmoid(logits).detach().cpu().numpy()
    pred_np = (probs_np >= 0.5).astype(int)

    acc = float(accuracy_score(y_true_np, pred_np))
    f1 = float(f1_score(y_true_np, pred_np, average="macro", zero_division=0))

    unique_labels = np.unique(y_true_np)
    if unique_labels.size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_np, probs_np))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "roc_auc": auc,
    }
