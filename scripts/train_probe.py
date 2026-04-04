#!/usr/bin/env python3
"""Train a probe on last-token prefill activations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.dataloader import (
    ActivationDataset,
    read_manifest,
    resolve_input_dim,
    resolve_sample_shape,
    resolve_split_info,
)
from loop_probe.configs import (
    build_probe_model,
    get_probe_config,
    probe_preset_choices,
)
from loop_probe.train_utils import (
    choose_device,
    evaluate_probe_outputs,
    resolve_classifier_layer,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument(
        "--probe-preset",
        choices=probe_preset_choices(),
        default="mlp",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Optional hidden width override when --probe-preset=mlp.",
    )
    parser.add_argument(
        "--mlp-depth",
        type=int,
        default=None,
        help="Optional number of hidden layers when --probe-preset=mlp.",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=None,
        help="Optional dropout override when --probe-preset=mlp.",
    )
    parser.add_argument(
        "--feature-key",
        default=None,
        help=(
            "Optional feature view key from a multi-view dataset manifest. "
            "If omitted, uses manifest default or legacy single-view fields."
        ),
    )
    parser.add_argument(
        "--classifier-mode",
        choices=("last_layer", "ensemble"),
        default="last_layer",
    )
    parser.add_argument(
        "--classifier-layer",
        type=int,
        default=-1,
        help=(
            "Layer index to use when --classifier-mode=last_layer and the "
            "selected feature view is stacked as [layer, hidden]."
        ),
    )
    parser.add_argument(
        "--score-rule",
        choices=("vote_fraction", "mean_prob"),
        default=None,
        help=(
            "How to aggregate per-layer ensemble scores at eval time. "
            "Defaults to vote_fraction for binary targets and mean_prob for "
            "probability targets."
        ),
    )
    parser.add_argument(
        "--train-balance-reference-data-dir",
        default=None,
        help=(
            "Optional dataset dir whose binary train labels define a balanced "
            "sampling distribution for the current train split. The current "
            "train split keeps all prompts; only the train sampler changes."
        ),
    )

    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", default=None)

    return parser.parse_args()


def _prepare_model_inputs(
    x: torch.Tensor,
    *,
    classifier_mode: str,
    resolved_classifier_layer: int | None,
) -> torch.Tensor:
    if classifier_mode == "last_layer":
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            if resolved_classifier_layer is None:
                raise SystemExit(
                    "Missing resolved classifier layer for stacked last_layer inputs."
                )
            return x[:, resolved_classifier_layer, :]
        raise SystemExit(
            "last_layer mode expects flat [batch, hidden] or stacked "
            f"[batch, layer, hidden] inputs, got shape {tuple(x.shape)}"
        )

    if classifier_mode == "ensemble":
        if x.ndim != 3:
            raise SystemExit(
                "ensemble mode requires stacked [batch, layer, hidden] inputs, "
                f"got shape {tuple(x.shape)}"
            )
        return x

    raise SystemExit(f"Unsupported classifier_mode '{classifier_mode}'.")


def _loss_targets(
    y: torch.Tensor,
    *,
    logits: torch.Tensor,
    classifier_mode: str,
) -> torch.Tensor:
    if classifier_mode == "ensemble":
        return y.unsqueeze(1).expand_as(logits)
    return y


def _evaluate(
    model,
    dataloader,
    device: torch.device,
    *,
    classifier_mode: str,
    resolved_classifier_layer: int | None,
    target_kind: str,
    score_rule: str,
) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for x, y in dataloader:
            x = _prepare_model_inputs(
                x.to(device),
                classifier_mode=classifier_mode,
                resolved_classifier_layer=resolved_classifier_layer,
            )
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    return evaluate_probe_outputs(
        labels_cat,
        logits_cat,
        classifier_mode=classifier_mode,
        target_kind=target_kind,
        score_rule=score_rule,
    )


def _write_jsonl(path: str, row: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _build_balanced_sampler(
    *,
    train_dataset: ActivationDataset,
    reference_data_dir: str,
    feature_key: str | None,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    reference_dataset = ActivationDataset(
        data_dir=reference_data_dir,
        split="train",
        feature_key=feature_key,
    )
    reference_labels: dict[int, int] = {}
    for sample_id, raw_label in zip(
        reference_dataset.sample_ids.tolist(),
        reference_dataset.y.tolist(),
    ):
        sample_id_i = int(sample_id)
        label_f = float(raw_label)
        label_i = int(round(label_f))
        if label_i not in (0, 1) or abs(label_f - float(label_i)) > 1e-6:
            raise SystemExit(
                "Balanced-train reference dataset must be binary-labeled, "
                f"got sample_id={sample_id_i} label={label_f!r}."
            )
        if sample_id_i in reference_labels:
            raise SystemExit(
                "Balanced-train reference dataset repeats sample_id "
                f"{sample_id_i}."
            )
        reference_labels[sample_id_i] = label_i

    missing_sample_ids: list[int] = []
    positive_count = 0
    negative_count = 0
    ordered_labels: list[int] = []
    for sample_id in train_dataset.sample_ids.tolist():
        sample_id_i = int(sample_id)
        label_i = reference_labels.get(sample_id_i)
        if label_i is None:
            missing_sample_ids.append(sample_id_i)
            continue
        ordered_labels.append(label_i)
        if label_i == 1:
            positive_count += 1
        else:
            negative_count += 1

    if missing_sample_ids:
        preview = missing_sample_ids[:10]
        raise SystemExit(
            "Balanced-train reference dataset is missing train sample_ids "
            f"needed by the current run: {preview}"
        )
    if positive_count < 1 or negative_count < 1:
        raise SystemExit(
            "Balanced-train reference dataset must contain both classes on the "
            f"current train split (pos={positive_count}, neg={negative_count})."
        )

    positive_weight = 0.5 / positive_count
    negative_weight = 0.5 / negative_count
    sample_weights = [
        positive_weight if label_i == 1 else negative_weight
        for label_i in ordered_labels
    ]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, {
        "num_positive": positive_count,
        "num_negative": negative_count,
    }


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _metric_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if _is_nan(float(value)):
            return default
        return float(value)
    return default


def _binary_primary_key(row: dict[str, object]) -> tuple[float, float]:
    return (
        _metric_float(row.get("eval_roc_auc"), default=float("-inf")),
        _metric_float(row.get("eval_macro_f1"), default=float("-inf")),
    )


def _binary_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        _metric_float(row.get("eval_pr_auc"), default=float("-inf")),
        _metric_float(row.get("eval_roc_auc"), default=float("-inf")),
        _metric_float(row.get("eval_macro_f1"), default=float("-inf")),
    )


def _probability_primary_key(row: dict[str, object]) -> tuple[float, float]:
    return (
        -_metric_float(row.get("eval_brier"), default=float("inf")),
        _metric_float(row.get("eval_spearman"), default=float("-inf")),
    )


def _probability_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        _metric_float(row.get("eval_top_20p_capture"), default=float("-inf")),
        _metric_float(row.get("eval_spearman"), default=float("-inf")),
        -_metric_float(row.get("eval_brier"), default=float("inf")),
    )


def _regression_primary_key(row: dict[str, object]) -> tuple[float, float]:
    return (
        -_metric_float(row.get("eval_mse"), default=float("inf")),
        _metric_float(row.get("eval_spearman"), default=float("-inf")),
    )


def _regression_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        _metric_float(row.get("eval_spearman"), default=float("-inf")),
        _metric_float(row.get("eval_top_20p_capture"), default=float("-inf")),
        -_metric_float(row.get("eval_mse"), default=float("inf")),
    )


def _within_relative_tolerance(
    value: float,
    best_value: float,
    *,
    relative_tolerance: float,
) -> bool:
    if not math.isfinite(value) or not math.isfinite(best_value):
        return False
    if best_value == 0.0:
        return abs(value) <= 1e-12
    return value <= (best_value * (1.0 + relative_tolerance))


def _select_best_loss_row(
    eval_rows: list[dict[str, object]],
    *,
    target_kind: str,
) -> dict[str, object] | None:
    if not eval_rows:
        return None
    if target_kind == "binary":
        return max(eval_rows, key=_binary_primary_key)
    if target_kind == "probability":
        return max(eval_rows, key=_probability_primary_key)
    if target_kind == "regression":
        return max(eval_rows, key=_regression_primary_key)
    raise SystemExit(f"Unsupported target_kind '{target_kind}'.")


def _select_best_rank_row(
    eval_rows: list[dict[str, object]],
    *,
    target_kind: str,
) -> dict[str, object] | None:
    if not eval_rows:
        return None
    if target_kind == "binary":
        return max(eval_rows, key=_binary_rank_key)
    if target_kind == "probability":
        best_brier = min(
            _metric_float(row.get("eval_brier"), default=float("inf"))
            for row in eval_rows
        )
        candidates = [
            row
            for row in eval_rows
            if _within_relative_tolerance(
                _metric_float(row.get("eval_brier"), default=float("inf")),
                best_brier,
                relative_tolerance=0.10,
            )
        ]
        return max(candidates or eval_rows, key=_probability_rank_key)
    if target_kind == "regression":
        best_mse = min(
            _metric_float(row.get("eval_mse"), default=float("inf"))
            for row in eval_rows
        )
        candidates = [
            row
            for row in eval_rows
            if _within_relative_tolerance(
                _metric_float(row.get("eval_mse"), default=float("inf")),
                best_mse,
                relative_tolerance=0.10,
            )
        ]
        return max(candidates or eval_rows, key=_regression_rank_key)
    raise SystemExit(f"Unsupported target_kind '{target_kind}'.")


def _checkpoint_payload(
    model,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    probe_config: dict[str, object] | None = None,
    feature_key: str | None = None,
    sample_shape: tuple[int, ...] | None = None,
    target_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }
    if probe_config is not None:
        payload["probe_config"] = probe_config
    if feature_key:
        payload["feature_key"] = feature_key
    if sample_shape is not None:
        payload["sample_shape"] = [int(dim) for dim in sample_shape]
    if target_spec is not None:
        payload["target_spec"] = target_spec
    return payload


def _resolve_target_spec(manifest: dict[str, object]) -> dict[str, object]:
    target_spec = manifest.get("target_spec")
    if isinstance(target_spec, dict):
        return target_spec
    label_spec = manifest.get("label_spec")
    if isinstance(label_spec, dict):
        return {
            "kind": "binary",
            "name": label_spec.get("target", "eventual_loop"),
            "horizon": label_spec.get("horizon"),
        }
    return {
        "kind": "binary",
        "name": "eventual_loop",
        "horizon": None,
    }


def _metric_bundle(
    prefix: str,
    metrics: dict[str, float],
    *,
    target_kind: str,
) -> dict[str, float]:
    if target_kind == "binary":
        return {
            f"{prefix}/accuracy": metrics["accuracy"],
            f"{prefix}/macro_f1": metrics["macro_f1"],
            f"{prefix}/positive_precision": metrics["positive_precision"],
            f"{prefix}/positive_recall": metrics["positive_recall"],
            f"{prefix}/positive_f1": metrics["positive_f1"],
            f"{prefix}/prevalence": metrics["prevalence"],
            f"{prefix}/roc_auc": metrics["roc_auc"],
            f"{prefix}/pr_auc": metrics["pr_auc"],
        }
    if target_kind == "probability":
        return {
            f"{prefix}/brier": metrics["brier"],
            f"{prefix}/mae": metrics["mae"],
            f"{prefix}/rmse": metrics["rmse"],
            f"{prefix}/target_mean": metrics["target_mean"],
            f"{prefix}/pred_mean": metrics["pred_mean"],
            f"{prefix}/spearman": metrics["spearman"],
            f"{prefix}/top_10p_capture": metrics["top_10p_capture"],
            f"{prefix}/top_20p_capture": metrics["top_20p_capture"],
        }
    if target_kind == "regression":
        return {
            f"{prefix}/mse": metrics["mse"],
            f"{prefix}/mae": metrics["mae"],
            f"{prefix}/rmse": metrics["rmse"],
            f"{prefix}/target_mean": metrics["target_mean"],
            f"{prefix}/pred_mean": metrics["pred_mean"],
            f"{prefix}/spearman": metrics["spearman"],
            f"{prefix}/top_10p_capture": metrics["top_10p_capture"],
            f"{prefix}/top_20p_capture": metrics["top_20p_capture"],
        }
    raise SystemExit(f"Unsupported target_kind '{target_kind}'.")


def _selection_metadata(
    *,
    target_kind: str,
    selection_kind: str,
) -> dict[str, object]:
    if selection_kind == "best_loss":
        if target_kind == "binary":
            return {
                "selection_metric": "roc_auc",
                "tie_breaker": "macro_f1",
            }
        if target_kind == "probability":
            return {
                "selection_metric": "brier",
                "tie_breaker": "spearman",
            }
        if target_kind == "regression":
            return {
                "selection_metric": "mse",
                "tie_breaker": "spearman",
            }
    elif selection_kind == "best_rank":
        if target_kind == "binary":
            return {
                "selection_metric": "pr_auc",
                "tie_breaker": "roc_auc",
                "selection_tertiary_metric": "macro_f1",
            }
        if target_kind == "probability":
            return {
                "selection_metric": "top_20p_capture",
                "tie_breaker": "spearman",
                "selection_constraint_metric": "brier",
                "selection_constraint_mode": "within_relative_tolerance_min",
                "selection_constraint_value": 0.10,
            }
        if target_kind == "regression":
            return {
                "selection_metric": "spearman",
                "tie_breaker": "top_20p_capture",
                "selection_constraint_metric": "mse",
                "selection_constraint_mode": "within_relative_tolerance_min",
                "selection_constraint_value": 0.10,
            }
    raise SystemExit(
        f"Unsupported selection metadata target_kind='{target_kind}' "
        f"selection_kind='{selection_kind}'."
    )


def _write_best_checkpoint_aliases(
    *,
    eval_rows: list[dict[str, object]],
    target_kind: str,
    selection_rule: str,
    rank_selection_rule: str,
    best_ckpt: str,
    best_loss_ckpt: str,
    best_rank_ckpt: str,
    best_metrics_json: str,
    best_loss_metrics_json: str,
    best_rank_metrics_json: str,
) -> None:
    if not eval_rows:
        return

    best_loss_row = _select_best_loss_row(eval_rows, target_kind=target_kind)
    best_rank_row = _select_best_rank_row(eval_rows, target_kind=target_kind)
    if best_loss_row is None or best_rank_row is None:
        raise RuntimeError("Expected non-empty eval rows to produce best checkpoints.")

    shutil.copy2(str(best_loss_row["checkpoint_path"]), best_ckpt)
    shutil.copy2(str(best_loss_row["checkpoint_path"]), best_loss_ckpt)
    shutil.copy2(str(best_rank_row["checkpoint_path"]), best_rank_ckpt)

    best_loss_payload = {
        "selection_kind": "best_loss",
        "selection_rule": selection_rule,
        **_selection_metadata(
            target_kind=target_kind,
            selection_kind="best_loss",
        ),
        **best_loss_row,
    }
    best_rank_payload = {
        "selection_kind": "best_rank",
        "selection_rule": rank_selection_rule,
        **_selection_metadata(
            target_kind=target_kind,
            selection_kind="best_rank",
        ),
        **best_rank_row,
    }
    for path, payload in (
        (best_metrics_json, best_loss_payload),
        (best_loss_metrics_json, best_loss_payload),
        (best_rank_metrics_json, best_rank_payload),
    ):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def _log_row_fields(
    prefix: str,
    metrics: dict[str, float],
    *,
    target_kind: str,
) -> dict[str, float]:
    payload = {
        f"{prefix}_{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    if prefix == "eval":
        payload.update(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
    return payload


def _print_metric_summary(
    *,
    epoch: int,
    train_loss: float,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float] | None,
    target_kind: str,
) -> str:
    parts = [
        f"epoch={epoch}",
        f"train_loss={train_loss:.6f}",
    ]
    if target_kind == "binary":
        parts.extend(
            [
                f"train_acc={train_metrics['accuracy']:.4f}",
                f"train_auc={train_metrics['roc_auc']:.4f}",
                f"train_pr_auc={train_metrics['pr_auc']:.4f}",
            ]
        )
        if eval_metrics is not None:
            parts.extend(
                [
                    f"eval_acc={eval_metrics['accuracy']:.4f}",
                    f"eval_f1={eval_metrics['macro_f1']:.4f}",
                    f"eval_auc={eval_metrics['roc_auc']:.4f}",
                    f"eval_pr_auc={eval_metrics['pr_auc']:.4f}",
                ]
            )
        return " ".join(parts)

    if target_kind == "probability":
        parts.extend(
            [
                f"train_brier={train_metrics['brier']:.4f}",
                f"train_mae={train_metrics['mae']:.4f}",
                f"train_spear={train_metrics['spearman']:.4f}",
            ]
        )
        if eval_metrics is not None:
            parts.extend(
                [
                    f"eval_brier={eval_metrics['brier']:.4f}",
                    f"eval_mae={eval_metrics['mae']:.4f}",
                    f"eval_spear={eval_metrics['spearman']:.4f}",
                    f"eval_top10={eval_metrics['top_10p_capture']:.4f}",
                ]
            )
        return " ".join(parts)

    if target_kind == "regression":
        parts.extend(
            [
                f"train_mse={train_metrics['mse']:.4f}",
                f"train_mae={train_metrics['mae']:.4f}",
                f"train_spear={train_metrics['spearman']:.4f}",
            ]
        )
        if eval_metrics is not None:
            parts.extend(
                [
                    f"eval_mse={eval_metrics['mse']:.4f}",
                    f"eval_mae={eval_metrics['mae']:.4f}",
                    f"eval_spear={eval_metrics['spearman']:.4f}",
                    f"eval_top10={eval_metrics['top_10p_capture']:.4f}",
                ]
            )
        return " ".join(parts)

    raise SystemExit(f"Unsupported target_kind '{target_kind}'.")


def _cosine_lr_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0

    clamped_step = min(max(step, 1), total_steps)
    if warmup_steps > 0 and clamped_step <= warmup_steps:
        return float(clamped_step) / float(warmup_steps)

    decay_steps = max(total_steps - warmup_steps, 1)
    decay_step = max(clamped_step - warmup_steps, 0)
    progress = min(max(decay_step / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def main() -> None:
    args = _parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be > 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be in [0, 1).")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min-lr-ratio must be in [0, 1].")
    if args.mlp_hidden_dim is not None and args.mlp_hidden_dim < 1:
        raise SystemExit("--mlp-hidden-dim must be >= 1.")
    if args.mlp_depth is not None and args.mlp_depth < 1:
        raise SystemExit("--mlp-depth must be >= 1.")
    if args.mlp_dropout is not None and not 0.0 <= args.mlp_dropout < 1.0:
        raise SystemExit("--mlp-dropout must be in [0, 1).")
    if (
        args.probe_preset != "mlp"
        and (
            args.mlp_hidden_dim is not None
            or args.mlp_depth is not None
            or args.mlp_dropout is not None
        )
    ):
        raise SystemExit(
            "--mlp-hidden-dim/--mlp-depth/--mlp-dropout can only be used with "
            "--probe-preset=mlp."
        )
    if args.classifier_mode != "last_layer" and args.classifier_layer != -1:
        raise SystemExit(
            "--classifier-layer is only valid when --classifier-mode=last_layer."
        )

    set_seed(args.seed)
    device = choose_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from dotenv import load_dotenv
    except Exception as exc:
        raise SystemExit(
            "python-dotenv is required. Install dependencies with `uv sync`."
        ) from exc

    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not found. Put it in .env or environment.")

    import wandb

    manifest = read_manifest(args.data_dir)
    target_spec = _resolve_target_spec(manifest)
    target_kind = str(target_spec.get("kind", "binary"))
    if target_kind not in ("binary", "probability", "regression"):
        raise SystemExit(f"Unsupported manifest target kind '{target_kind}'.")
    train_info, resolved_feature_key = resolve_split_info(
        manifest,
        split="train",
        feature_key=args.feature_key,
    )
    input_dim = resolve_input_dim(manifest, resolved_feature_key)
    sample_shape = resolve_sample_shape(manifest, resolved_feature_key)
    try:
        probe_cfg = get_probe_config(
            args.probe_preset,
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.mlp_dropout,
            depth=args.mlp_depth,
            classifier_mode=args.classifier_mode,
            classifier_layer=args.classifier_layer,
            score_rule=(
                args.score_rule
                if args.score_rule is not None
                else ("mean_prob" if target_kind != "binary" else None)
            ),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    resolved_classifier_layer: int | None = None
    if probe_cfg.classifier_mode == "last_layer" and len(sample_shape) == 2:
        resolved_classifier_layer = resolve_classifier_layer(
            int(sample_shape[0]),
            probe_cfg.classifier_layer,
        )

    train_dataset = ActivationDataset(
        data_dir=args.data_dir,
        split="train",
        feature_key=resolved_feature_key,
    )
    train_sampler: WeightedRandomSampler | None = None
    train_sampler_summary: dict[str, int] | None = None
    if args.train_balance_reference_data_dir:
        train_sampler, train_sampler_summary = _build_balanced_sampler(
            train_dataset=train_dataset,
            reference_data_dir=args.train_balance_reference_data_dir,
            feature_key=resolved_feature_key,
        )
        print(
            "train sampler balanced from "
            f"{args.train_balance_reference_data_dir} "
            f"(pos={train_sampler_summary['num_positive']}, "
            f"neg={train_sampler_summary['num_negative']})",
            flush=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_dataset = ActivationDataset(
        data_dir=args.data_dir,
        split="test",
        feature_key=resolved_feature_key,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise SystemExit("Training split is empty.")
    total_steps = args.epochs * steps_per_epoch

    num_pos = int(train_info.get("num_positive", 0))
    num_neg = int(train_info.get("num_negative", 0))
    if target_kind == "binary" and num_pos > 0 and num_neg > 0:
        pos_weight = torch.tensor(
            float(num_neg / num_pos),
            dtype=torch.float32,
            device=device,
        )
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

    model = build_probe_model(
        input_dim=input_dim,
        probe_cfg=probe_cfg,
        sample_shape=sample_shape,
    ).to(device)
    if target_kind == "binary":
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif target_kind == "probability":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_steps = 0
    if args.lr_scheduler == "cosine":
        warmup_steps = int(round(args.warmup_ratio * total_steps))
        if args.warmup_ratio > 0.0 and warmup_steps < 1:
            warmup_steps = 1
        warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "data_dir": args.data_dir,
            "feature_key": resolved_feature_key,
            "target_spec": target_spec,
            "input_dim": input_dim,
            "sample_shape": list(sample_shape),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pos_weight": float(pos_weight.item()),
            "classifier_mode": probe_cfg.classifier_mode,
            "classifier_layer": probe_cfg.classifier_layer,
            "resolved_classifier_layer": resolved_classifier_layer,
            "vote_rule": probe_cfg.vote_rule,
            "score_rule": probe_cfg.score_rule,
            "probe_config": probe_cfg.to_dict(),
            "train_balance_reference_data_dir": args.train_balance_reference_data_dir,
            "train_balance_reference_positive_count": (
                train_sampler_summary["num_positive"]
                if train_sampler_summary is not None
                else None
            ),
            "train_balance_reference_negative_count": (
                train_sampler_summary["num_negative"]
                if train_sampler_summary is not None
                else None
            ),
        },
    )

    metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
    best_ckpt = os.path.join(args.out_dir, "best.pt")
    best_loss_ckpt = os.path.join(args.out_dir, "best_loss.pt")
    best_rank_ckpt = os.path.join(args.out_dir, "best_rank.pt")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    best_metrics_json = os.path.join(args.out_dir, "best_metrics.json")
    best_loss_metrics_json = os.path.join(args.out_dir, "best_loss_metrics.json")
    best_rank_metrics_json = os.path.join(args.out_dir, "best_rank_metrics.json")
    eval_ckpt_dir = os.path.join(args.out_dir, "_eval_ckpts")
    os.makedirs(eval_ckpt_dir, exist_ok=True)

    with open(metrics_jsonl, "w", encoding="utf-8"):
        pass

    if target_kind == "binary":
        selection_rule = "max(roc_auc), tie_break=max(macro_f1)"
        rank_selection_rule = "max(pr_auc), tie_break=max(roc_auc), max(macro_f1)"
    elif target_kind == "probability":
        selection_rule = "min(brier), tie_break=max(spearman)"
        rank_selection_rule = (
            "within 10% of min(brier), max(top_20p_capture), "
            "tie_break=max(spearman), min(brier)"
        )
    else:
        selection_rule = "min(mse), tie_break=max(spearman)"
        rank_selection_rule = (
            "within 10% of min(mse), max(spearman), "
            "tie_break=max(top_20p_capture), min(mse)"
        )
    eval_rows: list[dict[str, object]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_logits = []
        epoch_labels = []
        lr_now = args.lr

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if args.lr_scheduler == "cosine":
                scale = _cosine_lr_factor(
                    global_step + 1,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                lr_now = args.lr * scale
                for group in optimizer.param_groups:
                    group["lr"] = lr_now

            x = _prepare_model_inputs(
                x.to(device),
                classifier_mode=probe_cfg.classifier_mode,
                resolved_classifier_layer=resolved_classifier_layer,
            )
            y = y.to(device)

            logits = model(x)
            loss_inputs = logits
            if target_kind == "regression":
                loss_inputs = torch.sigmoid(logits)
            loss = criterion(
                loss_inputs,
                _loss_targets(
                    y,
                    logits=logits,
                    classifier_mode=probe_cfg.classifier_mode,
                ),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(y.size(0))
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            global_step += 1

            # Store for epoch-level metrics
            epoch_logits.append(logits.detach().cpu())
            epoch_labels.append(y.detach().cpu())

            if batch_idx % args.log_every == 0:
                avg_loss = running_loss / max(seen, 1)
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "epoch": epoch,
                        "step": global_step,
                    },
                    step=global_step,
                )
                print(
                    f"epoch={epoch} batch={batch_idx} step={global_step} "
                    f"train_loss={avg_loss:.6f} lr={lr_now:.6e}",
                    flush=True,
                )

        train_loss_epoch = running_loss / max(seen, 1)
        
        # Compute full epoch training metrics (only once per epoch)
        epoch_logits_cat = torch.cat(epoch_logits, dim=0)
        epoch_labels_cat = torch.cat(epoch_labels, dim=0)
        train_metrics_epoch = evaluate_probe_outputs(
            epoch_labels_cat,
            epoch_logits_cat,
            classifier_mode=probe_cfg.classifier_mode,
            target_kind=target_kind,
            score_rule=probe_cfg.score_rule,
        )

        # Always log training metrics to wandb (every epoch)
        train_log_payload = {
            "train/loss_epoch": train_loss_epoch,
            "train/lr": lr_now,
            "epoch": epoch,
            "step": global_step,
        }
        train_log_payload.update(
            _metric_bundle(
                "train",
                train_metrics_epoch,
                target_kind=target_kind,
            )
        )
        wandb.log(train_log_payload, step=global_step)

        # Evaluate on test set based on eval_every
        if epoch % args.eval_every == 0:
            eval_metrics = _evaluate(
                model,
                test_loader,
                device,
                classifier_mode=probe_cfg.classifier_mode,
                resolved_classifier_layer=resolved_classifier_layer,
                target_kind=target_kind,
                score_rule=probe_cfg.score_rule,
            )
            
            # Log to metrics.jsonl
            log_row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss_epoch,
                "target_kind": target_kind,
                "target_name": target_spec.get("name"),
                "seed": args.seed,
                "lr": lr_now,
                "feature_key": resolved_feature_key,
                "sample_shape": list(sample_shape),
                "classifier_mode": probe_cfg.classifier_mode,
                "classifier_layer": probe_cfg.classifier_layer,
                "resolved_classifier_layer": resolved_classifier_layer,
                "vote_rule": probe_cfg.vote_rule,
                "score_rule": probe_cfg.score_rule,
                "train_balance_reference_data_dir": args.train_balance_reference_data_dir,
                "train_balance_reference_positive_count": (
                    train_sampler_summary["num_positive"]
                    if train_sampler_summary is not None
                    else None
                ),
                "train_balance_reference_negative_count": (
                    train_sampler_summary["num_negative"]
                    if train_sampler_summary is not None
                    else None
                ),
            }
            log_row.update(
                _log_row_fields(
                    "train",
                    train_metrics_epoch,
                    target_kind=target_kind,
                )
            )
            log_row.update(
                _log_row_fields(
                    "eval",
                    eval_metrics,
                    target_kind=target_kind,
                )
            )
            eval_ckpt_path = os.path.join(eval_ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                _checkpoint_payload(
                    model,
                    epoch,
                    global_step,
                    eval_metrics,
                    probe_config=probe_cfg.to_dict(),
                    feature_key=resolved_feature_key,
                    sample_shape=sample_shape,
                    target_spec=target_spec,
                ),
                eval_ckpt_path,
            )
            log_row["checkpoint_path"] = eval_ckpt_path
            _write_jsonl(metrics_jsonl, log_row)
            eval_rows.append(dict(log_row))
            _write_best_checkpoint_aliases(
                eval_rows=eval_rows,
                target_kind=target_kind,
                selection_rule=selection_rule,
                rank_selection_rule=rank_selection_rule,
                best_ckpt=best_ckpt,
                best_loss_ckpt=best_loss_ckpt,
                best_rank_ckpt=best_rank_ckpt,
                best_metrics_json=best_metrics_json,
                best_loss_metrics_json=best_loss_metrics_json,
                best_rank_metrics_json=best_rank_metrics_json,
            )

            # Log eval metrics to wandb
            eval_log_payload = {
                "epoch": epoch,
                "step": global_step,
            }
            eval_log_payload.update(
                _metric_bundle(
                    "eval",
                    eval_metrics,
                    target_kind=target_kind,
                )
            )
            wandb.log(eval_log_payload, step=global_step)

            print(
                _print_metric_summary(
                    epoch=epoch,
                    train_loss=train_loss_epoch,
                    train_metrics=train_metrics_epoch,
                    eval_metrics=eval_metrics,
                    target_kind=target_kind,
                ),
                flush=True,
            )
        else:
            # Print training metrics only
            print(
                _print_metric_summary(
                    epoch=epoch,
                    train_loss=train_loss_epoch,
                    train_metrics=train_metrics_epoch,
                    eval_metrics=None,
                    target_kind=target_kind,
                ),
                flush=True,
            )

        train_checkpoint_metrics = {"train_loss": train_loss_epoch}
        train_checkpoint_metrics.update(train_metrics_epoch)
        torch.save(
            _checkpoint_payload(
                model,
                epoch,
                global_step,
                train_checkpoint_metrics,
                probe_config=probe_cfg.to_dict(),
                feature_key=resolved_feature_key,
                sample_shape=sample_shape,
                target_spec=target_spec,
            ),
            last_ckpt,
        )

    run.finish()
    if eval_rows:
        _write_best_checkpoint_aliases(
            eval_rows=eval_rows,
            target_kind=target_kind,
            selection_rule=selection_rule,
            rank_selection_rule=rank_selection_rule,
            best_ckpt=best_ckpt,
            best_loss_ckpt=best_loss_ckpt,
            best_rank_ckpt=best_rank_ckpt,
            best_metrics_json=best_metrics_json,
            best_loss_metrics_json=best_loss_metrics_json,
            best_rank_metrics_json=best_rank_metrics_json,
        )
    else:
        print(
            "No eval checkpoints were logged (check --eval-every); best_metrics.json not written.",
            flush=True,
        )
    print(f"Saved checkpoints to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
