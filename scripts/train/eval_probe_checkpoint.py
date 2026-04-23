#!/usr/bin/env python3
"""Evaluate a trained torch probe checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.configs import ProbeConfig, build_probe_model, get_probe_config, probe_preset_choices
from loop_probe.dataloader import (
    make_dataloader,
    read_manifest,
    resolve_input_dim,
    resolve_sample_shape,
    resolve_split_info,
)
from loop_probe.train_utils import (
    choose_device,
    evaluate_probe_outputs,
    resolve_classifier_layer,
)
from loop_probe.stage_artifacts import stable_json_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--feature-key",
        default=None,
        help=(
            "Optional feature view key from a multi-view dataset manifest. "
            "If omitted, uses manifest default or legacy single-view fields."
        ),
    )
    parser.add_argument(
        "--probe-preset",
        choices=probe_preset_choices(),
        default=None,
        help="Fallback probe preset when checkpoint has no probe_config payload.",
    )
    parser.add_argument(
        "--classifier-mode",
        choices=("last_layer", "ensemble"),
        default=None,
        help="Fallback classifier mode for older checkpoints without this metadata.",
    )
    parser.add_argument(
        "--classifier-layer",
        type=int,
        default=None,
        help="Fallback classifier layer for older checkpoints without this metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _probe_cfg_from_checkpoint(
    payload: dict[str, Any],
    probe_preset: str | None,
    *,
    fallback_classifier_mode: str | None,
    fallback_classifier_layer: int | None,
) -> ProbeConfig:
    probe_cfg_raw = payload.get("probe_config")
    if isinstance(probe_cfg_raw, dict):
        probe_type = str(probe_cfg_raw.get("probe_type", "linear"))
        classifier_mode = probe_cfg_raw.get("classifier_mode", fallback_classifier_mode)
        classifier_layer = probe_cfg_raw.get("classifier_layer", fallback_classifier_layer)
        vote_rule = probe_cfg_raw.get("vote_rule", "majority")
        score_rule = probe_cfg_raw.get("score_rule", "vote_fraction")
        if probe_type == "linear":
            return ProbeConfig(
                probe_type="linear",
                classifier_mode=str(classifier_mode or "last_layer"),
                classifier_layer=int(
                    classifier_layer if classifier_layer is not None else -1
                ),
                vote_rule=str(vote_rule),
                score_rule=str(score_rule),
            )
        if probe_type == "mlp":
            hidden_dim = int(probe_cfg_raw.get("hidden_dim", 128))
            dropout = float(probe_cfg_raw.get("dropout", 0.1))
            depth = int(
                probe_cfg_raw.get(
                    "depth",
                    probe_cfg_raw.get("mlp_depth", 1),
                )
            )
            return ProbeConfig(
                probe_type="mlp",
                hidden_dim=hidden_dim,
                dropout=dropout,
                depth=depth,
                classifier_mode=str(classifier_mode or "last_layer"),
                classifier_layer=int(
                    classifier_layer if classifier_layer is not None else -1
                ),
                vote_rule=str(vote_rule),
                score_rule=str(score_rule),
            )
        raise SystemExit(
            f"Unsupported probe_type in checkpoint probe_config: '{probe_type}'"
        )

    try:
        if probe_preset is None:
            raise SystemExit(
                "Checkpoint has no probe_config payload; pass --probe-preset explicitly."
            )
        return get_probe_config(
            probe_preset,
            classifier_mode=fallback_classifier_mode,
            classifier_layer=fallback_classifier_layer,
        )
    except ValueError as exc:
        raise SystemExit(
            "Checkpoint has no probe_config payload; pass --probe-preset explicitly."
        ) from exc


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


def _resolve_target_spec_from_manifest(manifest: dict[str, Any]) -> dict[str, object]:
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


def _resolve_target_spec_from_checkpoint(payload: dict[str, Any]) -> dict[str, object] | None:
    target_spec = payload.get("target_spec")
    if isinstance(target_spec, dict):
        return target_spec
    return None


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.classifier_mode != "last_layer" and args.classifier_layer is not None:
        raise SystemExit(
            "--classifier-layer is only valid when --classifier-mode=last_layer."
        )

    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"Checkpoint payload missing state_dict: {args.checkpoint}")

    checkpoint_feature_key = payload.get("feature_key")
    manifest = read_manifest(args.data_dir)
    target_spec = _resolve_target_spec_from_manifest(manifest)
    checkpoint_target_spec = _resolve_target_spec_from_checkpoint(payload)
    if checkpoint_target_spec is not None and checkpoint_target_spec != target_spec:
        raise SystemExit(
            "Checkpoint target_spec does not match dataset target_spec: "
            f"{checkpoint_target_spec} vs {target_spec}"
        )
    target_kind = str(target_spec.get("kind", "binary"))
    if target_kind not in ("binary", "probability", "regression"):
        raise SystemExit(f"Unsupported manifest target kind '{target_kind}'.")
    resolved_feature_key_arg = args.feature_key
    if (
        (resolved_feature_key_arg is None or resolved_feature_key_arg == "")
        and isinstance(checkpoint_feature_key, str)
        and checkpoint_feature_key
        and isinstance(manifest.get("feature_views"), dict)
    ):
        resolved_feature_key_arg = checkpoint_feature_key

    split_info, resolved_feature_key = resolve_split_info(
        manifest,
        split=args.split,
        feature_key=resolved_feature_key_arg,
    )
    num_rows = int(split_info.get("num_rows", split_info.get("num_samples", 0)))
    if num_rows < 1:
        raise SystemExit(f"Split '{args.split}' in {args.data_dir} has no rows.")

    probe_cfg = _probe_cfg_from_checkpoint(
        payload,
        args.probe_preset,
        fallback_classifier_mode=args.classifier_mode,
        fallback_classifier_layer=args.classifier_layer,
    )
    input_dim = resolve_input_dim(manifest, resolved_feature_key)
    sample_shape = resolve_sample_shape(manifest, resolved_feature_key)
    payload_sample_shape = payload.get("sample_shape")
    if isinstance(payload_sample_shape, list):
        checkpoint_sample_shape = tuple(int(dim) for dim in payload_sample_shape)
        if checkpoint_sample_shape != sample_shape:
            raise SystemExit(
                "Checkpoint sample_shape does not match dataset sample_shape: "
                f"{checkpoint_sample_shape} vs {sample_shape}"
            )
    resolved_classifier_layer: int | None = None
    if probe_cfg.classifier_mode == "last_layer" and len(sample_shape) == 2:
        resolved_classifier_layer = resolve_classifier_layer(
            int(sample_shape[0]),
            probe_cfg.classifier_layer,
        )
    model = build_probe_model(
        input_dim=input_dim,
        probe_cfg=probe_cfg,
        sample_shape=sample_shape,
    )
    model.load_state_dict(payload["state_dict"], strict=True)

    device = choose_device(args.device)
    model = model.to(device)
    loader = make_dataloader(
        args.data_dir,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        feature_key=resolved_feature_key,
    )
    metrics = _evaluate(
        model,
        loader,
        device,
        classifier_mode=probe_cfg.classifier_mode,
        resolved_classifier_layer=resolved_classifier_layer,
        target_kind=target_kind,
        score_rule=probe_cfg.score_rule,
    )
    dataset = loader.dataset
    sample_ids: list[int] | None = None
    if hasattr(dataset, "sample_ids"):
        sample_ids = [int(value) for value in dataset.sample_ids.tolist()]

    out = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "feature_key": resolved_feature_key,
        "target_kind": target_kind,
        "target_spec": target_spec,
        "input_dim": input_dim,
        "sample_shape": list(sample_shape),
        "classifier_mode": probe_cfg.classifier_mode,
        "classifier_layer": probe_cfg.classifier_layer,
        "resolved_classifier_layer": resolved_classifier_layer,
        "vote_rule": probe_cfg.vote_rule,
        "score_rule": probe_cfg.score_rule,
        "num_rows": num_rows,
        "sample_ids": sample_ids,
        "sample_ids_sha256": stable_json_sha256(sample_ids) if sample_ids is not None else None,
        **metrics,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
