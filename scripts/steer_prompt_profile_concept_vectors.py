#!/usr/bin/env python3
"""Run prompt-prefill spherical steering from exported prompt-profile RFM vectors."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer

from loop_probe.adapters import (
    livecodebench_codegen,
    math_freeform,
    multiple_choice_gpqa,
    multiple_choice_mmlupro,
)
from loop_probe.collector import LcbSampleRecord
from loop_probe.labeling import first_ngram_loop_prefix_length
from loop_probe.rollout import resolve_sampling_defaults
from loop_probe.stage_artifacts import (
    build_steering_run_record,
    current_git_commit,
    stable_json_sha256,
    tensor_checksum_hex,
    write_stage_artifact_record,
)
from loop_probe.types import DatasetSpec

DEFAULT_T = 0.3
DEFAULT_HOOK_SITE = "prefill_layer_output_last_token"
DEFAULT_GRADER_VERSION_BY_TASK = {
    "math_freeform": "math_verify",
    "multiple_choice_gpqa": "structured_json_letter.v1",
    "multiple_choice_mmlupro": "structured_json_letter.v1",
}


@dataclass(frozen=True)
class SteeringPromptItem:
    sample_id: int
    prompt: str
    gold_answer: str | None = None
    gold_index: int | None = None
    question_id: str | None = None


@dataclass(frozen=True)
class LayerVectorPayload:
    layer: int
    vector: torch.Tensor
    record: dict[str, Any]
    vector_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-export-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["no_steer", "minus_v_spherical", "plus_v_spherical", "random_spherical"],
        choices=(
            "no_steer",
            "minus_v_spherical",
            "plus_v_spherical",
            "random_spherical",
        ),
    )
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--t", type=float, default=DEFAULT_T)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--hook-site", default=DEFAULT_HOOK_SITE)
    parser.add_argument("--livecodebench-repo", default="")
    parser.add_argument("--release-version", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _metric_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        value_f = float(value)
        if math.isnan(value_f):
            return default
        return value_f
    return default


def _stable_vector_bundle_hash(
    summary_payload: dict[str, Any],
    layer_records: dict[int, dict[str, Any]],
) -> str:
    return stable_json_sha256(
        {
            "summary": summary_payload,
            "layer_artifact_hashes": {
                str(layer): record["artifact_sha256"]
                for layer, record in sorted(layer_records.items())
            },
        }
    )


def _load_layer_payloads(
    vector_export_dir: Path,
    requested_layers: list[int] | None,
) -> tuple[dict[str, Any], dict[int, LayerVectorPayload], str]:
    summary_path = vector_export_dir / "summary.json"
    summary_payload = _read_json(summary_path)
    layer_entries = summary_payload.get("layers")
    if not isinstance(layer_entries, list) or not layer_entries:
        raise SystemExit(f"Vector summary has no layer entries: {summary_path}")

    selected_layers = {int(layer) for layer in requested_layers} if requested_layers else None
    layer_payloads: dict[int, LayerVectorPayload] = {}
    layer_records: dict[int, dict[str, Any]] = {}
    for entry in layer_entries:
        if not isinstance(entry, dict):
            continue
        layer = int(entry["layer"])
        if selected_layers is not None and layer not in selected_layers:
            continue
        record_path = Path(str(entry["record_path"]))
        record = _read_json(record_path)
        vector_payload = torch.load(str(entry["vector_path"]), map_location="cpu")
        vector = vector_payload.get("normalized_vector")
        if not isinstance(vector, torch.Tensor):
            raise SystemExit(f"Vector payload missing normalized_vector: {entry['vector_path']}")
        layer_payloads[layer] = LayerVectorPayload(
            layer=layer,
            vector=vector.detach().to(dtype=torch.float32, device="cpu"),
            record=record,
            vector_path=str(entry["vector_path"]),
        )
        layer_records[layer] = record

    if not layer_payloads:
        raise SystemExit("No layers were selected from the vector export bundle.")
    if selected_layers is not None:
        missing = sorted(selected_layers.difference(layer_payloads))
        if missing:
            raise SystemExit(f"Requested layer(s) missing from vector export bundle: {missing}")
    bundle_hash = _stable_vector_bundle_hash(summary_payload, layer_records)
    return summary_payload, layer_payloads, bundle_hash


def _resolve_source_manifest(layer_payloads: dict[int, LayerVectorPayload]) -> tuple[dict[str, Any], Path]:
    first_record = next(iter(layer_payloads.values())).record
    preprocessing = first_record.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise SystemExit("Vector bundle record is missing preprocessing metadata.")
    manifest_path_text = preprocessing.get("source_manifest_path")
    if not isinstance(manifest_path_text, str) or not manifest_path_text:
        raise SystemExit("Vector bundle preprocessing is missing source_manifest_path.")
    manifest_path = Path(manifest_path_text)
    return _read_json(manifest_path), manifest_path


def _load_archive_rows_by_sample_id(source_data_dir: Path, archive_relpath: str) -> dict[int, dict[str, Any]]:
    rows_by_sample_id: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl_rows(source_data_dir / archive_relpath):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, int):
            raise SystemExit("Prompt rollout archive row is missing integer sample_id.")
        if sample_id in rows_by_sample_id:
            raise SystemExit(f"Prompt rollout archive sample_id collision: {sample_id}")
        rows_by_sample_id[sample_id] = row
    return rows_by_sample_id


def _source_spec_for_split(source_manifest: dict[str, Any], split: str) -> DatasetSpec:
    spec_key = "test_spec" if split == "test" else "train_spec"
    payload = source_manifest.get(spec_key)
    if not isinstance(payload, dict):
        raise SystemExit(f"Source manifest is missing {spec_key}.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise SystemExit(f"Source manifest {spec_key} is missing dataset.")
    config = payload.get("config")
    split_name = payload.get("split")
    max_samples = payload.get("max_samples")
    return DatasetSpec(
        dataset=dataset,
        config=config if isinstance(config, str) or config is None else None,
        split=str(split_name),
        max_samples=int(max_samples) if isinstance(max_samples, int) else None,
    )


def _manifest_model_id(source_manifest: dict[str, Any]) -> str:
    rollout_cfg = source_manifest.get("rollout_config")
    if not isinstance(rollout_cfg, dict):
        raise SystemExit("Source manifest is missing rollout_config.")
    model_id = rollout_cfg.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise SystemExit("Source manifest rollout_config is missing model_id.")
    return model_id


def _manifest_rollout_value(source_manifest: dict[str, Any], key: str, *, default: Any = None) -> Any:
    rollout_cfg = source_manifest.get("rollout_config")
    if not isinstance(rollout_cfg, dict):
        return default
    return rollout_cfg.get(key, default)


def _prompt_hash(prompts: list[str]) -> str:
    return stable_json_sha256(prompts)


def _prompt_ids_hash(prompt_ids: list[int]) -> str:
    return stable_json_sha256(prompt_ids)


def _resolve_release_version(
    source_manifest: dict[str, Any],
    override: str | None,
) -> str:
    if override:
        return override
    payload = source_manifest.get("test_spec")
    if isinstance(payload, dict):
        dataset_name = payload.get("dataset")
        if isinstance(dataset_name, str) and dataset_name.startswith("livecodebench_"):
            return dataset_name.removeprefix("livecodebench_")
    return "release_v6"


def _load_math_gold(spec: DatasetSpec) -> dict[int, str]:
    return {
        record.sample_id: gold_answer
        for record, gold_answer in math_freeform.load_samples(
            spec,
            question_field="problem",
            answer_field="answer",
        )
    }


def _load_gpqa_gold(spec: DatasetSpec, seed: int) -> dict[int, str]:
    return {
        record.sample_id: gold_letter
        for record, _options, gold_letter in multiple_choice_gpqa.load_and_shuffle(
            spec,
            seed=seed,
        )
    }


def _load_mmlu_gold(spec: DatasetSpec) -> dict[int, tuple[str, int | None]]:
    return {
        record.sample_id: (gold_answer, gold_index)
        for record, _options, gold_answer, gold_index in multiple_choice_mmlupro.load_samples(spec)
    }


def _build_prompt_items(
    *,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    layer_payloads: dict[int, LayerVectorPayload],
    split: str,
    max_samples: int | None,
    livecodebench_repo: str,
    release_version: str,
) -> tuple[list[SteeringPromptItem], list[str], str, str, list[Any] | None]:
    first_record = next(iter(layer_payloads.values())).record
    benchmark = str(first_record["benchmark"])
    prompt_ids_payload = first_record.get("prompt_ids")
    if not isinstance(prompt_ids_payload, dict) or split not in prompt_ids_payload:
        raise SystemExit(f"Vector bundle record is missing prompt_ids.{split}.")
    prompt_ids = [int(value) for value in prompt_ids_payload[split]]
    if max_samples is not None:
        prompt_ids = prompt_ids[:max_samples]
    if not prompt_ids:
        raise SystemExit(f"No prompt IDs remain for split '{split}'.")
    prompt_id_hashes = first_record.get("prompt_id_hashes")
    if isinstance(prompt_id_hashes, dict):
        expected_prompt_ids_hash = prompt_id_hashes.get(split)
        if max_samples is None and isinstance(expected_prompt_ids_hash, str):
            actual_hash = _prompt_ids_hash(prompt_ids)
            if actual_hash != expected_prompt_ids_hash:
                raise SystemExit(
                    f"Prompt ID hash mismatch for split '{split}': "
                    f"{actual_hash} != {expected_prompt_ids_hash}"
                )

    preprocessing = first_record.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise SystemExit("Vector bundle preprocessing is missing.")
    source_data_dir_text = preprocessing.get("source_data_dir")
    if not isinstance(source_data_dir_text, str) or not source_data_dir_text:
        raise SystemExit("Vector bundle preprocessing is missing source_data_dir.")
    source_data_dir = Path(source_data_dir_text)
    archive_relpath = source_manifest.get("prompt_rollout_archive_file")
    if not isinstance(archive_relpath, str) or not archive_relpath:
        raise SystemExit("Source manifest is missing prompt_rollout_archive_file.")
    rows_by_sample_id = _load_archive_rows_by_sample_id(source_data_dir, archive_relpath)
    selected_rows = []
    prompts = []
    for sample_id in prompt_ids:
        row = rows_by_sample_id.get(sample_id)
        if row is None:
            raise SystemExit(f"Prompt rollout archive is missing sample_id={sample_id}.")
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise SystemExit(f"Prompt rollout archive row {sample_id} is missing prompt text.")
        selected_rows.append(row)
        prompts.append(prompt)

    prompt_text_hashes = first_record.get("prompt_text_hashes")
    if isinstance(prompt_text_hashes, dict):
        expected_prompt_hash = prompt_text_hashes.get(split)
        if max_samples is None and isinstance(expected_prompt_hash, str):
            actual_hash = _prompt_hash(prompts)
            if actual_hash != expected_prompt_hash:
                raise SystemExit(
                    f"Prompt text hash mismatch for split '{split}': "
                    f"{actual_hash} != {expected_prompt_hash}"
                )

    task_kind = source_manifest.get("task_kind")
    if not isinstance(task_kind, str):
        raise SystemExit("Source manifest is missing task_kind.")
    source_spec = _source_spec_for_split(source_manifest, split)
    benchmark_subset: list[Any] | None = None
    items: list[SteeringPromptItem] = []

    if task_kind == "math_freeform":
        gold_by_sample_id = _load_math_gold(source_spec)
        for row, prompt in zip(selected_rows, prompts):
            sample_id = int(row["sample_id"])
            gold_answer = gold_by_sample_id.get(sample_id)
            if gold_answer is None:
                raise SystemExit(f"Math source rows are missing sample_id={sample_id}.")
            items.append(
                SteeringPromptItem(
                    sample_id=sample_id,
                    prompt=prompt,
                    gold_answer=gold_answer,
                )
            )
        grader_version = DEFAULT_GRADER_VERSION_BY_TASK[task_kind]
    elif task_kind == "multiple_choice_gpqa":
        gpqa_seed = int(source_manifest.get("seed", 0))
        gold_by_sample_id = _load_gpqa_gold(source_spec, gpqa_seed)
        for row, prompt in zip(selected_rows, prompts):
            sample_id = int(row["sample_id"])
            gold_answer = gold_by_sample_id.get(sample_id)
            if gold_answer is None:
                raise SystemExit(f"GPQA source rows are missing sample_id={sample_id}.")
            items.append(
                SteeringPromptItem(
                    sample_id=sample_id,
                    prompt=prompt,
                    gold_answer=gold_answer,
                )
            )
        grader_version = DEFAULT_GRADER_VERSION_BY_TASK[task_kind]
    elif task_kind == "multiple_choice_mmlupro":
        gold_by_sample_id = _load_mmlu_gold(source_spec)
        for row, prompt in zip(selected_rows, prompts):
            sample_id = int(row["sample_id"])
            gold = gold_by_sample_id.get(sample_id)
            if gold is None:
                raise SystemExit(f"MMLU-Pro source rows are missing sample_id={sample_id}.")
            gold_answer, gold_index = gold
            items.append(
                SteeringPromptItem(
                    sample_id=sample_id,
                    prompt=prompt,
                    gold_answer=gold_answer,
                    gold_index=gold_index,
                )
            )
        grader_version = DEFAULT_GRADER_VERSION_BY_TASK[task_kind]
    elif task_kind == "livecodebench_codegen":
        if not livecodebench_repo:
            raise SystemExit("--livecodebench-repo is required for livecodebench steering.")
        benchmark, _ = livecodebench_codegen.load_benchmark(
            livecodebench_repo,
            release_version,
        )
        if source_spec.max_samples is not None:
            benchmark = benchmark[: source_spec.max_samples]
        benchmark_subset = []
        for row, prompt in zip(selected_rows, prompts):
            sample_id = int(row["sample_id"])
            if sample_id < 0 or sample_id >= len(benchmark):
                raise SystemExit(
                    f"LiveCodeBench sample_id={sample_id} is outside loaded benchmark range "
                    f"[0, {len(benchmark)})."
                )
            instance = benchmark[sample_id]
            benchmark_subset.append(instance)
            items.append(
                SteeringPromptItem(
                    sample_id=sample_id,
                    prompt=prompt,
                    question_id=str(instance.question_id),
                )
            )
        grader_commit = subprocess.run(
            ["git", "-C", livecodebench_repo, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        grader_version = f"livecodebench_codegen.{release_version}@{grader_commit}"
    else:
        raise SystemExit(f"Unsupported task_kind for steering: {task_kind}")

    _ = source_manifest_path
    return items, prompt_ids, benchmark, task_kind, benchmark_subset, grader_version


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def _resolve_transformer_layers(model: torch.nn.Module) -> tuple[torch.nn.ModuleList, str]:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    )
    for path in candidates:
        node: Any = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok and isinstance(node, torch.nn.ModuleList):
            return node, ".".join(path)
    raise SystemExit("Could not locate transformer block list for steering hooks.")


def _unit_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _spherical_steer_batch(
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    t: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pre_norm = hidden.norm(dim=-1)
    safe_pre_norm = pre_norm.clamp_min(1e-12)
    hidden_hat = hidden / safe_pre_norm.unsqueeze(-1)
    target_hat = target.unsqueeze(0).expand_as(hidden_hat)
    dot = (hidden_hat * target_hat).sum(dim=-1).clamp(-1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)
    use_linear = sin_theta.abs() < 1e-6
    coeff_hidden = torch.sin((1.0 - t) * theta) / sin_theta.clamp_min(1e-6)
    coeff_target = torch.sin(t * theta) / sin_theta.clamp_min(1e-6)
    steered_hat = coeff_hidden.unsqueeze(-1) * hidden_hat + coeff_target.unsqueeze(-1) * target_hat
    steered_hat_linear = _unit_normalize((1.0 - t) * hidden_hat + t * target_hat)
    steered_hat = torch.where(use_linear.unsqueeze(-1), steered_hat_linear, steered_hat)
    steered_hat = _unit_normalize(steered_hat)
    steered = steered_hat * pre_norm.unsqueeze(-1)
    post_norm = steered.norm(dim=-1)
    move_dot = (hidden_hat * steered_hat).sum(dim=-1).clamp(-1.0, 1.0)
    move_angle = torch.arccos(move_dot)
    return steered, {
        "pre_norm": pre_norm,
        "post_norm": post_norm,
        "start_angle": theta,
        "move_angle": move_angle,
        "norm_error": (post_norm - pre_norm).abs(),
    }


class _LayerStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.pre_norm_sum = 0.0
        self.post_norm_sum = 0.0
        self.start_angle_sum = 0.0
        self.move_angle_sum = 0.0
        self.norm_error_sum = 0.0
        self.max_norm_error = 0.0

    def update(self, stats: dict[str, torch.Tensor]) -> None:
        count = int(stats["pre_norm"].numel())
        if count == 0:
            return
        self.count += count
        self.pre_norm_sum += float(stats["pre_norm"].sum().item())
        self.post_norm_sum += float(stats["post_norm"].sum().item())
        self.start_angle_sum += float(stats["start_angle"].sum().item())
        self.move_angle_sum += float(stats["move_angle"].sum().item())
        self.norm_error_sum += float(stats["norm_error"].sum().item())
        self.max_norm_error = max(
            self.max_norm_error,
            float(stats["norm_error"].max().item()),
        )

    def to_json(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        denom = float(self.count)
        return {
            "count": self.count,
            "mean_pre_norm": self.pre_norm_sum / denom,
            "mean_post_norm": self.post_norm_sum / denom,
            "mean_start_angle": self.start_angle_sum / denom,
            "mean_move_angle": self.move_angle_sum / denom,
            "mean_norm_error": self.norm_error_sum / denom,
            "max_norm_error": self.max_norm_error,
        }


class PrefillLayerOutputSphericalController:
    """Steer the final prompt token at selected layer outputs during prefill only."""

    def __init__(
        self,
        *,
        layers: torch.nn.ModuleList,
        targets_by_layer: dict[int, torch.Tensor],
        t: float,
    ) -> None:
        self._layers = layers
        self._targets_by_layer = {
            int(layer): target.detach().to(dtype=torch.float32, device="cpu")
            for layer, target in targets_by_layer.items()
        }
        self._t = float(t)
        self._handles: list[Any] = []
        self._stats = {layer: _LayerStatsAccumulator() for layer in self._targets_by_layer}
        self.enabled = False

    def install(self) -> None:
        if self._handles:
            return
        for layer_idx in sorted(self._targets_by_layer):
            module = self._layers[layer_idx]
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        self.enabled = True

    def remove(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            handle.remove()
        self.enabled = False

    def reset_stats(self) -> None:
        self._stats = {layer: _LayerStatsAccumulator() for layer in self._targets_by_layer}

    def stats_json(self) -> dict[str, Any]:
        return {
            str(layer): accumulator.to_json()
            for layer, accumulator in sorted(self._stats.items())
        }

    def _make_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            if not self.enabled:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return output
            if hidden.size(1) <= 1:
                return output
            target = self._targets_by_layer[layer_idx].to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            last_token = hidden[:, -1, :]
            steered, stats = _spherical_steer_batch(last_token, target, t=self._t)
            hidden[:, -1, :] = steered.to(dtype=hidden.dtype)
            self._stats[layer_idx].update(
                {
                    key: value.detach().to(dtype=torch.float32, device="cpu")
                    for key, value in stats.items()
                }
            )
            return output

        return _hook


def _random_unit_vectors_like(
    layer_payloads: dict[int, LayerVectorPayload],
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    vectors: dict[int, torch.Tensor] = {}
    for layer in sorted(layer_payloads):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1009 + layer)
        noise = torch.randn(
            layer_payloads[layer].vector.shape,
            generator=generator,
            dtype=torch.float32,
        )
        vectors[layer] = _unit_normalize(noise.unsqueeze(0))[0]
    return vectors


def _targets_for_condition(
    condition_name: str,
    layer_payloads: dict[int, LayerVectorPayload],
    *,
    seed: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if condition_name == "no_steer":
        return {}, {"kind": "identity"}
    if condition_name == "minus_v_spherical":
        targets = {
            layer: (-payload.vector).detach().to(dtype=torch.float32, device="cpu")
            for layer, payload in layer_payloads.items()
        }
        return targets, {
            "kind": "bundle_signed_target",
            "signed_target": "-normalized_vector",
            "target_checksums": {
                str(layer): tensor_checksum_hex(target)
                for layer, target in sorted(targets.items())
            },
        }
    if condition_name == "plus_v_spherical":
        targets = {
            layer: payload.vector.detach().to(dtype=torch.float32, device="cpu")
            for layer, payload in layer_payloads.items()
        }
        return targets, {
            "kind": "bundle_signed_target",
            "signed_target": "+normalized_vector",
            "target_checksums": {
                str(layer): tensor_checksum_hex(target)
                for layer, target in sorted(targets.items())
            },
        }
    if condition_name == "random_spherical":
        targets = _random_unit_vectors_like(layer_payloads, seed=seed)
        return targets, {
            "kind": "random_unit_target",
            "seed": seed,
            "target_checksums": {
                str(layer): tensor_checksum_hex(target)
                for layer, target in sorted(targets.items())
            },
        }
    raise SystemExit(f"Unsupported steering condition: {condition_name}")


def _generate_finish_reason(
    generated_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
) -> str:
    eos_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, list):
        eos_ids.update(int(value) for value in eos_token_id)
    if any(token_id in eos_ids for token_id in generated_ids):
        return "eos"
    if len(generated_ids) >= max_new_tokens:
        return "length"
    return "stop"


def _effective_max_tokens(prompt_len: int, *, max_new_tokens: int, max_model_len: int) -> int:
    return min(max_new_tokens, max(0, max_model_len - prompt_len))


def _hit_max_model_len(
    *,
    prompt_len: int,
    token_count: int,
    finish_reason: str,
    max_model_len: int,
) -> bool:
    if finish_reason != "length":
        return False
    return prompt_len + token_count >= max_model_len


def _task_specific_correct(
    *,
    task_kind: str,
    response_text: str,
    item: SteeringPromptItem,
) -> bool:
    if task_kind == "math_freeform":
        return math_freeform.grade(response_text, item.gold_answer or "")
    if task_kind == "multiple_choice_gpqa":
        return multiple_choice_gpqa.grade(response_text, item.gold_answer or "")
    if task_kind == "multiple_choice_mmlupro":
        return multiple_choice_mmlupro.grade(
            response_text,
            item.gold_answer or "",
            item.gold_index,
        )
    raise ValueError(f"Unsupported direct grading task_kind={task_kind!r}")


def _aggregate_completion_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_rows = [row for row in rows if not row["prompt_too_long"]]
    graded_rows = [row for row in generated_rows if row.get("correct") is not None]
    looped_rows = [row for row in generated_rows if row["loop_flag"]]
    max_hit_rows = [row for row in generated_rows if row["max_length_hit"]]
    lengths = [int(row["token_count"]) for row in generated_rows]
    budget_rows = [row for row in generated_rows if row["effective_max_tokens"] > 0]
    accuracy = (
        float(sum(1 for row in graded_rows if row["correct"])) / float(len(graded_rows))
        if graded_rows
        else None
    )
    return {
        "num_prompts": len({int(row["sample_id"]) for row in rows}),
        "num_generated": len(generated_rows),
        "num_prompt_too_long": sum(1 for row in rows if row["prompt_too_long"]),
        "num_graded": len(graded_rows),
        "accuracy": accuracy,
        "loop_fraction": (
            float(len(looped_rows)) / float(len(generated_rows))
            if generated_rows
            else None
        ),
        "max_length_hit_fraction": (
            float(len(max_hit_rows)) / float(len(generated_rows))
            if generated_rows
            else None
        ),
        "over_half_budget_fraction": (
            float(sum(1 for row in budget_rows if row["budget_fraction"] > 0.5))
            / float(len(budget_rows))
            if budget_rows
            else None
        ),
        "avg_generation_length": (
            float(sum(lengths)) / float(len(lengths))
            if lengths
            else None
        ),
        "median_generation_length": (
            float(median(lengths))
            if lengths
            else None
        ),
    }


def _seed_summary(
    *,
    rows: list[dict[str, Any]],
    diagnostics_by_layer: dict[str, Any],
    native_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = _aggregate_completion_rows(rows)
    summary["diagnostics_by_layer"] = diagnostics_by_layer
    if native_metrics is not None:
        summary["native_metrics"] = native_metrics
    return summary


def _aggregate_condition_summaries(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "loop_fraction",
        "max_length_hit_fraction",
        "over_half_budget_fraction",
        "avg_generation_length",
        "median_generation_length",
    )
    aggregate: dict[str, Any] = {
        "seeds": sorted(int(seed) for seed in seed_summaries),
    }
    for metric_name in metric_names:
        values = [
            float(summary[metric_name])
            for summary in seed_summaries.values()
            if summary.get(metric_name) is not None
        ]
        if not values:
            continue
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return aggregate


def _load_model_and_tokenizer(
    *,
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: str | None,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=tokenizer_revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise SystemExit("Tokenizer has no pad_token/eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "revision": model_revision,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _run_condition_seed(
    *,
    condition_name: str,
    seed: int,
    items: list[SteeringPromptItem],
    task_kind: str,
    benchmark_subset: list[Any] | None,
    layer_payloads: dict[int, LayerVectorPayload],
    model: Any,
    tokenizer: Any,
    model_layers: torch.nn.ModuleList,
    controller_t: float,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    max_model_len: int,
    livecodebench_repo: str,
    release_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    do_sample = temperature > 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)
    targets_by_layer, condition_config = _targets_for_condition(
        condition_name,
        layer_payloads,
        seed=seed,
    )

    controller: PrefillLayerOutputSphericalController | None = None
    if targets_by_layer:
        controller = PrefillLayerOutputSphericalController(
            layers=model_layers,
            targets_by_layer=targets_by_layer,
            t=controller_t,
        )
        controller.install()
        controller.reset_stats()

    rows: list[dict[str, Any]] = []
    lcb_records: list[LcbSampleRecord] = []
    eos_token_id = tokenizer.eos_token_id

    try:
        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                batch_items = items[start : start + batch_size]
                prompt_token_ids = tokenizer(
                    [item.prompt for item in batch_items],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]

                valid_items: list[tuple[SteeringPromptItem, int, int]] = []
                for item, token_ids in zip(batch_items, prompt_token_ids):
                    prompt_len = len(token_ids)
                    effective_max_tokens = _effective_max_tokens(
                        prompt_len,
                        max_new_tokens=max_new_tokens,
                        max_model_len=max_model_len,
                    )
                    if effective_max_tokens < 1:
                        row = {
                            "sample_id": item.sample_id,
                            "condition_name": condition_name,
                            "seed": seed,
                            "prompt_too_long": True,
                            "prompt_token_count": prompt_len,
                            "effective_max_tokens": effective_max_tokens,
                            "token_count": 0,
                            "total_token_count": prompt_len,
                            "budget_fraction": 0.0,
                            "loop_flag": False,
                            "max_length_hit": False,
                            "first_loop_prefix_length": None,
                            "finish_reason": "prompt_too_long",
                            "response_text": "",
                            "correct": None,
                            "question_id": item.question_id,
                        }
                        rows.append(row)
                        if task_kind == "livecodebench_codegen":
                            lcb_records.append(
                                LcbSampleRecord(
                                    question_id=item.question_id or "",
                                    generation_index=0,
                                    code_output="",
                                    token_count=0,
                                    prompt_token_count=prompt_len,
                                    total_token_count=prompt_len,
                                    effective_max_tokens=effective_max_tokens,
                                    max_model_len=max_model_len,
                                    loop_flag=False,
                                    max_length_hit=False,
                                    finish_reason="prompt_too_long",
                                    prompt_too_long=True,
                                    first_loop_prefix_length=None,
                                )
                            )
                        continue
                    valid_items.append((item, prompt_len, effective_max_tokens))

                if not valid_items:
                    continue

                encoded = tokenizer(
                    [item.prompt for item, _, _ in valid_items],
                    return_tensors="pt",
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                padded_prompt_len = int(input_ids.size(1))
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else None,
                    "top_p": top_p,
                    "top_k": top_k,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": eos_token_id,
                    "use_cache": True,
                }
                generation_kwargs = {
                    key: value
                    for key, value in generation_kwargs.items()
                    if value is not None
                }
                outputs = model.generate(**generation_kwargs)
                if outputs.size(0) != len(valid_items):
                    raise RuntimeError(
                        f"Expected {len(valid_items)} generated rows, got {outputs.size(0)}."
                    )

                for row_idx, (item, prompt_len, effective_max_tokens) in enumerate(valid_items):
                    generated_ids = outputs[row_idx, padded_prompt_len:].tolist()
                    response_text = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    finish_reason = _generate_finish_reason(
                        generated_ids,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_token_id,
                    )
                    first_loop_prefix = first_ngram_loop_prefix_length(
                        generated_ids,
                        n=30,
                        k=20,
                    )
                    loop_flag = first_loop_prefix is not None
                    token_count = len(generated_ids)
                    total_token_count = prompt_len + token_count
                    max_length_hit = _hit_max_model_len(
                        prompt_len=prompt_len,
                        token_count=token_count,
                        finish_reason=finish_reason,
                        max_model_len=max_model_len,
                    )
                    row = {
                        "sample_id": item.sample_id,
                        "condition_name": condition_name,
                        "seed": seed,
                        "prompt_too_long": False,
                        "prompt_token_count": prompt_len,
                        "effective_max_tokens": effective_max_tokens,
                        "token_count": token_count,
                        "total_token_count": total_token_count,
                        "budget_fraction": (
                            float(token_count) / float(effective_max_tokens)
                            if effective_max_tokens > 0
                            else 0.0
                        ),
                        "loop_flag": loop_flag,
                        "max_length_hit": max_length_hit,
                        "first_loop_prefix_length": first_loop_prefix,
                        "finish_reason": finish_reason,
                        "response_text": response_text,
                        "correct": None,
                        "question_id": item.question_id,
                    }
                    if task_kind == "livecodebench_codegen":
                        lcb_records.append(
                            LcbSampleRecord(
                                question_id=item.question_id or "",
                                generation_index=0,
                                code_output=livecodebench_codegen.extract_code_output(
                                    response_text,
                                    repo_path=livecodebench_repo,
                                    model_id=model.config._name_or_path
                                    if hasattr(model.config, "_name_or_path")
                                    else model.name_or_path,
                                ),
                                token_count=token_count,
                                prompt_token_count=prompt_len,
                                total_token_count=total_token_count,
                                effective_max_tokens=effective_max_tokens,
                                max_model_len=max_model_len,
                                loop_flag=loop_flag,
                                max_length_hit=max_length_hit,
                                finish_reason=finish_reason,
                                prompt_too_long=False,
                                first_loop_prefix_length=first_loop_prefix,
                            )
                        )
                    else:
                        row["correct"] = _task_specific_correct(
                            task_kind=task_kind,
                            response_text=response_text,
                            item=item,
                        )
                    rows.append(row)
    finally:
        if controller is not None:
            controller.remove()

    native_metrics = None
    if task_kind == "livecodebench_codegen":
        if benchmark_subset is None:
            raise RuntimeError("LiveCodeBench steering requires benchmark_subset.")
        native_metrics, grading_by_key = livecodebench_codegen.evaluate_records(
            benchmark_subset,
            lcb_records,
            repo_path=livecodebench_repo,
            release_version=release_version,
        )
        grading_by_sample_id = {
            int(item.sample_id): bool(grading_by_key[(item.question_id or "", 0)])
            for item in items
            if not any(row["sample_id"] == item.sample_id and row["prompt_too_long"] for row in rows)
        }
        for row in rows:
            if row["prompt_too_long"]:
                continue
            row["correct"] = grading_by_sample_id.get(int(row["sample_id"]))

    diagnostics_by_layer = controller.stats_json() if controller is not None else {}
    seed_summary = _seed_summary(
        rows=rows,
        diagnostics_by_layer=diagnostics_by_layer,
        native_metrics=native_metrics,
    )
    return rows, seed_summary, native_metrics, condition_config


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")
    if not (0.0 <= args.t <= 1.0):
        raise SystemExit("--t must be in [0, 1].")

    vector_export_dir = Path(args.vector_export_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_payload, layer_payloads, vector_bundle_hash = _load_layer_payloads(
        vector_export_dir,
        args.layers,
    )
    source_manifest, source_manifest_path = _resolve_source_manifest(layer_payloads)

    benchmark = args.benchmark or str(next(iter(layer_payloads.values())).record["benchmark"])
    model_id = args.model_id or str(next(iter(layer_payloads.values())).record.get("model_id") or _manifest_model_id(source_manifest))
    record_model_revision = next(iter(layer_payloads.values())).record.get("model_revision")
    model_revision = args.model_revision if args.model_revision is not None else (
        str(record_model_revision) if record_model_revision else None
    )
    record_tokenizer_revision = next(iter(layer_payloads.values())).record.get("tokenizer_revision")
    tokenizer_revision = args.tokenizer_revision if args.tokenizer_revision is not None else (
        str(record_tokenizer_revision) if record_tokenizer_revision else None
    )

    release_version = _resolve_release_version(source_manifest, args.release_version)
    (
        items,
        prompt_ids,
        _bundle_benchmark,
        task_kind,
        benchmark_subset,
        grader_version,
    ) = _build_prompt_items(
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        layer_payloads=layer_payloads,
        split=args.split,
        max_samples=args.max_samples,
        livecodebench_repo=args.livecodebench_repo,
        release_version=release_version,
    )

    generation_temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(_manifest_rollout_value(source_manifest, "temperature", default=0.2))
    )
    resolved_top_p, resolved_top_k = resolve_sampling_defaults(
        model_id,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    hf_top_k = 0 if resolved_top_k < 0 else resolved_top_k
    max_new_tokens = (
        int(args.max_new_tokens)
        if args.max_new_tokens is not None
        else int(_manifest_rollout_value(source_manifest, "max_tokens", default=30000))
    )
    max_model_len = (
        int(args.max_model_len)
        if args.max_model_len is not None
        else int(_manifest_rollout_value(source_manifest, "max_model_len", default=40960))
    )

    prompt_hash = _prompt_hash([item.prompt for item in items])
    git_commit = current_git_commit(ROOT)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    model, tokenizer = _load_model_and_tokenizer(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        device=device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model_layers, layer_path = _resolve_transformer_layers(model)
    if max(layer_payloads) >= len(model_layers):
        raise SystemExit(
            f"Vector bundle references layer {max(layer_payloads)}, but model only has "
            f"{len(model_layers)} transformer blocks at {layer_path}."
        )

    config_payload = {
        "schema_name": "prompt_profile_rfm_steering_config.v1",
        "benchmark": benchmark,
        "split": args.split,
        "task_kind": task_kind,
        "vector_export_dir": str(vector_export_dir),
        "vector_bundle_hash": vector_bundle_hash,
        "selected_layers": sorted(layer_payloads),
        "prompt_ids": prompt_ids,
        "prompt_text_hash": prompt_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "git_commit": git_commit,
        "device": str(device),
        "dtype": args.dtype,
        "hook_site": args.hook_site,
        "t": float(args.t),
        "conditions": list(args.conditions),
        "seeds": [int(seed) for seed in args.seed],
        "generation_config": {
            "temperature": generation_temperature,
            "top_p": resolved_top_p,
            "top_k": hf_top_k,
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "batch_size": int(args.batch_size),
            "do_sample": bool(generation_temperature > 0.0),
        },
        "grader_version": grader_version,
        "source_manifest_path": str(source_manifest_path),
    }
    _write_json(out_dir / "config.json", config_payload)

    condition_summaries: dict[str, dict[str, Any]] = {}

    for condition_name in args.conditions:
        condition_dir = out_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        seed_summaries: dict[str, dict[str, Any]] = {}
        last_condition_config: dict[str, Any] | None = None
        for seed in args.seed:
            seed_dir = condition_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            rows, seed_summary, _native_metrics, condition_config = _run_condition_seed(
                condition_name=condition_name,
                seed=int(seed),
                items=items,
                task_kind=task_kind,
                benchmark_subset=benchmark_subset,
                layer_payloads=layer_payloads,
                model=model,
                tokenizer=tokenizer,
                model_layers=model_layers,
                controller_t=args.t,
                batch_size=args.batch_size,
                temperature=generation_temperature,
                top_p=resolved_top_p,
                top_k=hf_top_k,
                max_new_tokens=max_new_tokens,
                max_model_len=max_model_len,
                livecodebench_repo=args.livecodebench_repo,
                release_version=release_version,
            )
            last_condition_config = condition_config
            _write_jsonl(seed_dir / "completions.jsonl", rows)
            _write_json(seed_dir / "summary.json", seed_summary)
            seed_summaries[str(seed)] = seed_summary

        aggregate_summary = _aggregate_condition_summaries(seed_summaries)
        record = write_stage_artifact_record(
            condition_dir / "steering_run_record.json",
            build_steering_run_record(
                benchmark=benchmark,
                condition_name=condition_name,
                vector_artifact_hash=vector_bundle_hash,
                hook_site=args.hook_site,
                t=float(args.t),
                seeds=[int(seed) for seed in args.seed],
                prompt_ids=prompt_ids,
                prompt_text_hash=prompt_hash,
                generation_config=config_payload["generation_config"],
                grader_version=grader_version,
                output_path=str(condition_dir),
                git_commit=git_commit,
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                condition_config=last_condition_config,
            ),
        )
        condition_summary = {
            "condition_name": condition_name,
            "record": record,
            "aggregate": aggregate_summary,
            "seed_summaries": seed_summaries,
        }
        _write_json(condition_dir / "condition_summary.json", condition_summary)
        condition_summaries[condition_name] = condition_summary

    no_steer_summary = condition_summaries.get("no_steer")
    no_steer_accuracy = None
    if no_steer_summary is not None:
        accuracy_payload = no_steer_summary.get("aggregate", {}).get("accuracy")
        if isinstance(accuracy_payload, dict) and "mean" in accuracy_payload:
            no_steer_accuracy = float(accuracy_payload["mean"])
    if no_steer_accuracy is not None:
        for condition_name, condition_summary in condition_summaries.items():
            accuracy_payload = condition_summary.get("aggregate", {}).get("accuracy")
            if isinstance(accuracy_payload, dict) and "mean" in accuracy_payload:
                condition_summary["aggregate"]["accuracy_delta_vs_no_steer"] = (
                    float(accuracy_payload["mean"]) - no_steer_accuracy
                )
                _write_json(out_dir / condition_name / "condition_summary.json", condition_summary)

    overall_summary = {
        "schema_name": "prompt_profile_rfm_steering_summary.v1",
        "benchmark": benchmark,
        "split": args.split,
        "vector_bundle_hash": vector_bundle_hash,
        "conditions": condition_summaries,
        "config": config_payload,
    }
    _write_json(out_dir / "summary.json", overall_summary)


if __name__ == "__main__":
    main()
