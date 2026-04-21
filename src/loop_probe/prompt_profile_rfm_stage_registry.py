"""Shared retained-benchmark registry for the prompt-profile RFM stage."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def default_archive_source_root() -> Path:
    user = os.environ.get("USER") or getpass.getuser()
    return Path("/data/scratch") / user / "outputs" / "cot-loop-detection"


DEFAULT_ARCHIVE_SOURCE_ROOT = default_archive_source_root()
DEFAULT_FEATURE_KEY = "last_token_all_layers_stack_final"
DEFAULT_SAMPLE_SHAPE = (28, 2048)
DEFAULT_SOURCE_TARGET_NAME = "mean_relative_length"
DEFAULT_STAGE_LABEL_NAME = "majority_s_0.5"
STAGE_REGISTRY_VERSION = "prompt_profile_rfm_stage.v1"


@dataclass(frozen=True)
class PromptProfileRFMStageDataset:
    key: str
    display_name: str
    prompt_field: str
    archive_relpath: str
    train_count: int
    test_count: int
    train_prompt_ids_sha256: str
    test_prompt_ids_sha256: str
    train_prompt_text_sha256: str
    test_prompt_text_sha256: str
    active_stage: bool
    source_target_name: str = DEFAULT_SOURCE_TARGET_NAME
    stage_label_name: str = DEFAULT_STAGE_LABEL_NAME
    feature_key: str = DEFAULT_FEATURE_KEY
    sample_shape: tuple[int, int] = DEFAULT_SAMPLE_SHAPE

    def archive_data_dir(self, archive_source_root: Path | str | None = None) -> Path:
        base = Path(archive_source_root) if archive_source_root is not None else DEFAULT_ARCHIVE_SOURCE_ROOT
        return (base / self.archive_relpath).resolve()

    def manifest_path(self, archive_source_root: Path | str | None = None) -> Path:
        return self.archive_data_dir(archive_source_root) / "manifest.json"


@dataclass(frozen=True)
class PromptProfileRFMStageValidationResult:
    dataset: PromptProfileRFMStageDataset
    archive_source_root: str
    archive_data_dir: str
    manifest_path: str
    prompt_rollout_archive_file: str
    prompt_profile_files: dict[str, str]
    default_feature_key: str
    sample_shape: list[int]
    target_name: str | None
    train_prompt_ids: list[int]
    test_prompt_ids: list[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": {
                "key": self.dataset.key,
                "display_name": self.dataset.display_name,
                "prompt_field": self.dataset.prompt_field,
                "archive_relpath": self.dataset.archive_relpath,
                "train_count": self.dataset.train_count,
                "test_count": self.dataset.test_count,
                "train_prompt_ids_sha256": self.dataset.train_prompt_ids_sha256,
                "test_prompt_ids_sha256": self.dataset.test_prompt_ids_sha256,
                "train_prompt_text_sha256": self.dataset.train_prompt_text_sha256,
                "test_prompt_text_sha256": self.dataset.test_prompt_text_sha256,
                "active_stage": self.dataset.active_stage,
                "source_target_name": self.dataset.source_target_name,
                "stage_label_name": self.dataset.stage_label_name,
                "feature_key": self.dataset.feature_key,
                "sample_shape": list(self.dataset.sample_shape),
            },
            "archive_source_root": self.archive_source_root,
            "archive_data_dir": self.archive_data_dir,
            "manifest_path": self.manifest_path,
            "prompt_rollout_archive_file": self.prompt_rollout_archive_file,
            "prompt_profile_files": self.prompt_profile_files,
            "default_feature_key": self.default_feature_key,
            "sample_shape": self.sample_shape,
            "target_name": self.target_name,
            "train_prompt_ids": self.train_prompt_ids,
            "test_prompt_ids": self.test_prompt_ids,
        }


ALL_STAGE_DATASETS: tuple[PromptProfileRFMStageDataset, ...] = (
    PromptProfileRFMStageDataset(
        key="gpqa",
        display_name="GPQA",
        prompt_field="Question",
        archive_relpath="gpqa_mean_relative_from_archive_20260322/data",
        train_count=158,
        test_count=40,
        train_prompt_ids_sha256="a7668afb27b0e1e56f412d0e013c27733f9cb720feb0b093a483cb06dea912b5",
        test_prompt_ids_sha256="413eda0609b696963f5dd8241b2bcf02b606b1838f58e4198f890e1d15627fc5",
        train_prompt_text_sha256="de3910e0b5320f8c51374c183aab3ea9f4a05dffe3bbef1ad66f568f437d6421",
        test_prompt_text_sha256="8fd26361b6aa35558e23d1745d167f9d4adc575a758e41016ea579db16976606",
        active_stage=True,
    ),
    PromptProfileRFMStageDataset(
        key="math500",
        display_name="MATH-500",
        prompt_field="problem",
        archive_relpath="math_mean_relative_from_archive_20260323",
        train_count=400,
        test_count=100,
        train_prompt_ids_sha256="64498f19cf6933bd4c7687d8f2cbc179f54bc1633732f73a05553f2fff518710",
        test_prompt_ids_sha256="8165fdbb093b8b7f7b48a8aaa72ee75712e86bc8f056c3ec83e4f5a37e9999c4",
        train_prompt_text_sha256="4d4645c289fc264edd53ccaed51987cec3829989852279f41ae1356d516890cb",
        test_prompt_text_sha256="e654f5286d06739f44e0002041ddf83fd1eb567d9f6a313b71a8b241a799b81d",
        active_stage=True,
    ),
    PromptProfileRFMStageDataset(
        key="mmlu_pro",
        display_name="MMLU-Pro",
        prompt_field="problem",
        archive_relpath="mmlu_mean_relative_from_archive_20260323",
        train_count=640,
        test_count=160,
        train_prompt_ids_sha256="71291f7596ffd566f6f913f272e0c309e9d19a3c186e4969c5124b5f290f57de",
        test_prompt_ids_sha256="c07a4318c49a8b30d5d7f2d69e7c6b359b3eddc0c52f5a59cbeafa4a09f7110e",
        train_prompt_text_sha256="a8651546112d920301065465f143198675ef78cf0a2d90ea2c671189b957f827",
        test_prompt_text_sha256="d4f5fdc14a4fdd3cee9548cf86ada5aa3fc10e45515c382a3248bc7cd659f977",
        active_stage=True,
    ),
    PromptProfileRFMStageDataset(
        key="livecodebench",
        display_name="LiveCodeBench",
        prompt_field="problem",
        archive_relpath="livecodebench_mean_relative_from_archive_20260323",
        train_count=640,
        test_count=160,
        train_prompt_ids_sha256="71291f7596ffd566f6f913f272e0c309e9d19a3c186e4969c5124b5f290f57de",
        test_prompt_ids_sha256="c07a4318c49a8b30d5d7f2d69e7c6b359b3eddc0c52f5a59cbeafa4a09f7110e",
        train_prompt_text_sha256="aff770a20adcf388158e5f3bfed90898718a1c19c1cf3959dcdbf8aa41aeb08a",
        test_prompt_text_sha256="a26c5ca70aaa3e5f416c4b0e70524e0ffc2353af357ecde8dc27487d3d1d068a",
        active_stage=True,
    ),
    PromptProfileRFMStageDataset(
        key="aime",
        display_name="AIME",
        prompt_field="question",
        archive_relpath="aime_mean_relative_from_archive_20260322",
        train_count=48,
        test_count=12,
        train_prompt_ids_sha256="9f39c58a61ea9f3aea282b26fefee412732692b11f1e438c5a066e0ab91a0ed4",
        test_prompt_ids_sha256="9da066c391ab69220c841e9133b3929e4e0fcbbbc102b361fa0d7da332305b18",
        train_prompt_text_sha256="01366493c53b24e4bf9cf70906d44ebdf2349d3f061d0860cccb26a4b6d183ab",
        test_prompt_text_sha256="e89d907879cf19c5510f11e1246fca5483690b7881f10b3cff460f919ce2e3f4",
        active_stage=False,
    ),
)

DATASET_BY_KEY = {dataset.key: dataset for dataset in ALL_STAGE_DATASETS}


def active_stage_datasets() -> tuple[PromptProfileRFMStageDataset, ...]:
    return tuple(dataset for dataset in ALL_STAGE_DATASETS if dataset.active_stage)


def iter_stage_datasets(*, include_excluded: bool = False) -> tuple[PromptProfileRFMStageDataset, ...]:
    if include_excluded:
        return ALL_STAGE_DATASETS
    return active_stage_datasets()


def get_stage_dataset(key: str) -> PromptProfileRFMStageDataset:
    try:
        return DATASET_BY_KEY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_BY_KEY))
        raise KeyError(f"Unknown stage dataset '{key}'. Valid: {valid}") from exc


def stage_registry_payload(
    archive_source_root: Path | str | None = None,
    *,
    include_excluded: bool = False,
) -> dict[str, Any]:
    base = Path(archive_source_root) if archive_source_root is not None else DEFAULT_ARCHIVE_SOURCE_ROOT
    datasets = []
    for dataset in iter_stage_datasets(include_excluded=include_excluded):
        datasets.append(
            {
                "key": dataset.key,
                "display_name": dataset.display_name,
                "prompt_field": dataset.prompt_field,
                "archive_relpath": dataset.archive_relpath,
                "archive_data_dir": str(dataset.archive_data_dir(base)),
                "train_count": dataset.train_count,
                "test_count": dataset.test_count,
                "train_prompt_ids_sha256": dataset.train_prompt_ids_sha256,
                "test_prompt_ids_sha256": dataset.test_prompt_ids_sha256,
                "train_prompt_text_sha256": dataset.train_prompt_text_sha256,
                "test_prompt_text_sha256": dataset.test_prompt_text_sha256,
                "active_stage": dataset.active_stage,
                "source_target_name": dataset.source_target_name,
                "stage_label_name": dataset.stage_label_name,
                "feature_key": dataset.feature_key,
                "sample_shape": list(dataset.sample_shape),
            }
        )
    return {
        "schema_name": STAGE_REGISTRY_VERSION,
        "archive_source_root": str(base),
        "datasets": datasets,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _sample_ids_sha256(sample_ids: list[int]) -> str:
    body = json.dumps(sample_ids, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _prompt_text_sha256(prompts: list[str]) -> str:
    body = json.dumps(prompts, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _resolve_sample_shape(manifest: dict[str, Any], feature_key: str) -> list[int]:
    feature_views = manifest.get("feature_views")
    if isinstance(feature_views, dict):
        view_info = feature_views.get(feature_key)
        if isinstance(view_info, dict):
            sample_shape = view_info.get("sample_shape")
            if isinstance(sample_shape, list) and all(isinstance(x, int) for x in sample_shape):
                return sample_shape
    sample_shape = manifest.get("sample_shape")
    if isinstance(sample_shape, list) and all(isinstance(x, int) for x in sample_shape):
        return sample_shape
    raise SystemExit(f"Manifest is missing sample_shape for feature_key='{feature_key}'.")


def _resolve_split_sample_ids(data_dir: Path, split_name: str, feature_key: str) -> list[int]:
    manifest = _read_json(data_dir / "manifest.json")
    feature_views = manifest.get("feature_views")
    if isinstance(feature_views, dict):
        view_info = feature_views.get(feature_key)
        if not isinstance(view_info, dict):
            raise SystemExit(
                f"Feature key '{feature_key}' not found in manifest for {data_dir}."
            )
        split_info = view_info.get(split_name)
        if not isinstance(split_info, dict):
            raise SystemExit(f"Missing split '{split_name}' for feature_key='{feature_key}'.")
    else:
        split_info = manifest.get(split_name)
        if not isinstance(split_info, dict):
            raise SystemExit(f"Missing legacy split '{split_name}' in manifest for {data_dir}.")

    shard_relpaths = split_info.get("shards")
    if not isinstance(shard_relpaths, list) or not shard_relpaths:
        raise SystemExit(
            f"Manifest split '{split_name}' is missing shard metadata for {data_dir}."
        )
    sample_ids: list[int] = []
    for relpath in shard_relpaths:
        shard = torch_load(data_dir / relpath)
        shard_ids = shard.get("sample_ids")
        if shard_ids is None:
            raise SystemExit(f"Shard '{relpath}' is missing sample_ids.")
        sample_ids.extend(int(value) for value in shard_ids.tolist())
    return sample_ids


def validate_stage_dataset(
    dataset: PromptProfileRFMStageDataset,
    archive_source_root: Path | str | None = None,
) -> PromptProfileRFMStageValidationResult:
    base = Path(archive_source_root) if archive_source_root is not None else DEFAULT_ARCHIVE_SOURCE_ROOT
    data_dir = dataset.archive_data_dir(base)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest for dataset '{dataset.key}': {manifest_path}")

    manifest = _read_json(manifest_path)
    default_feature_key = manifest.get("default_feature_key")
    if default_feature_key != dataset.feature_key:
        raise SystemExit(
            f"Dataset '{dataset.key}' default_feature_key mismatch: "
            f"expected '{dataset.feature_key}', got '{default_feature_key}'."
        )

    sample_shape = _resolve_sample_shape(manifest, dataset.feature_key)
    if tuple(sample_shape) != dataset.sample_shape:
        raise SystemExit(
            f"Dataset '{dataset.key}' sample_shape mismatch: "
            f"expected {dataset.sample_shape}, got {tuple(sample_shape)}."
        )

    target_spec = manifest.get("target_spec")
    target_name = target_spec.get("name") if isinstance(target_spec, dict) else None
    if target_name != dataset.source_target_name:
        raise SystemExit(
            f"Dataset '{dataset.key}' target mismatch: "
            f"expected '{dataset.source_target_name}', got '{target_name}'."
        )

    prompt_profile_files = manifest.get("prompt_profile_files")
    if not isinstance(prompt_profile_files, dict):
        raise SystemExit(f"Dataset '{dataset.key}' manifest is missing prompt_profile_files.")
    required_prompt_profile_files: dict[str, str] = {}
    for split_name in ("train", "test"):
        relpath = prompt_profile_files.get(split_name)
        if not isinstance(relpath, str) or not relpath:
            raise SystemExit(
                f"Dataset '{dataset.key}' prompt_profile_files is missing split '{split_name}'."
            )
        candidate = data_dir / relpath
        if not candidate.exists():
            raise SystemExit(
                f"Dataset '{dataset.key}' prompt-profile file is missing: {candidate}"
            )
        required_prompt_profile_files[split_name] = relpath

    prompt_rollout_archive_file = manifest.get("prompt_rollout_archive_file")
    if not isinstance(prompt_rollout_archive_file, str) or not prompt_rollout_archive_file:
        raise SystemExit(
            f"Dataset '{dataset.key}' manifest is missing prompt_rollout_archive_file."
        )
    if not (data_dir / prompt_rollout_archive_file).exists():
        raise SystemExit(
            f"Dataset '{dataset.key}' prompt rollout archive is missing: "
            f"{data_dir / prompt_rollout_archive_file}"
        )
    archive_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_jsonl_rows(data_dir / prompt_rollout_archive_file):
        split = row.get("split")
        sample_id = row.get("sample_id")
        if not isinstance(split, str) or not isinstance(sample_id, int):
            raise SystemExit(
                f"Dataset '{dataset.key}' archive row is missing split/sample_id."
            )
        archive_rows_by_key[(split, sample_id)] = row

    train_prompt_ids = _resolve_split_sample_ids(data_dir, "train", dataset.feature_key)
    test_prompt_ids = _resolve_split_sample_ids(data_dir, "test", dataset.feature_key)
    if len(train_prompt_ids) != dataset.train_count:
        raise SystemExit(
            f"Dataset '{dataset.key}' train_count mismatch: "
            f"expected {dataset.train_count}, got {len(train_prompt_ids)}."
        )
    if len(test_prompt_ids) != dataset.test_count:
        raise SystemExit(
            f"Dataset '{dataset.key}' test_count mismatch: "
            f"expected {dataset.test_count}, got {len(test_prompt_ids)}."
        )
    if _sample_ids_sha256(train_prompt_ids) != dataset.train_prompt_ids_sha256:
        raise SystemExit(
            f"Dataset '{dataset.key}' train prompt-ID hash mismatch: "
            f"expected {dataset.train_prompt_ids_sha256}, "
            f"got {_sample_ids_sha256(train_prompt_ids)}."
        )
    if _sample_ids_sha256(test_prompt_ids) != dataset.test_prompt_ids_sha256:
        raise SystemExit(
            f"Dataset '{dataset.key}' test prompt-ID hash mismatch: "
            f"expected {dataset.test_prompt_ids_sha256}, "
            f"got {_sample_ids_sha256(test_prompt_ids)}."
        )
    train_prompts = []
    for sample_id in train_prompt_ids:
        row = archive_rows_by_key.get(("train", sample_id))
        prompt = None if row is None else row.get("prompt")
        if not isinstance(prompt, str):
            raise SystemExit(
                f"Dataset '{dataset.key}' is missing train prompt text for sample_id={sample_id}."
            )
        train_prompts.append(prompt)
    test_prompts = []
    for sample_id in test_prompt_ids:
        row = archive_rows_by_key.get(("test", sample_id))
        prompt = None if row is None else row.get("prompt")
        if not isinstance(prompt, str):
            raise SystemExit(
                f"Dataset '{dataset.key}' is missing test prompt text for sample_id={sample_id}."
            )
        test_prompts.append(prompt)
    if _prompt_text_sha256(train_prompts) != dataset.train_prompt_text_sha256:
        raise SystemExit(
            f"Dataset '{dataset.key}' train prompt-text hash mismatch: "
            f"expected {dataset.train_prompt_text_sha256}, "
            f"got {_prompt_text_sha256(train_prompts)}."
        )
    if _prompt_text_sha256(test_prompts) != dataset.test_prompt_text_sha256:
        raise SystemExit(
            f"Dataset '{dataset.key}' test prompt-text hash mismatch: "
            f"expected {dataset.test_prompt_text_sha256}, "
            f"got {_prompt_text_sha256(test_prompts)}."
        )

    return PromptProfileRFMStageValidationResult(
        dataset=dataset,
        archive_source_root=str(base),
        archive_data_dir=str(data_dir),
        manifest_path=str(manifest_path),
        prompt_rollout_archive_file=prompt_rollout_archive_file,
        prompt_profile_files=required_prompt_profile_files,
        default_feature_key=str(default_feature_key),
        sample_shape=sample_shape,
        target_name=str(target_name),
        train_prompt_ids=train_prompt_ids,
        test_prompt_ids=test_prompt_ids,
    )


def validate_stage_registry(
    archive_source_root: Path | str | None = None,
    *,
    include_excluded: bool = False,
) -> dict[str, Any]:
    base = Path(archive_source_root) if archive_source_root is not None else DEFAULT_ARCHIVE_SOURCE_ROOT
    return {
        "schema_name": "prompt_profile_rfm_stage_registry_validation.v1",
        "archive_source_root": str(base),
        "datasets": [
            validate_stage_dataset(dataset, base).to_json()
            for dataset in iter_stage_datasets(include_excluded=include_excluded)
        ],
    }


def torch_load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")
