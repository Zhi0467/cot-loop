from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ROOT)
DEFAULT_SUITE_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "rollout",
    "main_rollout_stats_suite.json",
)
CONFIG_SCHEMA_NAME = "main_rollout_stats_suite.config.v1"
CPU_POSTHOC_TASK_KINDS = frozenset({"livecodebench_codegen", "taco_codegen"})


@dataclass(frozen=True)
class MainRolloutSuiteConfig:
    model_id: str
    temperature: float
    num_generations: int
    max_tokens: int
    max_model_len: int
    tp: int = 1
    dp: int = 1
    dtype: str = "bfloat16"
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    seed: int = 0
    loop_n: int = 30
    loop_k: int = 20
    statistics: str = ""
    max_samples: int | None = None


@dataclass(frozen=True)
class MainRolloutDataset:
    key: str
    display_name: str
    task_kind: str
    dataset: str
    split: str
    dataset_config: str | None = None
    question_field: str | None = None
    answer_field: str | None = None
    starter_code_field: str | None = None
    record_id_field: str | None = None
    metadata_fields: tuple[str, ...] = ()
    row_filter: dict[str, object] | None = None
    prompt_format: str | None = None
    release_version: str | None = None
    lm_style_override: str | None = None
    max_samples: int | None = None
    requires_livecodebench_repo: bool = False


@dataclass(frozen=True)
class MainRolloutSuiteDefinition:
    config_path: str
    suite_config: MainRolloutSuiteConfig
    datasets: tuple[MainRolloutDataset, ...]


def _resolve_config_path(config_path: str | None) -> str:
    raw_path = config_path or DEFAULT_SUITE_CONFIG_PATH
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, raw_path))


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _required(raw: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in raw:
        raise ValueError(f"{context} is missing required field {key!r}.")
    return raw[key]


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return str(value)


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    return int(value)


def _statistics_to_csv(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        stats = [str(item).strip() for item in raw if str(item).strip()]
        if stats:
            return ",".join(stats)
    raise ValueError("suite_config.statistics must be a non-empty string or list.")


def _suite_config_from_dict(raw: dict[str, Any]) -> MainRolloutSuiteConfig:
    context = "suite_config"
    return MainRolloutSuiteConfig(
        model_id=str(_required(raw, "model_id", context=context)),
        temperature=float(_required(raw, "temperature", context=context)),
        num_generations=int(_required(raw, "num_generations", context=context)),
        max_tokens=int(_required(raw, "max_tokens", context=context)),
        max_model_len=int(_required(raw, "max_model_len", context=context)),
        tp=int(raw.get("tp", 1)),
        dp=int(raw.get("dp", 1)),
        dtype=str(raw.get("dtype", "bfloat16")),
        max_num_seqs=_optional_int(raw, "max_num_seqs"),
        max_num_batched_tokens=_optional_int(raw, "max_num_batched_tokens"),
        seed=int(raw.get("seed", 0)),
        loop_n=int(raw.get("loop_n", 30)),
        loop_k=int(raw.get("loop_k", 20)),
        statistics=_statistics_to_csv(_required(raw, "statistics", context=context)),
        max_samples=_optional_int(raw, "max_samples"),
    )


def _dataset_from_dict(raw: dict[str, Any], *, index: int) -> MainRolloutDataset:
    context = f"datasets[{index}]"
    metadata_fields = raw.get("metadata_fields", ())
    if not isinstance(metadata_fields, (list, tuple)):
        raise ValueError(f"{context}.metadata_fields must be a list when provided.")
    row_filter = raw.get("row_filter")
    if row_filter is not None and not isinstance(row_filter, dict):
        raise ValueError(f"{context}.row_filter must be a JSON object when provided.")
    return MainRolloutDataset(
        key=str(_required(raw, "key", context=context)),
        display_name=str(_required(raw, "display_name", context=context)),
        task_kind=str(_required(raw, "task_kind", context=context)),
        dataset=str(_required(raw, "dataset", context=context)),
        split=str(_required(raw, "split", context=context)),
        dataset_config=_optional_str(raw, "dataset_config"),
        question_field=_optional_str(raw, "question_field"),
        answer_field=_optional_str(raw, "answer_field"),
        starter_code_field=_optional_str(raw, "starter_code_field"),
        record_id_field=_optional_str(raw, "record_id_field"),
        metadata_fields=tuple(str(field) for field in metadata_fields),
        row_filter=row_filter,
        prompt_format=_optional_str(raw, "prompt_format"),
        release_version=_optional_str(raw, "release_version"),
        lm_style_override=_optional_str(raw, "lm_style_override"),
        max_samples=_optional_int(raw, "max_samples"),
        requires_livecodebench_repo=bool(raw.get("requires_livecodebench_repo", False)),
    )


def load_suite_definition(config_path: str | None = None) -> MainRolloutSuiteDefinition:
    resolved_path = _resolve_config_path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        raw = _require_mapping(json.load(handle), name=resolved_path)

    schema_name = raw.get("schema_name")
    if schema_name != CONFIG_SCHEMA_NAME:
        raise ValueError(
            f"{resolved_path} has schema_name={schema_name!r}; "
            f"expected {CONFIG_SCHEMA_NAME!r}."
        )
    suite_config = _suite_config_from_dict(
        _require_mapping(
            _required(raw, "suite_config", context=resolved_path),
            name="suite_config",
        )
    )
    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(f"{resolved_path} must define a non-empty datasets list.")
    datasets = tuple(
        _dataset_from_dict(_require_mapping(dataset, name=f"datasets[{index}]"), index=index)
        for index, dataset in enumerate(raw_datasets)
    )
    keys = [dataset.key for dataset in datasets]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"{resolved_path} has duplicate dataset keys: {duplicate_keys}.")
    return MainRolloutSuiteDefinition(
        config_path=resolved_path,
        suite_config=suite_config,
        datasets=datasets,
    )


SUITE_DEFINITION = load_suite_definition()
SUITE_CONFIG = SUITE_DEFINITION.suite_config
SUITE_DATASETS = SUITE_DEFINITION.datasets


def suite_dataset_keys(
    suite_definition: MainRolloutSuiteDefinition | None = None,
) -> list[str]:
    definition = suite_definition or SUITE_DEFINITION
    return [dataset.key for dataset in definition.datasets]


def get_suite_dataset(
    key: str,
    *,
    suite_definition: MainRolloutSuiteDefinition | None = None,
) -> MainRolloutDataset:
    definition = suite_definition or SUITE_DEFINITION
    for dataset in definition.datasets:
        if dataset.key == key:
            return dataset
    raise KeyError(f"Unknown suite dataset key: {key}")


def build_collect_env(
    dataset_key: str,
    thinking_mode: str,
    *,
    suite_config: MainRolloutSuiteConfig,
    output_root: str,
    livecodebench_repo: str | None = None,
    suite_definition: MainRolloutSuiteDefinition | None = None,
) -> dict[str, str]:
    dataset = get_suite_dataset(dataset_key, suite_definition=suite_definition)
    mode = thinking_mode.strip().lower()
    if mode not in {"on", "off"}:
        raise ValueError(f"Unsupported thinking mode {thinking_mode!r}; expected 'on' or 'off'.")
    if dataset.requires_livecodebench_repo and not livecodebench_repo:
        raise ValueError(f"{dataset.key} requires a LiveCodeBench repo path.")

    out_dir = os.path.join(output_root, dataset.key)
    out_path = os.path.join(out_dir, f"{dataset.key}__thinking_{mode}.json")
    effective_max_samples = (
        dataset.max_samples
        if dataset.max_samples is not None
        else suite_config.max_samples
    )

    env = {
        "TASK_KIND": dataset.task_kind,
        "DATASET": dataset.dataset,
        "SPLIT": dataset.split,
        "MODEL_ID": suite_config.model_id,
        "TEMPERATURE": str(suite_config.temperature),
        "NUM_GENERATIONS": str(suite_config.num_generations),
        "MAX_TOKENS": str(suite_config.max_tokens),
        "MAX_MODEL_LEN": str(suite_config.max_model_len),
        "TP": str(suite_config.tp),
        "DP": str(suite_config.dp),
        "DTYPE": suite_config.dtype,
        "SEED": str(suite_config.seed),
        "LOOP_N": str(suite_config.loop_n),
        "LOOP_K": str(suite_config.loop_k),
        "STATISTICS": suite_config.statistics,
        "THINKING_MODE": mode,
        "OUT": out_path,
    }
    if effective_max_samples is not None:
        env["MAX_SAMPLES"] = str(effective_max_samples)
    if suite_config.max_num_seqs is not None:
        env["MAX_NUM_SEQS"] = str(suite_config.max_num_seqs)
    if suite_config.max_num_batched_tokens is not None:
        env["MAX_NUM_BATCHED_TOKENS"] = str(suite_config.max_num_batched_tokens)
    if dataset.dataset_config is not None:
        env["DATASET_CONFIG"] = dataset.dataset_config
    if dataset.question_field is not None:
        env["QUESTION_FIELD"] = dataset.question_field
    if dataset.answer_field is not None:
        env["ANSWER_FIELD"] = dataset.answer_field
    if dataset.starter_code_field is not None:
        env["STARTER_CODE_FIELD"] = dataset.starter_code_field
    if dataset.record_id_field is not None:
        env["RECORD_ID_FIELD"] = dataset.record_id_field
    if dataset.metadata_fields:
        env["METADATA_FIELDS"] = ",".join(dataset.metadata_fields)
    if dataset.row_filter is not None:
        env["ROW_FILTER_JSON"] = json.dumps(dataset.row_filter, separators=(",", ":"))
    if dataset.prompt_format is not None:
        env["PROMPT_FORMAT"] = dataset.prompt_format
    if dataset.release_version is not None:
        env["RELEASE_VERSION"] = dataset.release_version
    if dataset.lm_style_override is not None:
        env["LM_STYLE_OVERRIDE"] = dataset.lm_style_override
    if dataset.requires_livecodebench_repo:
        env["LIVECODEBENCH_REPO"] = str(livecodebench_repo)
        env["LCB_NUM_PROCESS_EVALUATE"] = os.environ.get(
            "LCB_NUM_PROCESS_EVALUATE",
            "32",
        )
    if dataset.task_kind in CPU_POSTHOC_TASK_KINDS:
        env["DEFER_CPU_FINALIZE"] = "1"
    return env
