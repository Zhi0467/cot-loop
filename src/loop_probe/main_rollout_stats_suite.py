from __future__ import annotations

import json
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_STATISTICS = (
    "success_fraction,loop_fraction,avg_generation_length,"
    "avg_loop_generation_length,avg_first_loop_prefix_length,"
    "max_length_hit_fraction,loop_max_length_hit_fraction,"
    "max_length_hit_loop_fraction,generation_length_variance,"
    "max_length_hit_success_fraction,loop_success_fraction,"
    "avg_correct_generation_length,avg_wrong_generation_length"
)


@dataclass(frozen=True)
class MainRolloutSuiteConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    temperature: float = 0.2
    num_generations: int = 10
    max_tokens: int = 81920
    max_model_len: int = 40960
    tp: int = 1
    dp: int = 1
    dtype: str = "bfloat16"
    max_num_seqs: int | None = 10
    max_num_batched_tokens: int | None = 4096
    seed: int = 0
    statistics: str = DEFAULT_STATISTICS
    max_samples: int = 800


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
    row_filter: dict[str, dict[str, object]] | None = None
    prompt_format: str | None = None
    release_version: str | None = None
    lm_style_override: str | None = None
    requires_livecodebench_repo: bool = False
    requires_exclude_prompt_jsonl: bool = False


SUITE_DATASETS: tuple[MainRolloutDataset, ...] = (
    MainRolloutDataset(
        key="livecodebench",
        display_name="LiveCodeBench",
        task_kind="livecodebench_codegen",
        dataset="livecodebench/code_generation_lite",
        split="test",
        release_version="release_v6",
        lm_style_override="HFChatTemplate",
        requires_livecodebench_repo=True,
    ),
    MainRolloutDataset(
        key="livecodebench_extra",
        display_name="LiveCodeBench-extra",
        task_kind="livecodebench_codegen",
        dataset="livecodebench/code_generation_lite",
        split="test",
        release_version="release_v6",
        lm_style_override="HFChatTemplate",
        requires_livecodebench_repo=True,
        requires_exclude_prompt_jsonl=True,
    ),
    MainRolloutDataset(
        key="taco_hard",
        display_name="TACO-hard",
        task_kind="taco_codegen",
        dataset="BAAI/TACO",
        dataset_config="ALL",
        split="train",
        question_field="question",
        starter_code_field="starter_code",
        record_id_field="url",
        metadata_fields=(
            "difficulty",
            "source",
            "url",
            "skill_types",
            "name",
            "tags",
            "raw_tags",
            "input_output",
            "time_limit",
            "memory_limit",
            "Expected Time Complexity",
            "Expected Auxiliary Space",
        ),
        row_filter={"field_in": {"difficulty": ["HARD", "VERY_HARD"]}},
        prompt_format="chat_template",
    ),
    MainRolloutDataset(
        key="math_level5",
        display_name="MATH level-5",
        task_kind="math_freeform",
        dataset="SuperSecureHuman/competition_math_hf_dataset",
        split="train",
        question_field="problem",
        answer_field="solution",
        metadata_fields=("level", "type"),
        row_filter={"field_in": {"level": ["Level 5"]}},
        prompt_format="chat_template",
    ),
)


def suite_dataset_keys() -> list[str]:
    return [dataset.key for dataset in SUITE_DATASETS]


def get_suite_dataset(key: str) -> MainRolloutDataset:
    for dataset in SUITE_DATASETS:
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
    lcb_extra_exclude_prompt_jsonl: str | None = None,
) -> dict[str, str]:
    dataset = get_suite_dataset(dataset_key)
    mode = thinking_mode.strip().lower()
    if mode not in {"on", "off"}:
        raise ValueError(f"Unsupported thinking mode {thinking_mode!r}; expected 'on' or 'off'.")
    if dataset.requires_livecodebench_repo and not livecodebench_repo:
        raise ValueError(f"{dataset.key} requires a LiveCodeBench repo path.")
    if dataset.requires_exclude_prompt_jsonl and not lcb_extra_exclude_prompt_jsonl:
        raise ValueError(
            f"{dataset.key} requires an exclusion archive via --lcb-extra-exclude-prompt-jsonl."
        )

    out_dir = os.path.join(output_root, dataset.key)
    out_path = os.path.join(out_dir, f"{dataset.key}__thinking_{mode}.json")
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
        "STATISTICS": suite_config.statistics,
        "MAX_SAMPLES": str(suite_config.max_samples),
        "THINKING_MODE": mode,
        "OUT": out_path,
    }
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
    if dataset.requires_exclude_prompt_jsonl:
        env["EXCLUDE_PROMPT_JSONL"] = str(lcb_extra_exclude_prompt_jsonl)
    return env
