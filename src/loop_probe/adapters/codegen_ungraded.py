from __future__ import annotations

from typing import Any

from ._common import (
    collect_row_metadata,
    load_rows_from_dataset,
    resolve_sample_id,
    row_matches_filter,
)
from ..types import DatasetSpec, SampleRecord


def load_samples(
    spec: DatasetSpec,
    *,
    question_field: str = "question",
    starter_code_field: str | None = "starter_code",
    record_id_field: str | None = None,
    metadata_fields: list[str] | None = None,
    row_filter: dict[str, dict[str, object]] | None = None,
) -> list[SampleRecord]:
    if spec.max_samples is not None and spec.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")

    rows = load_rows_from_dataset(
        spec.dataset,
        config=spec.config,
        split=spec.split,
    )
    if rows:
        first_row = rows[0]
        if question_field not in first_row:
            raise SystemExit(
                f"Dataset '{spec.dataset}' must include '{question_field}'; "
                f"found {sorted(first_row.keys())}."
            )

    samples: list[SampleRecord] = []
    for idx, row in enumerate(rows):
        if not row_matches_filter(row, row_filter):
            continue
        if question_field not in row:
            raise SystemExit(f"Row {idx} is missing '{question_field}'.")
        prompt = row[question_field]
        if prompt is None:
            continue
        sample_id = resolve_sample_id(row.get("_source_sample_id", idx), idx)
        metadata = collect_row_metadata(row, metadata_fields=metadata_fields) or {}
        if starter_code_field and starter_code_field in row and row[starter_code_field] is not None:
            metadata[starter_code_field] = str(row[starter_code_field])
        samples.append(
            SampleRecord(
                sample_id=sample_id,
                prompt=str(prompt),
                source_split=spec.split,
                prompt_style="codegen_ungraded",
                record_id=(
                    str(row.get(record_id_field))
                    if record_id_field and row.get(record_id_field) is not None
                    else None
                ),
                metadata=metadata or None,
            )
        )
        if spec.max_samples is not None and len(samples) >= spec.max_samples:
            break
    return samples


def build_codegen_prompt(
    tokenizer: Any | None,
    prompt: str,
    *,
    starter_code: str | None,
    prompt_format: str,
) -> str:
    prompt_text = prompt.rstrip()
    starter = (starter_code or "").strip()
    if starter:
        prompt_text = (
            f"{prompt_text}\n\n"
            "Starter code:\n"
            f"{starter}"
        )
    if prompt_format == "raw":
        return prompt_text
    if tokenizer is None:
        raise SystemExit("Tokenizer is required for chat_template codegen prompts.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
