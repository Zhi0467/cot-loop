from __future__ import annotations

import os
from typing import Any

from datasets import load_dataset

from ._common import load_local_rows, resolve_sample_id
from ..types import DatasetSpec, SampleRecord

_MATH_VERIFY_CONFIGS: tuple[list[Any], list[Any]] | None = None


def preflight() -> None:
    try:
        import math_verify  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "math_freeform requires math_verify. Run `uv sync` or install "
            "`math-verify[antlr4-13-2]`."
        ) from exc


def _build_math_verify_configs(
    latex_extraction_config,
    expr_extraction_config,
) -> tuple[list[Any], list[Any]]:
    try:
        pred_config = [
            latex_extraction_config(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                equations=False,
                boxed="all",
                boxed_match_priority=0,
            ),
            expr_extraction_config(),
        ]
    except TypeError:
        try:
            from latex2sympy2_extended.math_normalization import NormalizationConfig
        except Exception:
            pred_config = [latex_extraction_config(), expr_extraction_config()]
        else:
            pred_config = [
                latex_extraction_config(
                    boxed_match_priority=0,
                    normalization_config=NormalizationConfig(
                        basic_latex=True,
                        units=True,
                        malformed_operators=False,
                        nits=False,
                        boxed="all",
                        equations=False,
                    ),
                ),
                expr_extraction_config(),
            ]

    gold_config = [latex_extraction_config(), expr_extraction_config()]
    return pred_config, gold_config


def _math_verify(pred: str, gold: str) -> bool:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        parse,
        verify,
    )

    global _MATH_VERIFY_CONFIGS
    if _MATH_VERIFY_CONFIGS is None:
        _MATH_VERIFY_CONFIGS = _build_math_verify_configs(
            LatexExtractionConfig,
            ExprExtractionConfig,
        )

    pred_config, gold_config = _MATH_VERIFY_CONFIGS
    try:
        parsed_pred = parse(pred, extraction_config=pred_config)
        parsed_gold = parse(gold, extraction_config=gold_config)
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return False


def grade(response: str, gold_answer: str) -> bool:
    return _math_verify(response, gold_answer)


def load_samples(
    spec: DatasetSpec,
    *,
    question_field: str = "question",
    answer_field: str = "answer",
) -> list[tuple[SampleRecord, str]]:
    if spec.max_samples is not None and spec.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")

    if os.path.isfile(spec.dataset):
        rows = load_local_rows(spec.dataset)
        if spec.max_samples is not None:
            rows = rows[: spec.max_samples]
    else:
        ds = load_dataset(spec.dataset, spec.config, split=spec.split)
        if question_field not in ds.column_names or answer_field not in ds.column_names:
            raise SystemExit(
                f"Dataset '{spec.dataset}' must include '{question_field}' and "
                f"'{answer_field}' columns; found {list(ds.column_names)}."
            )
        if spec.max_samples is not None:
            ds = ds.select(range(min(len(ds), spec.max_samples)))
        rows = list(ds)

    samples: list[tuple[SampleRecord, str]] = []
    for idx, row in enumerate(rows):
        if question_field not in row or answer_field not in row:
            raise SystemExit(
                f"Row {idx} is missing '{question_field}' or '{answer_field}'."
            )
        prompt = row[question_field]
        answer = row[answer_field]
        if prompt is None or answer is None:
            continue
        sample_id = resolve_sample_id(row.get("_source_sample_id", idx), idx)
        samples.append(
            (
                SampleRecord(
                    sample_id=sample_id,
                    prompt=str(prompt),
                    source_split=spec.split,
                ),
                str(answer),
            )
        )
    return samples
