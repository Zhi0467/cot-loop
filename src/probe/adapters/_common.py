from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset

_ANSWER_LINE_PATTERN = re.compile(r"^Answer: ([A-Z])$")
_ANSWER_PREFIX_PATTERN = re.compile(r"^(?:final\s+)?answer\s*[:：]\s*", re.IGNORECASE)
_ANSWER_IS_PATTERN = re.compile(
    r"^(?:(?:so|thus|therefore|hence|overall|in conclusion|conclusion|finally|"
    r"my answer|i choose|choose|the correct answer|the final answer)\s*[,:-]\s*)*"
    r"(?:the\s+)?answer\s+is\s+([A-Z])\b",
    re.IGNORECASE,
)
_BARE_LETTER_PATTERN = re.compile(r"^([A-Z])(?:\b|[).:,-])", re.IGNORECASE)
_BOXED_LETTER_PATTERN = re.compile(
    r"\\boxed\{\s*(?:\\text\{)?\s*(?:answer\s*[:：]\s*)?([A-Z])(?:\})?\s*\}",
    re.IGNORECASE,
)
_JSONISH_ANSWER_PATTERN = re.compile(
    r'^\{\s*["\']answer["\']\s*:\s*["\']?\s*([A-Z])\s*["\']?\s*\}$',
    re.IGNORECASE,
)
_LEADING_DECORATION_PATTERN = re.compile(r"^[>*#\-\s]+")
_WRAPPER_ONLY_LINE_PATTERN = re.compile(r"^(?:[`~$*_>|#-]+\s*)+$")


def resolve_sample_id(value: object, default: int) -> int:
    try:
        sample_id = int(value)
    except Exception:
        sample_id = default
    if sample_id < 0:
        return default
    return sample_id


def load_local_rows(path: str) -> list[dict[str, object]]:
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_rows_from_dataset(
    dataset: str,
    *,
    config: str | None,
    split: str,
) -> list[dict[str, object]]:
    def _load_hf_parquet_rows() -> list[dict[str, object]]:
        base = f"hf://datasets/{dataset}/"
        if config:
            base = f"{base}{config}/"
        ds = load_dataset(
            "parquet",
            data_files={split: f"{base}{split}-*.parquet"},
            split=split,
        )
        return list(ds)

    lower = dataset.lower()
    if lower.endswith(".parquet"):
        ds = load_dataset(
            "parquet",
            data_files={split: dataset},
            split=split,
        )
        return list(ds)
    if lower.endswith(".arrow"):
        ds = load_dataset(
            "arrow",
            data_files={split: dataset},
            split=split,
        )
        return list(ds)
    if os.path.isfile(dataset):
        return load_local_rows(dataset)
    if dataset == "BAAI/TACO":
        return _load_hf_parquet_rows()
    try:
        ds = load_dataset(dataset, config, split=split)
        return list(ds)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        try:
            return _load_hf_parquet_rows()
        except Exception as parquet_exc:
            raise RuntimeError(
                f"Dataset '{dataset}' could not be loaded via the retired script path "
                "or the HF parquet fallback."
            ) from parquet_exc


def parse_row_filter_json(raw: str | None) -> dict[str, dict[str, object]] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --row-filter-json payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--row-filter-json must decode to a JSON object.")
    normalized: dict[str, dict[str, object]] = {}
    for key in ("field_in", "field_ge", "field_le"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise SystemExit(f"--row-filter-json.{key} must be an object.")
        normalized[key] = value
    return normalized or None


def row_matches_filter(
    row: dict[str, object],
    row_filter: dict[str, dict[str, object]] | None,
) -> bool:
    if not row_filter:
        return True

    field_in = row_filter.get("field_in", {})
    for field, allowed_values in field_in.items():
        if not isinstance(field, str) or not field:
            raise SystemExit("--row-filter-json.field_in keys must be non-empty strings.")
        if not isinstance(allowed_values, list) or not allowed_values:
            raise SystemExit(
                f"--row-filter-json.field_in[{field!r}] must be a non-empty list."
            )
        value = row.get(field)
        if value is None:
            return False
        if isinstance(value, list):
            value_texts = {str(item).strip() for item in value}
            allowed_texts = {str(item).strip() for item in allowed_values}
            if not (value_texts & allowed_texts):
                return False
        else:
            if str(value).strip() not in {str(item).strip() for item in allowed_values}:
                return False

    for key, comparator in (("field_ge", lambda actual, bound: actual >= bound), ("field_le", lambda actual, bound: actual <= bound)):
        fields = row_filter.get(key, {})
        for field, bound in fields.items():
            if not isinstance(field, str) or not field:
                raise SystemExit(f"--row-filter-json.{key} keys must be non-empty strings.")
            try:
                bound_value = float(bound)
            except Exception as exc:
                raise SystemExit(
                    f"--row-filter-json.{key}[{field!r}] must be numeric."
                ) from exc
            value = row.get(field)
            try:
                actual_value = float(value)
            except Exception:
                return False
            if not comparator(actual_value, bound_value):
                return False
    return True


def collect_row_metadata(
    row: dict[str, object],
    *,
    metadata_fields: Iterable[str] | None,
) -> dict[str, object] | None:
    if metadata_fields is None:
        return None
    metadata: dict[str, object] = {}
    for field in metadata_fields:
        key = str(field).strip()
        if not key:
            continue
        if key in row:
            metadata[key] = _jsonable(row[key])
    return metadata or None


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def extract_answer_letter_from_last_lines(
    response: str,
    valid_letters: Iterable[str],
    *,
    max_lines: int = 6,
) -> str | None:
    allowed = {
        str(letter).strip().upper()
        for letter in valid_letters
        if str(letter).strip()
    }
    if not allowed:
        raise ValueError("valid_letters must contain at least one non-empty letter.")

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    match = _ANSWER_LINE_PATTERN.match(candidate)
    if not match:
        return None
    letter = match.group(1).upper()
    if letter in allowed:
        return letter
    return None


def extract_structured_answer_letter_from_last_lines(
    response: str,
    valid_letters: Iterable[str],
    *,
    max_lines: int = 12,
) -> str | None:
    letter, _ = extract_structured_answer_letter_and_mode_from_last_lines(
        response,
        valid_letters,
        max_lines=max_lines,
    )
    return letter


def extract_json_answer_letter_from_last_lines(
    response: str,
    valid_letters: Iterable[str],
    *,
    max_lines: int = 12,
) -> str | None:
    allowed = _normalize_allowed_letters(valid_letters)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None

    for candidate in reversed(lines[-max_lines:]):
        if _is_wrapper_only_line(candidate):
            continue
        return _extract_json_answer_field(candidate, allowed)
    return None


def extract_structured_answer_letter_and_mode_from_last_lines(
    response: str,
    valid_letters: Iterable[str],
    *,
    max_lines: int = 12,
) -> tuple[str | None, str | None]:
    allowed = _normalize_allowed_letters(valid_letters)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None, None

    for candidate in reversed(lines[-max_lines:]):
        if _is_wrapper_only_line(candidate):
            continue
        return _extract_terminal_answer_letter(candidate, allowed)
    return None, None


def _normalize_allowed_letters(valid_letters: Iterable[str]) -> set[str]:
    allowed = {
        str(letter).strip().upper()
        for letter in valid_letters
        if str(letter).strip()
    }
    if not allowed:
        raise ValueError("valid_letters must contain at least one non-empty letter.")
    return allowed


def _strip_markdown_answer_wrappers(candidate: str) -> str:
    stripped = candidate.strip()
    stripped = _LEADING_DECORATION_PATTERN.sub("", stripped)
    return stripped.replace("**", "").replace("__", "").replace("`", "").strip()


def _is_wrapper_only_line(candidate: str) -> bool:
    stripped = candidate.strip()
    return not stripped or bool(_WRAPPER_ONLY_LINE_PATTERN.match(stripped))


def _extract_terminal_answer_letter(
    candidate: str,
    allowed: set[str],
) -> tuple[str | None, str | None]:
    json_answer = _extract_json_answer_field(candidate, allowed)
    if json_answer is not None:
        return json_answer, "json_answer"

    normalized = _strip_markdown_answer_wrappers(candidate)
    match = _BOXED_LETTER_PATTERN.search(normalized)
    if match:
        letter = match.group(1).upper()
        if letter in allowed:
            return letter, "boxed_letter"

    answer_payload = normalized
    for _ in range(3):
        prefix_match = _ANSWER_PREFIX_PATTERN.match(answer_payload)
        if not prefix_match:
            break
        answer_payload = answer_payload[prefix_match.end() :].strip()
    bare_match = _BARE_LETTER_PATTERN.match(answer_payload)
    if bare_match:
        letter = bare_match.group(1).upper()
        if letter in allowed:
            return letter, "answer_field"

    answer_is_match = _ANSWER_IS_PATTERN.match(normalized)
    if answer_is_match:
        letter = answer_is_match.group(1).upper()
        if letter in allowed:
            return letter, "answer_is"
    return None, None


def _extract_json_answer_field(candidate: str, allowed: set[str]) -> str | None:
    stripped = _strip_markdown_answer_wrappers(candidate)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    fragment = stripped[start : end + 1]
    try:
        payload = json.loads(fragment)
    except json.JSONDecodeError:
        relaxed_match = _JSONISH_ANSWER_PATTERN.fullmatch(fragment)
        if not relaxed_match:
            return None
        letter = relaxed_match.group(1).upper()
        if letter in allowed:
            return letter
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("answer")
    if not isinstance(value, str):
        return None
    letter = value.strip().upper()
    if letter in allowed:
        return letter
    return None
