from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable

_ANSWER_LINE_PATTERN = re.compile(r"^Answer: ([A-Z])$")
_FINAL_ANSWER_LINE_PATTERN = re.compile(r"^(?:Final )?Answer: ([A-Z])$")
_ANSWER_FIELD_PATTERN = re.compile(
    r"(?i)(?:final\s+)?answer\s*[:：]\s*[*`\"']*\s*([A-Z])\b"
)
_THE_ANSWER_IS_PATTERN = re.compile(r"(?i)\b(?:the\s+)?answer\s+is\s+([A-Z])\b")
_BARE_LETTER_PATTERN = re.compile(r"^[\s>*`\"'(\[]*([A-Z])[\s<*`\"')\].:,-]*$")
_BOXED_LETTER_PATTERN = re.compile(
    r"\\boxed\{\s*(?:\\text\{)?\s*([A-Z])(?:\})?\s*\}",
)


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

    for candidate in reversed(lines[-max_lines:]):
        letter = _extract_structured_answer_letter(candidate, allowed)
        if letter is not None:
            return letter
    return None


def _extract_structured_answer_letter(candidate: str, allowed: set[str]) -> str | None:
    json_answer = _extract_json_answer_field(candidate, allowed)
    if json_answer is not None:
        return json_answer

    for pattern in (
        _ANSWER_LINE_PATTERN,
        _FINAL_ANSWER_LINE_PATTERN,
        _ANSWER_FIELD_PATTERN,
        _THE_ANSWER_IS_PATTERN,
        _BOXED_LETTER_PATTERN,
        _BARE_LETTER_PATTERN,
    ):
        match = pattern.search(candidate)
        if not match:
            continue
        letter = match.group(1).upper()
        if letter in allowed:
            return letter
    return None


def _extract_json_answer_field(candidate: str, allowed: set[str]) -> str | None:
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    fragment = candidate[start : end + 1]
    try:
        payload = json.loads(fragment)
    except json.JSONDecodeError:
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
