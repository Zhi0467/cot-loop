#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON_PATH = os.path.join(ROOT, "src", "probe", "adapters", "_common.py")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args()


def _load_common_module():
    spec = importlib.util.spec_from_file_location("probe_mc_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load parser helpers from {COMMON_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_letters_for_row(row: dict[str, object]) -> tuple[str, ...]:
    option_count = int(row["option_count"])
    return tuple(chr(ord("A") + idx) for idx in range(option_count))


def _normalize_gold_candidates(row: dict[str, object]) -> set[str]:
    return {
        str(value).strip().upper()
        for value in row.get("gold_candidates", [])
        if str(value).strip()
    }


def _parser_stats_template() -> dict[str, int]:
    return {
        "parsed": 0,
        "correct": 0,
        "stop_parsed": 0,
        "stop_correct": 0,
        "length_parsed": 0,
        "length_correct": 0,
    }


def _append_example(
    bucket: list[dict[str, object]],
    *,
    row: dict[str, object],
    prediction: str | None,
    mode: str | None,
    max_examples: int,
) -> None:
    if len(bucket) >= max_examples:
        return
    bucket.append(
        {
            "sample_id": row.get("sample_id"),
            "generation_index": row.get("generation_index"),
            "finish_reason": row.get("finish_reason"),
            "gold_candidates": row.get("gold_candidates"),
            "prediction": prediction,
            "mode": mode,
            "tail_lines": row.get("tail_lines"),
        }
    )


def _finalize_prompt_metrics(grouped: dict[int, list[bool]]) -> dict[str, float]:
    prompt_count = len(grouped)
    if prompt_count == 0:
        return {
            "prompt_count": 0,
            "sample0_correct_fraction": 0.0,
            "prompt_any_correct_fraction": 0.0,
            "prompt_majority_correct_fraction": 0.0,
            "prompt_mean_rollout_accuracy": 0.0,
        }

    sample0_hits = 0
    any_hits = 0
    majority_hits = 0
    mean_sum = 0.0
    for sample_correctness in grouped.values():
        if not sample_correctness:
            continue
        if sample_correctness[0]:
            sample0_hits += 1
        if any(sample_correctness):
            any_hits += 1
        if sum(sample_correctness) > len(sample_correctness) / 2:
            majority_hits += 1
        mean_sum += sum(sample_correctness) / len(sample_correctness)
    return {
        "prompt_count": prompt_count,
        "sample0_correct_fraction": sample0_hits / prompt_count,
        "prompt_any_correct_fraction": any_hits / prompt_count,
        "prompt_majority_correct_fraction": majority_hits / prompt_count,
        "prompt_mean_rollout_accuracy": mean_sum / prompt_count,
    }


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {input_path}")

    common = _load_common_module()
    legacy_extract = common.extract_answer_letter_from_last_lines
    json_extract = common.extract_json_answer_letter_from_last_lines
    structured_extract = common.extract_structured_answer_letter_and_mode_from_last_lines

    parser_stats = {
        "legacy_strict": _parser_stats_template(),
        "json_only": _parser_stats_template(),
        "terminal_structured": _parser_stats_template(),
    }
    prompt_groups: dict[str, dict[int, list[bool]]] = {
        "legacy_strict": defaultdict(list),
        "json_only": defaultdict(list),
        "terminal_structured": defaultdict(list),
    }
    finish_reason_counts: Counter[str] = Counter()
    structured_mode_counts: Counter[str] = Counter()
    structured_only_over_json: list[dict[str, object]] = []
    structured_only_over_legacy: list[dict[str, object]] = []
    structured_length_correct_examples: list[dict[str, object]] = []
    rows = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            rows += 1
            response_text = str(row.get("response_text", ""))
            valid_letters = _valid_letters_for_row(row)
            gold_candidates = _normalize_gold_candidates(row)
            finish_reason = str(row.get("finish_reason", "unknown") or "unknown")
            finish_reason_counts[finish_reason] += 1
            sample_id = int(row.get("sample_id", rows - 1))

            predictions: dict[str, tuple[str | None, bool]] = {}

            legacy_prediction = legacy_extract(response_text, valid_letters)
            legacy_correct = (
                legacy_prediction is not None and legacy_prediction in gold_candidates
            )
            predictions["legacy_strict"] = (legacy_prediction, legacy_correct)

            json_prediction = json_extract(response_text, valid_letters)
            json_correct = json_prediction is not None and json_prediction in gold_candidates
            predictions["json_only"] = (json_prediction, json_correct)

            structured_prediction, structured_mode = structured_extract(
                response_text,
                valid_letters,
            )
            structured_correct = (
                structured_prediction is not None
                and structured_prediction in gold_candidates
            )
            predictions["terminal_structured"] = (
                structured_prediction,
                structured_correct,
            )
            if structured_mode is not None:
                structured_mode_counts[structured_mode] += 1

            for parser_name, (prediction, is_correct) in predictions.items():
                stats = parser_stats[parser_name]
                if prediction is not None:
                    stats["parsed"] += 1
                    if finish_reason == "stop":
                        stats["stop_parsed"] += 1
                    elif finish_reason == "length":
                        stats["length_parsed"] += 1
                if is_correct:
                    stats["correct"] += 1
                    if finish_reason == "stop":
                        stats["stop_correct"] += 1
                    elif finish_reason == "length":
                        stats["length_correct"] += 1
                prompt_groups[parser_name][sample_id].append(is_correct)

            if structured_correct and not json_correct:
                _append_example(
                    structured_only_over_json,
                    row=row,
                    prediction=structured_prediction,
                    mode=structured_mode,
                    max_examples=args.max_examples,
                )
            if structured_correct and not legacy_correct:
                _append_example(
                    structured_only_over_legacy,
                    row=row,
                    prediction=structured_prediction,
                    mode=structured_mode,
                    max_examples=args.max_examples,
                )
            if structured_correct and finish_reason == "length":
                _append_example(
                    structured_length_correct_examples,
                    row=row,
                    prediction=structured_prediction,
                    mode=structured_mode,
                    max_examples=args.max_examples,
                )

    summary = {
        "input_jsonl": str(input_path),
        "rows": rows,
        "finish_reason_counts": dict(finish_reason_counts),
        "structured_mode_counts": dict(structured_mode_counts),
        "legacy_strict": parser_stats["legacy_strict"]
        | _finalize_prompt_metrics(prompt_groups["legacy_strict"])
        | {
            "parse_fraction": parser_stats["legacy_strict"]["parsed"] / rows if rows else 0.0,
            "correct_fraction": parser_stats["legacy_strict"]["correct"] / rows if rows else 0.0,
        },
        "json_only": parser_stats["json_only"]
        | _finalize_prompt_metrics(prompt_groups["json_only"])
        | {
            "parse_fraction": parser_stats["json_only"]["parsed"] / rows if rows else 0.0,
            "correct_fraction": parser_stats["json_only"]["correct"] / rows if rows else 0.0,
        },
        "terminal_structured": parser_stats["terminal_structured"]
        | _finalize_prompt_metrics(prompt_groups["terminal_structured"])
        | {
            "parse_fraction": parser_stats["terminal_structured"]["parsed"] / rows if rows else 0.0,
            "correct_fraction": parser_stats["terminal_structured"]["correct"] / rows if rows else 0.0,
        },
        "structured_only_over_json_examples": structured_only_over_json,
        "structured_only_over_legacy_examples": structured_only_over_legacy,
        "structured_length_correct_examples": structured_length_correct_examples,
    }

    out_path = (
        Path(args.out_json)
        if args.out_json
        else input_path.with_suffix(input_path.suffix + ".eval.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
