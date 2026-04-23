#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.adapters import multiple_choice_gpqa, multiple_choice_mmlupro
from loop_probe.adapters._common import extract_answer_letter_from_last_lines
from loop_probe.rollout import resolve_sampling_defaults
from loop_probe.types import DatasetSpec


_ANSWER_FIELD_RE = re.compile(
    r"(?i)(?:final\s+)?answer\s*[:：]\s*[*`\"']*\s*([A-Z])\b"
)
_THE_ANSWER_IS_RE = re.compile(r"(?i)\b(?:the\s+)?answer\s+is\s+([A-Z])\b")
_BARE_LETTER_RE = re.compile(r"^[\s>*`\"'(\[]*([A-Z])[\s<*`\"')\].:,-]*$")
_BOXED_LETTER_RE = re.compile(
    r"\\boxed\{\s*(?:\\text\{)?\s*([A-Z])(?:\})?\s*\}",
)


@dataclass(frozen=True)
class AuditItem:
    dataset_name: str
    sample_id: int
    prompt: str
    prompt_preview: str
    option_count: int
    gold_candidates: frozenset[str]
    strict_letters: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-kind",
        required=True,
        choices=("multiple_choice_gpqa", "multiple_choice_mmlupro"),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=10)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary-json", default="")
    parser.add_argument("--tail-lines", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    return parser.parse_args()


def _load_audit_items(args: argparse.Namespace, tokenizer) -> list[AuditItem]:
    spec = DatasetSpec(
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
    )
    items: list[AuditItem] = []
    if args.task_kind == "multiple_choice_gpqa":
        samples = multiple_choice_gpqa.load_and_shuffle(spec, args.seed)
        for record, options, gold_letter in samples:
            prompt = multiple_choice_gpqa.build_mcq_prompt(tokenizer, record.prompt, options)
            items.append(
                AuditItem(
                    dataset_name="GPQA",
                    sample_id=record.sample_id,
                    prompt=prompt,
                    prompt_preview=record.prompt[:200],
                    option_count=len(options),
                    gold_candidates=frozenset({gold_letter}),
                    strict_letters=tuple(multiple_choice_gpqa.GPQA_LETTERS[: len(options)]),
                )
            )
        return items

    samples = multiple_choice_mmlupro.load_samples(spec)
    for record, options, gold_answer, gold_index in samples:
        prompt = multiple_choice_mmlupro.build_mcq_prompt(tokenizer, record.prompt, options)
        valid_letters = tuple(multiple_choice_mmlupro.MMLUPRO_LETTERS[: len(options)])
        items.append(
            AuditItem(
                dataset_name="MMLU-Pro",
                sample_id=record.sample_id,
                prompt=prompt,
                prompt_preview=record.prompt[:200],
                option_count=len(options),
                gold_candidates=frozenset(
                    _mmlu_gold_candidates(
                        gold_answer=gold_answer,
                        gold_index=gold_index,
                        valid_letters=valid_letters,
                    )
                ),
                strict_letters=valid_letters,
            )
        )
    return items


def _mmlu_gold_candidates(
    *,
    gold_answer: str,
    gold_index: int | None,
    valid_letters: tuple[str, ...],
) -> set[str]:
    candidates: set[str] = set()
    gold_answer = gold_answer.strip().upper()
    if gold_answer:
        parsed = extract_answer_letter_from_last_lines(
            gold_answer,
            valid_letters,
            max_lines=1,
        )
        candidates.add(parsed or gold_answer)
    if gold_index is not None and 0 <= gold_index < len(valid_letters):
        candidates.add(valid_letters[gold_index])
    return candidates


def _tail_lines(text: str, limit: int) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()][-limit:]


def _extract_json_answer(line: str, allowed: set[str]) -> str | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    fragment = line[start : end + 1]
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


def _relaxed_prediction_from_tail(
    response: str,
    valid_letters: Iterable[str],
    *,
    tail_limit: int,
) -> tuple[str | None, str | None]:
    allowed = {str(letter).strip().upper() for letter in valid_letters if str(letter).strip()}
    lines = _tail_lines(response, tail_limit)
    for line in reversed(lines):
        json_answer = _extract_json_answer(line, allowed)
        if json_answer is not None:
            return json_answer, "json_answer"
        match = _ANSWER_FIELD_RE.search(line)
        if match:
            letter = match.group(1).upper()
            if letter in allowed:
                return letter, "answer_field"
        match = _THE_ANSWER_IS_RE.search(line)
        if match:
            letter = match.group(1).upper()
            if letter in allowed:
                return letter, "answer_is"
        match = _BOXED_LETTER_RE.search(line)
        if match:
            letter = match.group(1).upper()
            if letter in allowed:
                return letter, "boxed_letter"
        match = _BARE_LETTER_RE.match(line)
        if match:
            letter = match.group(1).upper()
            if letter in allowed:
                return letter, "bare_letter"
    return None, None


def main() -> None:
    args = _parse_args()
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary_json) if args.out_summary_json else out_jsonl.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    items = _load_audit_items(args, tokenizer)
    if not items:
        raise SystemExit("No items loaded for audit.")

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit("vLLM is required for this audit.") from exc

    top_p, top_k = resolve_sampling_defaults(args.model_id)
    llm_kwargs = {
        "model": args.model_id,
        "tensor_parallel_size": 1,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    llm = LLM(**llm_kwargs)

    strict_parsed = 0
    strict_correct = 0
    relaxed_parsed = 0
    relaxed_correct = 0
    finish_reason_counter: Counter[str] = Counter()
    relaxed_mode_counter: Counter[str] = Counter()
    last_line_counter: Counter[str] = Counter()
    rows_written = 0

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for item in items:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=args.max_tokens,
                n=args.num_generations,
                repetition_penalty=1.0,
                seed=args.seed + item.sample_id,
            )
            outputs = llm.generate([item.prompt], sampling_params)
            if len(outputs) != 1:
                raise RuntimeError(f"Expected 1 prompt output, got {len(outputs)}.")
            result = outputs[0]
            if len(result.outputs) != args.num_generations:
                raise RuntimeError(
                    f"Expected {args.num_generations} generations, got {len(result.outputs)}."
                )
            for generation_index, sample in enumerate(result.outputs):
                response_text = str(getattr(sample, "text", ""))
                token_ids = getattr(sample, "token_ids", None) or []
                strict_prediction = extract_answer_letter_from_last_lines(
                    response_text,
                    item.strict_letters,
                )
                strict_is_correct = (
                    strict_prediction is not None
                    and strict_prediction in item.gold_candidates
                )
                relaxed_prediction, relaxed_mode = _relaxed_prediction_from_tail(
                    response_text,
                    item.strict_letters,
                    tail_limit=args.tail_lines,
                )
                relaxed_is_correct = (
                    relaxed_prediction is not None
                    and relaxed_prediction in item.gold_candidates
                )
                tail_lines = _tail_lines(response_text, args.tail_lines)
                last_line = tail_lines[-1] if tail_lines else ""
                finish_reason = str(
                    getattr(sample, "finish_reason", None)
                    or getattr(result, "finish_reason", None)
                    or getattr(sample, "stop_reason", None)
                    or "unknown"
                )
                finish_reason_counter[finish_reason] += 1
                if strict_prediction is not None:
                    strict_parsed += 1
                if strict_is_correct:
                    strict_correct += 1
                if relaxed_prediction is not None:
                    relaxed_parsed += 1
                if relaxed_is_correct:
                    relaxed_correct += 1
                if relaxed_mode:
                    relaxed_mode_counter[relaxed_mode] += 1
                last_line_counter[last_line] += 1
                payload = {
                    "dataset_name": item.dataset_name,
                    "sample_id": item.sample_id,
                    "generation_index": generation_index,
                    "option_count": item.option_count,
                    "gold_candidates": sorted(item.gold_candidates),
                    "prompt_preview": item.prompt_preview,
                    "response_text": response_text,
                    "token_count": len(token_ids),
                    "finish_reason": finish_reason,
                    "strict_prediction": strict_prediction,
                    "strict_correct": strict_is_correct,
                    "relaxed_prediction": relaxed_prediction,
                    "relaxed_mode": relaxed_mode,
                    "relaxed_correct": relaxed_is_correct,
                    "tail_lines": tail_lines,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rows_written += 1
            print(
                f"[audit-mcq-terminal-format] processed {item.dataset_name} sample {item.sample_id}",
                flush=True,
            )

    summary = {
        "task_kind": args.task_kind,
        "dataset": args.dataset,
        "model_id": args.model_id,
        "temperature": args.temperature,
        "num_generations": args.num_generations,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "rows_written": rows_written,
        "strict_parse_fraction": strict_parsed / rows_written if rows_written else 0.0,
        "strict_correct_fraction": strict_correct / rows_written if rows_written else 0.0,
        "relaxed_parse_fraction": relaxed_parsed / rows_written if rows_written else 0.0,
        "relaxed_correct_fraction": relaxed_correct / rows_written if rows_written else 0.0,
        "finish_reason_counts": dict(finish_reason_counter.most_common()),
        "relaxed_mode_counts": dict(relaxed_mode_counter.most_common()),
        "top_last_lines": [
            {"line": line, "count": count}
            for line, count in last_line_counter.most_common(20)
        ],
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
