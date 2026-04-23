#!/usr/bin/env python3
"""Compute looping fraction and average response length using model tokenizers."""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

from transformers import AutoTokenizer
from utils import _math_verify, has_ngram_loop


def load_tokenizer(model_id: str, cache: Dict[str, object]):
    if model_id not in cache:
        cache[model_id] = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )
    return cache[model_id]


def load_answer_map(path: str) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "answer" not in row:
                raise ValueError("Answer map rows must include 'id' and 'answer'.")
            answers[str(row["id"])] = str(row["answer"])
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--data",
        default="",
        help="Optional dataset JSONL with 'id' and 'answer' for grading.",
    )
    args = parser.parse_args()

    tok_cache: Dict[str, object] = {}
    stats: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "loop": 0, "token_sum": 0, "correct": 0, "graded": 0}
    )
    answer_map: Optional[Dict[str, str]] = None
    if args.data:
        answer_map = load_answer_map(args.data)

    with open(args.generations, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            model_id = row["model_id"]
            temperature = float(row["temperature"])
            text = row.get("text", "")

            tok = load_tokenizer(model_id, tok_cache)
            token_ids = tok.encode(text, add_special_tokens=False)

            looping = has_ngram_loop(token_ids, n=args.n, k=args.k)
            key = (model_id, temperature)
            stats[key]["count"] += 1
            stats[key]["loop"] += 1 if looping else 0
            stats[key]["token_sum"] += len(token_ids)
            gold = row.get("answer")
            if gold is None and answer_map is not None:
                row_id = row.get("id")
                if row_id is not None:
                    gold = answer_map.get(str(row_id))
            if gold is not None:
                result = _math_verify(text, str(gold))
                if result is not None:
                    stats[key]["graded"] += 1
                    if result:
                        stats[key]["correct"] += 1

    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            [
                "model_id",
                "temperature",
                "num_samples",
                "loop_fraction",
                "avg_tokens",
                "num_correct",
                "accuracy",
            ]
        )
        for (model_id, temperature) in sorted(stats.keys()):
            s = stats[(model_id, temperature)]
            count = int(s["count"])
            loop = int(s["loop"])
            token_sum = float(s["token_sum"])
            correct = int(s.get("correct", 0))
            graded = int(s.get("graded", 0))
            loop_frac = (loop / count) if count else 0.0
            avg_tokens = (token_sum / count) if count else 0.0
            accuracy = (correct / graded) if graded else 0.0
            writer.writerow(
                [model_id, temperature, count, loop_frac, avg_tokens, correct, accuracy]
            )



if __name__ == "__main__":
    main()
