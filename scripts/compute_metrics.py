#!/usr/bin/env python3
"""Compute looping fraction and average response length using model tokenizers."""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Tuple

from transformers import AutoTokenizer


def has_ngram_loop(token_ids, n=30, k=20) -> bool:
    if len(token_ids) < n:
        return False

    base = 1000003
    mod = 1 << 64
    mask = mod - 1

    pow_n = pow(base, n, mod)
    h = 0
    for t in token_ids[:n]:
        h = (h * base + (t + 1)) & mask

    counts = {h: 1}
    for i in range(n, len(token_ids)):
        out_t = token_ids[i - n] + 1
        in_t = token_ids[i] + 1
        h = (h * base + in_t - (out_t * pow_n)) & mask
        c = counts.get(h, 0) + 1
        if c >= k:
            return True
        counts[h] = c

    return False


def load_tokenizer(model_id: str, cache: Dict[str, object]):
    if model_id not in cache:
        cache[model_id] = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )
    return cache[model_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    tok_cache: Dict[str, object] = {}
    stats: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "loop": 0, "token_sum": 0}
    )

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

    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            [
                "model_id",
                "temperature",
                "num_samples",
                "loop_fraction",
                "avg_tokens",
            ]
        )
        for (model_id, temperature) in sorted(stats.keys()):
            s = stats[(model_id, temperature)]
            count = int(s["count"])
            loop = int(s["loop"])
            token_sum = float(s["token_sum"])
            loop_frac = (loop / count) if count else 0.0
            avg_tokens = (token_sum / count) if count else 0.0
            writer.writerow([model_id, temperature, count, loop_frac, avg_tokens])



if __name__ == "__main__":
    main()
