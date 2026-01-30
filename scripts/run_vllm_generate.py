#!/usr/bin/env python3
"""Generate AIME responses with vLLM for Figure 1 reproduction."""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, question: str) -> str:
    user_msg = (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_temps(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-rollouts", action="store_true")
    parser.add_argument("--delete-rollouts", action="store_true")
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--temps", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    if not args.no_rollouts and not args.out:
        raise SystemExit("Provide --out or set --no-rollouts.")
    if args.no_rollouts and args.delete_rollouts:
        raise SystemExit("--delete-rollouts requires rollouts to be written.")
    if not args.metrics_out and args.delete_rollouts:
        raise SystemExit("--delete-rollouts requires --metrics-out.")

    data = load_jsonl(args.data)
    if not data:
        raise SystemExit("Dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    gen_config = GenerationConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    top_k = gen_config.top_k if gen_config.top_k is not None else -1
    if top_k == 0:
        top_k = -1

    prompts = [build_prompt(tokenizer, row["question"]) for row in data]

    llm_kwargs = {
        "model": args.model_id,
        "tensor_parallel_size": args.tp,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)

    temps = parse_temps(args.temps)

    metrics: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "loop": 0, "token_sum": 0}
    )

    out_f = None
    if not args.no_rollouts:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")

    try:
        for temperature in temps:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=args.max_tokens,
                n=args.n,
                repetition_penalty=1.0,
                seed=args.seed,
            )

            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            dt = time.time() - t0

            for row, out in zip(data, outputs):
                for idx, sample in enumerate(out.outputs):
                    token_ids = getattr(sample, "token_ids", None)
                    if not token_ids:
                        token_ids = tokenizer.encode(sample.text, add_special_tokens=False)

                    if args.metrics_out:
                        key = (args.model_id, float(temperature))
                        metrics[key]["count"] += 1
                        metrics[key]["token_sum"] += len(token_ids)
                        if has_ngram_loop(token_ids, n=args.loop_n, k=args.loop_k):
                            metrics[key]["loop"] += 1

                    if out_f is not None:
                        record = {
                            "model_id": args.model_id,
                            "question_id": row.get("id"),
                            "temperature": temperature,
                            "sample_idx": idx,
                            "text": sample.text,
                            "finish_reason": sample.finish_reason,
                            "elapsed_s": dt,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if out_f is not None:
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8", newline="") as metrics_f:
            writer = csv.writer(metrics_f)
            writer.writerow(
                [
                    "model_id",
                    "temperature",
                    "num_samples",
                    "loop_fraction",
                    "avg_tokens",
                ]
            )
            for (model_id, temperature) in sorted(metrics.keys()):
                s = metrics[(model_id, temperature)]
                count = int(s["count"])
                loop = int(s["loop"])
                token_sum = float(s["token_sum"])
                loop_frac = (loop / count) if count else 0.0
                avg_tokens = (token_sum / count) if count else 0.0
                writer.writerow([model_id, temperature, count, loop_frac, avg_tokens])

    if args.delete_rollouts and args.out and os.path.exists(args.out):
        os.remove(args.out)


if __name__ == "__main__":
    main()
