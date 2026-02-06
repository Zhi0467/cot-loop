#!/usr/bin/env python3
"""Generate AIME responses with vLLM for Figure 1 reproduction.

Examples (sbatch)
  # QwQ-32B (TP=8, 8 GPUs)
  sbatch --export=ALL,MODEL_ID=Qwen/QwQ-32B,TP=8,DP=1,NUM_REPETITION=1,METRICS_OUT=outputs/qwq32b_metrics.rep1.csv \
    scripts/run_vllm_generate.sbatch

  # OpenThinker3-7B (DP=8, 8 GPUs)
  sbatch --export=ALL,MODEL_ID=open-thoughts/OpenThinker3-7B,TP=1,DP=8,NUM_REPETITION=1,METRICS_OUT=outputs/openthinker3_7b_metrics.rep1.csv \
    scripts/run_vllm_generate.sbatch

  # OpenThinker3-1.5B (DP=8, 8 GPUs)
  sbatch --export=ALL,MODEL_ID=open-thoughts/OpenThinker3-1.5B,TP=1,DP=8,NUM_REPETITION=1,METRICS_OUT=outputs/openthinker3_1p5b_metrics.rep1.csv \
    scripts/run_vllm_generate.sbatch

Smoke tests (small scale)
  # QwQ-32B (TP=8, 1 sample, 1 temp)
  sbatch --export=ALL,MODEL_ID=Qwen/QwQ-32B,TP=8,DP=1,NUM_REPETITION=1,TEMPS=0,N=2,MAX_TOKENS=256,METRICS_OUT=outputs/qwq32b_metrics_smoke.rep1.csv \
    scripts/run_vllm_generate.sbatch

  # OpenThinker3-7B (DP=8, 1 sample, 1 temp)
  sbatch --export=ALL,MODEL_ID=open-thoughts/OpenThinker3-7B,TP=1,DP=8,NUM_REPETITION=1,TEMPS=0,N=2,MAX_TOKENS=256,METRICS_OUT=outputs/openthinker3_7b_metrics_smoke.rep1.csv \
    scripts/run_vllm_generate.sbatch

  # OpenThinker3-1.5B (DP=8, 1 sample, 1 temp)
  sbatch --export=ALL,MODEL_ID=open-thoughts/OpenThinker3-1.5B,TP=1,DP=8,NUM_REPETITION=1,TEMPS=0,N=2,MAX_TOKENS=256,METRICS_OUT=outputs/openthinker3_1p5b_metrics_smoke.rep1.csv \
    scripts/run_vllm_generate.sbatch
"""

import argparse
import multiprocessing as mp
import os
import queue as queue_module
from collections import defaultdict
from typing import Dict, Tuple

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from utils import (
    _math_verify,
    add_repetition_suffix,
    build_prompt,
    get_visible_devices,
    has_ngram_loop,
    load_jsonl,
    merge_metric_dicts,
    parse_temps,
    suppress_sem_unlink_errors,
    write_metrics,
)


def run_generate(args: argparse.Namespace) -> Dict[Tuple[str, float], Dict[str, float]]:
    data = load_jsonl(args.data)
    if not data:
        raise SystemExit("Dataset is empty.")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard-idx must be in [0, num_shards).")
    if args.num_shards > 1:
        data = [row for i, row in enumerate(data) if i % args.num_shards == args.shard_idx]
        if not data:
            raise SystemExit("Shard is empty. Check --shard-idx/--num-shards.")
    if any("answer" not in row for row in data):
        raise SystemExit("Dataset rows must include 'answer' for math-verify grading.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    gen_config = GenerationConfig.from_pretrained(
        args.model_id,
    )

    top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    top_k = gen_config.top_k if gen_config.top_k is not None else -1
    if top_k == 0:
        top_k = -1

    prompts = [
        build_prompt(tokenizer, row["question"], args.num_repetition) for row in data
    ]
    answers = [row["answer"] for row in data]

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
        lambda: {"count": 0, "loop": 0, "token_sum": 0, "correct": 0, "graded": 0}
    )

    for temperature in temps:
        n = 1 if temperature == 0.0 else args.n
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=args.max_tokens,
            n=n,
            repetition_penalty=1.0,
            seed=args.seed,
        )

        outputs = llm.generate(prompts, sampling_params)
        if len(outputs) != len(answers):
            raise SystemExit(
                f"Expected {len(answers)} outputs, but got {len(outputs)}."
            )

        for idx, out in enumerate(outputs):
            gold = answers[idx]
            for sample in out.outputs:
                token_ids = getattr(sample, "token_ids", None)
                if not token_ids:
                    token_ids = tokenizer.encode(sample.text, add_special_tokens=False)

                key = (args.model_id, float(temperature))
                metrics[key]["count"] += 1
                metrics[key]["token_sum"] += len(token_ids)
                if has_ngram_loop(token_ids, n=args.loop_n, k=args.loop_k):
                    metrics[key]["loop"] += 1
                result = _math_verify(sample.text, gold)
                if result is None:
                    continue
                metrics[key]["graded"] += 1
                if result:
                    metrics[key]["correct"] += 1

    if args.metrics_out:
        write_metrics(metrics, args.metrics_out)

    return metrics


def dp_worker(
    args: argparse.Namespace,
    device: str,
    metrics_queue: "mp.queues.SimpleQueue",
) -> None:
    suppress_sem_unlink_errors()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    metrics = run_generate(args)
    metrics_queue.put(dict(metrics))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--temps", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--num-repetition", type=int, default=1)
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

    if args.dp < 1:
        raise SystemExit("--dp must be >= 1.")
    if args.num_repetition < 1:
        raise SystemExit("--num-repetition must be >= 1.")

    if args.metrics_out:
        args.metrics_out = add_repetition_suffix(args.metrics_out, args.num_repetition)

    if args.dp == 1:
        if args.metrics_out and os.path.exists(args.metrics_out):
            os.remove(args.metrics_out)
        run_generate(args)
        return

    if args.tp != 1:
        raise SystemExit("Data-parallel runs require --tp 1.")

    if args.metrics_out and os.path.exists(args.metrics_out):
        os.remove(args.metrics_out)

    devices = get_visible_devices()
    if len(devices) < args.dp:
        raise SystemExit(
            f"Requested dp={args.dp}, but only {len(devices)} visible GPU(s)."
        )

    ctx = mp.get_context("spawn")
    processes = []
    metrics_queue: "mp.queues.Queue" = ctx.Queue()
    for rank in range(args.dp):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.shard_idx = rank
        worker_args.num_shards = args.dp
        worker_args.metrics_out = ""

        p = ctx.Process(
            target=dp_worker,
            args=(worker_args, devices[rank], metrics_queue),
        )
        p.start()
        processes.append(p)

    failures = []
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failures.append(p.exitcode)

    if failures:
        raise SystemExit(f"DP worker(s) failed: {failures}")

    metrics_shards = []
    for _ in range(args.dp):
        try:
            metrics_shards.append(metrics_queue.get(timeout=5))
        except queue_module.Empty:
            raise SystemExit("Missing metrics from DP worker(s).")

    if args.metrics_out:
        merged = merge_metric_dicts(metrics_shards)
        write_metrics(merged, args.metrics_out)


if __name__ == "__main__":
    main()
