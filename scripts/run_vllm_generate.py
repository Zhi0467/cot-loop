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
import csv
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
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


def build_prompt(tokenizer, question: str, num_repetition: int) -> str:
    user_msg = (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    if num_repetition > 1:
        user_msg = user_msg * num_repetition
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_temps(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def add_repetition_suffix(path: str, num_repetition: int) -> str:
    if not path:
        return path
    root, ext = os.path.splitext(path)
    if re.search(r"\.rep\d+$", root):
        return path
    suffix = f"rep{num_repetition}"
    if ext:
        return f"{root}.{suffix}{ext}"
    return f"{path}.{suffix}"


def get_visible_devices() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return [v.strip() for v in visible.split(",") if v.strip() != ""]
    try:
        import torch  # type: ignore

        count = torch.cuda.device_count()
        return [str(i) for i in range(count)]
    except Exception:
        return []


def suppress_sem_unlink_errors() -> None:
    if os.environ.get("VLLM_SUPPRESS_SEM_UNLINK_ERRORS") != "1":
        return
    try:
        import multiprocessing.synchronize as mp_sync  # type: ignore
    except Exception:
        return
    if not hasattr(mp_sync, "_cleanup"):
        return
    if getattr(mp_sync, "_vllm_cleanup_patched", False):
        return

    orig_cleanup = mp_sync._cleanup

    def _cleanup(name: str) -> None:
        try:
            orig_cleanup(name)
        except FileNotFoundError:
            pass

    mp_sync._cleanup = _cleanup  # type: ignore[attr-defined]
    mp_sync._vllm_cleanup_patched = True

    try:
        import multiprocessing.resource_tracker as rt  # type: ignore
    except Exception:
        return
    if getattr(rt, "_vllm_rt_patched", False):
        return

    orig_register = rt.register
    orig_unregister = rt.unregister

    def _register(name: str, rtype: str) -> None:
        if rtype == "semaphore":
            return
        orig_register(name, rtype)

    def _unregister(name: str, rtype: str) -> None:
        if rtype == "semaphore":
            return
        orig_unregister(name, rtype)

    rt.register = _register  # type: ignore[assignment]
    rt.unregister = _unregister  # type: ignore[assignment]
    rt._vllm_rt_patched = True


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


def write_metrics(metrics: Dict[Tuple[str, float], Dict[str, float]], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as metrics_f:
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
            loop = float(s["loop"])
            token_sum = float(s["token_sum"])
            loop_frac = (loop / count) if count else 0.0
            avg_tokens = (token_sum / count) if count else 0.0
            writer.writerow([model_id, temperature, count, loop_frac, avg_tokens])


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

        for out in outputs:
            for sample in out.outputs:
                token_ids = getattr(sample, "token_ids", None)
                if not token_ids:
                    token_ids = tokenizer.encode(sample.text, add_special_tokens=False)

                key = (args.model_id, float(temperature))
                metrics[key]["count"] += 1
                metrics[key]["token_sum"] += len(token_ids)
                if has_ngram_loop(token_ids, n=args.loop_n, k=args.loop_k):
                    metrics[key]["loop"] += 1

    if args.metrics_out:
        write_metrics(metrics, args.metrics_out)

    return metrics


def merge_metric_dicts(
    shards: List[Dict[Tuple[str, float], Dict[str, float]]],
) -> Dict[Tuple[str, float], Dict[str, float]]:
    stats: Dict[Tuple[str, float], Dict[str, float]] = {}
    for shard in shards:
        for key, s in shard.items():
            out = stats.setdefault(key, {"count": 0, "loop": 0.0, "token_sum": 0.0})
            out["count"] += int(s["count"])
            out["loop"] += float(s["loop"])
            out["token_sum"] += float(s["token_sum"])
    return stats


def dp_worker(
    args: argparse.Namespace,
    device: str,
    metrics_queue: "mp.queues.SimpleQueue",
) -> None:
    suppress_sem_unlink_errors()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    metrics = run_generate(args)
    metrics_queue.put(metrics)


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
    metrics_queue: "mp.queues.SimpleQueue" = ctx.SimpleQueue()
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
