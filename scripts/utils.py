import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_MATH_VERIFY_CONFIGS: Optional[Tuple[List[Any], List[Any]]] = None


def _build_math_verify_configs(LatexExtractionConfig, ExprExtractionConfig) -> Tuple[
    List[Any], List[Any]
]:
    try:
        pred_config = [
            LatexExtractionConfig(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                equations=False,
                boxed="all",
                boxed_match_priority=0,
            ),
            ExprExtractionConfig(),
        ]
    except TypeError:
        try:
            from latex2sympy2_extended.math_normalization import NormalizationConfig
        except Exception:
            pred_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        else:
            pred_config = [
                LatexExtractionConfig(
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
                ExprExtractionConfig(),
            ]

    gold_config = [LatexExtractionConfig(), ExprExtractionConfig()]
    return pred_config, gold_config


def _math_verify(pred: str, gold: str) -> Optional[bool]:
    try:
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    except Exception:
        return None

    global _MATH_VERIFY_CONFIGS
    if _MATH_VERIFY_CONFIGS is None:
        _MATH_VERIFY_CONFIGS = _build_math_verify_configs(
            LatexExtractionConfig, ExprExtractionConfig
        )

    pred_config, gold_config = _MATH_VERIFY_CONFIGS
    try:
        parsed_pred = parse(pred, extraction_config=pred_config)
        parsed_gold = parse(gold, extraction_config=gold_config)
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return False


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
        "You must put your final answer within \\boxed{}."
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
                "num_correct",
                "accuracy",
            ]
        )
        for (model_id, temperature) in sorted(metrics.keys()):
            s = metrics[(model_id, temperature)]
            count = int(s["count"])
            loop = float(s["loop"])
            token_sum = float(s["token_sum"])
            correct = int(s.get("correct", 0))
            graded = int(s.get("graded", count))
            loop_frac = (loop / count) if count else 0.0
            avg_tokens = (token_sum / count) if count else 0.0
            accuracy = (correct / graded) if graded else 0.0
            writer.writerow(
                [model_id, temperature, count, loop_frac, avg_tokens, correct, accuracy]
            )


def merge_metric_dicts(
    shards: List[Dict[Tuple[str, float], Dict[str, float]]],
) -> Dict[Tuple[str, float], Dict[str, float]]:
    stats: Dict[Tuple[str, float], Dict[str, float]] = {}
    for shard in shards:
        for key, s in shard.items():
            out = stats.setdefault(
                key, {"count": 0, "loop": 0.0, "token_sum": 0.0, "correct": 0, "graded": 0}
            )
            out["count"] += int(s["count"])
            out["loop"] += float(s["loop"])
            out["token_sum"] += float(s["token_sum"])
            out["correct"] += int(s.get("correct", 0))
            out["graded"] += int(s.get("graded", 0))
    return stats
