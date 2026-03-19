from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable

LABEL_TARGET_CHOICES = ("eventual_loop", "loop_by_horizon")


@dataclass(frozen=True)
class RolloutTerminalStats:
    length: int
    relative_length: float
    cap_hit: int
    loop_flag: int
    first_loop_prefix: int | None


def first_ngram_loop_prefix_length(
    token_ids: Iterable[int],
    *,
    n: int = 30,
    k: int = 20,
) -> int | None:
    token_ids = list(token_ids)
    if len(token_ids) < n:
        return None

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
        counts[h] = c
        if c >= k:
            return i + 1

    return None


def has_ngram_loop(token_ids: Iterable[int], n: int = 30, k: int = 20) -> bool:
    return first_ngram_loop_prefix_length(token_ids, n=n, k=k) is not None


def label_from_rollout(
    token_ids: Iterable[int],
    *,
    loop_n: int,
    loop_k: int,
    label_target: str = "eventual_loop",
    label_horizon: int | None = None,
) -> int:
    if label_target not in LABEL_TARGET_CHOICES:
        raise ValueError(
            f"Unknown label_target '{label_target}'. Valid: {LABEL_TARGET_CHOICES}"
        )

    first_loop_prefix = first_ngram_loop_prefix_length(
        token_ids,
        n=loop_n,
        k=loop_k,
    )
    if label_target == "eventual_loop":
        return int(first_loop_prefix is not None)

    if label_horizon is None or label_horizon < 1:
        raise ValueError(
            "label_horizon must be a positive integer when "
            "label_target='loop_by_horizon'."
        )
    return int(first_loop_prefix is not None and first_loop_prefix <= label_horizon)


def labels_from_rollouts(
    rollout_token_ids: list[list[int]],
    *,
    loop_n: int,
    loop_k: int,
    label_target: str = "eventual_loop",
    label_horizon: int | None = None,
) -> list[int]:
    return [
        label_from_rollout(
            token_ids,
            loop_n=loop_n,
            loop_k=loop_k,
            label_target=label_target,
            label_horizon=label_horizon,
        )
        for token_ids in rollout_token_ids
    ]


def rollout_terminal_stats(
    token_ids: Iterable[int],
    *,
    effective_max_tokens: int,
    loop_n: int,
    loop_k: int,
) -> RolloutTerminalStats:
    if effective_max_tokens < 1:
        raise ValueError("effective_max_tokens must be >= 1.")

    tokens = list(token_ids)
    length = len(tokens)
    first_loop_prefix = first_ngram_loop_prefix_length(
        tokens,
        n=loop_n,
        k=loop_k,
    )
    return RolloutTerminalStats(
        length=length,
        relative_length=float(length) / float(effective_max_tokens),
        cap_hit=int(length >= effective_max_tokens),
        loop_flag=int(first_loop_prefix is not None),
        first_loop_prefix=first_loop_prefix,
    )


def aggregate_prompt_profile(
    rollout_token_ids: list[list[int]],
    *,
    effective_max_tokens: int,
    loop_n: int,
    loop_k: int,
    tail_threshold: float,
) -> dict[str, object]:
    if not rollout_token_ids:
        raise ValueError("aggregate_prompt_profile requires at least one rollout.")
    if not 0.0 < tail_threshold <= 1.0:
        raise ValueError("tail_threshold must be in (0, 1].")

    stats = [
        rollout_terminal_stats(
            token_ids,
            effective_max_tokens=effective_max_tokens,
            loop_n=loop_n,
            loop_k=loop_k,
        )
        for token_ids in rollout_token_ids
    ]
    num_rollouts = len(stats)
    lengths = [stat.length for stat in stats]
    relative_lengths = [stat.relative_length for stat in stats]
    cap_hits = [stat.cap_hit for stat in stats]
    loop_flags = [stat.loop_flag for stat in stats]
    first_loop_prefix_lengths = [stat.first_loop_prefix for stat in stats]
    tail_hits = [int(stat.relative_length >= tail_threshold) for stat in stats]
    mu_log_rel = sum(math.log1p(stat.relative_length) for stat in stats) / float(
        num_rollouts
    )

    return {
        "num_rollouts": num_rollouts,
        "effective_max_tokens": int(effective_max_tokens),
        "lengths": lengths,
        "relative_lengths": relative_lengths,
        "cap_hits": cap_hits,
        "loop_flags": loop_flags,
        "first_loop_prefix_lengths": first_loop_prefix_lengths,
        "tail_hits": tail_hits,
        "mean_length": sum(lengths) / float(num_rollouts),
        "mean_relative_length": sum(relative_lengths) / float(num_rollouts),
        "p_cap": sum(cap_hits) / float(num_rollouts),
        "p_loop": sum(loop_flags) / float(num_rollouts),
        "mu_log_rel": mu_log_rel,
        "tail_threshold": float(tail_threshold),
        "s_tail": sum(tail_hits) / float(num_rollouts),
    }
