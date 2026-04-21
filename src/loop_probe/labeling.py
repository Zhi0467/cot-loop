from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable

LABEL_TARGET_CHOICES = ("eventual_loop", "loop_by_horizon")
PROMPT_PROFILE_TARGET_CHOICES = (
    "s_tail",
    "p_loop",
    "p_cap",
    "mean_relative_length",
    "loop_budget_share",
    "majority_tail",
)


@dataclass(frozen=True)
class RolloutTerminalStats:
    length: int
    relative_length: float
    cap_hit: int
    loop_flag: int
    first_loop_prefix: int | None


@dataclass(frozen=True)
class LoopTriggerSpan:
    ngram_start_positions: tuple[int, ...]
    trigger_start: int
    trigger_end: int
    ngram_token_ids: tuple[int, ...]

    @property
    def first_loop_prefix(self) -> int:
        return self.trigger_end + 1


@dataclass
class _HashedNgramGroup:
    representative_start: int
    start_positions: list[int]


def first_ngram_loop_prefix_length(
    token_ids: Iterable[int],
    *,
    n: int = 30,
    k: int = 20,
) -> int | None:
    if n < 1:
        raise ValueError("n must be >= 1.")
    if k < 2:
        raise ValueError("k must be >= 2.")
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


def _same_ngram(
    token_ids: list[int],
    left_start: int,
    right_start: int,
    *,
    n: int,
) -> bool:
    for offset in range(n):
        if token_ids[left_start + offset] != token_ids[right_start + offset]:
            return False
    return True


def find_ngram_loop_trigger(
    token_ids: Iterable[int],
    *,
    n: int = 30,
    k: int = 20,
) -> LoopTriggerSpan | None:
    if n < 1:
        raise ValueError("n must be >= 1.")
    if k < 2:
        raise ValueError("k must be >= 2.")
    token_ids = list(token_ids)
    if len(token_ids) < n:
        return None

    base = 1000003
    mod = 1 << 64
    mask = mod - 1

    pow_n = pow(base, n, mod)
    h = 0
    for token_id in token_ids[:n]:
        h = (h * base + (token_id + 1)) & mask

    occurrences: dict[int, list[_HashedNgramGroup]] = {
        h: [_HashedNgramGroup(representative_start=0, start_positions=[0])]
    }

    for i in range(n, len(token_ids)):
        out_t = token_ids[i - n] + 1
        in_t = token_ids[i] + 1
        h = (h * base + in_t - (out_t * pow_n)) & mask
        start = i - n + 1
        bucket = occurrences.setdefault(h, [])
        for group in bucket:
            if not _same_ngram(
                token_ids,
                group.representative_start,
                start,
                n=n,
            ):
                continue
            group.start_positions.append(start)
            if len(group.start_positions) >= k:
                return LoopTriggerSpan(
                    ngram_start_positions=tuple(group.start_positions),
                    trigger_start=start,
                    trigger_end=start + n - 1,
                    ngram_token_ids=tuple(token_ids[start : start + n]),
                )
            break
        else:
            bucket.append(
                _HashedNgramGroup(
                    representative_start=start,
                    start_positions=[start],
                )
            )

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


def cap_hit_from_finish_reason(
    finish_reason: str | None,
    *,
    length: int,
    effective_max_tokens: int,
) -> int:
    if isinstance(finish_reason, str):
        normalized = finish_reason.strip().lower()
        if normalized and normalized != "unknown":
            return int(normalized == "length")
    return int(length >= effective_max_tokens)


def rollout_terminal_stats(
    token_ids: Iterable[int],
    *,
    effective_max_tokens: int,
    loop_n: int,
    loop_k: int,
    finish_reason: str | None = None,
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
        cap_hit=cap_hit_from_finish_reason(
            finish_reason,
            length=length,
            effective_max_tokens=effective_max_tokens,
        ),
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
    finish_reasons: list[str | None] | None = None,
) -> dict[str, object]:
    if not rollout_token_ids:
        raise ValueError("aggregate_prompt_profile requires at least one rollout.")
    if not 0.0 < tail_threshold <= 1.0:
        raise ValueError("tail_threshold must be in (0, 1].")
    if finish_reasons is not None and len(finish_reasons) != len(rollout_token_ids):
        raise ValueError("finish_reasons must align 1:1 with rollout_token_ids.")

    stats = [
        rollout_terminal_stats(
            token_ids,
            effective_max_tokens=effective_max_tokens,
            loop_n=loop_n,
            loop_k=loop_k,
            finish_reason=(
                finish_reasons[idx]
                if finish_reasons is not None
                else None
            ),
        )
        for idx, token_ids in enumerate(rollout_token_ids)
    ]
    num_rollouts = len(stats)
    lengths = [stat.length for stat in stats]
    relative_lengths = [stat.relative_length for stat in stats]
    cap_hits = [stat.cap_hit for stat in stats]
    loop_flags = [stat.loop_flag for stat in stats]
    first_loop_prefix_lengths = [stat.first_loop_prefix for stat in stats]
    tail_hits = [int(stat.relative_length >= tail_threshold) for stat in stats]
    tail_hit_count = sum(tail_hits)
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
        "tail_hit_count": int(tail_hit_count),
        "majority_tail": int(tail_hit_count > (num_rollouts / 2.0)),
        "mean_length": sum(lengths) / float(num_rollouts),
        "mean_relative_length": sum(relative_lengths) / float(num_rollouts),
        "loop_budget_share": sum(
            stat.loop_flag * stat.relative_length for stat in stats
        )
        / float(num_rollouts),
        "p_cap": sum(cap_hits) / float(num_rollouts),
        "p_loop": sum(loop_flags) / float(num_rollouts),
        "mu_log_rel": mu_log_rel,
        "tail_threshold": float(tail_threshold),
        "s_tail": sum(tail_hits) / float(num_rollouts),
    }


def profile_target_name(
    profile_target: str,
    *,
    tail_threshold: float,
) -> str:
    if profile_target == "s_tail":
        threshold_text = format(float(tail_threshold), "g")
        return f"s_{threshold_text}"
    if profile_target == "mean_relative_length":
        return "mean_relative_length"
    if profile_target == "loop_budget_share":
        return "loop_budget_share"
    if profile_target == "p_loop":
        return "p_loop"
    if profile_target == "p_cap":
        return "p_cap"
    if profile_target == "majority_tail":
        threshold_text = format(float(tail_threshold), "g")
        return f"majority_s_{threshold_text}"
    raise ValueError(
        f"Unknown prompt-profile target '{profile_target}'. "
        f"Valid: {PROMPT_PROFILE_TARGET_CHOICES}"
    )


def profile_target_value(
    profile: dict[str, object],
    *,
    profile_target: str,
) -> float:
    if profile_target == "s_tail":
        return float(profile["s_tail"])
    if profile_target == "mean_relative_length":
        return float(profile["mean_relative_length"])
    if profile_target == "loop_budget_share":
        return float(profile["loop_budget_share"])
    if profile_target == "p_loop":
        return float(profile["p_loop"])
    if profile_target == "p_cap":
        return float(profile["p_cap"])
    if profile_target == "majority_tail":
        return float(profile["majority_tail"])
    raise ValueError(
        f"Unknown prompt-profile target '{profile_target}'. "
        f"Valid: {PROMPT_PROFILE_TARGET_CHOICES}"
    )
