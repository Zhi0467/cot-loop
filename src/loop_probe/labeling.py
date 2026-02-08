from collections.abc import Iterable


def has_ngram_loop(token_ids: Iterable[int], n: int = 30, k: int = 20) -> bool:
    token_ids = list(token_ids)
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


def labels_from_rollouts(
    rollout_token_ids: list[list[int]],
    *,
    loop_n: int,
    loop_k: int,
) -> list[int]:
    return [
        int(has_ngram_loop(token_ids, n=loop_n, k=loop_k))
        for token_ids in rollout_token_ids
    ]
