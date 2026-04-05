# Understand Where `loop` And `max_length` Come From

Last updated: 2026-04-03 23:15 UTC

## Purpose

This note fixes one recurring ambiguity in the cot-loop repo:

- what exactly the saved `loop` signal means;
- what exactly the saved `max_length` / cap-hit signal means;
- where those signals are defined in code;
- why they overlap strongly without being the same object.

The object here is the current saved rollout and prompt-profile surface, not a new theory of why the model loops.

## Bottom Line

- `loop` is a token-level repeated-`n`-gram detector on one rollout.
- `max_length` is a terminal budget-hit event on one rollout, but the repo has two closely related cap surfaces:
  - the older rollout-stat path asks whether prompt plus generation hit `max_model_len`;
  - the prompt-profile path asks whether generation hit `effective_max_tokens = min(max_tokens, max_model_len - prompt_len)`.
- they are related, but they are not the same label:
  - cap hits are usually a subset of loops on the saved bundle;
  - many loops still terminate before the cap.
- prompt-level quantities such as `p_loop`, `p_cap`, `mean_relative_length`, and `majority_s_0.5` are all built from those rollout-level events, but they ask different questions.

So if a plot or table shows both `loop` and `max_length`, it is not duplicating the same signal twice. It is showing a broader failure event and a narrower terminal event.

## Exact Rollout-Level Definitions

### `loop`

The rollout-level loop bit is defined in `src/loop_probe/labeling.py`.

- `first_ngram_loop_prefix_length(token_ids, n=30, k=20)` scans the generated token IDs with a rolling hash.
- it marks a loop when some `30`-token `n`-gram has appeared at least `20` times in the same rollout.
- `has_ngram_loop(...)` is just the boolean version of that test.
- `rollout_terminal_stats(...)` stores:
  - `loop_flag = 1` if that repeated-`n`-gram condition fires;
  - `first_loop_prefix` = the earliest generated-token prefix length where the detector first fires.

Important:

- this is a token-pattern detector, not a semantic judgment;
- it does not require the rollout to hit the cap;
- it does not use `finish_reason`.

### `max_length` / cap-hit

The repo uses two closely related but not perfectly identical cap objects.

#### Prompt-profile path

This is the cap object used by the repeated-rollout prompt-profile datasets.

- it is defined in `src/loop_probe/labeling.py`;
- `cap_hit_from_finish_reason(...)` returns `1` when `finish_reason == "length"` if that metadata exists;
- if `finish_reason` is missing, it falls back to `length >= effective_max_tokens`.

So this prompt-profile cap object means:

- "this rollout exhausted the allowed generation budget for this prompt."

#### Rollout-stat path

This is the older `max_length_hit` object used by `scripts/collect_model_stats.py` and the cross-dataset rollout reports.

- `_hit_max_model_len(...)` returns `1` only when:
  - `finish_reason == "length"`, and
  - `prompt_len + token_count >= max_model_len`.

So this rollout-stat object means:

- "this rollout ran long enough that prompt plus generation hit the model context ceiling."

Those two cap objects often overlap, but they can differ when `max_tokens` binds before `max_model_len`.

### `effective_max_tokens`

This is the budget the rollout actually had available for generation.

In `scripts/build_probe_dataset.py`, for each prompt:

```python
effective_max_tokens = min(max_tokens, max_model_len - prompt_len)
```

So the effective budget is not always just the global `MAX_TOKENS` setting.

- if the prompt is short, the effective budget is usually `max_tokens`;
- if the prompt is long enough to eat into the context window, the effective budget becomes smaller.

That is why the repo keeps `effective_max_tokens` as a saved field instead of assuming one global ceiling for every prompt.

This is also why the prompt-profile `cap_hit` and the rollout-stat `max_length_hit` should not be treated as textually interchangeable without checking which path produced the artifact.

## Where These Fields Are Written

The repeated-rollout path in `scripts/build_probe_dataset.py` writes three relevant layers of state.

### Per-rollout archive

`diagnostics/prompt_rollout_archive.jsonl`

Each rollout row stores:

- `finish_reason`
- `length`
- `relative_length = length / effective_max_tokens`
- `cap_hit`
- `loop_flag`
- `tail_hit`
- `first_loop_prefix_length`

This is the canonical place to inspect the raw rollout-level events.

### Per-prompt prompt-profile rows

`diagnostics/train_prompt_profile.jsonl`
`diagnostics/test_prompt_profile.jsonl`

Each prompt row aggregates repeated rollouts into:

- `p_loop = E[1{loop}]`
- `p_cap = E[1{cap_hit}]`
- `mean_relative_length = E[L / E]`
- `majority_tail = 1[sum_r 1[L_r / E >= t] > n / 2]`

with `t = 0.5` for the current `majority_s_0.5` surface.

### Projection / visualization exports

The projection exporters reuse those same saved fields.

- `docs/prompt-profile-projection.md` describes the prompt-level projection path.
- `docs/prefill-activation-visualization.md` describes the earlier rollout/prompt visualization path.

So the visual panels are not inventing a separate notion of loop or max-length. They are reading the same saved rollout archive and prompt-profile aggregates.

## How The Prompt-Level Targets Relate

These names are easy to conflate, so here is the clean split.

### `p_loop`

- prompt-level loop frequency;
- fraction of repeated rollouts whose `loop_flag == 1`.

### `p_cap`

- prompt-level cap-hit frequency;
- fraction of repeated rollouts whose `cap_hit == 1`.

### `mean_relative_length`

- prompt-level average budget usage;
- mean of `length / effective_max_tokens` across repeated rollouts.

### `majority_s_0.5`

- prompt-level binary "often long" label;
- `1` if more than half of repeated rollouts satisfy `length / effective_max_tokens >= 0.5`.

This is the key distinction:

- `p_loop` is the direct loop-frequency target;
- `p_cap` is the direct hard-cap target;
- `mean_relative_length` is a dense budget-usage target;
- `majority_s_0.5` is not "loop" and not "cap hit." It is a coarse thresholded long-rollout label.

## Why `loop` And `max_length` Overlap Without Collapsing

The saved common-policy rollout bundle already shows the pattern clearly.

Source:

- `outputs/qwen3_1p7b_cross_dataset_rollout_stats/cross_dataset_rollout_summary.csv`

Important:

- in this table, `max_length_hit` is the rollout-stat object from `scripts/collect_model_stats.py`;
- it means prompt plus generation reached `max_model_len`;
- it is not literally the same column as prompt-profile `cap_hit`, even though both are cap-style terminal events.

| Dataset | Loop fraction | Max-length-hit fraction | Among loops, cap-hit fraction | Among cap hits, loop fraction |
| --- | ---: | ---: | ---: | ---: |
| `MATH-500` | `0.042` | `0.022` | `0.524` | `1.000` |
| `AIME` | `0.167` | `0.150` | `0.900` | `1.000` |
| `GPQA` | `0.313` | `0.202` | `0.645` | `1.000` |
| `MMLU-Pro` | `0.069` | `0.031` | `0.442` | `1.000` |
| `LiveCodeBench` | `0.253` | `0.209` | `0.820` | `0.995` |

Read:

- almost every cap hit is a loop on this saved bundle;
- but a substantial fraction of loops still stop before the cap, especially on `MATH-500`, `GPQA`, and `MMLU-Pro`.

So the clean mental model is:

- cap-hit is the narrower, terminal event;
- loop is the broader degeneracy event;
- prompt-profile `p_cap` is therefore usually a stricter and sparser target than `p_loop`.

This is also why older notes keep saying that direct `p_cap` is too narrow for the main objective: many bad looping rollouts never become literal cap hits.

## Why The Visual Structure Looks Different

The repo's visualization notes already give the same qualitative read.

From `docs/prefill-activation-visualization.md` on the saved GPQA pilot:

- correctness is the clearest broad structure in the 2D prompt plane;
- max-length risk is visible but fragmented across multiple prompt islands;
- loop risk is broader and more mixed than max-length risk.

That is consistent with the definitions above.

- `max_length` asks for a very specific terminal behavior;
- `loop` fires earlier and more often;
- therefore the `loop` surface can spread across prompts that are already degenerating but do not all run all the way to the ceiling.

## What This Note Does And Does Not Claim

What this note does fix:

- the exact code-path meaning of `loop`;
- the exact code-path meaning of `max_length` / cap hit;
- the exact meaning of `effective_max_tokens`;
- the relationship between rollout-level events and prompt-level targets.

What this note does not claim:

- why the model loops mechanistically;
- that `max_length` causes looping;
- that `loop` and `cap_hit` should be treated as interchangeable labels;
- that `majority_s_0.5` is a direct loop label.

## Practical Reading Rule

When reading the current cot-loop artifacts:

- if you care about the broader failure event, look at `loop_flag` / `p_loop`;
- if you care about literal budget exhaustion, look at `cap_hit` / `p_cap` / `max_length_hit`;
- if you care about how long the rollout tends to run even when it does not literally hit the ceiling, look at `mean_relative_length`;
- if you see `majority_s_0.5`, read it as "often long under this budget" rather than "often loops."
