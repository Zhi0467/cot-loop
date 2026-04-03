author: Zhi
date: 2026-04-03

This document is a plan to understand where loop and max-length come from.

Clarification:

- we call both loop and max-length hits degenerate rollouts;
- our earlier statistics-collection work showed that max-length hits are usually a subset of looped rollouts;
- mean rollout length plus binary length-threshold labels can proxy this regime and are easier to predict at prefill time;
- but the research question here is not "which proxy is easiest to predict?" The research question is where these degenerate rollouts enter the model family in the first place.

Companion note:

- for the exact saved definitions of `loop`, prompt-profile `cap_hit` / `p_cap`, rollout-stat `max_length_hit`, `effective_max_tokens`, and `majority_s_0.5`, see `docs/understand-where-loop-and-max-length-come-from.md`.

## Objective

Test the hypothesis that the base model should have very few degenerate rollouts, and that this behavior is introduced mainly during SFT or RLVR.

Plainly:

- base -> learns semantics and syntax;
- SFT -> may start introducing format-following or answer-shape behaviors that lengthen or destabilize reasoning;
- RLVR -> may further amplify those behaviors if the reward and policy dynamics favor pathological long trajectories.

The job is to measure that progression, not assume it.

## Main Axis

Use the OLMo-3 progression:

1. `Olmo-3-7B`
2. `Olmo-3-7B-Instruct-SFT`
3. `Olmo-3-7B-Instruct`

Interpret that as:

- base
- SFT
- RLVR

## Collection Path

Reuse the current statistics-collection module rather than inventing a second path.

Current repo surfaces to reuse:

- `scripts/collect_model_stats.py`
- `scripts/build_cross_dataset_rollout_report.py`
- `outputs/qwen3_1p7b_cross_dataset_rollout_stats/`
- `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`

The requirement is that the OLMo collection should be as reproducible as the existing Qwen3 bundle.

That means:

- same metric schema;
- same output JSON/CSV shape where possible;
- one explicit report bundle per checkpoint;
- one progression summary built from those checkpoint bundles rather than from thread text.

## Dataset And Decode Contract

Default choice:

- use the same benchmark families as the repaired Qwen3 common-policy bundle so the new result is directly comparable:
  - `MATH-500`
  - `AIME`
  - `GPQA`
  - `MMLU-Pro`
  - capped `LiveCodeBench release_v6`

Default decode rule:

- keep one shared decode policy across all three checkpoints unless an OLMo-specific runtime constraint forces a documented deviation.

The point is not to search for each checkpoint's best decode. The point is to compare the progression under one fixed measurement contract.

## Statistics To Track

At minimum, carry forward the current rollout-stat bundle:

- `success_fraction`
- `loop_fraction`
- `max_length_hit_fraction`
- `loop_max_length_hit_fraction`
- `max_length_hit_loop_fraction`
- `max_length_hit_success_fraction`
- `loop_success_fraction`
- `avg_generation_length`
- `avg_loop_generation_length`
- `avg_first_loop_prefix_length`
- `avg_correct_generation_length`
- `avg_wrong_generation_length`
- `generation_length_variance`

Also keep the sanity/context fields:

- prompt count
- generated count
- graded count
- prompt-too-long count
- prompt length statistics if available

These metrics are enough to answer three separate questions:

1. does degenerate behavior exist at all?
2. does it rise from base -> SFT -> RLVR?
3. is the rise mostly "more looping", "more literal cap hits", or both?

## Visualization Plan

### Within One Checkpoint

Replace the current bar-style overlap view with a Sankey plot when the goal is to show subset and overlap structure such as:

- correct vs wrong
- loop vs non-loop
- max-length-hit vs not

Use Sankey when the question is:

- how much of `max_length` sits inside `loop`?
- how much of `loop` is still correct?
- how much failure mass moves between these buckets?

### Across Checkpoints

Do not use Sankey for the progression question.

For base -> SFT -> RLVR, use line plots.

Each metric above should be plotted across:

- checkpoint stage
- dataset

because the research question is about progression, not one checkpoint's internal partition.

## Concrete Run Plan

1. Materialize a clean OLMo stats workspace on the GPU node.
2. Run the existing stats collector for `Olmo-3-7B`.
3. Run the same collector for `Olmo-3-7B-Instruct-SFT`.
4. Run the same collector for `Olmo-3-7B-Instruct`.
5. Build one per-checkpoint report bundle in the current JSON/CSV/PDF style.
6. Add one progression summary artifact:
   - line plots for each metric across base -> SFT -> RLVR;
   - per-dataset tables;
   - one short conclusion on whether degeneracy is already present in base or introduced later.

## Decision Rule

The hypothesis is supported if:

- the base checkpoint has little or no degenerate rollout mass under the shared contract;
- SFT or RLVR shows a clear increase in loop rate, cap-hit rate, or both;
- the increase is visible across multiple datasets rather than being driven by one outlier only.

The hypothesis is weakened if:

- the base model already shows substantial degeneracy under the same measurement rule;
- or the progression is flat and noisy rather than stage-linked.

## Deliverables

We want three layers of output:

1. raw reproducible stats bundles per checkpoint
2. readable per-checkpoint report(s)
3. one progression artifact that answers the actual hypothesis directly

The final conclusion should stay on this object:

- where degenerate rollouts enter the model family
- whether the progression points to base, SFT, RLVR, or no clean stage break

not on the older prompt-profile target-selection question.
