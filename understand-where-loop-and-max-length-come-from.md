author: Zhi
date: 2026-04-03

This document is a plan to understand where loop and max-length come from.

Clarification:

- here "degenerate rollouts" means the older rollout-statistics object already used in this repo: looping rollouts plus max-length-hit rollouts under the shared collector contract;
- the intended statistics are the ones we already collected for the Qwen3 common-policy bundle via `scripts/collect_model_stats.py`, `src/loop_probe/collector.py`, and `scripts/build_cross_dataset_rollout_report.py`;
- that older line already established the qualitative pattern we care about:
  - max-length hits usually sit inside looped rollouts;
  - looped rollouts are much longer than the average rollout;
  - looped and max-length-hit rollouts are much less accurate on the harder datasets;
- mean rollout length plus binary length-threshold labels are still useful prompt-level proxies, but this note is not about choosing among those proxies. This note is about where the degenerate-rollout regime enters the model family.

## Existing Reference Surface

Use the older rollout-statistics bundle as the reference object, not thread memory.

Current repo surfaces to reuse:

- `scripts/collect_model_stats.py`
- `src/loop_probe/collector.py`
- `scripts/build_cross_dataset_rollout_report.py`
- `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`
- `outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf`

If anyone needs the exact saved field definitions, the background appendix is `docs/understand-where-loop-and-max-length-come-from.md`. That appendix is not the main object here; it is just the definitions sheet behind the rollout-stat bundle.

The per-checkpoint bundle should carry forward the same metric family:

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

Keep the raw counts too:

- prompt count
- generated count
- graded count
- prompt-too-long count
- looped count
- max-length-hit count
- overlap counts between loop, max-length hit, and correctness

## Plan

The hypothesis is the base model should barely have any degenerate rollouts. The key question is whether this behavior is introduced during SFT, amplified during RLVR, or already present in the base model.

So we do this:

1. Take `Olmo-3-7B`, `Olmo-3-7B-Instruct-SFT`, and `Olmo-3-7B-Instruct` as the progression axis from base -> SFT -> RLVR.
2. For each checkpoint, collect the same rollout-stat bundle we already used for Qwen3.
   - reuse the same report shape and metric schema;
   - reuse the same benchmark family where possible: `MATH-500`, `AIME`, `GPQA`, `MMLU-Pro`, and capped `LiveCodeBench release_v6`;
   - keep one shared decode policy across the three checkpoints unless an OLMo-specific runtime constraint forces a documented deviation.
3. Replace the within-checkpoint overlap visualization with a Sankey plot when the question is subset structure:
   - correct vs wrong;
   - loop vs non-loop;
   - max-length-hit vs not.
4. Collect stats for all three checkpoints using the GPU node and build one JSON/CSV/PDF bundle per checkpoint in the current collector/report style.
5. Compare the checkpoints directly.
   - for the progression question, do not use Sankey;
   - use line plots for each metric across base -> SFT -> RLVR, broken out by dataset.
6. Draw conclusions on the hypothesis.
   - supported if the base checkpoint has little degenerate-rollout mass and SFT or RLVR shows a clear rise in loop rate, max-length-hit rate, or both across multiple datasets;
   - weakened if the base model already shows substantial degeneracy under the same contract, or if the progression is flat and noisy rather than stage-linked.

## Deliverables

We want three layers of output:

1. raw reproducible stats bundles per checkpoint
2. readable per-checkpoint report(s)
3. one progression artifact that answers the actual hypothesis directly

The final conclusion should stay on this object:

- where degenerate rollouts enter the model family;
- whether the progression points to base, SFT, RLVR, or no clean stage break.
