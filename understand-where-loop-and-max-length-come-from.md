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

## Execution Update (2026-04-04)

The first execution pass changed one important assumption in this plan.

- A forced shared `raw` prompt contract across all three checkpoints is **not** the right object for OLMo.
- A direct `MATH-500` probe on samples `4` and `5` with `max_tokens=2048` showed:
  - base `raw` produces substantive long-form math completions (`2048` tokens with `finish_reason=length` on sample `4`, `1711` tokens with `finish_reason=stop` on sample `5`);
  - SFT `raw` and RLVR `raw` often collapse to an instruction echo or blank (`24` / `1` tokens for SFT, `31` / `1` tokens for RLVR on those same samples);
  - SFT `chat_template` and RLVR `chat_template` produce normal math solutions on the same prompts.
- So the corrected comparison object is:
  - base checkpoint on native `raw`;
  - SFT checkpoint on native `chat_template`;
  - RLVR checkpoint on native `chat_template`;
  - same collector and same decode policy otherwise.

The first bounded collection under that corrected object already landed:

- output root: `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/math8_cap4096_gen1_pilot/`
- dataset slice: first `8` `MATH-500` prompts
- decode settings: `temperature=0.2`, `num_generations=1`, `max_tokens=4096`

Observed rollout-stat bundle on that slice:

- base `raw`:
  - `4 / 8` correct
  - `1 / 8` looped
  - `0 / 8` max-length hits
  - `avg_generation_length = 2136.125`
- SFT `chat_template`:
  - `4 / 8` correct
  - `0 / 8` looped
  - `0 / 8` max-length hits
  - `avg_generation_length = 371.875`
- RLVR `chat_template`:
  - `5 / 8` correct
  - `0 / 8` looped
  - `0 / 8` max-length hits
  - `avg_generation_length = 495.0`

The larger bounded follow-up under that same corrected object also landed:

- output root: `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/math32_cap4096_gen1_pilot/`
- dataset slice: first `32` `MATH-500` prompts
- decode settings: unchanged (`temperature=0.2`, `num_generations=1`, `max_tokens=4096`)

Observed rollout-stat bundle on that larger slice:

- base `raw`:
  - `9 / 32` correct
  - `3 / 32` looped
  - `0 / 32` max-length hits
  - `avg_generation_length = 2375.65625`
- SFT `chat_template`:
  - `17 / 32` correct
  - `0 / 32` looped
  - `0 / 32` max-length hits
  - `avg_generation_length = 348.5`
- RLVR `chat_template`:
  - `16 / 32` correct
  - `0 / 32` looped
  - `0 / 32` max-length hits
  - `avg_generation_length = 1146.0`

Current interpretation:

- the larger corrected slice keeps the same qualitative result as the 8-prompt pilot: observed looping is still confined to the base checkpoint;
- on these bounded `MATH-500` slices, SFT and RLVR do not show loop or max-length-hit mass at all under the corrected prompt surfaces;
- this is now enough to reject the earlier all-raw comparison, but it is still not the final multi-dataset answer to the full note hypothesis;
- the next expansion should keep the corrected prompt-surface split fixed and scale the same rollout-stat object outward, rather than revisiting the invalid shared-raw contract.

## Fairness Correction (2026-04-04 05:11 UTC)

Wangzhi's follow-up exposed one more confound in the first bounded pilots.

- The prompt surface had been corrected, but the sampling contract was still too loose relative to the older Qwen3 rollout-stat bundle.
- Those earlier bounded pilots used:
  - `temperature=0.2`
  - `num_generations=1`
  - `max_tokens=4096`
  - `max_model_len=65536`
  - implicit `top_p` / `top_k` inherited from each checkpoint's `generation_config`
- That meant the comparison was no longer close to the Qwen3 common-policy bundle we were supposed to reuse.
- Worse, the inherited sampling defaults were not even shared across the OLMo stages:
  - base OLMo effectively ran with `top_p=1.0`, `top_k=-1`
  - instruct SFT / RLVR ran with `top_p=0.95`, `top_k=-1`

So the collector now supports explicit sampling overrides, and the corrected follow-up object is:

- same dataset slice and same collector;
- same loop detector;
- same sampling contract across checkpoints:
  - `temperature=0.2`
  - `top_p=0.95`
  - `top_k=-1`
  - `num_generations=10`
  - `max_model_len=40960`
- still the same native prompt-surface split:
  - base `raw`
  - SFT `chat_template`
  - RLVR `chat_template`

The first Qwen3-like bounded rerun with `top_k=20` was only a short diagnostic. Wangzhi then clarified that `top_k` should stay at its default `-1`, so the live canonical rerun is now under:

- `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/math8_qwen3like_gen10_ctx40960_topkneg1/`

Current state of that rerun:

- `2089` SFT finished in `00:01:12` with:
  - `41 / 80` correct
  - `0 / 80` looped
  - `0 / 80` max-length hits
  - `avg_generation_length = 379.8625`
- `2090` RLVR finished in `00:01:46` with:
  - `50 / 80` correct
  - `0 / 80` looped
  - `0 / 80` max-length hits
  - `avg_generation_length = 538.4125`
- `2088` base is still running and is again the only long pole

The superseded `top_k=20` diagnostic is still useful as a short runtime sanity check:

- SFT finished in `00:01:12` with `42 / 80` correct, `0 / 80` loops, `0 / 80` max-length hits, `avg_generation_length = 376.8625`
- RLVR finished in `00:01:55` with `50 / 80` correct, `0 / 80` loops, `0 / 80` max-length hits, `avg_generation_length = 562.7`
- base had to be canceled when the `top_k=-1` clarification arrived

So the current fairness rule is:

- do **not** force an identical wrapper when that wrapper is broken for one stage;
- do keep the collector, dataset slice, loop detector, and sampling contract fixed;
- treat prompt surface as a separate interface choice that must stay native unless a direct probe shows otherwise.

## Deliverables

We want three layers of output:

1. raw reproducible stats bundles per checkpoint
2. readable per-checkpoint report(s)
3. one progression artifact that answers the actual hypothesis directly

The final conclusion should stay on this object:

- where degenerate rollouts enter the model family;
- whether the progression points to base, SFT, RLVR, or no clean stage break.
