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

## Temperature-0.1 Collection Queue (2026-04-04 05:39 UTC)

Wangzhi then made one more execution correction:

- stop arguing from `MATH-500` alone;
- keep collecting across datasets;
- use `temperature=0.1` instead of `0.2`.

So the stale `0.2` base rerun (`2088`) was canceled and replaced with one bounded cross-dataset queue under:

- output root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/bound8_temp0p1_gen10_ctx40960_topkneg1/`
- shared decode contract:
  - `temperature=0.1`
  - `top_p=0.95`
  - `top_k=-1`
  - `num_generations=10`
  - `max_tokens=40960`
  - `max_model_len=40960`
  - `max_samples=8` per dataset for this bounded collection pass
- prompt surfaces:
  - `base/raw`
  - `SFT/chat_template`
  - `RLVR/chat_template`
  - `LiveCodeBench` stays raw-string prompt generation for every stage because the benchmark prompt builder itself is raw and the collector disallows chat-template wrapping there

The live queue is now:

- base chain:
  - `2091` `MATH-500`
  - `2092` `AIME`
  - `2093` `GPQA`
  - `2094` `MMLU-Pro`
  - `2095` `LiveCodeBench`
- SFT chain:
  - `2096` `MATH-500`
  - `2097` `AIME`
  - `2098` `GPQA`
  - `2099` `MMLU-Pro`
  - `2100` `LiveCodeBench`
- RLVR chain:
  - `2101` `MATH-500`
  - `2102` `AIME`
  - `2103` `GPQA`
  - `2104` `MMLU-Pro`
  - `2105` `LiveCodeBench`

State at queue launch checkpoint:

- `2096` already finished `MATH-500` cleanly and advanced the SFT chain to `2097` (`AIME`);
- `2101` RLVR is still on the `MATH-500` leg;
- `2091` base is still on the `MATH-500` leg and remains the runtime long pole.

The runtime estimate from the canceled `0.2` base log is still useful operationally:

- the base `MATH-500` leg is roughly an `85`-`95` minute job on this bounded `8`-prompt contract;
- the instruct-stage legs are minute-scale and should clear the later dataset queue much sooner.

Wangzhi then narrowed the thread objective again:

- if base is the long pole, skip it for now;
- in this thread, finish collecting only the SFT and RLVR stats.

So the base chain (`2091`-`2095`) has now been canceled, and the live collection object for this thread is just:

- SFT:
  - `2097` `AIME`
  - `2098` `GPQA`
  - `2099` `MMLU-Pro`
  - `2100` `LiveCodeBench`
- RLVR:
  - `2102` `AIME`
  - `2103` `GPQA`
  - `2104` `MMLU-Pro`
  - `2105` `LiveCodeBench`

So the immediate deliverable for this thread is no longer a three-stage progression table. It is the instruct-stage rollout-stat bundle across the remaining datasets, with base deferred because it is currently the runtime bottleneck rather than the most decision-relevant object.

That instruct-stage bundle is now complete under the bounded `temperature=0.1` contract:

- SFT:
  - `MATH-500`: `41 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 379.125`
  - `AIME`: `17 / 80` correct, `1 / 80` looped, `1 / 80` max-length hits, `avg_generation_length = 1136.05`
  - `GPQA`: `27 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 7.0`
  - `MMLU-Pro`: `36 / 80` correct, `1 / 80` looped, `1 / 80` max-length hits, `avg_generation_length = 590.55`
  - `LiveCodeBench`: `0 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 155.4125`
- RLVR:
  - `MATH-500`: `50 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 555.4125`
  - `AIME`: `38 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 7293.2`
  - `GPQA`: `43 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 291.0625`
  - `MMLU-Pro`: `0 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 6.15`
  - `LiveCodeBench`: `8 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 766.0125`

So on this bounded instruct-only pass:

- RLVR stayed loop-free and max-length-hit-free on every collected dataset;
- the only observed degeneracy mass was on SFT, and even there it was light (`1 / 80` on `AIME`, `1 / 80` on `MMLU-Pro`);
- that does **not** yet make the instruct-side read scientifically clean across datasets.

Two follow-up cautions matter before treating this bundle as directly comparable to the earlier Qwen3 surface:

- `RLVR / MMLU-Pro` came back at `0 / 80` correct with `avg_generation_length = 6.15` under the same `chat_template` surface that was otherwise working on the other MCQ tasks. That shape looks more like a terminal-answer or grading mismatch than a stable capability read.
- `LiveCodeBench` is on a different interface from the MCQ tasks by construction: it stays on the benchmark's raw-string prompt builder for every stage, so its numbers should not be read as one more ordinary MCQ point.
- `LiveCodeBench` is also not obviously a total generation failure. The saved RLVR JSON includes native codegen metrics of `pass@1 = 0.10` and `pass@10 = 0.375` on this `8`-problem slice, and the paired `livecodebench__lcb_records.json` file contains substantive code outputs rather than blanks.
- the public `Olmo-3-7B-Instruct` model card is also a weak sanity check against reading these two suspect rows too literally: Ai2 reports nonzero coding and knowledge performance on that surface, so a total collapse on bounded `MMLU-Pro` or `LiveCodeBench` is not the default expectation. That is still only a sanity check, not a matched eval, because their table uses different benchmarks and settings.

So the current live question is no longer only "what do we do next about the deferred base stage?" It is also "do we need a terminal-format / grading sanity pass on the weird instruct-side datasets, especially `MMLU-Pro`, before making a stronger SFT-versus-RLVR claim?"

There is also a real smaller-model fallback, but it is not within the same OLMo 3 family.

- the public OLMo 3 instruct release currently exposes the progression only at `7B` and `32B`, so there is no smaller same-family `Olmo-3-*-Instruct-SFT -> Olmo-3-*-Instruct` ladder to substitute into this exact note;
- the smallest public progression with closely related post-training stages is instead the April 2025 OLMo 2 `1B` chain:
  - `OLMo-2-0425-1B`
  - `OLMo-2-0425-1B-SFT`
  - `OLMo-2-0425-1B-RLVR1`
  - `OLMo-2-0425-1B-Instruct`
- that would still change families and training details, so it should be treated as a fallback debug object after auditing the current OLMo 3 bounded outputs, not as a silent replacement for them.

## Local Audit Before The OLMo 2 Pivot (2026-04-04 07:25 UTC)

I checked the weird rows against the local collector / adapter code before trying to trust or rerun them blindly.

One concrete bug explains why the current `LiveCodeBench` row should not be treated as canonical for OLMo:

- our `livecodebench_codegen` adapter was silently defaulting every non-Qwen model family to `CodeQwenInstruct` LM style when no explicit override was supplied;
- the official LiveCodeBench repo does **not** recommend that fallback. Their README says new model families should be added explicitly in `lcb_runner/lm_styles.py` and `prompts/generation.py` before evaluation is treated as benchmark-comparable;
- so the bounded OLMo 3 `LiveCodeBench` rows were not on a model-native codegen surface. That means `SFT / LiveCodeBench = 0 / 80` is not strong evidence about where degeneration enters the model family. It is confounded by the benchmark wrapper.

I patched the adapter to fail fast on that case instead of silently reusing the Qwen codegen style for unrelated model families. Future non-Qwen `LiveCodeBench` runs now require an explicit `--lm-style-override` backed by actual LiveCodeBench prompt / extraction support.

`MMLU-Pro` is a different story. The local code audit does **not** show the same kind of easy explanation there:

- the current grader already accepts terminal JSON, boxed letters, bare letters, and `answer is X` style endings;
- the older parser audit we already saved in-repo (`outputs/mcq_parser_pilot_2026-03-16/mcq_parser_pilot_summary.md`) showed that on the JSON-contract `MMLU-Pro` pilot, strict JSON and the tightened terminal parser were identical;
- so `RLVR / MMLU-Pro = 0 / 80` is unlikely to be explained by "the model forgot the exact JSON wrapper" alone.

That row is still suspicious, but the remaining possibilities are narrower:

- a real model-side prompt/interface mismatch on that dataset;
- a strongly wrong-but-short response pattern on this tiny `8`-prompt slice;
- or some sample-level issue in the saved completions that only direct output inspection can settle.

That last step is currently blocked by infrastructure, not by missing local analysis:

- `ssh tianhaowang-gpu0` still times out during banner exchange, so I cannot inspect the saved remote `MMLU-Pro` completions yet;
- the same SSH failure also blocks launching the OLMo 2 `1B` fallback sweep right now.

So the current honest read is:

- `LiveCodeBench` has a confirmed wrapper bug and should be removed from any OLMo stage claim until we have a model-native LM style for that family;
- `MMLU-Pro` still needs direct saved-output inspection before I can call the `0 / 80` row either a true model behavior or a bad interface surface;
- OLMo 2 `1B` remains the right cheaper fallback once the node is reachable again, but I cannot run it from this workstation while the GPU node is rejecting SSH.

## Why The New Lengths Are Much Shorter Than Qwen3

The correct comparison object here is the saved Qwen3 common-policy bundle:

- `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`

That bundle is still the right reference surface, but it is larger and much more degenerate than this bounded OLMo instruct-only slice.

Two things are happening at once:

1. This OLMo pass is only the first `8` prompts per dataset, while the Qwen3 v2 bundle is the full saved rollout surface.
2. More importantly, the Qwen3 v2 means are inflated by much heavier loop / cap mass.

Concrete examples:

- `AIME`: Qwen3 v2 has `avg_generation_length = 20340.26`, `loop_fraction = 0.150`, `max_length_hit_fraction = 0.1267`; current OLMo gives SFT `1136.05` with `1 / 80` loop and RLVR `7293.2` with `0 / 80` loops.
- `GPQA`: Qwen3 v2 has `avg_generation_length = 9687.16`, `loop_fraction = 0.1641`, `max_length_hit_fraction = 0.0692`; current OLMo gives SFT `7.0` and RLVR `291.06`, both with `0 / 80` loops and `0 / 80` max-length hits.
- `MMLU-Pro`: Qwen3 v2 has `avg_generation_length = 3702.36`, `loop_fraction = 0.0456`, `max_length_hit_fraction = 0.0095`; current OLMo gives SFT `590.55` with `1 / 80` loop and RLVR `6.15` with `0 / 80`.

So the short lengths are not mainly coming from a silent prompt-contract change. The current GPQA / MMLU-Pro adapters still use the same JSON-answer surface as the saved Qwen3 v2 bundle. Much of the gap is plausibly behavioral: far less loop / cap mass on this bounded OLMo instruct slice, plus some prompts where the model simply emits the requested short JSON answer and stops. But `RLVR / MMLU-Pro = 0 / 80` with `avg_generation_length = 6.15` is still suspicious enough that the dataset-by-dataset semantic read should stay provisional until the terminal-answer surface is audited.

One automation footnote:

- I attempted to rebuild per-stage and instruct-only progression reports after the queue finished.
- The clean GPU-node checkout's default `python3` lacks `matplotlib`, so the report scripts fail at import time.
- The raw JSON bundle is complete, and this note is now the readable summary surface for the finished instruct-only pass.

## Deliverables

We want three layers of output:

1. raw reproducible stats bundles per checkpoint
2. readable per-checkpoint report(s)
3. one progression artifact that answers the actual hypothesis directly

The final conclusion should stay on this object:

- where degenerate rollouts enter the model family;
- whether the progression points to base, SFT, RLVR, or no clean stage break.

## Remote Audit Correction And OLMo 2 Follow-Through (2026-04-04 22:07 UTC)

This supersedes the earlier "SSH blocked / drop LiveCodeBench" state.

The GPU node is reachable again, so I finished the first direct audit on the suspicious OLMo 3 rows instead of reasoning only from aggregate shapes.

`MMLU-Pro` first:

- the weird `RLVR / MMLU-Pro = 0 / 80` row was a grader artifact, not a real zero-accuracy collapse;
- saved RLVR completions often end in relaxed JSON-like forms such as `{"answer": I}` that carry the correct terminal letter but are not strict JSON;
- our old parser rejected that exact form even when the answer letter matched, so I patched `_extract_json_answer_field` in `src/loop_probe/adapters/_common.py` to accept it;
- I then reran the bounded RLVR audit under the same contract and wrote the corrected row to:
  - `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/mmlu_audit_temp0p1_gen10_ctx40960_topkneg1/rlvr/mmlu_pro.json`
- corrected RLVR `MMLU-Pro` metrics on that same `8`-prompt / `80`-rollout object:
  - `27 / 80` correct
  - `0 / 80` looped
  - `0 / 80` max-length hits
  - `avg_generation_length = 6.15`
- so the old suspicious part of that row was grading, not degeneracy;
- the matching SFT audit is also finished now:
  - output: `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/mmlu_audit_temp0p1_gen10_ctx40960_topkneg1/sft/mmlu_pro.json`
  - `34 / 80` correct
  - `1 / 80` looped
  - `1 / 80` max-length hits
  - `avg_generation_length = 589.9375`

`LiveCodeBench` next:

- the earlier fail-fast guard was too narrow; OLMo does have a model-native evaluation path here;
- instruct-stage OLMo tokenizers expose a chat template, and LiveCodeBench only requires a raw string prompt plus code extraction, so the right fix is an explicit OLMo path rather than banning the benchmark;
- `src/loop_probe/adapters/livecodebench_codegen.py` now does three OLMo-specific things:
  - maps OLMo instruct checkpoints to a new `HFChatTemplate` style and raw/base checkpoints to `GenericBase`;
  - builds the prompt by wrapping LiveCodeBench's standard generic system message plus question template inside the model's own tokenizer chat template;
  - extracts code with a fenced-code parse first, then falls back to raw-code extraction so bare code completions are still captured;
- remote smoke confirmed that `allenai/Olmo-3-7B-Instruct` now serializes the prompt through the model-native chat surface and returns extractable code on that path;
- the bounded SFT rerun is already finished on that native path:
  - output: `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/livecodebench_hfchat_temp0p1_gen10_ctx40960/sft/livecodebench.json`
  - `6 / 80` correct
  - `1 / 80` looped
  - `1 / 80` max-length hits
  - `avg_generation_length = 656.5375`
- the first RLVR rerun did generate successfully enough to write the full records checkpoint, but the grading step exceeded the original `32G` host-memory reservation and was killed by Slurm;
- I resumed grading from that saved checkpoint as job `2210` with `64G` host memory, and that finished cleanly;
- final bounded RLVR native row:
  - output: `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/livecodebench_hfchat_temp0p1_gen10_ctx40960/rlvr/livecodebench.json`
  - `22 / 80` correct
  - `0 / 80` looped
  - `0 / 80` max-length hits
  - `avg_generation_length = 1317.15`

With the 7B audit unblocked, I also started the requested smaller fallback:

- released OLMo 2 progression:
  - `allenai/OLMo-2-0425-1B`
  - `allenai/OLMo-2-0425-1B-SFT`
  - `allenai/OLMo-2-0425-1B-RLVR1`
  - `allenai/OLMo-2-0425-1B-Instruct`
- one mechanical correction was needed first: the `1B` checkpoints only support `max_model_len=4096`, so the first `40960`-context submissions were canceled and relaunched at `4096`;
- active bounded output root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/olmo2_1b_degeneration_origin_progression/bound8_temp0p1_gen10_ctx4096_topkneg1/`

The fallback is already far enough along to show a stage-shaped degeneration pattern.

Completed rows on disk so far:

- base:
  - `MATH-500`: `3 / 80` correct, `27 / 80` looped, `29 / 80` max-length hits, `avg_generation_length = 1536.3375`
  - `AIME`: `0 / 80` correct, `48 / 80` looped, `48 / 80` max-length hits, `avg_generation_length = 2456.0875`
  - `GPQA`: `2 / 80` correct, `42 / 80` looped, `46 / 80` max-length hits, `avg_generation_length = 2265.7125`
  - `MMLU-Pro`: `6 / 80` correct, `37 / 80` looped, `49 / 80` max-length hits, `avg_generation_length = 2491.0875`
- SFT:
  - `MATH-500`: `8 / 80` correct, `7 / 80` looped, `9 / 80` max-length hits, `avg_generation_length = 891.3875`
  - `AIME`: `0 / 80` correct, `11 / 80` looped, `11 / 80` max-length hits, `avg_generation_length = 1664.6875`
  - `GPQA`: `20 / 80` correct, `11 / 80` looped, `11 / 80` max-length hits, `avg_generation_length = 785.2375`
  - `MMLU-Pro`: `16 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 136.6125`
- RLVR1:
  - `MATH-500`: `14 / 80` correct, `1 / 80` looped, `1 / 80` max-length hits, `avg_generation_length = 538.025`
- instruct:
  - `MATH-500`: `22 / 80` correct, `1 / 80` looped, `1 / 80` max-length hits, `avg_generation_length = 614.4`
  - `AIME`: `0 / 80` correct, `3 / 80` looped, `5 / 80` max-length hits, `avg_generation_length = 1196.7125`
  - `GPQA`: `10 / 80` correct, `5 / 80` looped, `10 / 80` max-length hits, `avg_generation_length = 908.4375`
  - `MMLU-Pro`: `8 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 85.575`

So the cheaper OLMo 2 fallback is already producing the kind of stage progression this note is trying to isolate:

- strong degeneracy in base;
- reduced but still present loop / cap mass in SFT;
- much smaller loop / cap mass in `RLVR1` / instruct on the first completed `MATH-500` leg.

The bounded OLMo 2 `1B` ladder is now complete on the same `8`-prompt / `80`-rollout contract:

- base:
  - `LiveCodeBench`: `0 / 80` correct, `32 / 80` looped, `62 / 80` max-length hits, `avg_generation_length = 2134.2125`
- SFT:
  - `LiveCodeBench`: `0 / 80` correct, `15 / 80` looped, `23 / 80` max-length hits, `avg_generation_length = 1149.9`
- RLVR1:
  - `AIME`: `3 / 80` correct, `4 / 80` looped, `5 / 80` max-length hits, `avg_generation_length = 1212.6`
  - `GPQA`: `9 / 80` correct, `1 / 80` looped, `1 / 80` max-length hits, `avg_generation_length = 406.0125`
  - `MMLU-Pro`: `0 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 64.975`
  - `LiveCodeBench`: `0 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 412.8875`
- instruct:
  - `LiveCodeBench`: `0 / 80` correct, `0 / 80` looped, `0 / 80` max-length hits, `avg_generation_length = 510.7875`

So the full fallback read is sharper than the broken original OLMo 3 bundle:

- loop / cap mass is heaviest in base;
- SFT still carries substantial degeneracy on several datasets, including `LiveCodeBench`;
- `RLVR1` and instruct largely remove loop / cap mass on this bounded object, even though some datasets are still simply low-accuracy rather than degenerate.

The collaborator-facing summary surface for this whole arc now lives at:

- `docs/olmo-degeneration-origin-audit-2026-04-04.md`
- `outputs/olmo_degeneration_origin_audit_20260404/olmo_degeneration_origin_audit_20260404.pdf`
