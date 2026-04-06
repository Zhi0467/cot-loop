# OLMo Degeneration-Origin Audit - 2026-04-04

Update:
- the larger OLMo2 fallback follow-up now lives at `docs/olmo2-1b-fifty-prompt-rerun-2026-04-05.md`;
- use that newer note for stage conclusions on the OLMo2 ladder;
- keep this older note as the smaller `8`-prompt audit that motivated the rerun.

## Scope

This note closes the bounded OLMo audit that started from `understand-where-loop-and-max-length-come-from.md`.

Question:
- where do degenerate rollouts first enter the stage progression?
- are the fishy OLMo3 rows real model behavior, or evaluation-surface bugs?

Objects covered here:
- corrected OLMo3 `7B` bounded audit;
- bounded OLMo2 `1B` fallback ladder on the same collector family;
- one durable PDF artifact for collaborator review.

## What Was Actually Wrong

Two evaluation bugs were mixed into the first bounded OLMo3 read.

1. `RLVR / MMLU-Pro = 0 / 80` was a grader bug.
   - Saved RLVR completions often ended with relaxed forms like `{"answer": I}`.
   - The old parser rejected that even when the terminal answer letter was correct.
   - `_extract_json_answer_field` in `src/loop_probe/adapters/_common.py` now accepts that form.

2. `SFT / LiveCodeBench = 0 / 80` was initially on the wrong wrapper surface.
   - The old adapter silently defaulted non-Qwen models to a Qwen-specific `CodeQwenInstruct` style.
   - `src/loop_probe/adapters/livecodebench_codegen.py` now has an OLMo-native path:
     - instruct checkpoints use the tokenizer chat template to serialize the benchmark prompt into the model's own preferred text format;
     - raw/base checkpoints stay on the generic raw-string path;
     - code extraction tries fenced blocks first and falls back to raw code.

So the corrected rows below should replace the earlier fishy OLMo3 table. The older `0 / 80` `MMLU-Pro` RLVR row and the Qwen-wrapped OLMo `LiveCodeBench` rows should not be cited anymore.

## Corrected OLMo3 `7B` Read

### Base-versus-instruct pilot on `MATH-500`

The `7B` base checkpoint was too slow to finish the five-dataset sweep under the shared `n=10` bounded contract, so the clean base evidence is still the corrected `32`-prompt `MATH-500` pilot:

| Stage | Prompt surface | Correct | Looped | Max-length hits | Avg generation length |
| --- | --- | --- | --- | --- | --- |
| base | `raw` | `9 / 32` | `3 / 32` | `0 / 32` | `2375.7` |
| SFT | `chat_template` | `17 / 32` | `0 / 32` | `0 / 32` | `348.5` |
| RLVR | `chat_template` | `16 / 32` | `0 / 32` | `0 / 32` | `1146.0` |

This is the corrected object that retired the earlier all-raw comparison.

### Corrected instruct-only bounded bundle

Shared contract:
- `temperature=0.1`
- `top_p=0.95`
- `top_k=-1`
- `num_generations=10`
- `max_tokens=max_model_len=40960`
- first `8` prompts per dataset

Corrected rows:

| Stage | Dataset | Correct | Looped | Max-length hits | Avg generation length |
| --- | --- | --- | --- | --- | --- |
| SFT | `MATH-500` | `41 / 80` | `0 / 80` | `0 / 80` | `379.1` |
| SFT | `AIME` | `17 / 80` | `1 / 80` | `1 / 80` | `1136.1` |
| SFT | `GPQA` | `27 / 80` | `0 / 80` | `0 / 80` | `7.0` |
| SFT | `MMLU-Pro` | `34 / 80` | `1 / 80` | `1 / 80` | `589.9` |
| SFT | `LiveCodeBench` | `6 / 80` | `1 / 80` | `1 / 80` | `656.5` |
| RLVR | `MATH-500` | `50 / 80` | `0 / 80` | `0 / 80` | `555.4` |
| RLVR | `AIME` | `38 / 80` | `0 / 80` | `0 / 80` | `7293.2` |
| RLVR | `GPQA` | `43 / 80` | `0 / 80` | `0 / 80` | `291.1` |
| RLVR | `MMLU-Pro` | `27 / 80` | `0 / 80` | `0 / 80` | `6.15` |
| RLVR | `LiveCodeBench` | `22 / 80` | `0 / 80` | `0 / 80` | `1317.2` |

Read:
- the corrected OLMo3 bounded bundle does not support "RLVR introduces the degeneracy";
- the only remaining loop / cap mass in this bounded instruct comparison is on SFT, and even there it is light;
- `RLVR / MMLU-Pro` stays short, but it is no longer a zero-accuracy failure row once the relaxed parser bug is fixed.

## Bounded OLMo2 `1B` Fallback Ladder

Because the `7B` base stage was too slow for a clean full five-dataset sweep, I also ran the April 2025 OLMo2 `1B` ladder under the same bounded collector family. One mechanical correction was needed first: these checkpoints only support `max_model_len=4096`, so the first `40960` submissions were canceled and relaunched at `4096`.

Shared contract:
- `temperature=0.1`
- `top_p=0.95`
- `top_k=-1`
- `num_generations=10`
- `max_tokens=max_model_len=4096`
- first `8` prompts per dataset

Full ladder:

| Stage | Dataset | Correct | Looped | Max-length hits | Avg generation length |
| --- | --- | --- | --- | --- | --- |
| base | `MATH-500` | `3 / 80` | `27 / 80` | `29 / 80` | `1536.3` |
| base | `AIME` | `0 / 80` | `48 / 80` | `48 / 80` | `2456.1` |
| base | `GPQA` | `2 / 80` | `42 / 80` | `46 / 80` | `2265.7` |
| base | `MMLU-Pro` | `6 / 80` | `37 / 80` | `49 / 80` | `2491.1` |
| base | `LiveCodeBench` | `0 / 80` | `32 / 80` | `62 / 80` | `2134.2` |
| SFT | `MATH-500` | `8 / 80` | `7 / 80` | `9 / 80` | `891.4` |
| SFT | `AIME` | `0 / 80` | `11 / 80` | `11 / 80` | `1664.7` |
| SFT | `GPQA` | `20 / 80` | `11 / 80` | `11 / 80` | `785.2` |
| SFT | `MMLU-Pro` | `16 / 80` | `0 / 80` | `0 / 80` | `136.6` |
| SFT | `LiveCodeBench` | `0 / 80` | `15 / 80` | `23 / 80` | `1149.9` |
| RLVR1 | `MATH-500` | `14 / 80` | `1 / 80` | `1 / 80` | `538.0` |
| RLVR1 | `AIME` | `3 / 80` | `4 / 80` | `5 / 80` | `1212.6` |
| RLVR1 | `GPQA` | `9 / 80` | `1 / 80` | `1 / 80` | `406.0` |
| RLVR1 | `MMLU-Pro` | `0 / 80` | `0 / 80` | `0 / 80` | `65.0` |
| RLVR1 | `LiveCodeBench` | `0 / 80` | `0 / 80` | `0 / 80` | `412.9` |
| instruct | `MATH-500` | `22 / 80` | `1 / 80` | `1 / 80` | `614.4` |
| instruct | `AIME` | `0 / 80` | `3 / 80` | `5 / 80` | `1196.7` |
| instruct | `GPQA` | `10 / 80` | `5 / 80` | `10 / 80` | `908.4` |
| instruct | `MMLU-Pro` | `8 / 80` | `0 / 80` | `0 / 80` | `85.6` |
| instruct | `LiveCodeBench` | `0 / 80` | `0 / 80` | `0 / 80` | `510.8` |

Read:
- base is heavily degenerate on every dataset in this bounded ladder;
- SFT is still materially degenerate, especially on `AIME`, `GPQA`, and `LiveCodeBench`;
- `RLVR1` and final instruct are much cleaner on loop / max-length mass, even where accuracy is still weak.

That means the cheap full-ladder control already shows the stage shape the original note was trying to isolate: degeneration is heaviest in base, reduced but still present in SFT, and much smaller after RL-style post-training.

## April 4 Conclusion Before The 50-Prompt Rerun

What is now proved:
- the fishy OLMo3 rows were not just "weird but maybe real"; two of them were genuine evaluation-surface bugs, and the corrected rows are now on disk;
- bounded OLMo3 does not show RLVR adding loop / cap mass relative to SFT;
- bounded OLMo2 `1B` does show a clear stage progression from heavy base degeneracy to much cleaner `RLVR1` / instruct behavior.

What is still not proved:
- a full OLMo3 `7B` base -> SFT -> RLVR five-dataset progression under one bounded contract, because the `7B` base checkpoint is still too slow to finish that object honestly;
- any stronger claim that low accuracy itself disappears after RL-style post-training. Some `1B` rows remain low-accuracy while already being non-degenerate.

At that stage, before the later `50`-prompt OLMo2 rerun and the Qwen base raw control, the next honest step was:
- scale the bounded OLMo2 ladder to a larger prompt slice; or
- revive the OLMo3 base stage only on a smaller / more targeted object where the runtime is actually tractable.

## Canonical Artifacts

- Durable working note: `understand-where-loop-and-max-length-come-from.md`
- Report PDF: `outputs/olmo_degeneration_origin_audit_20260404/olmo_degeneration_origin_audit_20260404.pdf`
- OLMo3 corrected remote roots:
  - `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/mmlu_audit_temp0p1_gen10_ctx40960_topkneg1/`
  - `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/livecodebench_hfchat_temp0p1_gen10_ctx40960/`
- OLMo2 fallback remote root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/olmo2_1b_degeneration_origin_progression/bound8_temp0p1_gen10_ctx4096_topkneg1/`

## April 6 Follow-up

These bullets supersede the April 4 "next step" framing above. They are the reason this note is now historical rather than the main stage-conclusion surface.

Two later clarifications matter if this audit note is cited on its own:

- the older Qwen reference object it is being compared against is the repaired v2 bundle `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`, with the long-horizon contract `temperature=0.2`, `num_generations=10`, `max_tokens=81920`, and `max_model_len=40960`;
- the OLMo2 fallback here is therefore **not** a literal horizon match, because `OLMo-2-0425-1B*` only supports `4096`.

There is now also a same-family Qwen base control on the old rollout surface:

- model: `Qwen/Qwen3-1.7B-Base`
- sampler and dataset family: same v2 surface as above
- context limit: `32768`, not `40960`, because the base checkpoint advertises the shorter limit and vLLM rejects the instruct horizon without an unsafe override
- first finished long-horizon row: `MATH-500` base raw on that control is `243/500` correct, `19/500` looped, and `22/500` max-length-hit; compared with the old instruct reference (`3628/5000` correct, `147/5000` looped, `73/5000` max-length-hit), the base row is already worse on both loop rate and max-hit rate
- text-level pathology: Qwen base raw does degenerate on MCQ mainly by repeating the answer-format instruction tail (`Do not output anything else`, `Do not explain your answer`, `Do not output anything that is not JSON`) rather than by repeating OLMo-style math-derivation lines
- tiny `LiveCodeBench` style probe: the old LM-style mismatch is a real provenance caveat, but it is not an easy fix for base `LiveCodeBench`; on the same first `2` `release_v6` prompts, both `GenericBase` and `CodeQwenInstruct` looped to the cap with `pass@1 = 0`, with `GenericBase` drifting from plausible code into repeated problem text while `CodeQwenInstruct` collapsed into repeated `# YOUR CODE HERE`
- status caveat: this is still only the first finished same-family Qwen row; the remaining `AIME`, `GPQA`, `MMLU-Pro`, and `LiveCodeBench` base collectors are still live in the working note plus dated `roadmap.md` checkpoints, not packaged as a finished five-dataset companion note yet

And the OLMo2 ladder in this audit now has a dedicated figure bundle:

- `outputs/olmo2_1b_progression_bound50_20260406/olmo2_1b_progression_bound50.pdf`
- `outputs/olmo2_1b_progression_bound50_20260406/figures/progression_rates.png`
- `outputs/olmo2_1b_progression_bound50_20260406/figures/stage_overlap_sankey.png`
