# Prompt-Profile Length Mechanism Note

Last updated: 2026-04-09 02:39 UTC

## Question

Why can prompt length predict completion length at all on the natural `mean_relative_length` regression surface, if the baseline really is just prompt length and there is no implementation bug?

## Short Answer

Because on this fixed setup, longer prompts are often the prompts that already contain more work.

- In `AIME` and `MATH-500`, longer prompts usually mean more mathematical setup, more notation, more conditions, or more cases to track.
- In `LiveCodeBench`, longer prompts usually mean a longer programming spec with more edge cases or more formatting requirements.
- In `MMLU-Pro`, length still carries some signal, but it is weaker and less clean than on the math or code datasets.
- `GPQA` is the important exception: raw length is weak there, and prompt structure matters more than raw length.

So prompt length is not acting like a magical cause. It is acting like a rough visible summary of how much answer-work the prompt is likely to demand from this fixed model and decode policy.

## What This Audit Ran

This note is based on a new audit over the finished natural regression rerun:

- source regression root:
  - `outputs/weeks/2026-W14/prompt_profile_natural_regression_rerun_20260405/` on the saved remote archive
- new audit bundle:
  - `outputs/weeks/2026-W15/prompt_profile_length_mechanism_20260409/`
- GPU execution:
  - `slurm/mechanism_analysis/run_prompt_profile_length_mechanism_audit.sbatch`
  - `2` GPUs on `tianhaowang-gpu0`
  - copied local log: `outputs/weeks/2026-W15/prompt_profile_length_mechanism_20260409/logs/prompt-profile-length-mech-2372.out`
- exact object:
  - target `mean_relative_length`
  - natural prompt-disjoint train/test split
  - frozen `best_loss` checkpoints for `last_layer` and `ensemble`

The audit compared:

1. raw prompt length only
2. a linear prompt-shape regressor using:
   - `prompt_token_count`
   - `log_token_length`
   - `char_length`
   - `newline_count`
   - `digit_count`
   - `dollar_count`
   - `choice_count`
3. a nonlinear prompt-shape tree on the same prompt features
4. the saved activation probes (`last_layer`, `ensemble`)

## Cross-Dataset Read

### Mean test metrics

| Model | Mean `top_20p_capture` | Mean `RMSE` | Mean `Spearman` |
| --- | ---: | ---: | ---: |
| prompt length | `0.261` | `0.180` | `0.410` |
| prompt-shape linear | `0.264` | `0.174` | `0.452` |
| prompt-shape tree | `0.288` | `0.189` | `0.392` |
| activation `last_layer` | `0.257` | `0.179` | `0.423` |
| activation `ensemble` | `0.304` | `0.177` | `0.586` |

Main read:

- raw prompt length really does carry held-out signal on this surface;
- adding a few cheap prompt-shape features improves the prompt-only baseline a bit, which means the old length-only row was compressing real prompt structure;
- the activation `ensemble` is still the best screening score overall;
- but the reason prompt length works is no longer mysterious: a lot of the signal is already visible on the prompt surface.
- Athena's plain-English read agrees with the same local picture:
  - longer prompts are often just the prompts with more work packed into them;
  - that is a real pattern on `AIME`, `MATH-500`, `MMLU-Pro`, and much of `LiveCodeBench`;
  - `GPQA` is the important place where raw length itself mostly stops working.

## Dataset By Dataset

### `AIME`

This is the cleanest “longer prompt, longer answer” case.

- prompt-length test correlation:
  - Pearson `0.456`
  - Spearman `0.676`
- prompt-length quartiles:
  - `71 tokens -> 0.422`
  - `96 tokens -> 0.674`
  - `130 tokens -> 0.726`
  - `336 tokens -> 0.765`
- the same quartiles also become much more symbolic:
  - mean `dollar_count` rises from `8.5` to `37.3`

Plain reading: the long `AIME` prompts are often the denser TeX-heavy olympiad problems. Those prompts naturally make this model write longer derivations, so prompt length already works as a good rough clue.

### `MATH-500`

This is similar to `AIME`, but milder.

- prompt-length test correlation:
  - Pearson `0.346`
  - Spearman `0.461`
- prompt-length quartiles:
  - `43 tokens -> 0.125`
  - `64 tokens -> 0.214`
  - `91 tokens -> 0.195`
  - `196 tokens -> 0.328`
- the longest quartile is also visibly more structured:
  - mean `newline_count` rises from `5.0` to `13.2`
  - mean `dollar_count` rises from `2.6` to `11.0`

Plain reading: longer `MATH-500` prompts usually carry more setup, more notation, or more separate conditions, so the model tends to write more before it is done.

### `LiveCodeBench`

Prompt length works here too, but only as a rough proxy.

- prompt-length test correlation:
  - Pearson `0.332`
  - Spearman `0.405`
- prompt-length quartiles:
  - `351 tokens -> 0.236`
  - `467 tokens -> 0.390`
  - `591 tokens -> 0.547`
  - `853 tokens -> 0.462`
- prompt size tracks much longer specs:
  - mean `newline_count` rises from `52.6` to `84.2`
- activation ensemble still wins clearly:
  - prompt length `top_20p_capture = 0.248`
  - activation ensemble `top_20p_capture = 0.345`

Plain reading: longer coding prompts usually mean longer specs and more cases to satisfy, so prompt length points in the right direction. But it is not enough by itself, because two code tasks with similar length can still differ a lot in what they require.

### `MMLU-Pro`

Prompt length has some signal, but this is not a simple or dominant length story.

- prompt-length test correlation:
  - Pearson `0.332`
  - Spearman `0.393`
- prompt-length quartiles:
  - `163 tokens -> 0.075`
  - `208 tokens -> 0.095`
  - `237 tokens -> 0.137`
  - `395 tokens -> 0.183`
- activation ensemble still helps on screening:
  - prompt length `top_20p_capture = 0.292`
  - activation ensemble `top_20p_capture = 0.364`

Plain reading: once the multiple-choice scaffold is mostly fixed, prompt length still tracks how much detail is in the stem, but it is only part of the story.

### `GPQA`

This is the important exception, and it is the clearest proof that raw length is not the right explanation by itself.

- prompt-length test correlation:
  - Pearson `-0.069`
  - Spearman `0.116`
- prompt-length quartiles:
  - `153 tokens -> 0.197`
  - `207 tokens -> 0.259`
  - `248 tokens -> 0.377`
  - `407 tokens -> 0.218`

The longest prompts are **not** the riskiest prompts. The medium-long prompts are.

The prompt-shape models are the useful clue here:

- prompt length `top_20p_capture = 0.168`
- prompt-shape linear `top_20p_capture = 0.184`
- prompt-shape tree `top_20p_capture = 0.284`
- activation ensemble `top_20p_capture = 0.240`

Plain reading: in `GPQA`, what matters is not raw length. It is what kind of long prompt it is. Prompt structure already separates the medium-long multi-line scientific prompts that make the model ramble from the very longest prompts that are long for other reasons.

## Athena Read

I sent Athena this exact narrow question with the new note and summary bundle attached:

- why should prompt length predict completion length at all on this natural regression surface, if there is no implementation bug?
- answer it in plain English, dataset by dataset;
- stay on the literal input-output pair `prompt length -> completion length`.

Athena agreed with the local read and phrased it the same way:

- on this fixed setup, longer prompts often already contain more written work;
- that makes the model write more before it finishes;
- raw length works best when "longer prompt" usually means "more of the same kind of task";
- raw length breaks when prompts can be long for different reasons, which is exactly what `GPQA` shows.

The copied Athena context note is:

- `outputs/weeks/2026-W15/prompt_profile_length_mechanism_20260409/athena_length_mechanism_context.md`

## What Is Actually Proved

- There is a real prompt-length to completion-length correlation on most of these datasets. This is not an implementation bug artifact.
- The old prompt-length baseline was hiding real prompt-surface structure. Cheap prompt-shape features improve the prompt-only baseline.
- The correlation is benchmark-specific:
  - near-monotone on `AIME`
  - moderate on `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`
  - weak and non-monotone on `GPQA`
- The activation ensemble is still the strongest overall screening score, so prompt-only surface cues do not explain everything.

## What This Does **Not** Prove

- It does **not** prove that prompt length itself is the causal driver. The cleaner reading is that prompt length is usually standing in for prompt workload or prompt type.
- It does **not** prove that one tiny prompt-shape model is the full metadata ceiling. The current prompt-only baselines are still cheap baselines, not a final ceiling.
- It does **not** prove that activations stop helping once prompt shape is included. The ensemble still wins overall on `top_20p_capture`, especially on `LiveCodeBench`.
- It does **not** justify a universal slogan like “longer prompt always means longer completion.” `GPQA` already breaks that.

## Bottom Line

The simplest honest answer is:

Prompt length predicts completion length here because, on most of these datasets, longer prompts are usually the prompts with more work packed into them. The model then writes longer answers. That is a real effect on `AIME`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`. `GPQA` shows the important caveat: raw length alone is weak there, and prompt structure matters more than raw length.

So the old result was not pointing to a hidden bug. It was pointing to a real correlation between visible prompt workload and visible completion length on this fixed setup.
