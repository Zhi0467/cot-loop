# Prompt-Profile Metadata Mechanism Context

Last updated: 2026-04-06 07:02 UTC

## Exact question

Explain why the prompt-only metadata predictors are strong on the current prompt-profile surfaces. The collaborator does **not** want another generic summary that says "prompt-only features are strong" or "use a stronger baseline." The answer should reason about the mechanism.

## Current canonical objects

- Regression head:
  - target: `mean_relative_length`
  - split: natural prompt-disjoint train/test
  - sampler: natural
  - canonical rerun: `2215`, which exactly reproduced locked run `2043`
- Binary head:
  - target: `majority_s_0.5`
  - split: balanced train / natural test
  - current best activation surface: `ensemble h256 d1`

## Relevant target definitions from code

From `src/loop_probe/labeling.py`:

- `mean_relative_length = mean(length / effective_max_tokens)` across rollouts
- `p_cap = mean(cap_hit)`
- `p_loop = mean(loop_flag)`
- `majority_s_0.5 = 1` iff a strict majority of rollouts have `relative_length >= 0.5`

On this April surface, `effective_max_tokens = 30000` is fixed, so both main targets are fundamentally about long completions under a constant budget.

## Existing report surface

- Combined audit note:
  - `docs/prompt-profile-combined-audit-2026-04-05.md`
- Cheap prompt-stat audit bundle:
  - `outputs/prompt_profile_metadata_audit_20260405/`
- New mechanism summary:
  - `outputs/prompt_profile_metadata_mechanism_20260406/metadata_mechanism_summary.json`

## Already-established facts

- The old reported "metadata baseline" is only a train-fit 1D prompt-length model on this fixed-budget surface.
- Cheap single prompt features already rival or beat prompt length on some datasets.
- The current activation results therefore prove lift over prompt length on some slices, but not yet lift over strong prompt-only controls in general.

## New mechanism findings

### 1. The binary label is very close to a thresholded version of the regression label

Using the finished binary test sets, treat the **ground-truth** `mean_relative_length` itself as a scorer for `majority_s_0.5`:

| Dataset | `PR-AUC(mean_relative_length -> majority_s_0.5)` | `PR-AUC(p_cap -> majority_s_0.5)` | `PR-AUC(p_loop -> majority_s_0.5)` |
| --- | ---: | ---: | ---: |
| `AIME` | `1.000` | `0.793` | `0.690` |
| `GPQA` | `1.000` | `0.286` | `0.333` |
| `MATH-500` | `0.917` | `0.353` | `0.145` |
| `MMLU-Pro` | `0.833` | `0.506` | `0.506` |
| `LiveCodeBench` | `0.972` | `0.658` | `0.663` |

Interpretation:

- `majority_s_0.5` is much closer to a thresholded long-rollout / relative-length object than to a loop-probability object.
- This immediately explains why prompt features that predict long completion risk also transfer well to the binary head.

### 2. The useful prompt-only cues are dataset-specific

Single-feature audit highlights:

| Dataset | Regression best cheap feature (`top_20p_capture`) | Binary best cheap feature (`PR-AUC`) |
| --- | --- | --- |
| `GPQA` | `newline_count = 0.242` | `newline_count = 0.167` |
| `AIME` | `char_length = 0.332` | `dollar_count = 0.937` |
| `MATH-500` | `prompt_token_count = 0.290` | `dollar_count = 0.117` |
| `MMLU-Pro` | `dollar_count = 0.300` | `newline_count = 0.506` |
| `LiveCodeBench` | `prompt_token_count = 0.248` | `char_length = 0.590` |

So there is no single universal prompt-only feature. Different datasets hand the target away through different surface cues.

### 3. Some of the strongest rows are real structure, some are tiny-prevalence artifacts

- `AIME` binary:
  - the high-`dollar_count` examples are exactly dense TeX-heavy olympiad problems
  - example top row: a geometry problem with `dollar_count = 76`, `prompt_token_count = 597`, `mean_relative_length = 0.779`, `majority_s_0.5 = 1`
  - coarse test pattern:
    - `dollar_count = 4`: `majority_s_0.5 rate = 0.0`, mean `mean_relative_length = 0.479`
    - `dollar_count = 46` or `76`: `majority_s_0.5 rate = 1.0`, mean `mean_relative_length ≈ 0.78-0.82`
  - this looks like a real derivation-length cue, not an implementation bug

- `GPQA` regression:
  - length is visibly non-monotone
  - test prompt-length quartiles by mean target:
    - Q1: `0.197`
    - Q2: `0.259`
    - Q3: `0.377`
    - Q4: `0.218`
  - the riskiest examples are medium-long structured physics / quantum prompts with many line breaks, not the absolute longest formal prompts
  - example high-risk row:
    - `prompt_token_count = 246`, `newline_count = 18`, `mean_relative_length = 0.582`
  - example very long but safer row:
    - `prompt_token_count = 916`, `newline_count = 16`, `mean_relative_length = 0.182`
  - this suggests prompt family / formatting structure, not a simple "longer prompt -> longer answer" law

- `MMLU-Pro` binary:
  - the apparent `newline_count` strength is fragile
  - the `PR-AUC 0.506` row is mostly driven by a rare outlier bucket under tiny prevalence
  - exact test breakdown:
    - `newline_count = 17`: `n = 152`, `majority_s_0.5 rate = 0.0066`
    - `newline_count = 27`: `n = 1`, `majority_s_0.5 rate = 1.0`
  - the positive example is a long itemized finance-style prompt
  - this is still evidence that prompt formatting matters, but it should **not** be overread as a robust global feature law

- `LiveCodeBench` regression / binary:
  - prompt length and character count are the main cheap cues
  - quartile pattern is not monotone at the extreme top end:
    - mean `mean_relative_length` rises from Q1 to Q3, then drops in Q4
  - this looks like "problem-spec length / specification complexity" more than loop-specific structure

### 4. A multifeature prompt-shape model helps on some surfaces, but not uniformly

Simple additive prompt-shape model using:

- `prompt_token_count`
- `log_token_length`
- `char_length`
- `newline_count`
- `digit_count`
- `dollar_count`
- `choice_count`

compared with 1D prompt length:

| Dataset | Regression `top_20p_capture` length -> shape | Binary `PR-AUC` length -> shape |
| --- | --- | --- |
| `GPQA` | `0.168 -> 0.193` | `0.082 -> 0.092` |
| `AIME` | `0.306 -> 0.306` | `0.898 -> 0.808` |
| `MATH-500` | `0.290 -> 0.310` | `0.099 -> 0.108` |
| `MMLU-Pro` | `0.292 -> 0.303` | `0.110 -> 0.031` |
| `LiveCodeBench` | `0.248 -> 0.239` | `0.577 -> 0.577` |

Interpretation:

- the point is **not** that one simple additive prompt-only model dominates everything
- the point is that the current targets let several cheap surface cues become predictive, often in dataset-specific and nonlinear ways
- some apparent metadata wins are stable structural cues; others are brittle because test prevalence is tiny

## Working hypothesis to evaluate

The strongest explanation is likely:

1. Both current targets are dominated by long-completion risk under a fixed budget.
2. That long-completion risk is often already encoded in prompt surface structure:
   - TeX / symbolic density on olympiad math
   - itemized / tabular formatting on some MMLU-Pro prompts
   - specification length and test-case density on LiveCodeBench
   - structured scientific formatting on the riskier GPQA prompts
3. Because `majority_s_0.5` is so close to thresholded `mean_relative_length`, any prompt feature that predicts the regression head can automatically look strong on the binary head too.
4. The remaining activation claim therefore has to be residual:
   - do activations add lift **after** conditioning on prompt shape?
   - do they help inside matched prompt-shape strata?

## Specific questions for Athena

Please answer these directly:

1. Is the explanation above the most plausible reading of the current code + data surface?
2. What is the cleanest causal story for why the metadata predictors are strong, dataset by dataset?
3. Which current rows look like genuine structure, and which look like tiny-sample / prevalence artifacts that should not be overinterpreted?
4. Given the code and target definitions, what is the most honest narrow claim we can make right now about activation lift?
5. What is the next cheapest decisive analysis to separate:
   - prompt-shape-driven long-completion risk
   - loop-specific signal
   - residual activation lift
