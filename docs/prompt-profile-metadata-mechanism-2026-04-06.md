# Prompt-Profile Metadata Mechanism Note

Last updated: 2026-04-06 07:02 UTC

## Question

Why are the prompt-only metadata predictors so strong on the current prompt-profile surfaces?

## Short Answer

They are strong mainly because both current targets are dominated by fixed-budget long-completion risk, and prompt surface structure already encodes a lot of the latent prompt family / expected derivation verbosity that drives that risk.

The key split is:

- `mean_relative_length` is normalized completion length under a fixed `effective_max_tokens = 30000`
- `majority_s_0.5` is a majority-over-rollouts threshold on that same relative-length object

So prompt-only cues that predict long completions naturally look strong on both the regression and binary heads, even before activations enter.

## Current Objects

- Regression:
  - target: `mean_relative_length`
  - split: natural prompt-disjoint train/test
  - sampler: natural
- Binary:
  - target: `majority_s_0.5`
  - split: balanced train / natural test
  - current best activation surface: `ensemble h256 d1`

## Main Evidence

### 1. The binary head is mostly downstream of the regression head

On the finished binary test sets, using the ground-truth `mean_relative_length` itself as a scorer for `majority_s_0.5` gives:

| Dataset | `PR-AUC(mean_relative_length -> majority_s_0.5)` | `PR-AUC(p_cap -> majority_s_0.5)` | `PR-AUC(p_loop -> majority_s_0.5)` |
| --- | ---: | ---: | ---: |
| `AIME` | `1.000` | `0.793` | `0.690` |
| `GPQA` | `1.000` | `0.286` | `0.333` |
| `MATH-500` | `0.917` | `0.353` | `0.145` |
| `MMLU-Pro` | `0.833` | `0.506` | `0.506` |
| `LiveCodeBench` | `0.972` | `0.658` | `0.663` |

This is the clearest reason the metadata predictors look strong. The binary label is much closer to a thresholded long-completion target than to a loop-probability target.

### 2. The useful prompt-only cues are dataset-specific

Best cheap single-feature cue on the saved audit bundle:

| Dataset | Regression cue | Binary cue | Main read |
| --- | --- | --- | --- |
| `AIME` | `char_length 0.332` | `dollar_count 0.937` | symbolic / TeX density marks long derivations |
| `GPQA` | `newline_count 0.242` | `newline_count 0.167` | prompt structure matters more than raw length |
| `MATH-500` | `prompt_token_count 0.290` | `dollar_count 0.117` | mostly ordinary problem-length effect |
| `MMLU-Pro` | `dollar_count 0.300` | `newline_count 0.506` | some structure is real, but the binary row is fragile |
| `LiveCodeBench` | `prompt_token_count 0.248` | `char_length 0.590` | specification length / complexity drives output length |

There is no single universal metadata feature. Different prompt families hand the target away through different cheap surface cues.

### 3. The strongest rows are a mix of real structure and prevalence artifacts

#### Real structure

- `AIME` binary `dollar_count`
  - the high-value examples are dense TeX-heavy olympiad prompts
  - example top row: `dollar_count = 76`, `prompt_token_count = 597`, `mean_relative_length = 0.779`, `majority_s_0.5 = 1`
  - this is a real derivation-length cue, not a code bug
- `GPQA` regression `newline_count`
  - the effect is non-monotone in raw prompt length
  - prompt-length quartile means are `0.197, 0.259, 0.377, 0.218`
  - the riskiest prompts are medium-long multi-line scientific prompts, not the absolute longest formal ones
- `LiveCodeBench` prompt length / char length
  - this mostly looks like specification length, test-case density, and coding-task complexity
- `MATH-500` regression prompt length
  - weaker, but consistent with the ordinary "longer math statement -> longer derivation" story

#### Rows to treat cautiously

- `MMLU-Pro` binary `newline_count = 0.506`
  - this is mostly a tiny-prevalence artifact
  - in the test set, `newline_count = 17` covers `152` rows with positive rate `0.0066`
  - `newline_count = 27` is a single positive outlier
  - the row is still a clue that prompt formatting matters, but it is not a stable global law
- Any universal claim like:
  - "newlines are strong everywhere"
  - "longer prompt always means longer completion"
  - both `GPQA` and `LiveCodeBench` already show non-monotone quartile behavior

### 4. One additive prompt-shape model is not the whole answer

A simple seven-feature prompt-shape model using:

- `prompt_token_count`
- `log_token_length`
- `char_length`
- `newline_count`
- `digit_count`
- `dollar_count`
- `choice_count`

only helps modestly on some slices and can even get worse on others. That is useful evidence in itself. It means the prompt-only signal is benchmark-specific and nonlinear, not one clean universal prompt-length ceiling.

Examples:

- `GPQA` regression `top_20p_capture`: `0.168 -> 0.193`
- `MATH-500` regression `top_20p_capture`: `0.290 -> 0.310`
- `AIME` binary `PR-AUC`: `0.898 -> 0.808`
- `MMLU-Pro` binary `PR-AUC`: `0.110 -> 0.031`

So the current lesson is not "one additive prompt-only baseline solves the whole problem." The lesson is that the targets let several cheap surface cues become predictive, often in dataset-specific ways.

## Athena Read

Athena's deep read agreed with the main local hypothesis, but sharpened two points:

1. The surface features are mostly proxies for latent prompt family / expected derivation verbosity, not direct causes in themselves.
2. The current activation claim is only residual against raw prompt length, not yet against a serious prompt-shape control.

Athena's causal summary:

- latent task subtype / expected derivation length
  - influences prompt surface cues
  - influences long completion under a fixed budget
  - which drives `mean_relative_length`
  - which then almost thresholds into `majority_s_0.5`

That is why prompt-only metadata looks strong on both heads.

## Narrowest Honest Claim

The current results support only this narrow activation-lift claim:

- activations beat the old 1D prompt-length baseline on some slices of the current targets
- so they contain some signal about long-completion propensity beyond raw prompt length alone

The current results do **not** yet support:

- lift over a strong prompt-shape baseline in general
- a loop-specific hidden-state claim
- or a claim that binary-head lift is anything more than lift on a thresholded `mean_relative_length` object

## Cheapest Decisive Next Check

The cheapest next analysis is a residualized conditional-lift audit on the regression head:

1. Fit a stronger prompt-shape model for `mean_relative_length` on the existing frozen splits.
2. Score the natural test set.
3. Evaluate activation lift only:
   - on residual `mean_relative_length`, or
   - inside narrow prompt-shape-risk bins
4. Then check whether higher activation score is associated with higher `p_loop` after controlling for:
   - prompt shape
   - and true `mean_relative_length`

Why this is the right next test:

- `majority_s_0.5` is already too downstream of `mean_relative_length` to cleanly separate the mechanism
- this check needs no new rollouts
- and it is the cheapest way to distinguish:
  - prompt-shape-driven long-completion risk
  - residual long-completion signal
  - genuinely loop-specific activation signal

## Artifacts

- Combined surface note: `docs/prompt-profile-combined-audit-2026-04-05.md`
- Metadata audit bundle: `outputs/prompt_profile_metadata_audit_20260405/`
- Mechanism summary bundle: `outputs/prompt_profile_metadata_mechanism_20260406/`
- Athena consult log: `.agent/runtime/consult_history/1775257834.444509.jsonl`
