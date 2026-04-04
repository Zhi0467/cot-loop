# Balanced Prompt-Profile Regression Rerun

Superseded on 2026-04-04 23:12 UTC.

This note trained regression on the downsampled balanced binary subset, not on the full saved Qwen3 train splits.
Use `docs/prompt-profile-balanced-regression-corrected-2026-04-04.md` and
`outputs/prompt_profile_balanced_regression_corrected_20260404/` for the corrected
full-train sampler rerun.

Generated: 2026-04-04 UTC

## Object

This note covers the missing balanced-only regression object for the prompt-profile line:

- target: `mean_relative_length`
- split contract: reuse the exact prompt IDs from the saved balanced `majority_s_0.5` binary split
- train split: balanced on the binary `majority_s_0.5` label
- test split: natural
- evaluation surface: prompt-prefill only, same fixed April saved rollout archives

Important nuance: the regression target itself is not "balanced." The balancing applies only to which prompt IDs are included in train/test, because the regression rerun reuses the binary split and relabels those same prompts with `mean_relative_length`.

## Run Contract

- Source rollout archive root: `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404/`
- Balanced regression out root: `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404_regression_balanced/`
- Model / prompt surface: same locked April prompt-profile surface (`Qwen/Qwen3-1.7B`, prompt-prefill stacked all-layer last-token activations, `temperature=0.2`, `num_generations=4`, `max_tokens=30000`)
- Probe family: default `mlp` preset, so hidden width `128`, depth `1`, dropout `0.1`
- Views:
  - `ensemble`: one MLP per layer with `mean_prob` aggregation
  - `last_layer`: one MLP on the final-layer slice
- Seeds: `0, 1, 2`
- Epoch selection rule: frozen `best_loss`
- Primary regression read: `top_20p_capture`
- Secondary calibration read: `RMSE`
- Diagnostic only: `Spearman`

Metadata baseline in this note means a train-fit linear regressor on prompt-only features. Because `effective_max_tokens` is fixed at `30000` on this surface, the only nontrivial metadata baseline is prompt length.

## Subset Sizes

The reused binary split sizes are:

| Dataset | Train prompts | Train `majority_s_0.5` positives | Test prompts | Test `majority_s_0.5` positives |
| --- | ---: | ---: | ---: | ---: |
| GPQA | 18 | 9 | 40 | 2 |
| AIME | 48 | 24 | 12 | 7 |
| MATH-500 | 44 | 22 | 100 | 4 |
| MMLU-Pro | 14 | 7 | 160 | 2 |
| LiveCodeBench | 350 | 175 | 160 | 54 |

These counts matter for interpretation: `GPQA` and especially `MMLU-Pro` still have very small balanced train sets.

## Metric Definitions

- `top_10p_capture` / `top_20p_capture`: sort held-out prompts by predicted `mean_relative_length`, keep the top `10%` or `20%`, and divide the sum of actual `mean_relative_length` on those prompts by the total held-out actual `mean_relative_length` mass.
- `RMSE`: pointwise error on aligned held-out prompts.
- `Spearman`: monotone-order diagnostic on aligned held-out prompts. It is not the headline metric.

## Main Read

- Screening (`top_20p_capture`): ensemble is the best single global regression surface on this balanced rerun.
  - Cross-dataset mean `top_20p_capture`: ensemble `0.344`, prompt length `0.262`, last-layer `0.257`
  - Ensemble beats last-layer on `4 / 5`
  - Ensemble beats prompt length on `4 / 5`
  - The only non-win against prompt length is `AIME`
- Calibration (`RMSE`): the picture is different.
  - Cross-dataset mean `RMSE`: last-layer `0.191`, ensemble `0.205`, prompt length `0.265`
  - Last-layer beats ensemble on `GPQA`, `MATH-500`, and `MMLU-Pro`
  - Ensemble beats last-layer on `AIME` and `LiveCodeBench`
- Diagnostic ordering (`Spearman`): ensemble stays much stronger than last-layer on every dataset and beats prompt length on `4 / 5`

So the balanced rerun sharpens the same split that was already emerging earlier: if the regression lane is used as a screening score, keep the ensemble. If the regression lane is used as calibrated point prediction, last-layer is still competitive and sometimes better.

## Results Tables

### `top_10p_capture` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.081` | `0.109` | `0.143` |
| AIME | `0.223` | `0.221` | `0.204` |
| MATH-500 | `0.151` | `0.128` | `0.191` |
| MMLU-Pro | `0.169` | `0.084` | `0.284` |
| LiveCodeBench | `0.137` | `0.156` | `0.169` |

Cross-dataset mean `top_10p_capture`: ensemble `0.198`, prompt length `0.152`, last-layer `0.140`.

### `top_20p_capture` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.177` | `0.220 +/- 0.047` | `0.296 +/- 0.033` |
| AIME | `0.306` | `0.321 +/- 0.020` | `0.305 +/- 0.017` |
| MATH-500 | `0.290` | `0.234 +/- 0.097` | `0.346 +/- 0.016` |
| MMLU-Pro | `0.292` | `0.200 +/- 0.051` | `0.436 +/- 0.016` |
| LiveCodeBench | `0.248` | `0.308 +/- 0.008` | `0.336 +/- 0.007` |

### `RMSE` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.244` | `0.170 +/- 0.005` | `0.188 +/- 0.009` |
| AIME | `0.176` | `0.189 +/- 0.004` | `0.183 +/- 0.006` |
| MATH-500 | `0.253` | `0.182 +/- 0.017` | `0.227 +/- 0.025` |
| MMLU-Pro | `0.359` | `0.168 +/- 0.010` | `0.191 +/- 0.013` |
| LiveCodeBench | `0.291` | `0.246 +/- 0.005` | `0.236 +/- 0.009` |

### `Spearman` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `-0.116` | `0.127 +/- 0.109` | `0.541 +/- 0.088` |
| AIME | `0.676` | `0.483 +/- 0.138` | `0.639 +/- 0.090` |
| MATH-500 | `0.461` | `0.258 +/- 0.403` | `0.744 +/- 0.011` |
| MMLU-Pro | `0.392` | `0.116 +/- 0.159` | `0.580 +/- 0.053` |
| LiveCodeBench | `0.405` | `0.778 +/- 0.005` | `0.806 +/- 0.007` |

## Interpretation

- `GPQA`: the balanced rerun still strongly favors ensemble for screening, but last-layer has lower `RMSE`.
- `AIME`: this remains the awkward dataset. Screening is effectively a tie between prompt length and last-layer, and prompt length still has the best `RMSE`.
- `MATH-500`: ensemble clearly wins on capture, last-layer clearly wins on `RMSE`.
- `MMLU-Pro`: ensemble is the strongest screening surface by a wide margin, while last-layer still has lower `RMSE`.
- `LiveCodeBench`: ensemble wins on both screening and `RMSE`, but last-layer is now much closer than on the other datasets.

## Recommendation

- If we keep `mean_relative_length` as a regression lane for catching degenerate / long rollouts, use `ensemble` and report `top_20p_capture` first.
- If we care more about calibrated point prediction than screening, the honest story is mixed, and `last_layer` remains competitive enough that it should not be written off from this balanced rerun.
- `AIME` is still the dataset preventing a clean global "ensemble beats prompt length everywhere" claim on the balanced regression object.

## Artifacts

- PDF report: `outputs/prompt_profile_balanced_regression_20260404/prompt_profile_balanced_regression_20260404.pdf`
- LaTeX source: `outputs/prompt_profile_balanced_regression_20260404/prompt_profile_balanced_regression_20260404.tex`
- Copied summary ledger: `outputs/prompt_profile_balanced_regression_20260404/remote_summary/`
