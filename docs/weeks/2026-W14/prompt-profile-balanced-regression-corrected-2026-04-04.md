# Balanced Prompt-Profile Regression Rerun (Corrected Full-Train Sampler Surface)

Status: non-canonical after 2026-04-04 23:45 UTC.

Wangzhi later corrected the training object itself: `mean_relative_length` should stay on the natural distribution and natural sampler because it is a continuous target, not a binary-label problem. Keep this note only as a side analysis of what changed when the full regression train/test splits were preserved but train-time sampling was still balanced using natural `majority_s_0.5` labels.

Use `docs/weeks/2026-W14/prompt-profile-full-train-results-2026-04-04.md` or
`docs/weeks/2026-W14/prompt-profile-full-surface-update-2026-04-04.md` for the canonical regression lane.

Generated: 2026-04-04 UTC

## Correction

The earlier note `docs/weeks/2026-W14/prompt-profile-balanced-regression-2026-04-04.md` was not the same train object as the previous Qwen3 run. It relabeled the downsampled balanced binary subset and therefore trained regression on much smaller train sets (`18 / 48 / 44 / 14 / 350` prompts across `GPQA / AIME / MATH-500 / MMLU-Pro / LiveCodeBench`).

This corrected rerun keeps the full saved Qwen3 `mean_relative_length` train/test splits and uses the binary `majority_s_0.5` labels only to balance the train-time sampler. Test stays natural. No regression-label downsampling is applied.

## Object

- target: `mean_relative_length`
- split contract:
  - train prompts: full saved Qwen3 shared-archive train split
  - train balancing: weighted sampler from the natural `majority_s_0.5` labels
  - test prompts: full saved Qwen3 shared-archive natural test split
  - no regression-label downsampling
- prompt surface: same locked April prompt-profile setup (`Qwen/Qwen3-1.7B`, prompt-prefill stacked all-layer last-token activations, `temperature=0.2`, `num_generations=4`, `max_tokens=30000`)
- probe family: default `mlp` preset, so hidden width `128`, depth `1`, dropout `0.1`
- views:
  - `ensemble`: one MLP per layer with `mean_prob` aggregation
  - `last_layer`: one MLP on the final-layer slice
- seeds: `0, 1, 2`
- checkpoint rule: frozen `best_loss`
- primary regression read: `top_20p_capture`
- secondary calibration read: `RMSE`
- diagnostic only: `Spearman`

Metadata baseline in this note means a train-fit linear regressor on prompt-only features. Because `effective_max_tokens` is fixed at `30000` on this surface, the only nontrivial metadata baseline is prompt length.

## Exact Train/Test Contract

| Dataset | Correct train prompts | Correct test prompts | Natural binary train pos | Natural binary train neg | Natural binary test pos | Natural binary test neg | Withdrawn note train prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPQA | 158 | 40 | 9 | 149 | 2 | 38 | 18 |
| AIME | 48 | 12 | 24 | 24 | 7 | 5 | 48 |
| MATH-500 | 400 | 100 | 22 | 378 | 4 | 96 | 44 |
| MMLU-Pro | 640 | 160 | 7 | 633 | 2 | 158 | 14 |
| LiveCodeBench | 640 | 160 | 175 | 465 | 54 | 106 | 350 |

The train-balance mechanism is therefore:

- keep all `158 / 48 / 400 / 640 / 640` regression train prompts;
- derive binary prevalence from the natural `majority_s_0.5` labels on those same train prompts;
- use those binary labels only to weight train sampling toward a balanced positive/negative mix.

## Metric Definitions

- Each held-out prompt `i` gets a score `s_i = predicted mean_relative_length`.
- Each held-out prompt also has mass `y_i = actual mean_relative_length` from the saved rollout archive.
- `top_10p_capture` / `top_20p_capture`: sort held-out prompts by `s_i`, keep the top `10%` or `20%`, and divide the captured actual mass by total held-out mass:
  - `sum_{i in top-k by s} y_i / sum_j y_j`
- `RMSE`: pointwise error on aligned held-out prompts.
- `Spearman`: monotone-order diagnostic on aligned held-out prompts. It is not the headline metric.

## Main Read

- Screening (`top_20p_capture`): ensemble is still the best single global regression surface.
  - Cross-dataset mean `top_20p_capture`: ensemble `0.345`, last-layer `0.299`, prompt length `0.261`
  - Ensemble beats last-layer on all `5 / 5`
  - Ensemble beats prompt length on `4 / 5`
  - `AIME` is still the only non-win against prompt length on that metric
- Calibration (`RMSE`): the full-train correction changes the story materially.
  - Cross-dataset mean `RMSE`: prompt length `0.180`, ensemble `0.192`, last-layer `0.195`
  - Prompt length is now the best calibrated baseline on `GPQA`, `AIME`, `MATH-500`, and `MMLU-Pro`
  - `LiveCodeBench` is the only dataset where ensemble has the best `RMSE`
- Diagnostic ordering (`Spearman`): ensemble is strongest on all `5 / 5`
  - Cross-dataset mean `Spearman`: ensemble `0.680`, last-layer `0.472`, prompt length `0.410`

So the corrected full-train object supports the regression lane only as a screening score. Restoring the full Qwen3 train counts helps the activation probes on the screening read, but it helps the prompt-length calibration baseline even more strongly on several datasets.

## What Changed Materially From The Withdrawn Note

- The withdrawn note undertrained `GPQA`, `MATH-500`, and `MMLU-Pro` by collapsing them to `18`, `44`, and `14` train prompts.
- Once the full train counts are restored, the prompt-length baseline becomes much stronger on `RMSE`:
  - `GPQA`: `0.244 -> 0.165`
  - `MATH-500`: `0.253 -> 0.154`
  - `MMLU-Pro`: `0.359 -> 0.126`
- The activation controls also move on the screening read:
  - `last_layer top_20p_capture` rises from `0.234 -> 0.305` on `MATH-500`
  - `last_layer top_20p_capture` rises from `0.200 -> 0.363` on `MMLU-Pro`
  - ensemble stays the best screening surface overall, but the calibration claim from the withdrawn note does not survive the full-train correction

## Results Tables

### `top_10p_capture` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.070` | `0.101 +/- 0.009` | `0.169 +/- 0.009` |
| AIME | `0.223` | `0.194 +/- 0.032` | `0.195 +/- 0.015` |
| MATH-500 | `0.151` | `0.157 +/- 0.025` | `0.186 +/- 0.003` |
| MMLU-Pro | `0.169` | `0.205 +/- 0.092` | `0.263 +/- 0.013` |
| LiveCodeBench | `0.137` | `0.160 +/- 0.004` | `0.171 +/- 0.000` |

Cross-dataset mean `top_10p_capture`: ensemble `0.197`, last-layer `0.164`, prompt length `0.150`.

### `top_20p_capture` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.168` | `0.246 +/- 0.031` | `0.306 +/- 0.039` |
| AIME | `0.306` | `0.268 +/- 0.037` | `0.305 +/- 0.017` |
| MATH-500 | `0.290` | `0.305 +/- 0.046` | `0.337 +/- 0.011` |
| MMLU-Pro | `0.292` | `0.363 +/- 0.130` | `0.445 +/- 0.005` |
| LiveCodeBench | `0.248` | `0.312 +/- 0.004` | `0.330 +/- 0.016` |

### `RMSE` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.165` | `0.171 +/- 0.011` | `0.206 +/- 0.023` |
| AIME | `0.176` | `0.192 +/- 0.002` | `0.187 +/- 0.002` |
| MATH-500 | `0.154` | `0.192 +/- 0.031` | `0.199 +/- 0.034` |
| MMLU-Pro | `0.126` | `0.195 +/- 0.026` | `0.149 +/- 0.004` |
| LiveCodeBench | `0.279` | `0.224 +/- 0.007` | `0.219 +/- 0.004` |

### `Spearman` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| GPQA | `0.116` | `0.246 +/- 0.054` | `0.584 +/- 0.052` |
| AIME | `0.676` | `0.361 +/- 0.185` | `0.718 +/- 0.077` |
| MATH-500 | `0.461` | `0.549 +/- 0.260` | `0.760 +/- 0.008` |
| MMLU-Pro | `0.393` | `0.412 +/- 0.187` | `0.538 +/- 0.003` |
| LiveCodeBench | `0.405` | `0.790 +/- 0.005` | `0.800 +/- 0.005` |

## Interpretation

- `GPQA`: ensemble is still clearly the best screening surface, but the corrected full-train prompt-length baseline is now the best calibrated predictor.
- `AIME`: this remains the awkward dataset. Screening is essentially a tie between prompt length and ensemble, while prompt length still has the best `RMSE`.
- `MATH-500`: ensemble is best for screening, but prompt length is best for calibration once the full train split is restored.
- `MMLU-Pro`: ensemble is strongest on both capture metrics and `Spearman`, but prompt length is still the best calibrated baseline by `RMSE`.
- `LiveCodeBench`: ensemble is the strongest overall surface here. It wins both screening and `RMSE`.

## Recommendation

- Treat this note as sensitivity analysis only, not as the default regression contract.
- If `mean_relative_length` stays in the project as a screening score for catching degenerate or long rollouts, use the natural-split / natural-sampler regression lane instead of this balanced-sampler variant.
- The substantive lesson from this side analysis is narrower:
  - restoring the full train counts matters;
  - even after that fix, prompt length remains the stronger calibration baseline on most datasets.

## Artifacts

- Corrected PDF report: `outputs/weeks/2026-W14/prompt_profile_balanced_regression_corrected_20260404/prompt_profile_balanced_regression_corrected_20260404.pdf`
- Corrected LaTeX source: `outputs/weeks/2026-W14/prompt_profile_balanced_regression_corrected_20260404/prompt_profile_balanced_regression_corrected_20260404.tex`
- Corrected copied summary ledger: `outputs/weeks/2026-W14/prompt_profile_balanced_regression_corrected_20260404/remote_summary/`
- Slurm log: `outputs/weeks/2026-W14/prompt_profile_balanced_regression_corrected_20260404/logs/prompt-profile-regression-balanced-2214.out`
- Superseded note for provenance only: `docs/weeks/2026-W14/prompt-profile-balanced-regression-2026-04-04.md`
