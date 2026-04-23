# Natural Prompt-Profile Regression Rerun (Natural Split / Natural Sampler)

Last updated: 2026-04-05 00:21 UTC

## Executive Summary

- This note is the fresh regression-only rerun for the object Wangzhi ultimately asked to keep:
  - target: `mean_relative_length`
  - split: natural prompt-disjoint train/test
  - sampler: natural
  - views: `ensemble` and `last_layer`
- Runtime object:
  - Slurm job `2215`
  - branch head at launch: `b9247a1` (`slurm: add natural regression rerun wrapper`)
  - node: `tianhaowang-gpu0`
  - GPUs: `2`
- Data object:
  - reuse the saved April shared archives from the locked full-train run under `full_train_locked_pair_20260404`
  - retrain the regression probes from scratch on the current branch
  - do not rerun the binary head here
- Main result:
  - the natural regression surface reproduced the original locked `2043` regression ledger exactly
  - max absolute difference was `0.0` across all five datasets for:
    - prompt-only regression metadata baselines
    - `ensemble` aggregate metrics at `best_loss` and `best_rank`
    - `last_layer` aggregate metrics at `best_loss` and `best_rank`
- So the current statement is stronger than “natural split is the right conceptual object.”
  - it is now the right object and a verified reproducible object on the current PR branch
  - the balanced-regression notes remain side analyses only

## Why This Rerun Exists

The thread corrected the regression contract twice.

- First correction:
  - the downsampled balanced binary subset was the wrong regression dataset
- Second correction:
  - even the full-count balanced-sampler rerun was still the wrong default object
- Final collaborator decision:
  - `mean_relative_length` should stay on the natural train/test split with natural sampling because it is a continuous target, not a binary-label problem

This rerun therefore answers a narrow but important question:

- if we retrain the regression lane today on the current branch, under the natural split and natural sampler, do the April regression conclusions still hold?

The answer is yes, exactly.

## Rerun Contract

- Source shared-archive root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404`
- Fresh rerun output root:
  - `outputs/weeks/2026-W14/prompt_profile_natural_regression_rerun_20260405/`
- Model and rollout object:
  - `Qwen/Qwen3-1.7B`
  - `temperature=0.2`
  - `num_generations=4`
  - `max_tokens=30000`
  - prompt-prefill stacked all-layer last-token activations only
- Probe family:
  - default `mlp`
  - hidden width `128`
  - depth `1`
  - dropout `0.1`
- Optimizer / seed object:
  - seeds `0,1,2`
  - epochs `10`
  - batch size `256`
  - learning rate `1e-4`
  - weight decay `0.1`
- Reporting rule:
  - keep `best_loss` as the main checkpoint
  - keep `best_rank` diagnostic only

## Exact Reproduction Check

The fresh summary bundle was compared directly against the original locked full-train regression summary.

- Original regression ledger:
  - `outputs/weeks/2026-W14/prompt_profile_full_train_locked_pair_20260404/remote_summary/cross_dataset_summary.json`
- Fresh rerun ledger:
  - `outputs/weeks/2026-W14/prompt_profile_natural_regression_rerun_20260405/remote_summary/cross_dataset_summary.json`

Compared objects:

- all regression metadata-baseline test metrics for:
  - `prompt_length`
  - `effective_budget`
  - `prompt_length_plus_effective_budget`
- all aggregated regression probe metrics for:
  - `ensemble`
  - `last_layer`
  - both `best_loss` and `best_rank`
  - all five datasets

Result:

- max absolute difference = `0.0`
- no dataset, view, metric, or baseline moved

So the old natural regression note was not just still conceptually preferred. It was numerically stable under a clean retrain on the current branch.

## Fixed Dataset Surface

| Dataset | Train prompts | Test prompts | Task kind |
| --- | ---: | ---: | --- |
| `GPQA` | `158` | `40` | `multiple_choice_gpqa` |
| `AIME` | `48` | `12` | `math_freeform` |
| `MATH-500` | `400` | `100` | `math_freeform` |
| `MMLU-Pro` | `640` | `160` | `multiple_choice_mmlupro` |
| `LiveCodeBench` | `640` | `160` | `livecodebench_codegen` |

## Metric Definitions

- Regression score:
  - each held-out prompt gets one scalar score `s_i = predicted mean_relative_length`
- Actual mass:
  - each held-out prompt contributes `y_i = actual mean_relative_length`
- `top_10p_capture` / `top_20p_capture`:
  - sort held-out prompts by descending `s_i`
  - keep the top `10%` or `20%`
  - report `sum_{i in kept} y_i / sum_j y_j`
  - here “mass” means total realized relative-length weight across held-out prompts, not a prompt count and not a binary positive count
- `RMSE`:
  - pointwise error on aligned held-out prompt pairs
- `Spearman`:
  - monotone-order diagnostic on those same aligned prompt pairs
  - it is not the task definition
- Metadata baseline:
  - train-fit prompt-only linear regression on `prompt_token_count` and `effective_max_tokens`
  - because `effective_max_tokens=30000` is constant on this surface, the only nontrivial feature is prompt length

## Results

### `top_10p_capture` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.070` | `0.089 +/- 0.006` | `0.132 +/- 0.019` |
| `AIME` | `0.223` | `0.221 +/- 0.004` | `0.204 +/- 0.015` |
| `MATH-500` | `0.151` | `0.117 +/- 0.061` | `0.129 +/- 0.033` |
| `MMLU-Pro` | `0.169` | `0.094 +/- 0.062` | `0.209 +/- 0.022` |
| `LiveCodeBench` | `0.137` | `0.156 +/- 0.000` | `0.170 +/- 0.004` |

Cross-dataset mean:

- prompt length: `0.150`
- last-layer: `0.135`
- ensemble: `0.169`

### `top_20p_capture` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.168` | `0.216 +/- 0.039` | `0.247 +/- 0.018` |
| `AIME` | `0.306` | `0.321 +/- 0.020` | `0.305 +/- 0.017` |
| `MATH-500` | `0.290` | `0.243 +/- 0.094` | `0.262 +/- 0.058` |
| `MMLU-Pro` | `0.292` | `0.185 +/- 0.098` | `0.355 +/- 0.025` |
| `LiveCodeBench` | `0.248` | `0.317 +/- 0.010` | `0.340 +/- 0.005` |

Cross-dataset mean:

- prompt length: `0.261`
- last-layer: `0.256`
- ensemble: `0.302`

### `RMSE` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.165` | `0.164 +/- 0.003` | `0.160 +/- 0.002` |
| `AIME` | `0.176` | `0.189 +/- 0.004` | `0.183 +/- 0.006` |
| `MATH-500` | `0.154` | `0.169 +/- 0.018` | `0.162 +/- 0.002` |
| `MMLU-Pro` | `0.126` | `0.138 +/- 0.005` | `0.133 +/- 0.000` |
| `LiveCodeBench` | `0.279` | `0.242 +/- 0.011` | `0.248 +/- 0.018` |

Cross-dataset mean:

- prompt length: `0.180`
- last-layer: `0.180`
- ensemble: `0.177`

### `Spearman` at `best_loss`

| Dataset | Prompt length | Last-layer | Ensemble |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.116` | `0.175 +/- 0.225` | `0.419 +/- 0.045` |
| `AIME` | `0.676` | `0.483 +/- 0.138` | `0.639 +/- 0.090` |
| `MATH-500` | `0.461` | `0.158 +/- 0.426` | `0.351 +/- 0.199` |
| `MMLU-Pro` | `0.393` | `-0.019 +/- 0.293` | `0.348 +/- 0.034` |
| `LiveCodeBench` | `0.405` | `0.784 +/- 0.006` | `0.805 +/- 0.002` |

Cross-dataset mean:

- prompt length: `0.410`
- last-layer: `0.316`
- ensemble: `0.512`

## Interpretation

- Operational screening read:
  - on `top_20p_capture`, ensemble beats prompt length on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`
  - it loses on `AIME` and `MATH-500`
  - it beats `last_layer` on `4 / 5`, with only `AIME` favoring `last_layer`
- Calibration read:
  - on `RMSE`, ensemble beats prompt length only on `GPQA` and `LiveCodeBench`
  - prompt length still wins on `AIME`, `MATH-500`, and `MMLU-Pro`
- Diagnostic ordering:
  - `Spearman` still looks strongest for ensemble on `GPQA` and `LiveCodeBench`
  - but this remains tertiary and should not drive the headline claim
- Most important practical result from the rerun:
  - the mixed regression story did not wash out, sharpen, or drift on the current branch
  - it reproduced exactly

## Recommendation

- Keep the regression lane on the natural split and natural sampler.
- Keep `top_20p_capture` as the primary regression metric for this lane.
- Keep `RMSE` as calibration context.
- Keep `Spearman` diagnostic only.
- Do not cite the balanced-regression notes as the canonical regression answer.
- If the regression lane is tuned further, do it on this exact natural surface:
  - small layer-subset sweep
  - small capacity sweep
  - no more balancing detours

## Artifacts

- Fresh report bundle:
  - `outputs/weeks/2026-W14/prompt_profile_natural_regression_rerun_20260405/`
- Fresh copied summary ledger:
  - `outputs/weeks/2026-W14/prompt_profile_natural_regression_rerun_20260405/remote_summary/`
- Fresh Slurm log:
  - `outputs/weeks/2026-W14/prompt_profile_natural_regression_rerun_20260405/logs/prompt-profile-regression-natural-2215.out`
- Original locked two-head note for background context:
  - `docs/weeks/2026-W14/prompt-profile-full-train-results-2026-04-04.md`
- Combined April surface note for the regression + binary split:
  - `docs/weeks/2026-W14/prompt-profile-full-surface-update-2026-04-04.md`
