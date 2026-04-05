# Prompt-Profile Combined Audit: Natural Regression, Balanced Binary, And Metadata Controls

Last updated: 2026-04-05 02:10 UTC

## Executive Summary

- This is the current combined prompt-profile surface.
  - Regression: `mean_relative_length` on the natural prompt-disjoint split with natural sampling.
  - Binary: `majority_s_0.5` on balanced train / natural test, with the current recommended activation surface `ensemble h256 d1`.
  - Audit: a new cheap prompt-stat baseline pass plus an Athena code audit of the prompt-profile build / label / train / summarize path.
- The regression lane is now pinned cleanly.
  - Slurm `2215` reran the canonical natural regression object from the current PR head.
  - It reproduced the original locked `2043` regression ledger exactly, with max absolute difference `0.0` across prompt-length baselines plus `ensemble` and `last_layer` at both frozen checkpoints.
- The binary lane is still the cleaner deployment-facing head.
  - On the balanced binary capacity sweep, the best single global activation surface remains `ensemble h256 d1`.
  - Cross-dataset mean test `PR-AUC`: prompt length `0.350`, `ensemble h256 d1` `0.518`, `last_layer h256 d2` `0.430`.
- The metadata story is now narrower and more honest than the earlier PDFs.
  - The main report’s “metadata baseline” is really a train-fit 1D prompt-length baseline, because `effective_max_tokens=30000` is fixed.
  - Cheap prompt-shape features often match or beat raw prompt length, and in a few cases they rival or beat the activation probes too.
  - So the current activation results establish lift over prompt length on some surfaces, not yet lift over strong prompt-only controls in general.

## Current Objects

### Regression

- Target: `mean_relative_length`
- Split: natural prompt-disjoint train/test
- Sampler: natural
- Views: `ensemble`, `last_layer`
- Canonical artifact:
  - `docs/prompt-profile-natural-regression-rerun-2026-04-05.md`
  - `outputs/prompt_profile_natural_regression_rerun_20260405/prompt_profile_natural_regression_rerun_20260405.pdf`

### Binary

- Target: `majority_s_0.5`
- Split: train balanced by downsampling, test natural
- Current recommended activation surface: `ensemble h256 d1`
- Capacity-control artifact:
  - `docs/prompt-profile-binary-capacity-controls-2026-04-04.md`
  - `outputs/prompt_profile_binary_capacity_controls_20260404/prompt_profile_binary_capacity_controls_20260404.pdf`

## Regression Read

Primary metric here is screening, not calibration.

### Frozen `best_loss` regression table

| Dataset | Prompt length `top_20p_capture` | Ensemble `top_20p_capture` | Prompt length `RMSE` | Ensemble `RMSE` |
| --- | ---: | ---: | ---: | ---: |
| `GPQA` | `0.168` | `0.247 +/- 0.018` | `0.165` | `0.160 +/- 0.002` |
| `AIME` | `0.306` | `0.305 +/- 0.017` | `0.176` | `0.183 +/- 0.006` |
| `MATH-500` | `0.290` | `0.262 +/- 0.058` | `0.154` | `0.162 +/- 0.002` |
| `MMLU-Pro` | `0.292` | `0.355 +/- 0.025` | `0.126` | `0.133 +/- 0.000` |
| `LiveCodeBench` | `0.248` | `0.340 +/- 0.005` | `0.279` | `0.248 +/- 0.018` |

Cross-dataset means:

- `top_20p_capture`: prompt length `0.261`, ensemble `0.302`
- `RMSE`: prompt length `0.180`, ensemble `0.177`

### Regression interpretation

- On the operational screening read, ensemble beats prompt length on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`.
- On calibration, prompt length still wins on `AIME`, `MATH-500`, and `MMLU-Pro`.
- So this lane is still useful as a prompt-conditioned long-completion risk score.
- It is still not a clean calibrated regressor, and it is not yet evidence of a loop-specific hidden-state signal by itself.

## Binary Read

### Current binary recommendation table

| Dataset | Test prevalence | Prompt length `PR-AUC` | Ensemble `h256 d1` `PR-AUC` | Last-layer `h256 d2` `PR-AUC` |
| --- | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.0657` | `0.5833` | `0.3176` |
| `AIME` | `0.5833` | `0.8976` | `0.9120` | `0.8479` |
| `MATH-500` | `0.0400` | `0.1002` | `0.1663` | `0.1322` |
| `MMLU-Pro` | `0.0125` | `0.1104` | `0.2116` | `0.0959` |
| `LiveCodeBench` | `0.3375` | `0.5760` | `0.7143` | `0.7575` |

Cross-dataset mean test `PR-AUC`:

- prompt length `0.350`
- `ensemble h256 d1` `0.518`
- `last_layer h256 d2` `0.430`

### Binary interpretation

- `majority_s_0.5` is still the cleaner finished head.
- `ensemble h256 d1` remains the best single global activation surface.
- But the prompt-length baseline is too strong on `AIME` to support a broad “activation-only lift” story there.
- `LiveCodeBench` and `GPQA` remain the cleanest current binary activation wins.

## Metadata Audit

### What the current reported baseline really is

- The main report baseline is intentionally narrow:
  - train-fit prompt-only scorer on `prompt_token_count` and `effective_max_tokens`
  - on this run, `effective_max_tokens=30000` is fixed
  - so the effective control is just prompt length
- That means the current report does **not** yet compare activations against a strong prompt-only metadata ceiling.

### Cheap prompt-stat audit

The new audit used the saved prompt archives directly:

- regression source root:
  - `outputs/prompt_profile_natural_regression_rerun_20260405/` via the reused shared archives
- binary source root:
  - `outputs/prompt_profile_full_train_locked_pair_20260404/`
- saved audit bundle:
  - `outputs/prompt_profile_metadata_audit_20260405/`

Best raw prompt feature by dataset:

| Dataset | Regression best raw feature by `top_20p_capture` | Binary best raw feature by `PR-AUC` |
| --- | --- | --- |
| `GPQA` | `newline_count 0.242` | `newline_count 0.167` |
| `AIME` | `char_length 0.332` | `dollar_count 0.937` |
| `MATH-500` | `prompt_token_count 0.290` | `dollar_count 0.117` |
| `MMLU-Pro` | `dollar_count 0.300` | `newline_count 0.506` |
| `LiveCodeBench` | `prompt_token_count 0.248` | `char_length 0.590` |

Important comparisons against raw prompt length:

- `AIME` binary:
  - prompt length `PR-AUC 0.898`
  - dollar count `PR-AUC 0.937`
  - current recommended activation surface `ensemble h256 d1 0.912`
- `MMLU-Pro` binary:
  - prompt length `PR-AUC 0.110`
  - newline count `PR-AUC 0.506`
  - current recommended activation surface `ensemble h256 d1 0.212`
- `GPQA` regression:
  - prompt length `top_20p_capture 0.168`
  - newline count `0.242`
  - current ensemble `0.247`

So the cheap-feature audit already shows that the current 1D prompt-length control is missing real prompt-shape structure.

### Prompt-length quartiles

The quartile picture is also informative:

- `AIME`:
  - prompt-length quartiles rise almost monotonically in both `mean_relative_length` and `majority_s_0.5`
  - the longest quartile reaches mean relative length `0.765` and binary positive rate `1.000`
- `MMLU-Pro`:
  - the longest quartile moves from mean relative length `0.075` to `0.183`
  - binary positive rate moves from `0.000` to `0.050`
- `LiveCodeBench`:
  - the middle-to-long quartiles are the riskiest
  - binary positive rate rises from `0.100` in the shortest quartile to `0.538` in the third quartile
- `GPQA`:
  - the effect is non-monotone
  - the third quartile is riskiest, and the very longest prompts get safer again
  - this argues for prompt-family structure, not a simple linear length effect

## Athena Audit

Athena reviewed the current prompt-profile code path and the cheap prompt-stat audit in deep mode.

Files reviewed included:

- `docs/prompt-profile-natural-regression-rerun-2026-04-05.md`
- `docs/prompt-profile-full-train-results-2026-04-04.md`
- `scripts/run_prompt_profile_full_train.py`
- `scripts/summarize_prompt_profile_full_train.py`
- `scripts/train_probe.py`
- `scripts/train_metadata_residual_probe.py`
- `src/loop_probe/labeling.py`
- `scripts/audit_prompt_metadata_correlations.py`
- `outputs/prompt_profile_metadata_audit_20260405/metadata_correlation_summary.json`

Main Athena conclusions:

- The strongest explanation is conceptual, not a hidden implementation bug.
- The present targets are close to fixed-budget long-completion risk:
  - `mean_relative_length` is output length scaled by a constant budget
  - `majority_s_0.5` is effectively “does this prompt usually cross about half the budget?”
- So prompt-only structure can be strong even before activations enter.
- The missing control is a stronger prompt-shape baseline, not another prompt-length row.
  - token length, log token length, char length
  - newline, digit, dollar counts
  - prompt-template / subject / family identity where available
  - nonlinearities or interactions, because `GPQA` is already visibly non-monotone
- The current results still support narrower claims:
  - ensemble remains better than `last_layer` on most current surfaces
  - `ensemble h256 d1` is still the right current binary default
  - `LiveCodeBench` remains the clearest regression and binary activation win
- What they do **not** yet support is a general claim of hidden-state lift over strong prompt-only controls, or a clean loop-specific early-warning claim.

## Recommended Next Checks

Priority order from the combined local audit plus Athena:

1. Replace the current 1D prompt-length control with a strong train-fit prompt-shape baseline on the same frozen splits.
2. Evaluate activation probes as residuals over that metadata model, not only as stand-alone predictors.
3. Re-evaluate inside matched strata of prompt shape:
   - prompt-length quantile
   - newline / dollar / digit bins
   - subject or template family where available
4. Use family-held-out splits where possible, not only prompt-disjoint random splits.
5. Decompose “long completion risk” from “loop-specific risk” more explicitly before making stronger mechanistic claims.

## Current Honest Claim

The combined current claim should be:

- Regression:
  - useful as a screening score for prompt-conditioned long-completion risk
  - still mixed against even the narrow prompt-length baseline
- Binary:
  - `majority_s_0.5` is the cleaner finished head
  - `ensemble h256 d1` is the best current global activation surface
- Metadata:
  - the current report establishes lift over 1D prompt length on some surfaces
  - it does **not** yet establish lift over strong prompt-only controls in general

## Artifacts

- Combined audit PDF:
  - `outputs/prompt_profile_combined_audit_20260405/prompt_profile_combined_audit_20260405.pdf`
- Combined audit TeX:
  - `outputs/prompt_profile_combined_audit_20260405/prompt_profile_combined_audit_20260405.tex`
- Metadata audit bundle:
  - `outputs/prompt_profile_metadata_audit_20260405/`
