# Prompt-Profile Full Train Results And Binary Capacity Follow-Up

Last updated: 2026-04-05 00:54 UTC

## Executive Summary

- This report now keeps the whole current object in one place:
  - the first locked prompt-profile full-train run on the fixed five-dataset saved surface;
  - the later balanced-binary capacity follow-up on the same saved April `majority_s_0.5` data.
- Fresh regression verification:
  - Slurm `2215` reran the regression lane only on the current branch with the natural train/test split and natural sampler.
  - That rerun reproduced the original regression ledger exactly.
  - Use `docs/weeks/2026-W14/prompt-profile-natural-regression-rerun-2026-04-05.md` for the explicit reproducibility check and the fresh regression-only artifact bundle.
- Runtime object: Slurm job `2043` on `2` GPUs, completed at `2026-04-04 00:10 UTC`.
- Model/policy object: `Qwen/Qwen3-1.7B`, `temperature=0.2`, `num_generations=4`, `max_tokens=30000`, prompt-prefill activations only, loop detector `n=30`, `k=20`.
- Split contract: regression keeps the natural prompt-disjoint train/test split; binary relabel keeps test natural but downsample-balances the train split to `50/50`.
- Continuous head (`mean_relative_length`): if the use is screening degenerate prompts, the right headline metric is held-out top-k capture on the frozen `best_loss` checkpoint, not `Spearman`. On `top_20p_capture`, ensemble beats the train-fit prompt-length baseline on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`, but not on `AIME` or `MATH-500`. On `RMSE`, the win shrinks to `GPQA` and `LiveCodeBench` only.
- Binary head (`majority_s_0.5`): this is still the cleaner finished surface. Ensemble `PR-AUC` beats the prompt-length baseline on all five datasets.
- Balanced-binary capacity follow-up:
  - the locked run above used the default `h128 d1` binary probes;
  - the later `2106` / `2107` / `2108` follow-up reran only the balanced binary head with `h128 d1`, `h256 d1`, and `h256 d2` on the same saved April relabeled data;
  - on that follow-up, the best single global binary ensemble surface is `h256 d1`, not default `h128 d1` and not deeper `h256 d2`;
  - this changes the recommended binary probe family only; it does not change the regression lane above.
- Recommendation from the combined surface:
  - if the goal is deployment-facing degenerate-prompt screening, lead with `majority_s_0.5` using ensemble `h256 d1`;
  - if `mean_relative_length` is still reported, use `top_20p_capture` first, `RMSE` second, and `Spearman` only as a tertiary diagnostic.

## Exact Question

What does the first locked full-train pass say about the current two-head prompt-profile surface under the fixed saved contract?

- regression head: `mean_relative_length`
- binary head: `majority_s_0.5`
- feature surface: prompt-prefill activations only
- views compared: layerwise `ensemble` versus `last_layer`
- checkpoint rule: keep `best_loss` as the frozen reporting checkpoint and keep `best_rank` diagnostic-only

## Frozen Run Contract

- Data object: reuse the saved March prompt-profile archives. This run does not reroll prompts and does not rebuild the prompt-profile surface from scratch.
- Optimizer seeds: `0`, `1`, `2`. All ensemble/last-layer means below are means across those three seeds, with `+/-` showing seed standard deviation.
- Dataset seed and prompt splits stay fixed. Only optimizer seed varies.

| Dataset | Task kind | Train prompts | Test prompts | Runtime note |
| --- | --- | ---: | ---: | --- |
| `GPQA` | `multiple_choice_gpqa` | `158` | `40` | saved prompt field `Question` |
| `AIME` | `math_freeform` | `48` | `12` | saved prompt field `question` |
| `MATH-500` | `math_freeform` | `400` | `100` | saved prompt field `problem` |
| `MMLU-Pro` | `multiple_choice_mmlupro` | `640` | `160` | saved prompt field `problem` |
| `LiveCodeBench` | `livecodebench_codegen` | `640` | `160` | saved prompt field `problem` |

### Split And Balancing Contract

- Regression `mean_relative_length`
  - no balancing on train or test; the shared archive keeps the natural prompt-disjoint split shown above.
- Binary `majority_s_0.5`
  - relabeled from the same prompts with `--balance-train downsample --balance-test none`.
  - train prevalence is therefore exactly `0.5` on every dataset.
  - test prevalence stays natural: `GPQA 0.050`, `AIME 0.583`, `MATH-500 0.040`, `MMLU-Pro 0.0125`, `LiveCodeBench 0.3375`.

## Definitions

### Targets

- `mean_relative_length`
  - For prompt `i`, this is the repeated-rollout mean of `L_r / E`, where `L_r` is the rollout length and `E` is the prompt's effective token budget.
- `majority_s_0.5`
  - For prompt `i`, this is `1` if a strict majority of the repeated rollouts satisfy `L_r / E >= 0.5`, otherwise `0`.

### Views

- `ensemble` for regression
  - one MLP per layer with `mean_prob` aggregation.
- `ensemble` for binary
  - one MLP per layer with `vote_fraction` aggregation over per-layer `0/1` predictions.
- `last_layer`
  - the same probe family restricted to the final layer only.

### Evaluation pairs

- Regression eval pair
  - each held-out prompt `i` contributes one aligned pair `(\hat y_i, y_i)`, where `y_i` is that same prompt's realized `mean_relative_length`.
- Binary eval pair
  - each held-out prompt `i` contributes one aligned pair `(\hat s_i, y_i)`, where `y_i` is that same prompt's realized `majority_s_0.5` label.

### Metrics

- `score`
  - regression lane: each held-out prompt gets one scalar score `s_i = \hat y_i`.
  - for regression `ensemble`, `s_i` is the mean of the per-layer MLP predictions; for regression `last_layer`, it is the final-layer MLP prediction; for the prompt-length baseline, it is the linear-model prediction from prompt length, with the budget term constant on this run.
  - binary lane: each held-out prompt gets one scalar positive score `s_i = \hat s_i`; `PR-AUC` ranks these scores, and the threshold table applies one train-chosen threshold to them.
- `top_10p_capture` / `top_20p_capture`
  - for held-out prompts `i=1,...,n`, sort by descending regression score `s_i`, keep the top `ceil(0.1 n)` or `ceil(0.2 n)`, and compute `sum_{i in kept} y_i / sum_{i=1}^n y_i`.
  - here `y_i` is the realized `mean_relative_length` for that same held-out prompt.
  - "mass" means total realized relative-length weight across prompts, not a prompt count and not a binary positive count.
  - concrete example: on `GPQA`, test has `40` prompts, so `top_20p_capture` keeps the top `8` prompts by predicted `mean_relative_length`; ensemble `0.247` means those `8` prompts contain `24.7%` of the total realized relative-length mass on the `40`-prompt test split.
- `RMSE`
  - pointwise error on those aligned held-out prompt pairs.
- `Spearman`
  - monotone ordering agreement on those same aligned prompt pairs. It is not the task definition.
- `PR-AUC`
  - ranking metric for the binary head across thresholds.

### Metadata baseline

- This report uses train-fit prompt-only baselines:
  - allowed inputs: `prompt_token_count`, and separately `effective_max_tokens`;
  - excluded inputs: no activations, no prompt text, no dataset/source identity, no model identifier, no architecture features, no generated output features.
- Regression baseline
  - standardized linear regression fit on the train split, evaluated once on held-out test.
- Binary baseline
  - train-fit 1D score rule, with direction chosen by train `PR-AUC` and threshold chosen by train macro-F1 / positive-F1 / accuracy, evaluated once on held-out test.
- Important fixed-run nuance
  - `effective_max_tokens=30000` is constant on this run, so prompt length is the only nontrivial metadata feature here. The joint regression control collapses to prompt length, and the binary `effective_budget` control is effectively constant.

## Regression Results: `mean_relative_length`

The continuous head can be read in two different ways:

- screening read
  - use `top_k` capture because the product question is "do we catch degenerate prompts?"
- calibration read
  - use `RMSE` because the target is still continuous

This report keeps the checkpoint frozen and changes only the reporting lens.

### Screening Table: Top-10 Capture

| Dataset | Ensemble `top_10p_capture` | Last-layer `top_10p_capture` | Prompt-length baseline `top_10p_capture` |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.132 +/- 0.019` | `0.089 +/- 0.006` | `0.070` |
| `AIME` | `0.204 +/- 0.015` | `0.221 +/- 0.004` | `0.223` |
| `MATH-500` | `0.129 +/- 0.033` | `0.117 +/- 0.061` | `0.151` |
| `MMLU-Pro` | `0.209 +/- 0.022` | `0.094 +/- 0.062` | `0.169` |
| `LiveCodeBench` | `0.170 +/- 0.004` | `0.156 +/- 0.000` | `0.137` |

### Screening Table: Top-20 Capture

| Dataset | Ensemble `top_20p_capture` | Last-layer `top_20p_capture` | Prompt-length baseline `top_20p_capture` |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.247 +/- 0.018` | `0.216 +/- 0.039` | `0.168` |
| `AIME` | `0.305 +/- 0.017` | `0.321 +/- 0.020` | `0.306` |
| `MATH-500` | `0.262 +/- 0.058` | `0.243 +/- 0.094` | `0.290` |
| `MMLU-Pro` | `0.355 +/- 0.025` | `0.185 +/- 0.098` | `0.292` |
| `LiveCodeBench` | `0.340 +/- 0.005` | `0.317 +/- 0.010` | `0.248` |

### Calibration And Diagnostic Table

| Dataset | Ensemble `RMSE` | Last-layer `RMSE` | Prompt-length baseline `RMSE` | Ensemble `Spearman` | Last-layer `Spearman` | Prompt-length baseline `Spearman` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.160 +/- 0.002` | `0.164 +/- 0.003` | `0.165` | `0.419 +/- 0.045` | `0.175 +/- 0.225` | `0.116` |
| `AIME` | `0.183 +/- 0.006` | `0.189 +/- 0.004` | `0.176` | `0.639 +/- 0.090` | `0.483 +/- 0.138` | `0.676` |
| `MATH-500` | `0.162 +/- 0.002` | `0.169 +/- 0.018` | `0.154` | `0.351 +/- 0.199` | `0.158 +/- 0.426` | `0.461` |
| `MMLU-Pro` | `0.133 +/- 0.000` | `0.138 +/- 0.005` | `0.126` | `0.348 +/- 0.034` | `-0.019 +/- 0.293` | `0.393` |
| `LiveCodeBench` | `0.248 +/- 0.018` | `0.242 +/- 0.011` | `0.279` | `0.805 +/- 0.002` | `0.784 +/- 0.006` | `0.405` |

### Regression Interpretation

- Screening read, ensemble versus prompt length
  - wins: `GPQA`, `MMLU-Pro`, `LiveCodeBench`
  - losses: `AIME`, `MATH-500`
- Calibration read, ensemble versus prompt length
  - wins: `GPQA`, `LiveCodeBench`
  - losses: `AIME`, `MATH-500`, `MMLU-Pro`
- Ensemble versus last-layer
  - on `top_20p_capture`, ensemble wins `4 / 5`; only `AIME` slightly favors `last_layer`
  - on `RMSE`, ensemble also wins `4 / 5`, but the exception is different: `LiveCodeBench`
- Most important split
  - `LiveCodeBench` flips depending on the object:
    - `last_layer` is slightly better calibrated by `RMSE`
    - ensemble is better for screening by both `top_10p_capture` and `top_20p_capture`
- Practical read
  - `mean_relative_length` is still a usable score, but it is not a uniform activation-lift story over prompt length. It is mixed even after freezing the target and the run contract.

## Binary Results: `majority_s_0.5`

This is the cleaner finished head from the locked pair.

### Ranking Table

| Dataset | Ensemble `PR-AUC` | Last-layer `PR-AUC` | Prompt-length baseline `PR-AUC` | Test prevalence |
| --- | ---: | ---: | ---: | ---: |
| `GPQA` | `0.420 +/- 0.029` | `0.230 +/- 0.278` | `0.066` | `0.050` |
| `AIME` | `0.922 +/- 0.024` | `0.904 +/- 0.028` | `0.898` | `0.583` |
| `MATH-500` | `0.135 +/- 0.005` | `0.151 +/- 0.071` | `0.100` | `0.040` |
| `MMLU-Pro` | `0.184 +/- 0.037` | `0.056 +/- 0.012` | `0.110` | `0.013` |
| `LiveCodeBench` | `0.711 +/- 0.036` | `0.712 +/- 0.023` | `0.576` | `0.338` |

### Threshold Table: Ensemble Versus Prompt-Length Baseline

| Dataset | Ensemble precision | Ensemble recall | Ensemble macro-F1 | Prompt-length precision | Prompt-length recall | Prompt-length macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.110 +/- 0.050` | `1.000 +/- 0.000` | `0.382 +/- 0.244` | `0.038` | `0.500` | `0.286` |
| `AIME` | `0.926 +/- 0.128` | `0.762 +/- 0.218` | `0.798 +/- 0.044` | `0.800` | `0.571` | `0.667` |
| `MATH-500` | `0.091 +/- 0.020` | `1.000 +/- 0.000` | `0.444 +/- 0.060` | `0.079` | `0.750` | `0.458` |
| `MMLU-Pro` | `0.117 +/- 0.073` | `0.833 +/- 0.289` | `0.562 +/- 0.067` | `0.029` | `1.000` | `0.397` |
| `LiveCodeBench` | `0.609 +/- 0.046` | `0.827 +/- 0.091` | `0.747 +/- 0.014` | `0.494` | `0.722` | `0.646` |

### Binary Interpretation

- Ensemble `PR-AUC` beats prompt length on all five datasets.
- The cleanest wins are:
  - `AIME`
    - strong ranking and strong fixed-threshold behavior
  - `MMLU-Pro`
    - large ranking gain despite tiny prevalence
  - `LiveCodeBench`
    - large ranking gain with solid precision/recall
- Caveats by dataset
  - `GPQA`
    - real ranking lift, but only `5%` test prevalence, so threshold metrics are unstable
  - `MATH-500`
    - ranking lift over prompt length exists, but the absolute threshold-quality story is still weak

## Follow-Up: Balanced-Binary Capacity Controls

This section is separate from the frozen locked `2043` tables above.

- Why it exists:
  - Wangzhi explicitly asked for the newer train-balanced / test-natural binary surface, with probe size called out literally rather than inherited from the old default.
- What stayed fixed:
  - same saved April relabeled binary data under `outputs/weeks/2026-W14/prompt_profile_full_train_locked_pair_20260404/`
  - same binary target `majority_s_0.5`
  - same train-balanced / test-natural split contract
  - same prompt-prefill feature surface with sample shape `[28, 2048]`
  - same seeds `0,1,2`
  - same optimizer settings `epochs=15`, `batch_size=256`, `lr=1e-4`, `weight_decay=0.05`, `dropout=0.1`
- What changed:
  - default binary probes `h128 d1`
  - width-only control `h256 d1`
  - width+depth control `h256 d2`
- Follow-up jobs:
  - `2106`: `LiveCodeBench`-only pilot caused by the first dataset-list bug
  - `2107`: corrected five-dataset `h256 d2` rerun
  - `2108`: width-only `h256 d1` control

### Balanced-Binary Ranking Table: Test `PR-AUC`

| Dataset | Test prevalence | Prompt length | Ensemble `h128 d1` | Ensemble `h256 d1` | Ensemble `h256 d2` | Last-layer `h128 d1` | Last-layer `h256 d1` | Last-layer `h256 d2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.0657` | `0.4198` | `0.5833` | `0.5025` | `0.2298` | `0.3071` | `0.3176` |
| `AIME` | `0.5833` | `0.8976` | `0.9218` | `0.9120` | `0.9090` | `0.9043` | `0.8436` | `0.8479` |
| `MATH-500` | `0.0400` | `0.1002` | `0.1354` | `0.1663` | `0.1596` | `0.1506` | `0.1315` | `0.1322` |
| `MMLU-Pro` | `0.0125` | `0.1104` | `0.1837` | `0.2116` | `0.2023` | `0.0559` | `0.0701` | `0.0959` |
| `LiveCodeBench` | `0.3375` | `0.5760` | `0.7111` | `0.7143` | `0.6865` | `0.7116` | `0.7452` | `0.7575` |

### Balanced-Binary Mean `PR-AUC`

| Surface | `best_loss` mean test `PR-AUC` | `best_rank` mean test `PR-AUC` |
| --- | ---: | ---: |
| Prompt length baseline | `0.3500` | `0.3500` |
| Ensemble `h128 d1` | `0.4744` | `0.4744` |
| Ensemble `h256 d1` | `0.5175` | `0.5385` |
| Ensemble `h256 d2` | `0.4920` | `0.5215` |
| Last-layer `h128 d1` | `0.4104` | `0.4104` |
| Last-layer `h256 d1` | `0.4195` | `0.4552` |
| Last-layer `h256 d2` | `0.4302` | `0.4359` |

### Balanced-Binary Threshold Behavior For The Recommended Surface

Held fixed to ensemble `h256 d1` at the frozen `best_loss` checkpoint.

| Dataset | Test prevalence | Positive precision | Positive recall | Positive `F1` | Macro `F1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.2143` | `1.0000` | `0.3452` | `0.6069` |
| `AIME` | `0.5833` | `0.9048` | `0.4762` | `0.5639` | `0.6143` |
| `MATH-500` | `0.0400` | `0.0539` | `0.6667` | `0.0996` | `0.4391` |
| `MMLU-Pro` | `0.0125` | `0.1249` | `1.0000` | `0.2176` | `0.5746` |
| `LiveCodeBench` | `0.3375` | `0.6143` | `0.6543` | `0.5138` | `0.5809` |

### Balanced-Binary Follow-Up Interpretation

- What is now proved:
  - for the per-layer `ensemble`, width is the useful change and added depth hurts
  - `ensemble h256 d1` is the best single global binary surface under the frozen `best_loss` rule
  - `ensemble h256 d1` also stays ahead of `h256 d2` under `best_rank`, so the main ensemble choice is not just a checkpoint-selection accident
  - `ensemble h256 d1` still beats the prompt-length baseline on all five datasets by `PR-AUC`
- What did not stay stable:
  - the `last_layer` ranking is not robust in the same way
  - under `best_loss`, `last_layer h256 d2` is slightly ahead of `h256 d1`
  - under `best_rank`, `last_layer h256 d1` is ahead of `h256 d2`
- Practical consequence:
  - the robust tuning conclusion is about the ensemble, not about promoting a separate last-layer champion
  - threshold metrics on the natural test split remain recall-heavy on rare-positive datasets, so `PR-AUC` should stay primary and threshold metrics should stay diagnostic
  - the next honest tuning step is layer subset / view selection on the same balanced binary data, not another vague capacity increase

## What This Run Does And Does Not Show

- This run does show
  - the locked pair can be trained end-to-end on the full saved surface under one frozen contract
  - `majority_s_0.5` is the cleaner finished activation-lift head today
  - `mean_relative_length` is usable as a screening score, but only with a mixed cross-dataset read
- This run does not show
  - that `Spearman` should be the primary regression metric
  - that `mean_relative_length` beats prompt length uniformly
  - that `majority_s_0.5` is a pure loop label
  - that `p_loop` should be reopened as the default train target after the fact

## Relation To The Earlier Probe Surface

- The March target-choice bundle was not using this exact contract.
- Continuous target-choice runs (`mean_relative_length` and `p_loop`)
  - used prompt-disjoint natural train/test splits with no balancing.
  - were typically reported from `best_rank`, not the frozen `best_loss` contract used here.
- Older March `majority_s_0.5` pilot/control runs
  - were also mostly natural-train rather than explicitly balanced.
  - saved train prevalences in the cross-dataset bundle were `GPQA 0.0569`, `AIME 0.5000`, `MATH-500 0.0550`, `MMLU-Pro 0.0109`, and `LiveCodeBench 0.2734`.
- So if the question is "did we at least balance the train split before target switching?", the answer is: not on the old March probe/control bundle except where the underlying split happened to land near balanced. The explicit balanced-train contract starts with the locked April binary relabel path used in this report.

## Recommendation

- Default finished surface from this run
  - `majority_s_0.5`
- Default binary probe family after the balanced follow-up
  - ensemble `h256 d1`
- How to report the continuous head
  - if the use is screening high-risk prompts, report `top_20p_capture` first, `top_10p_capture` second if tighter budget matters, `RMSE` as calibration context, and `Spearman` only as a tertiary diagnostic
- Most honest one-line summary
  - on the frozen locked run, the binary head is a `5 / 5` prompt-length-baseline win by `PR-AUC`, while the continuous head is a `3 / 5` win by screening capture and only a `2 / 5` win by calibration error
  - on the balanced-binary probe follow-up, the better global ensemble default is `h256 d1`, not deeper `h256 d2`

## Artifact Bundle

Copied ledger for this run:
- `outputs/weeks/2026-W14/prompt_profile_full_train_locked_pair_20260404/remote_summary/`
- `outputs/weeks/2026-W14/prompt_profile_full_train_locked_pair_20260404/regression_summary.csv`
- `outputs/weeks/2026-W14/prompt_profile_full_train_locked_pair_20260404/binary_summary.csv`
- `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/capacity_comparison_summary.json`
- `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/capacity_comparison_metrics.csv`
- `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/prompt_profile_binary_capacity_controls_20260404.pdf`
