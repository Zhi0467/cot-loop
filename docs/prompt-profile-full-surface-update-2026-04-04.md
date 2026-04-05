# Prompt-Profile Full Surface Update: Locked Pair Plus Balanced Binary Capacity Controls

Last updated: 2026-04-05 00:54 UTC

## Executive Summary

- This report puts the whole April surface back in one place.
  - locked full-train run: Slurm `2043`
  - balanced binary capacity reruns: Slurm `2107` and `2108`
- Fresh regression verification:
  - Slurm `2215` reran the regression lane only on the current branch with the natural train/test split and natural sampler.
  - That rerun reproduced the original regression ledger exactly, so the regression section below is now both the original April result and a verified current-branch reproduction.
  - Use `docs/prompt-profile-natural-regression-rerun-2026-04-05.md` for the regression-only rerun note and artifact bundle.
- The regression lane and the balanced binary rerun are different objects.
  - `mean_relative_length` comes from the locked full-train run and keeps the natural prompt-disjoint train/test split.
  - `majority_s_0.5` keeps test natural but downsample-balances train to `50/50`.
  - the follow-up retrains only changed binary probe capacity; they did not rerun or rebalance the continuous target.
- Regression read on the frozen `best_loss` checkpoint:
  - if the use is screening degenerate prompts, the main metric is `top_20p_capture`, not `Spearman`
  - ensemble beats the train-fit prompt-length baseline on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`
  - ensemble still loses to the prompt-length baseline on `AIME` and `MATH-500`
- Binary read on the locked run:
  - ensemble `PR-AUC` beats the prompt-length baseline on all five datasets
- Binary capacity-control read on the same saved April binary data:
  - for `ensemble`, width is the useful change and extra depth hurts
  - for `last_layer`, extra depth helps modestly on top of width
  - if one single global binary ensemble surface is needed today, it should be `h256 d1`, not `h256 d2`
- Secondary sanity check on that binary choice:
  - under `best_rank`, the ensemble ordering stays the same (`h256 d1 0.539 > h256 d2 0.522 > h128 d1 0.474`)
  - the `last_layer` ordering does not stay fixed across checkpoint rules, so that view should remain a cheap control rather than the promoted tuning target
- Threshold caveat on the balanced binary reruns:
  - train is balanced but test is natural
  - rare-positive datasets therefore stay recall-heavy at the train-fit threshold
  - keep `PR-AUC` primary and treat threshold metrics as behavior diagnostics only
- Most honest combined summary:
  - the cleaner deployment-facing head is still `majority_s_0.5`
  - the regression lane is still mixed and should be reported as a screening score, not as a blanket win over prompt length

## Exact Objects

This report combines two April result surfaces that were split across separate PDFs in-thread.

### Object A: Locked full-train run

- Slurm job: `2043`
- Runtime: `2026-04-03 23:57 UTC` to `2026-04-04 00:10 UTC`
- Targets:
  - regression: `mean_relative_length`
  - binary: `majority_s_0.5`
- Views:
  - `ensemble`
  - `last_layer`
- Fixed reporting rule:
  - keep `best_loss` as the headline checkpoint
  - keep `best_rank` diagnostic-only

### Object B: Balanced binary capacity reruns

- Slurm jobs:
  - `2107`: corrected five-dataset `h256 d2` retrain
  - `2108`: width-only `h256 d1` control
- Runtime:
  - `2107`: `2026-04-04 05:57:19 UTC` to `2026-04-04 06:04:55 UTC`
  - `2108`: `2026-04-04 06:13:02 UTC` to `2026-04-04 06:22:07 UTC`
- Fixed object:
  - same saved prompt-prefill activations as the April locked run
  - same binary target `majority_s_0.5`
  - same train-balanced / test-natural split contract
  - same seeds `0,1,2`
  - same optimizer settings
- Only intentional change:
  - probe capacity for the binary head

### Why Regression Is Not "Balanced"

- `mean_relative_length` is a continuous target, so the April locked run keeps the natural prompt-disjoint split.
- The explicit train-balanced / test-natural contract applies to the binary relabel `majority_s_0.5`.
- So the follow-up capacity reruns are binary-only by design. They do not replace the regression lane; they sit next to it.

## Frozen Dataset Surface

| Dataset | Train prompts | Test prompts | Task kind | Prompt field |
| --- | ---: | ---: | --- | --- |
| `GPQA` | `158` | `40` | `multiple_choice_gpqa` | `Question` |
| `AIME` | `48` | `12` | `math_freeform` | `question` |
| `MATH-500` | `400` | `100` | `math_freeform` | `problem` |
| `MMLU-Pro` | `640` | `160` | `multiple_choice_mmlupro` | `problem` |
| `LiveCodeBench` | `640` | `160` | `livecodebench_codegen` | `problem` |

Common model/policy object:

- model: `Qwen/Qwen3-1.7B`
- `temperature=0.2`
- `num_generations=4`
- `max_tokens=30000`
- feature surface: prompt-prefill activations only
- loop detector: `n=30`, `k=20`

## Definitions

### Targets

- `mean_relative_length`
  - for prompt `i`, this is the repeated-rollout mean of `L_r / E`, where `L_r` is rollout length and `E` is the prompt's effective token budget
- `majority_s_0.5`
  - for prompt `i`, this is `1` if a strict majority of repeated rollouts satisfy `L_r / E >= 0.5`, else `0`

### Views

- regression `ensemble`
  - one MLP per layer with `mean_prob` aggregation
- binary `ensemble`
  - one MLP per layer across all `28` saved layers with `vote_fraction` aggregation
- `last_layer`
  - the same probe family restricted to the final layer only

### Probe Families For The Binary Reruns

- default April binary probe: `hidden_dim=128`, `depth=1`, `dropout=0.1`
- width-only control: `hidden_dim=256`, `depth=1`, `dropout=0.1`
- width+depth control: `hidden_dim=256`, `depth=2`, `dropout=0.1`
- optimizer settings kept fixed:
  - `epochs=15`
  - `lr=1e-4`
  - `weight_decay=0.05`

### Score, Capture, And Mass

- regression `score`
  - each held-out prompt gets one scalar score `s_i = yhat_i`
- binary `score`
  - each held-out prompt gets one scalar positive score `s_i = shat_i`
- `top_10p_capture` / `top_20p_capture`
  - sort held-out prompts by descending regression score
  - keep the top `ceil(0.1 n)` or `ceil(0.2 n)`
  - compute `sum_{i in kept} y_i / sum_i y_i`
  - here `y_i` is the realized `mean_relative_length` for that same held-out prompt
- "mass"
  - total realized relative-length weight across prompts
  - not a prompt count
  - not a binary positive count

Concrete example:

- `GPQA` test has `40` prompts
- `top_20p_capture` keeps the top `8` prompts by predicted `mean_relative_length`
- ensemble `0.247` means those `8` prompts contain `24.7%` of the total realized relative-length mass on that `40`-prompt test split

### Metadata Baseline

- allowed inputs:
  - `prompt_token_count`
  - `effective_max_tokens`
- excluded inputs:
  - no activations
  - no prompt text
  - no dataset identity
  - no architecture features
  - no generated output features
- regression baseline:
  - train-fit standardized linear regression, evaluated once on held-out test
- binary baseline:
  - train-fit 1D score rule, with direction chosen by train `PR-AUC` and threshold chosen by train macro-F1 / positive-F1 / accuracy, evaluated once on held-out test
- fixed-run nuance:
  - `effective_max_tokens=30000` is constant here, so prompt length is the only nontrivial metadata feature on this run

## Split And Balancing Contract

### Regression `mean_relative_length`

- train: natural
- test: natural
- prompt-disjoint split stays fixed

### Binary `majority_s_0.5`

- relabel from the same prompts with `--balance-train downsample --balance-test none`
- train prevalence is therefore exactly `0.5` on every dataset
- test prevalence stays natural:
  - `GPQA 0.0500`
  - `AIME 0.5833`
  - `MATH-500 0.0400`
  - `MMLU-Pro 0.0125`
  - `LiveCodeBench 0.3375`

## Regression Results: Locked Full-Train `mean_relative_length`

This lane comes only from the locked full-train run. The binary reruns did not change it.

### Screening Table: `top_10p_capture`

| Dataset | Ensemble | Last-layer | Prompt-length baseline |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.132 +/- 0.019` | `0.089 +/- 0.006` | `0.070` |
| `AIME` | `0.204 +/- 0.015` | `0.221 +/- 0.004` | `0.223` |
| `MATH-500` | `0.129 +/- 0.033` | `0.117 +/- 0.061` | `0.151` |
| `MMLU-Pro` | `0.209 +/- 0.022` | `0.094 +/- 0.062` | `0.169` |
| `LiveCodeBench` | `0.170 +/- 0.004` | `0.156 +/- 0.000` | `0.137` |

### Screening Table: `top_20p_capture`

| Dataset | Ensemble | Last-layer | Prompt-length baseline |
| --- | ---: | ---: | ---: |
| `GPQA` | `0.247 +/- 0.018` | `0.216 +/- 0.039` | `0.168` |
| `AIME` | `0.305 +/- 0.017` | `0.321 +/- 0.020` | `0.306` |
| `MATH-500` | `0.262 +/- 0.058` | `0.243 +/- 0.094` | `0.290` |
| `MMLU-Pro` | `0.355 +/- 0.025` | `0.185 +/- 0.098` | `0.292` |
| `LiveCodeBench` | `0.340 +/- 0.005` | `0.317 +/- 0.010` | `0.248` |

### Calibration And Diagnostic Table

| Dataset | Ensemble `RMSE` | Last-layer `RMSE` | Prompt-length `RMSE` | Ensemble `Spearman` | Last-layer `Spearman` | Prompt-length `Spearman` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.160 +/- 0.002` | `0.164 +/- 0.003` | `0.165` | `0.419 +/- 0.045` | `0.175 +/- 0.225` | `0.116` |
| `AIME` | `0.183 +/- 0.006` | `0.189 +/- 0.004` | `0.176` | `0.639 +/- 0.090` | `0.483 +/- 0.138` | `0.676` |
| `MATH-500` | `0.162 +/- 0.002` | `0.169 +/- 0.018` | `0.154` | `0.351 +/- 0.199` | `0.158 +/- 0.426` | `0.461` |
| `MMLU-Pro` | `0.133 +/- 0.000` | `0.138 +/- 0.005` | `0.126` | `0.348 +/- 0.034` | `-0.019 +/- 0.293` | `0.393` |
| `LiveCodeBench` | `0.248 +/- 0.018` | `0.242 +/- 0.011` | `0.279` | `0.805 +/- 0.002` | `0.784 +/- 0.006` | `0.405` |

### Regression Interpretation

- screening read, ensemble versus prompt length:
  - wins: `GPQA`, `MMLU-Pro`, `LiveCodeBench`
  - losses: `AIME`, `MATH-500`
- calibration read, ensemble versus prompt length:
  - wins: `GPQA`, `LiveCodeBench`
  - losses: `AIME`, `MATH-500`, `MMLU-Pro`
- ensemble versus `last_layer`:
  - on `top_20p_capture`, ensemble wins `4 / 5`
  - on `RMSE`, ensemble also wins `4 / 5`
  - the exception flips by object on `LiveCodeBench`
- practical read:
  - `mean_relative_length` is usable as a screening score
  - it is not a uniform activation-lift win over prompt length
  - `Spearman` should stay diagnostic only

## Binary Results: Locked Full-Train `majority_s_0.5`

This is the default April binary head before the capacity controls.

### Ranking Table: Locked Run

| Dataset | Ensemble `PR-AUC` | Last-layer `PR-AUC` | Prompt-length baseline `PR-AUC` | Test prevalence |
| --- | ---: | ---: | ---: | ---: |
| `GPQA` | `0.420 +/- 0.029` | `0.230 +/- 0.278` | `0.066` | `0.050` |
| `AIME` | `0.922 +/- 0.024` | `0.904 +/- 0.028` | `0.898` | `0.583` |
| `MATH-500` | `0.135 +/- 0.005` | `0.151 +/- 0.071` | `0.100` | `0.040` |
| `MMLU-Pro` | `0.184 +/- 0.037` | `0.056 +/- 0.012` | `0.110` | `0.013` |
| `LiveCodeBench` | `0.711 +/- 0.036` | `0.712 +/- 0.023` | `0.576` | `0.338` |

### Threshold Table: Locked Ensemble Versus Prompt-Length Baseline

| Dataset | Ensemble precision | Ensemble recall | Ensemble macro-F1 | Prompt precision | Prompt recall | Prompt macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.110 +/- 0.050` | `1.000 +/- 0.000` | `0.382 +/- 0.244` | `0.038` | `0.500` | `0.286` |
| `AIME` | `0.926 +/- 0.128` | `0.762 +/- 0.218` | `0.798 +/- 0.044` | `0.800` | `0.571` | `0.667` |
| `MATH-500` | `0.091 +/- 0.020` | `1.000 +/- 0.000` | `0.444 +/- 0.060` | `0.079` | `0.750` | `0.458` |
| `MMLU-Pro` | `0.117 +/- 0.073` | `0.833 +/- 0.289` | `0.562 +/- 0.067` | `0.029` | `1.000` | `0.397` |
| `LiveCodeBench` | `0.609 +/- 0.046` | `0.827 +/- 0.091` | `0.747 +/- 0.014` | `0.494` | `0.722` | `0.646` |

### Locked Binary Interpretation

- ensemble `PR-AUC` beats the prompt-length baseline on all five datasets
- the cleanest wins are `AIME`, `MMLU-Pro`, and `LiveCodeBench`
- the rare-positive datasets still have unstable threshold metrics, so `PR-AUC` remains the main ranking object

## Balanced Binary Capacity Controls On The Same April Binary Data

This section is the follow-up rerun surface Wangzhi asked for after the train-balanced / test-natural contract was made explicit.

### Probe Families Compared

| Surface | Hidden dim | Depth | View definition |
| --- | ---: | ---: | --- |
| Default `h128 d1` | `128` | `1` | same April default |
| Width-only `h256 d1` | `256` | `1` | same data, wider MLP |
| Width+depth `h256 d2` | `256` | `2` | same data, wider and deeper MLP |

### Ranking Table: Test `PR-AUC`

| Dataset | Test prevalence | Prompt length | Ensemble `h128 d1` | Ensemble `h256 d1` | Ensemble `h256 d2` | Last-layer `h128 d1` | Last-layer `h256 d1` | Last-layer `h256 d2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.0657` | `0.4198` | `0.5833` | `0.5025` | `0.2298` | `0.3071` | `0.3176` |
| `AIME` | `0.5833` | `0.8976` | `0.9218` | `0.9120` | `0.9090` | `0.9043` | `0.8436` | `0.8479` |
| `MATH-500` | `0.0400` | `0.1002` | `0.1354` | `0.1663` | `0.1596` | `0.1506` | `0.1315` | `0.1322` |
| `MMLU-Pro` | `0.0125` | `0.1104` | `0.1837` | `0.2116` | `0.2023` | `0.0559` | `0.0701` | `0.0959` |
| `LiveCodeBench` | `0.3375` | `0.5760` | `0.7111` | `0.7143` | `0.6865` | `0.7116` | `0.7452` | `0.7575` |

### Cross-Dataset Mean `PR-AUC`

| Surface | Mean test `PR-AUC` |
| --- | ---: |
| Prompt length baseline | `0.3500` |
| Ensemble `h128 d1` | `0.4744` |
| Ensemble `h256 d1` | `0.5175` |
| Ensemble `h256 d2` | `0.4920` |
| Last-layer `h128 d1` | `0.4104` |
| Last-layer `h256 d1` | `0.4195` |
| Last-layer `h256 d2` | `0.4302` |

### Secondary Check: Mean Test `PR-AUC` At `best_rank`

| Surface | Mean test `PR-AUC` |
| --- | ---: |
| Prompt length baseline | `0.3500` |
| Ensemble `h128 d1` | `0.4744` |
| Ensemble `h256 d1` | `0.5385` |
| Ensemble `h256 d2` | `0.5215` |
| Last-layer `h128 d1` | `0.4104` |
| Last-layer `h256 d1` | `0.4552` |
| Last-layer `h256 d2` | `0.4359` |

This secondary checkpoint rule keeps the global ensemble recommendation intact: `h256 d1` still stays ahead of `h256 d2`. The `last_layer` ranking does move, which is another reason not to treat that control view as the main tuning conclusion.

### Threshold Behavior On Natural Test For The Recommended Surface

Held fixed to `ensemble h256 d1` at the frozen `best_loss` checkpoint.

| Dataset | Test prevalence | Positive precision | Positive recall | Positive `F1` | Macro `F1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.2143` | `1.0000` | `0.3452` | `0.6069` |
| `AIME` | `0.5833` | `0.9048` | `0.4762` | `0.5639` | `0.6143` |
| `MATH-500` | `0.0400` | `0.0539` | `0.6667` | `0.0996` | `0.4391` |
| `MMLU-Pro` | `0.0125` | `0.1249` | `1.0000` | `0.2176` | `0.5746` |
| `LiveCodeBench` | `0.3375` | `0.6143` | `0.6543` | `0.5138` | `0.5809` |

Because training is balanced but evaluation is natural, these threshold metrics stay recall-heavy on rare-positive datasets. That is expected behavior on this object, not evidence that the operating point is suddenly well-calibrated. Use them as a threshold-shape check, not as the headline comparison.

### Capacity-Control Interpretation

- for `ensemble`:
  - width helps relative to the default
  - added depth hurts relative to width-only on all five datasets
  - best global ensemble surface is `h256 d1`
- for `last_layer`:
  - width is mixed
  - added depth helps modestly on top of width on all five datasets
  - best global last-layer surface is `h256 d2`
- important nuance:
  - the `last_layer` ordering is not stable across checkpoint rules
  - `h256 d2` wins at `best_loss`, but `h256 d1` wins at `best_rank`
  - the robust tuning conclusion is therefore about the ensemble, not about promoting a separate last-layer champion
- so the earlier `h256 d2` result mixed one good change with one bad one:
  - good change for ensemble: width
  - bad change for ensemble: extra depth

## Relation To The Earlier March Probe Surface

- The March target-choice bundle did not use this exact April reporting contract.
- Continuous target-choice runs were usually reported from `best_rank`, not the frozen `best_loss` checkpoint used here.
- The older March `majority_s_0.5` probe/control bundle was not explicitly train-balanced in the way the April binary relabel is.
- Saved train prevalences in that older March bundle were:
  - `GPQA 0.0569`
  - `AIME 0.5000`
  - `MATH-500 0.0550`
  - `MMLU-Pro 0.0109`
  - `LiveCodeBench 0.2734`
- The inherited default probe family was the same `h128 d1` MLP family.
- So the April balanced binary reruns are a fresh object, not a report rewrite on the March target-choice bundle.

## Combined Recommendation

- If the project needs one deployment-facing prompt-risk screen today:
  - lead with binary `majority_s_0.5`
  - use `ensemble h256 d1` as the current single global ensemble default
- If the project keeps a cheap comparison control:
  - keep `last_layer h256 d2` only as the frozen-`best_loss` control
  - do not present it as a stable tuning winner across checkpoint rules
- If the regression lane is reported:
  - use `top_20p_capture` first
  - use `RMSE` as calibration context
  - keep `Spearman` diagnostic only
- The next honest tuning step is not another vague size increase.
  - the open question now is layer subset, not whether to keep making every per-layer MLP deeper

## Artifact Bundle

- Combined note: `docs/prompt-profile-full-surface-update-2026-04-04.md`
- Combined PDF: `outputs/prompt_profile_full_surface_update_20260404/prompt_profile_full_surface_update_20260404.pdf`
- Combined TeX: `outputs/prompt_profile_full_surface_update_20260404/prompt_profile_full_surface_update_20260404.tex`
- Underlying locked full-train bundle:
  - `outputs/prompt_profile_full_train_locked_pair_20260404/`
- Underlying binary capacity bundle:
  - `outputs/prompt_profile_binary_capacity_controls_20260404/`
