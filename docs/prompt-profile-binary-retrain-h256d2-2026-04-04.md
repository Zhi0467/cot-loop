# Prompt-Profile Binary Retrain: `h256 d2` On Balanced `majority_s_0.5`

Last updated: 2026-04-04 06:18 UTC

## Executive Summary

- This note covers a new binary-only retrain on the saved April `majority_s_0.5` data. It is not a new target-choice run and it does not overwrite the earlier locked-pair report.
- Exact runtime:
  - pilot `2106`: `2026-04-04 05:54:44 UTC` to `2026-04-04 05:56:55 UTC`
  - corrected all-dataset rerun `2107`: `2026-04-04 05:57:19 UTC` to `2026-04-04 06:04:55 UTC`
- Fixed object:
  - same saved prompt-prefill activations from the April locked run
  - same binary target `majority_s_0.5`
  - same train-balanced / test-natural split contract
  - same optimizer seeds `0,1,2`
  - same `epochs=15`, `lr=1e-4`, `weight_decay=0.05`, `dropout=0.1`
- The only intentional model change was probe capacity:
  - old default: `hidden_dim=128`, `depth=1`
  - retrain: `hidden_dim=256`, `depth=2`
- Main read on held-out `PR-AUC`:
  - `h256 d2` is not a uniform win
  - ensemble improves on `GPQA`, `MATH-500`, and `MMLU-Pro`, but drops on `AIME` and `LiveCodeBench`
  - last-layer improves on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`, but drops on `AIME` and `MATH-500`
- Best current global ranking surface from this rerun:
  - `ensemble h256 d2`
  - mean test `PR-AUC = 0.492`, versus `0.474` for default ensemble, `0.430` for `last_layer h256 d2`, and `0.350` for the prompt-length baseline
- Limitation:
  - this rerun tuned probe capacity only
  - it did not sweep which layers to keep
  - `ensemble` still means one MLP per layer across all `28` saved layers, and `last_layer` still means final layer only

## Exact Question

If we keep the saved April balanced binary data fixed, does moving from the inherited default probe family to a larger `h256 d2` family materially improve the binary `majority_s_0.5` screen?

## Run Contract

- Source root: `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404`
- Retrain root: `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404_binary_h256d2`
- Binary target: `majority_s_0.5`
- Train balance: downsample to `50/50`
- Test balance: natural
- Views:
  - `ensemble`: one MLP per layer, aggregated by `vote_fraction`
  - `last_layer`: final-layer-only MLP
- Important bug note:
  - `2106` only covered `LiveCodeBench`, because repeated `--dataset` flags collapsed to the last dataset only
  - commit `913021c` fixed that bug, and `2107` is the real five-dataset rerun

## What Changed Versus The April Default

| Setting | Default binary run | Retrain |
| --- | --- | --- |
| Hidden dim | `128` | `256` |
| MLP depth | `1` | `2` |
| Dropout | `0.1` | `0.1` |
| Epochs | `15` | `15` |
| Learning rate | `1e-4` | `1e-4` |
| Weight decay | `0.05` | `0.05` |
| Layer set | all `28` layers for ensemble; final layer for `last_layer` | unchanged |
| Data / split | saved April balanced binary data | unchanged |

So this is a capacity comparison, not a new data object and not a layer-subset sweep.

## Ranking Results: Test `PR-AUC`

| Dataset | Test prevalence | Prompt length | Ensemble `h128 d1` | Ensemble `h256 d2` | Last-layer `h128 d1` | Last-layer `h256 d2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.0657` | `0.4198` | `0.5025` | `0.2298` | `0.3176` |
| `AIME` | `0.5833` | `0.8976` | `0.9218` | `0.9090` | `0.9043` | `0.8479` |
| `MATH-500` | `0.0400` | `0.1002` | `0.1354` | `0.1596` | `0.1506` | `0.1322` |
| `MMLU-Pro` | `0.0125` | `0.1104` | `0.1837` | `0.2023` | `0.0559` | `0.0959` |
| `LiveCodeBench` | `0.3375` | `0.5760` | `0.7111` | `0.6865` | `0.7116` | `0.7575` |

## Cross-Dataset Mean `PR-AUC`

| Surface | Mean test `PR-AUC` |
| --- | ---: |
| Prompt length baseline | `0.3500` |
| Default ensemble `h128 d1` | `0.4744` |
| Default `last_layer h128 d1` | `0.4104` |
| Retrain ensemble `h256 d2` | `0.4920` |
| Retrain `last_layer h256 d2` | `0.4302` |

## Retrain Threshold Behavior

These threshold metrics are secondary here because the train split is balanced while the test split is natural. On the rare-positive datasets, high recall / low precision is expected.

| Dataset | View | Precision | Recall | Positive-F1 | Macro-F1 | Selected epochs |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `GPQA` | `ensemble` | `0.1538` | `1.0000` | `0.2667` | `0.5487` | `4;7;5` |
| `GPQA` | `last_layer` | `0.1065` | `1.0000` | `0.1862` | `0.3375` | `6;15;6` |
| `AIME` | `ensemble` | `1.0000` | `0.4762` | `0.6330` | `0.6840` | `4;9;4` |
| `AIME` | `last_layer` | `0.8778` | `0.6667` | `0.7564` | `0.7494` | `12;6;15` |
| `MATH-500` | `ensemble` | `0.0761` | `1.0000` | `0.1414` | `0.4008` | `12;8;8` |
| `MATH-500` | `last_layer` | `0.0759` | `1.0000` | `0.1407` | `0.3842` | `15;7;11` |
| `MMLU-Pro` | `ensemble` | `0.0794` | `1.0000` | `0.1470` | `0.5336` | `12;1;7` |
| `MMLU-Pro` | `last_layer` | `0.0330` | `0.5000` | `0.0611` | `0.4920` | `15;15;3` |
| `LiveCodeBench` | `ensemble` | `0.6766` | `0.5494` | `0.5895` | `0.7057` | `3;3;14` |
| `LiveCodeBench` | `last_layer` | `0.6014` | `0.7901` | `0.6828` | `0.7396` | `15;15;15` |

## Interpretation

- Proven now:
  - `h256 d2` raises the best cross-dataset mean `PR-AUC` for both views
  - `ensemble h256 d2` is the strongest overall ranking surface in this rerun
  - within the retrained family, ensemble beats `last_layer` on `4 / 5` datasets; `LiveCodeBench` is the exception
- Also proven:
  - the effect is not monotone by dataset
  - `AIME` is worse under the larger family for both views on `PR-AUC`
  - `LiveCodeBench` splits the other way: larger `last_layer` is the best surface there, while larger ensemble is worse than the default ensemble
- Important threshold caveat:
  - rare-positive datasets (`GPQA`, `MATH-500`, `MMLU-Pro`) naturally push the retrained classifiers toward high recall and weak precision under the train-fit threshold
  - that is why `PR-AUC` remains the primary comparison object for this rerun
- What this rerun did not answer:
  - whether we should drop early layers, keep only a subset of layers, or tune the layer set per dataset
  - this pass only changed MLP capacity, not the ensemble layer membership

## Recommendation

- For the current balanced binary object, keep `PR-AUC` as the main decision metric.
- If one single global surface is needed today, use `ensemble h256 d2`.
- If we are willing to split by view or by dataset:
  - keep the ensemble family for `GPQA`, `AIME`, `MATH-500`, and `MMLU-Pro`
  - keep `last_layer h256 d2` as the strongest current `LiveCodeBench` binary surface
- The next honest tuning step, if we want more than this, is not another vague “bigger probe” change. It is a small layer-subset / view sweep on the same balanced binary data.

## Artifact Bundle

- Metrics CSV: `outputs/prompt_profile_binary_retrain_h256d2_20260404/binary_retrain_metrics.csv`
- Summary JSON: `outputs/prompt_profile_binary_retrain_h256d2_20260404/binary_retrain_summary.json`
- Slurm logs:
  - `outputs/prompt_profile_binary_retrain_h256d2_20260404/logs/prompt-profile-binary-retrain-2106.out`
  - `outputs/prompt_profile_binary_retrain_h256d2_20260404/logs/prompt-profile-binary-retrain-2107.out`
