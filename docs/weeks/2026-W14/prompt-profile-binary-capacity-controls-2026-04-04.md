# Prompt-Profile Binary Capacity Controls On Balanced `majority_s_0.5`

Last updated: 2026-04-04 06:31 UTC

## Executive Summary

- This note compares three probe families on the same saved April binary data:
  - default `h128 d1`
  - width-only control `h256 d1`
  - width+depth control `h256 d2`
- Exact runtime:
  - `2106`: `LiveCodeBench`-only pilot caused by the first dataset-list bug
  - `2107`: corrected five-dataset `h256 d2` rerun, `2026-04-04 05:57:19 UTC` to `2026-04-04 06:04:55 UTC`
  - `2108`: width-only `h256 d1` control, `2026-04-04 06:13:02 UTC` to `2026-04-04 06:22:07 UTC`
- Fixed object:
  - same saved prompt-prefill activations
  - same binary target `majority_s_0.5`
  - same train-balanced / test-natural split contract
  - same seeds `0,1,2`
  - same `epochs=15`, `batch_size=256`, `lr=1e-4`, `weight_decay=0.05`, `dropout=0.1`
- Main read on held-out `PR-AUC`:
  - for the per-layer `ensemble`, width is the useful change and added depth hurts
  - for `last_layer`, added depth helps modestly on top of width
- Cross-dataset mean test `PR-AUC`:
  - prompt length baseline: `0.350`
  - ensemble `h128 d1`: `0.474`
  - ensemble `h256 d1`: `0.518`
  - ensemble `h256 d2`: `0.492`
  - last-layer `h128 d1`: `0.410`
  - last-layer `h256 d1`: `0.420`
  - last-layer `h256 d2`: `0.430`
- Best single global surface from this control sweep:
  - `ensemble h256 d1`
- Secondary sanity check:
  - under `best_rank`, the ensemble ordering stays the same (`h256 d1 0.539 > h256 d2 0.522 > h128 d1 0.474`)
  - the `last_layer` ordering does not stay fixed across checkpoint rules, so that view should remain a cheap control rather than the promoted tuning target
- Threshold caveat:
  - on the natural test split, rare-positive datasets still show recall-heavy operating points after train balancing
  - keep `PR-AUC` primary and treat threshold metrics as behavior diagnostics only
- Limitation:
  - this still does not answer which layers to keep
  - it only separates width from depth while keeping the same two view definitions (`ensemble` over all `28` layers, or final `last_layer`)

## Exact Question

On the saved balanced binary object, did the earlier `h256 d2` shifts come from width, depth, or only the combination?

## Run Contract

- Source data root: `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404`
- Binary target: `majority_s_0.5`
- Train balance: downsample to `50/50`
- Test balance: natural
- Feature surface:
  - stacked prompt-prefill activations with sample shape `[28, 2048]`
- Views:
  - `ensemble`: one MLP per layer across all `28` saved layers, aggregated by `vote_fraction`
  - `last_layer`: final-layer-only MLP
- Probe families:
  - `h128 d1`: `hidden_dim=128`, `depth=1`
  - `h256 d1`: `hidden_dim=256`, `depth=1`
  - `h256 d2`: `hidden_dim=256`, `depth=2`
- Optimization held fixed:
  - `epochs=15`, `batch_size=256`, `lr=1e-4`, `weight_decay=0.05`, `dropout=0.1`

## Ranking Results: Test `PR-AUC`

| Dataset | Test prevalence | Prompt length | Ensemble `h128 d1` | Ensemble `h256 d1` | Ensemble `h256 d2` | Last-layer `h128 d1` | Last-layer `h256 d1` | Last-layer `h256 d2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.0657` | `0.4198` | `0.5833` | `0.5025` | `0.2298` | `0.3071` | `0.3176` |
| `AIME` | `0.5833` | `0.8976` | `0.9218` | `0.9120` | `0.9090` | `0.9043` | `0.8436` | `0.8479` |
| `MATH-500` | `0.0400` | `0.1002` | `0.1354` | `0.1663` | `0.1596` | `0.1506` | `0.1315` | `0.1322` |
| `MMLU-Pro` | `0.0125` | `0.1104` | `0.1837` | `0.2116` | `0.2023` | `0.0559` | `0.0701` | `0.0959` |
| `LiveCodeBench` | `0.3375` | `0.5760` | `0.7111` | `0.7143` | `0.6865` | `0.7116` | `0.7452` | `0.7575` |

## Cross-Dataset Mean `PR-AUC`

| Surface | Mean test `PR-AUC` |
| --- | ---: |
| Prompt length baseline | `0.3500` |
| Ensemble `h128 d1` | `0.4744` |
| Ensemble `h256 d1` | `0.5175` |
| Ensemble `h256 d2` | `0.4920` |
| Last-layer `h128 d1` | `0.4104` |
| Last-layer `h256 d1` | `0.4195` |
| Last-layer `h256 d2` | `0.4302` |

## Secondary Check: `best_rank` Mean Test `PR-AUC`

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

## Threshold Behavior On Natural Test For The Recommended Surface

Held fixed to `ensemble h256 d1` at the frozen `best_loss` checkpoint.

| Dataset | Test prevalence | Positive precision | Positive recall | Positive `F1` | Macro `F1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.0500` | `0.2143` | `1.0000` | `0.3452` | `0.6069` |
| `AIME` | `0.5833` | `0.9048` | `0.4762` | `0.5639` | `0.6143` |
| `MATH-500` | `0.0400` | `0.0539` | `0.6667` | `0.0996` | `0.4391` |
| `MMLU-Pro` | `0.0125` | `0.1249` | `1.0000` | `0.2176` | `0.5746` |
| `LiveCodeBench` | `0.3375` | `0.6143` | `0.6543` | `0.5138` | `0.5809` |

Because training is balanced but evaluation is natural, these threshold metrics stay recall-heavy on rare-positive datasets. That is expected behavior on this object, not evidence that the operating point is suddenly well-calibrated. Use them as a threshold-shape check, not as the headline comparison.

## What Width And Depth Actually Did

- Ensemble:
  - width helps relative to the default on `GPQA`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`, with only a small drop on `AIME`
  - added depth hurts relative to width-only on all five ensemble slices
  - so the best global ensemble surface is `h256 d1`, not `h256 d2`
- Last-layer:
  - width helps on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`, but hurts on `AIME` and `MATH-500`
  - added depth then improves on top of width on all five last-layer slices, though sometimes only slightly
  - so the best global last-layer surface is `h256 d2`

## Interpretation

- Proven now:
  - `ensemble h256 d1` is the strongest single global surface in this capacity sweep under the frozen `best_loss` rule
  - `ensemble h256 d1` also stays ahead of `h256 d2` under the secondary `best_rank` rule, so the main ensemble choice is not a checkpoint-selection artifact
  - `ensemble h256 d1` still beats the prompt-length baseline on all five datasets by `PR-AUC`
  - `last_layer h256 d2` is the best last-layer variant only under the frozen `best_loss` rule, and even there it is still weaker than `ensemble h256 d1` as a global choice
- Also proven:
  - the earlier `h256 d2` ensemble result was mixing one good change with one bad one
  - the good change was width
  - the bad change, for ensemble, was the extra depth
- Important nuance:
  - the `last_layer` ordering is not stable across checkpoint rules (`h256 d2` wins at `best_loss`, `h256 d1` wins at `best_rank`)
  - so the robust tuning conclusion is about the ensemble, not about promoting a separate last-layer champion
- Important caveat:
  - threshold metrics on the natural test split remain recall-heavy on rare-positive datasets because train balance and test prevalence differ
  - keep `PR-AUC` primary and treat threshold metrics as diagnostic only
  - this still says nothing about whether we should keep all `28` layers in the ensemble
  - the next honest tuning step is a small layer-subset / view sweep on the same balanced binary data, not another vague capacity increase

## Recommendation

- If one single global binary surface is needed today, use `ensemble h256 d1`.
- Keep `PR-AUC` as the main comparison object for this balanced-train / natural-test binary surface.
- Keep `best_loss` as the frozen main checkpoint rule and `best_rank` as a sanity check, not as a reason to reopen the global ensemble choice.
- If the note still needs a cheap `last_layer` control under the same frozen rule, use `h256 d2`, but do not present that as a stable tuning winner.

## Artifact Bundle

- PDF: `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/prompt_profile_binary_capacity_controls_20260404.pdf`
- TeX: `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/prompt_profile_binary_capacity_controls_20260404.tex`
- Summary JSON: `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/capacity_comparison_summary.json`
- Summary CSV: `outputs/weeks/2026-W14/prompt_profile_binary_capacity_controls_20260404/capacity_comparison_metrics.csv`
- Intermediate `h256 d2` note:
  - `docs/weeks/2026-W14/prompt-profile-binary-retrain-h256d2-2026-04-04.md`
  - use this only as the raw `2106` / `2107` rerun record, not as the current recommendation surface
