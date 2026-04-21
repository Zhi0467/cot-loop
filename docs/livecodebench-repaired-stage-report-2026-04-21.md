# LiveCodeBench Repaired Stage Report

Last updated: 2026-04-21 21:41 UTC

## Bottom Line

- The repaired `LiveCodeBench` stage is now frozen as a report-style deliverable rather than only a thread summary.
- On the repaired prompt-level `majority_s_0.5` object, current single-seed layerwise RFM is real and competitive:
  - it is clearly above the prompt-only and activation-linear baselines;
  - it is roughly tied with activation MLP last-layer under the `best_rank` checkpoint rule;
  - it is slightly below that same MLP row under the `best_loss` checkpoint rule.
- The exported benchmark-local vector bundle is stable enough to treat as a genuine stage-2 object:
  - all `28` layers clear mean bootstrap cosine `>= 0.781`;
  - the weakest `95%` low bound is `0.693`;
  - late layers `23-26` are the most coherent.
- The first larger benchmark-local spherical steering table is negative:
  - all four conditions stay at `0 / 32` `pass@1`;
  - `no_steer` has loop fraction `0.03125`;
  - `minus_v_spherical`, `plus_v_spherical`, and `random_spherical` are all worse on loop fraction.
  - but this table is only the bounded prefill-last-token spherical sub-lane, not the full block-specific linear+spherical figure contract.

## Object

- This uses the same prompt-level label family as the other probe surfaces:
  - `majority_s_0.5`
  - prompt is positive iff a strict majority of its `4` saved rollouts have `relative_length > 0.5`
- The March archive still stores an archive-time tail bit at `0.9`, so this stage recomputes the stage label from saved rollout lengths rather than trusting the archive bit directly.
- Frozen `LiveCodeBench` repaired split:
  - source train pool: `640` prompts
  - fit-train: `280` prompts, `140` positives
  - validation: `128` prompts, `35` positives
  - test: `160` prompts, `54` positives
- Feature view:
  - `last_token_all_layers_stack_final`
  - sample shape `[28, 2048]`

## Detector Read

- RFM detector artifact root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
- Selected row:
  - layer `27`
  - bandwidth `100`
  - validation `PR-AUC 0.6555`
  - test `PR-AUC 0.7055`
  - test `ROC-AUC 0.8590`
- Repaired prompt-only baselines:
  - `prompt_length`: test `PR-AUC 0.5771`, `ROC-AUC 0.7201`
  - `prompt_shape_linear`: test `PR-AUC 0.5871`, `ROC-AUC 0.7290`
  - `prompt_shape_tree`: test `PR-AUC 0.3732`, `ROC-AUC 0.5751`
- Repaired activation baselines:
  - linear `last_layer`: mean test `PR-AUC 0.4163`, `ROC-AUC 0.5486`
  - linear `ensemble`: mean test `PR-AUC 0.5698`, `ROC-AUC 0.7475`
  - MLP `h256 d1 ensemble` (`best_rank`): mean test `PR-AUC 0.6637`, `ROC-AUC 0.8458`
  - MLP `h256 d1 last_layer` (`best_rank`): mean test `PR-AUC 0.7055`, `ROC-AUC 0.8435`
  - MLP `h256 d1 last_layer` (`best_loss`): mean test `PR-AUC 0.7147`, `ROC-AUC 0.8489`

## Direction Quality Read

- Vector-export summary root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/vector_exports/summary.json`
- Fixed-hyperparameter direction bootstrap:
  - `100` replay fits per layer
  - all `28` layers clear mean cosine `>= 0.781`
  - weakest low bound: `0.693`
- Cosine definition:
  - fix one reference vector by fitting the chosen RFM once on the original full fit-train split for that layer;
  - then, for each bootstrap replay, refit the same RFM on a with-replacement resample of those fit-train prompts;
  - extract that replay's signed normalized vector from the top eigenvector of the symmetrized final `M`;
  - choose the sign by fit-train `PR-AUC` then `ROC-AUC`;
  - then measure cosine between that bootstrap-refit vector and the fixed reference exported signed normalized vector for that same layer.
- Most stable layers:
  - `23`, `24`, `25`, `26`
- Selected detector layer `27` is still usable rather than pathological:
  - mean cosine `0.786`
  - low bound `0.734`
  - held-out 1D projection test `PR-AUC 0.7150`

## Steering Read

- Steering artifact root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_controls32_t0p3_seed0_20260421/`
- Fixed protocol:
  - prompt-prefill last-token hook surface
  - spherical steering
  - `t = 0.3`
  - same `32` held-out prompt IDs across all conditions
  - bounded pilot decode cap `max_new_tokens = 1024`
- Final `32`-prompt table:
  - `no_steer`: `0 / 32` `pass@1`, loop fraction `0.03125`
  - `minus_v_spherical`: `0 / 32`, loop fraction `0.28125`
  - `plus_v_spherical`: `0 / 32`, loop fraction `0.125`
  - `random_spherical`: `0 / 32`, loop fraction `0.34375`
- Current honest read:
  - there is no accuracy movement anywhere;
  - every steered condition is worse than baseline on loop fraction;
  - this closes the first larger bounded prefill-last-token spherical control table as a negative result, not an in-flight pilot;
  - it does not yet answer the full figure contract because linear steering, all-prefill-token steering, and the full decode budget were not part of this run.

## Deliverables

- Report bundle:
  - `../outputs/livecodebench_repaired_stage_report_apr21/`
- Main PDF:
  - `../outputs/livecodebench_repaired_stage_report_apr21/livecodebench_repaired_stage_report_apr21.pdf`
- Machine-readable summary:
  - `../outputs/livecodebench_repaired_stage_report_apr21/livecodebench_stage_summary.json`
- Figures:
  - `../outputs/livecodebench_repaired_stage_report_apr21/detector_comparison.png`
  - `../outputs/livecodebench_repaired_stage_report_apr21/direction_stability.png`
  - `../outputs/livecodebench_repaired_stage_report_apr21/steering_controls.png`

## What Is Still Open

- If the detector ranking itself needs to be frozen more tightly, the remaining honest follow-up is matching multiseed or split-seed RFM sensitivity, not more debate about the repaired object definition.
- If the steering lane continues on `LiveCodeBench`, it needs a new control or a new hypothesis:
  - `shuffled_label_spherical`
  - a different hook surface
  - or a direction-selection change grounded in the stability surface
- If the project returns to cross-benchmark transfer, it should do so explicitly after deciding whether the other repaired benchmark objects are science-worthy or only provenance-worthy at their current positive counts.
