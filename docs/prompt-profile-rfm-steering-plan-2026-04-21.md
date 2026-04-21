# Prompt-Profile RFM Steering Stage Plan

Last updated: 2026-04-21 18:05 UTC

## Bottom Line

- The next cot-loop stage is not another prompt-profile archive rebuild and not another trigger-attention cleanup pass.
- It is a benchmark-local RFM detector plus steering stage on the frozen `Qwen/Qwen3-1.7B` prompt-profile surface that already underlies the April prompt-profile notes.
- The collaborator-facing stage benchmark set is now `GPQA`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`.
  - `AIME` stays out of this stage because, on this object, it mostly acts like a prompt-visible workload case rather than the predictor question we care about; that is already the repo's collaborator-facing read in `docs/prompt-profile-unified-report-2026-04-09.md`.
- The attached PDF was directionally right, but the repo reality had drifted in two important ways:
  - upstream PR `#10` is already merged (`2026-04-21 04:35 UTC`, merge commit `2fcb7b7`), so trigger-attention is no longer the live GitHub blocker surface;
  - the real saved activation/archive surface is the March `2026-03-22` to `2026-03-23` prompt-profile bundle reused by the trigger-attention replay, not a vague later April rebuild.

## Frozen Execution Surface

- Model and decode policy:
  - `MODEL_ID=Qwen/Qwen3-1.7B`
  - `TEMPERATURE=0.2`
  - `NUM_GENERATIONS=4`
  - `MAX_TOKENS=30000`
  - `MAX_MODEL_LEN=40960`
- Saved prompt-disjoint archive counts exist for five benchmarks, but the active RFM/steering stage keeps only the retained four-benchmark comparison set:
  - `GPQA`: `158 / 40`
  - `MATH-500`: `400 / 100`
  - `MMLU-Pro`: `640 / 160`
  - `LiveCodeBench`: `640 / 160`
- Saved but intentionally excluded from this collaborator-facing stage:
  - `AIME`: `48 / 12`
- Feature surface:
  - prompt-prefill last-token stacked activations
  - manifest default feature key: `last_token_all_layers_stack_final`
  - per-prompt tensor shape: `[layer, hidden] = [28, 2048]`
- Label for this stage:
  - binary `majority_s_0.5`
  - per prompt, this is `1` when a majority of the saved rollouts exceed `50%` of the generation budget
  - this is a long-rollout / budget-usage risk proxy, not a pure loop label
- Execution reality on the frozen March archives:
  - `GPQA`: train `156/2`, test `40/0` for negative/positive prompts
  - `MATH-500`: train `397/3`, test `100/0`
  - `MMLU-Pro`: train `639/1`, test `160/0`
  - `LiveCodeBench`: train `606/34`, test `152/8`
- Immediate consequence:
  - the frozen archive contract still names the retained four-benchmark set, but on the current March `majority_s_0.5` object only `LiveCodeBench` has a non-degenerate held-out positive slice;
  - `GPQA`, `MATH-500`, and `MMLU-Pro` can still be kept in provenance tables, but they do not currently support a meaningful held-out binary detector comparison without changing the label object or the split policy.
- Exact saved archive roots currently surfaced by the trigger-attention code for the retained stage benchmarks:
  - `GPQA`: `/data/scratch/murphy/outputs/cot-loop-detection/gpqa_mean_relative_from_archive_20260322/data`
  - `MATH-500`: `/data/scratch/murphy/outputs/cot-loop-detection/math_mean_relative_from_archive_20260323`
  - `MMLU-Pro`: `/data/scratch/murphy/outputs/cot-loop-detection/mmlu_mean_relative_from_archive_20260323`
  - `LiveCodeBench`: `/data/scratch/murphy/outputs/cot-loop-detection/livecodebench_mean_relative_from_archive_20260323`
- Saved but excluded from the active stage table:
  - `AIME`: `/data/scratch/murphy/outputs/cot-loop-detection/aime_mean_relative_from_archive_20260322`

## What The Repo Already Has

- Saved stacked-tensor prompt-profile datasets plus manifest metadata:
  - `scripts/build_probe_dataset.py`
  - `scripts/relabel_prompt_profile_dataset.py`
  - `src/loop_probe/dataloader.py`
- Existing prompt-profile training/report surfaces:
  - `scripts/run_prompt_profile_full_train.py`
  - `scripts/summarize_prompt_profile_full_train.py`
  - `scripts/train_metadata_residual_probe.py`
  - `scripts/build_prompt_profile_unified_report.py`
- Existing activation-side linear controls, distinct from the prompt-only metadata baselines:
  - `src/loop_probe/probes/linear_probe.py`
  - `scripts/run_prompt_profile_full_train.py`
  - `docs/prompt-profile-unified-report-2026-04-09.md`
- Existing rollout-stat collector for steering evaluation metrics:
  - `scripts/collect_model_stats.py`
- Important negative fact:
  - the repo now has stage-0 RFM scaffolding:
    - shared retained-benchmark registry
    - emit / validate CLI surfaces
    - machine-readable artifact helpers
    - first node-side validation artifact under `outputs/prompt_profile_rfm_stage0_registry_validation_20260421/`;
  - the repo now also has a live native RFM detector path:
    - `src/loop_probe/rfm.py`
    - `scripts/train_prompt_profile_rfm.py`
    - `slurm/run_prompt_profile_rfm.sbatch`;
  - there is still no live steering runner in current `scripts/`, `src/`, or `slurm/`;
  - matched March-split baseline tooling also now exists:
    - `scripts/materialize_prompt_profile_stage_binary_data.py`
    - `scripts/train_probe.py` with explicit `--train-split` / `--eval-split`
    - `scripts/eval_probe_checkpoint.py` for arbitrary split evaluation;
  - older RFM results still exist only in archived PR2 output artifacts, so the detector and steering parts of this stage remain a new implementation lane rather than a parameter-only follow-up.

## What This Stage Is Trying To Prove

- First, whether a layerwise RFM built on the existing prompt-prefill tensors is competitive with the current MLP surfaces on the same held-out `majority_s_0.5` task.
- Second, whether the exported benchmark-specific RFM directions are stable enough to be treated as steering vectors rather than only as a by-product of a nonlinear classifier.
- Third, whether those benchmark-specific directions can be used for small, controlled spherical steering interventions that reduce long-rollout risk without simply destroying accuracy.
- Fourth, whether one averaged "verbose" vector built from those benchmark-local directions has measurable transfer value on a genuinely different benchmark.
- Detector ranking and steering utility are related, but they are not the same gate for this stage. The steering story should continue even if RFM does not become the single best detector.
- Likewise, a strong RFM detector is not by itself evidence that the exported direction is stable or useful for steering. Direction quality needs its own diagnostics.

This stage is not trying to prove a mechanistic explanation of looping, and it is not yet the right place to claim prompt-shape-controlled causal lift.

## Stage Sequence

### Stage 0: Freeze Provenance And The Four-Benchmark Registry

- Add one committed retained-benchmark registry that names, in one place:
  - archive root
  - prompt field
  - train/test counts
  - feature key
  - label head
- Keep that registry to the active four-benchmark stage surface:
  - `GPQA`
  - `MATH-500`
  - `MMLU-Pro`
  - `LiveCodeBench`
- Fail closed if the archive root, manifest, sample shape, or saved prompt IDs do not match the frozen contract.
- Pull the current hard-coded archive roots out of analysis-only scripts and into that shared registry before any RFM training starts.
- Commit one machine-readable artifact schema for:
  - registry validation records
  - per-benchmark per-layer RFM vector bundles
  - steering-run ledgers
- Keep that schema aligned with `docs/prompt-profile-rfm-artifact-schema-2026-04-21.md` so later experiment lookup does not depend on Slack thread history.

### Stage 1: Add RFM As A Sibling Detector

- Add a native RFM path on top of the existing stacked tensors, rather than re-extracting activations.
- Committed execution surface now exists:
  - `src/loop_probe/rfm.py`
  - `scripts/train_prompt_profile_rfm.py`
  - `slurm/run_prompt_profile_rfm.sbatch`
- Train one RFM per layer on the saved `[N, L, H]` prompt-prefill tensors.
- Use the PDF hyperparameter grid, but keep selection inside the train side only:
  - bandwidth in `{1, 10, 100}`
  - `lambda = 1e-3`
  - `T = 5` fixed
- Default solver is now `solve`, not `cholesky`.
  - The first non-degenerate `LiveCodeBench` full run failed under the original `cholesky` default because the kernel matrix was not positive-definite.
- Emit held-out tables against the existing baseline families on this same binary surface:
  - prompt-only baselines
  - activation-side linear baselines
  - activation-side MLP baselines
  - `PR-AUC`, `ROC-AUC`, prevalence, and threshold diagnostics as secondary context
- Use bootstrap intervals for the detector metrics, especially on smaller held-out sets such as `GPQA`.
- First on-node execution receipts (`2026-04-21`):
  - `GPQA` smoke `2756` proved the full artifact path end to end.
  - `GPQA` full sweep `2757` also finished cleanly, but it confirmed the archive-level problem above: held-out test positives are `0`, so `test PR-AUC` / `ROC-AUC` are `NaN` and the resulting table is not scientifically useful.
  - `LiveCodeBench` full sweep on patched head `ea06bb7` (`2760`) is the first viable stage-1 detector artifact:
    - output root: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_20260421/`
    - fit-train / val / test counts: `54 / 128 / 160`
    - positives: `27 / 7 / 8`
    - selected layer: `18`
    - selected bandwidth: `100`
    - validation `PR-AUC`: `0.3921`
    - test `PR-AUC`: `0.1021`
    - test `ROC-AUC`: `0.7352`
    - test positive `F1`: `0.1429`
- Important comparison note:
  - the April prompt-only / linear / MLP `majority_s_0.5` tables cannot be reused as the stage-1 baseline comparison, because they were trained on a different binary object with much higher held-out prevalence (for example `LiveCodeBench` test prevalence `0.3375` there versus `0.05` here);
  - the matched March-split tooling for that rerun is now in-tree, so the next honest detector comparison is to execute prompt-only, activation-linear, and activation-MLP baselines on this exact March-reconstructed split before drawing any RFM-versus-baseline conclusion.

### Stage 2: Export Signed Concept Vectors And Check Direction Quality

- After the RFM tables exist, export one benchmark-local concept-vector bundle.
- Proposed new surface:
  - `scripts/export_prompt_profile_rfm_vectors.py`
- Save, for each benchmark and layer:
  - chosen hyperparameters
  - train/validation provenance
  - direction vector
  - sign convention
  - validation score
  - vector norm before normalization
- Sign convention must be explicit:
  - positive direction means higher predicted `majority_s_0.5` risk;
  - the anti-risk direction is therefore `-v`;
  - for spherical steering, the target pole is `mu_l = -normalize(v_l)`.
- Before any steering claim, add lightweight direction-coherence diagnostics:
  - bootstrap or seed cosine stability for repeated estimates of the same layerwise direction
  - cross-layer cosine structure within each benchmark
  - cross-benchmark cosine alignment layerwise across the retained benchmark set
  - held-out 1D projection separation under the declared sign convention
- The projection score is:
  - `s_i,l = h_i,l^T v_l`
- This 1D score does not need to beat the full RFM detector. It only needs to show that the exported signed direction has a meaningful relationship to the label.

### Stage 3: Extend The Unified Detector Report Before Steering

- Add RFM results plus direction-quality diagnostics to the current prompt-profile report surface before running steering claims.
- Compare, on the same held-out prompts:
  - prompt length baseline
  - strongest cheap prompt feature where available
  - activation linear `last_layer`
  - activation linear `ensemble`
  - activation MLP `last_layer`
  - MLP `ensemble`
  - RFM
- Add one supplementary direction table with:
  - benchmark
  - layer
  - validation score
  - vector norm before normalization
  - bootstrap cosine stability
  - projection separation score
  - sign convention check
  - cross-benchmark cosine where relevant
- Keep this report separate from trigger-attention narrative. The attention note is background context, not validation for the RFM stage.

### Stage 4: In-Distribution Spherical Steering On Benchmark-Specific Vectors

- Proposed new surfaces:
  - `scripts/steer_prompt_profile_concept_vectors.py`
  - `slurm/run_prompt_profile_rfm_steering.sbatch`
- For the first pass, switch from additive steering to norm-preserving spherical steering.
- Use one fixed angular strength:
  - `t = 0.3`
- Do not grid-search `t` yet. First answer whether the direction works at all.
- A fixed `t` does not imply the same Euclidean movement at every layer or on every example, so log starting angles and realized angular movement.
- Normalize each exported layerwise vector and define the anti-risk target as:
  - `mu_l = -normalize(v_l)`
- Prefer a prefill residual-activation hook route for the first causal test:
  - train on prompt-prefill residual activations
  - intervene on the same prompt-prefill residual activation surface during the prefill forward pass
- Do not start with KV-cache intervention if the activation-hook route is feasible, because that changes the steering surface away from the probe surface.
- Do not add a separate top-`k` layer-selection rule or probe gating in the first pass.
  - Use the exported signed per-layer benchmark-local bundle directly.
  - If layer ablations or controllers matter later, treat them as follow-ups after the first steering table exists.
- Run paired evaluation on each benchmark test set under these conditions:
  - `no_steer`
  - `minus_v_spherical`
  - `plus_v_spherical`
  - `random_spherical` with matched angular intervention protocol
  - `shuffled_label_spherical` if implementation time allows
- Report one steering table with:
  - average completion length
  - median completion length
  - `>50%`-budget fraction
  - loop fraction
  - max-length-hit fraction
  - accuracy
  - accuracy delta
  - number of evaluated prompts
- Add bootstrap intervals for the key steering deltas, especially on `GPQA`.
- Add one diagnostics table with:
  - mean starting angle
  - mean angular movement
  - mean pre-intervention norm
  - mean post-intervention norm
  - norm-preservation error
- The diagnostics table should verify that the spherical implementation is actually norm-preserving in practice.

### Stage 5: External Averaged-Vector Test

- Only after the benchmark-local steering table exists, build one external-benchmark control:
  - sign-align the retained four benchmark-local vectors layerwise
  - normalize them layerwise
  - average them into one "average verbose vector"
  - pick one benchmark outside the retained four-benchmark detector / in-distribution steering set
  - apply that average vector to the external benchmark with the same spherical intervention protocol
- Keep the angular strength fixed at the same `t = 0.3` for the first OOD pass.
- Report the same paired metric table plus accuracy delta.

### Stage 6: Minimal Controls And Stop Rules

- Keep the first steering pass unconditional:
  - no probe gating
  - no online controller
  - no top-`k` layer selection
- Do not stop the steering story just because RFM trails the current MLP detector on held-out `PR-AUC`.
  - That ranking belongs in the detector table, but it is not itself a steering stop rule.
- Stop or flag the steering story if this happens:
  - steering mainly shortens generations but collapses accuracy, or
  - `minus_v_spherical` does not outperform random or shuffled-label controls, or
  - `plus_v_spherical` produces the same improvement as `minus_v_spherical`, or
  - steering does not reduce loop / max-hit / `majority_s_0.5` rates relative to the no-steer and control-direction runs, or
  - direction-coherence diagnostics show unstable or sign-flipping vectors, or
  - effects appear only on GPQA-sized samples without bootstrap support.

## Backlog For This Stage

### P0: Must Land Before Any Steering Claim

- Add the benchmark registry and manifest/count validator.
- Implement the native layerwise RFM trainer on existing prompt-profile tensors.
- Emit benchmark-local RFM tables against the activation linear, activation MLP, and prompt-only baselines already on the repo surface.
- Export signed per-layer vector artifacts with explicit provenance.
- Add direction-coherence diagnostics for the exported vectors.
- Extend the unified report so RFM and the direction diagnostics live on the normal prompt-profile surface.

### P1: First Steering Pass

- Add the benchmark-local spherical steering runner with fixed `t = 0.3`.
- Match the intervention surface to the probe surface through prompt-prefill residual hooks if feasible.
- Use the exported signed per-layer bundle directly; do not add a top-`k` layer rule, probe gate, or controller in the first pass.
- Run paired benchmark-wise evaluation on held-out test prompts under `no_steer`, `minus_v_spherical`, `plus_v_spherical`, `random_spherical`, and `shuffled_label_spherical` where feasible.
- Report length, loop, max-hit, `majority_s_0.5`, accuracy, and bootstrap deltas in one table.
- Report angular-move and norm-preservation diagnostics in a separate table.

### P2: Second-Pass Follow-Ups

- External-benchmark steering with the averaged "verbose" vector built from the retained four benchmark-local bundles.
- Stronger prompt-shape controls or residualized analysis, because `majority_s_0.5` is still partly prompt-visible.
- Layer-selection ablations, controller designs, or `t` sweeps only if the first fixed spherical steering table shows real signal.

## What This Stage Should Not Claim

- Do not claim the merged trigger-attention note proves the RFM directions are mechanistically correct.
- Do not claim a global activation-only win until RFM is compared against the strongest cheap prompt features on the same split.
- Do not collapse "best detector" and "best steering direction" into the same claim. This stage measures both.
- Do not call `majority_s_0.5` a pure loop label. It is still a long-rollout / budget-usage proxy.
- Do not say steering "fixes loops" unless loop and max-hit rates drop without an unacceptable accuracy collapse.

## Deliverable Shape

- One committed stage-plan note: this file.
- One backlog/roadmap update that makes the stage executable without Slack archaeology.
- After implementation, one unified detector report first, then one steering report.
