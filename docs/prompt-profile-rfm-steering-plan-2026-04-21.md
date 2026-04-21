# Prompt-Profile RFM Steering Stage Plan

Last updated: 2026-04-21 21:41 UTC

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
- Archive-versus-stage label correction:
  - the saved March prompt-profile archives were built with `archive_tail_threshold = 0.9`
  - this stage is about `majority_s_0.5`
  - on archives that still have rollout `relative_length`, the stage label should therefore be recomputed from saved rollouts rather than read from the archive's saved `0.9` label
- First fully repaired stage object:
  - `LiveCodeBench` is now the first benchmark rerun all the way through that rollout-recomputed path
  - repaired split: fit-train / val / test counts `280 / 128 / 160`
  - repaired positives: `140 / 35 / 54`
- Immediate consequence:
  - the earlier `54 / 128 / 160` with `27 / 7 / 8` `LiveCodeBench` table is withdrawn as a stale object
  - the retained four-benchmark registry is still the stage contract, but only `LiveCodeBench` has been rerun end to end on the repaired `majority_s_0.5` object so far
  - quick repaired-materialization check on the other retained benchmarks (`2026-04-21`) still leaves them tiny:
    - `GPQA`: train `126/7`, val `32/2`, test `40/2`
    - `MATH-500`: train `320/18`, val `80/4`, test `100/4`
    - `MMLU-Pro`: train `512/6`, val `128/1`, test `160/2`
  - so `LiveCodeBench` remains the only repaired benchmark with a genuinely non-tiny stage object right now
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
  - the repo now also has a live vector-export surface:
    - `scripts/export_prompt_profile_rfm_vectors.py`;
  - the repo now also has a live steering runner:
    - `scripts/steer_prompt_profile_concept_vectors.py`
    - `slurm/run_prompt_profile_rfm_steering.sbatch`;
  - matched March-split baseline tooling also now exists:
    - `scripts/materialize_prompt_profile_stage_binary_data.py`
    - `scripts/train_probe.py` with explicit `--train-split` / `--eval-split`
    - `scripts/eval_probe_checkpoint.py` for arbitrary split evaluation;
  - older RFM results still exist only in archived PR2 output artifacts, so the detector and steering parts of this stage remain a new implementation lane rather than a parameter-only follow-up.

## What This Stage Is Trying To Prove

- First, whether a layerwise RFM built on the existing prompt-prefill tensors is competitive with the current MLP surfaces on the same held-out `majority_s_0.5` task.
- Second, whether the exported benchmark-specific RFM directions are stable enough to be treated as steering vectors rather than only as a by-product of a nonlinear classifier.
- Third, whether those benchmark-specific directions can be used for small, controlled block-specific steering interventions, with linear and spherical variants reported side by side, that reduce long-rollout risk without simply destroying accuracy.
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
- First repaired stage-1 detector receipt (`2026-04-21`):
  - repaired detector root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
  - repaired split:
    - fit-train / val / test counts `280 / 128 / 160`
    - positives `140 / 35 / 54`
  - selected detector row:
    - layer `27`
    - bandwidth `100`
    - validation `PR-AUC 0.6555`
    - test `PR-AUC 0.7055`
    - test `ROC-AUC 0.8590`
    - test positive `F1 0.2222`
- Important comparison note:
  - the older March `LiveCodeBench` comparison bundle under `.../livecodebench_majority_s0p5_seed0_20260421/` is superseded for the same reason as the old detector row: it was built on the stale archive `0.9` label instead of the rollout-recomputed stage label
  - the honest detector comparison now has to use the repaired stage materialization and the repaired prompt-only / activation reruns on the same prompt IDs
- Repaired `LiveCodeBench` baseline receipts (`2026-04-21`):
  - materialized comparison object:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
    - fit-train / val / test counts `280 / 128 / 160`
    - positives `140 / 35 / 54`
  - prompt-only baseline bundle:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_prompt_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
    - `prompt_length`: test `PR-AUC 0.5771`, test `ROC-AUC 0.7201`
    - `prompt_shape_linear`: test `PR-AUC 0.5871`, test `ROC-AUC 0.7290`
    - `prompt_shape_tree`: test `PR-AUC 0.3732`, test `ROC-AUC 0.5751`
  - activation baseline bundle:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
    - `best_rank` mean test `PR-AUC` / `ROC-AUC` across seeds `0/1/2`:
      - linear `last_layer`: `0.4163` / `0.5486`
      - linear `ensemble`: `0.5698` / `0.7475`
      - MLP `h256 d1 last_layer`: `0.7055` / `0.8435`
      - MLP `h256 d1 ensemble`: `0.6637` / `0.8458`
    - `best_loss` mean test `PR-AUC` / `ROC-AUC` across seeds `0/1/2`:
      - MLP `h256 d1 last_layer`: `0.7147` / `0.8489`
- Current honest read on the repaired object:
  - RFM is clearly above the cheap prompt-only baselines and above the activation linear baselines on the repaired split
  - `h256 d1` MLP last-layer is now essentially tied with the current single-seed RFM on `PR-AUC`, and it is slightly ahead if the detector comparison uses the activation `best_loss` mean
  - because RFM has only a repaired single-seed row so far while the activation baselines now have a repaired multiseed table, the detector lane is not yet a clean RFM win or a clean MLP win
  - that still does not close the steering question, because the plan keeps steering utility separate from detector ranking

### Stage 2: Export Signed Concept Vectors And Check Direction Quality

- After the RFM tables exist, export one benchmark-local concept-vector bundle.
- Live export surface:
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
- Current stage-2 status:
  - repaired LiveCodeBench export summary exists at:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/vector_exports/summary.json`
  - the bundle already includes:
    - signed per-layer vectors for all `28` layers
    - prompt-ID provenance
    - preprocessing metadata including archive/stage tail-threshold distinction
    - raw and normalized checksums
    - held-out 1D projection separation metrics
    - cross-layer cosine structure
    - direction-bootstrap replay records for all `28` layers
  - finished bootstrap-stability read on the repaired `LiveCodeBench` object:
    - `100` replay fits per layer under the selected detector hyperparameters
    - all `28` layers clear mean cosine `>= 0.781` against their exported signed reference vector
    - the weakest `95%` low bound is still `0.693`
    - late layers `23-26` are the most direction-stable (`0.867` to `0.909` mean cosine)
    - validation selection still peaks at layer `27`, which remains stable but less extreme on cosine (`0.786`, low `0.734`)
  - this stage is now only partial because cross-benchmark cosine alignment is still missing
  - and that missing alignment surface is constrained by the repaired data reality above: outside `LiveCodeBench`, the current repaired splits are still tiny-positive objects rather than obviously robust transfer bundles

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
- First report-style LiveCodeBench deliverable now exists:
  - note:
    - `docs/livecodebench-repaired-stage-report-2026-04-21.md`
  - bundle:
    - `outputs/livecodebench_repaired_stage_report_apr21/`
  - what it freezes:
    - the repaired `LiveCodeBench` prompt object
    - the detector comparison against repaired prompt-only / activation baselines
    - the direction-bootstrap stability surface
    - the first finished larger benchmark-local steering table
- That report does not replace the later unified detector report for the whole prompt-profile surface, but it is now the collaborator-facing artifact for "finish LiveCodeBench" on this stage.

### Stage 4: In-Distribution Block-Specific Steering On Benchmark-Specific Vectors

- Committed steering surfaces:
  - `scripts/steer_prompt_profile_concept_vectors.py`
  - `slurm/run_prompt_profile_rfm_steering.sbatch`
- Follow the figure contract literally:
  - each transformer block `l` is steered by its own exported block-specific vector `v_l`
  - do not collapse that into one shared global direction or a top-`k` layer selector
- Run the two block-specific steering variants in parallel:
  - linear steering
    - anti-risk: `h_l' = h_l - epsilon v_l`
    - sign-flip control: `h_l' = h_l + epsilon v_l`
  - spherical steering
    - anti-risk target pole: `mu_l = -normalize(v_l)`
    - sign-flip control: `+normalize(v_l)`
- The current repo only has the spherical branch implemented. Linear block-specific steering is still missing and should be treated as open work rather than implied by the current report.
- Use one fixed angular strength:
  - `t = 0.3`
- Do not grid-search `t` yet. First answer whether the direction works at all.
- A fixed `t` does not imply the same Euclidean movement at every layer or on every example, so log starting angles and realized angular movement.
- Prefer a prefill residual-activation hook route for the first causal test:
  - train on prompt-prefill residual activations
  - intervene on the same prompt-prefill residual activation surface during the prefill forward pass
- Do not start with KV-cache intervention if the activation-hook route is feasible, because that changes the steering surface away from the probe surface.
- Be explicit about timing:
  - the current in-repo runner modifies only the last prompt token during prefill
  - it does not yet steer decode steps token by token
  - if the intended contract is decode-step steering, that controller is still missing and has to land before the steering stage is called complete
- Do not add a separate top-`k` layer-selection rule or probe gating in the first pass.
  - Use the exported signed per-layer benchmark-local bundle directly.
  - If layer ablations or controllers matter later, treat them as follow-ups after the first steering table exists.
- Run paired evaluation on each benchmark test set under these conditions:
  - `no_steer`
  - `minus_v_linear`
  - `plus_v_linear`
  - `minus_v_spherical`
  - `plus_v_spherical`
  - `random_linear`
  - `random_spherical` with matched angular intervention protocol
  - `shuffled_label_linear` and `shuffled_label_spherical` if implementation time allows
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
- Current stage-4 status:
  - the benchmark-local steering runner now exists in-repo:
    - `scripts/steer_prompt_profile_concept_vectors.py`
    - `slurm/run_prompt_profile_rfm_steering.sbatch`
  - what exists today is narrower than the intended stage contract:
    - spherical only
    - hook site `prefill_layer_output_last_token`
    - prefill-once last-prompt-token intervention
    - bounded pilot decode cap `max_new_tokens = 1024`
  - so the existing LiveCodeBench steering table is a bounded prefill-only spherical pilot, not the full block-specific linear+spherical stage verdict
  - first repaired `LiveCodeBench` smoke root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_smoke_t0p3_n8_seed0_20260421_fix2/`
  - completed smoke conditions:
    - `no_steer`
    - `minus_v_spherical`
  - what that smoke actually proves:
    - the prompt-hash checks, prompt-surface recovery, prefill-layer spherical hook, generation path, `LiveCodeBench` grading path, and steering ledgers all run end to end on the repaired object
  - what it does not prove:
    - it is only an `8`-prompt smoke and is not a useful scientific steering table by itself
    - it uses the bounded pilot cap `max_new_tokens = 1024`, not the archive-side `30000`
    - it is prefill-only rather than decode-step steering
    - both conditions stayed at `0 / 8` `pass@1` with mean completion length `1024`
    - `minus_v_spherical` increased loop fraction from `0.0` to `0.375` on that tiny slice, so the first honest steering claim still requires a larger held-out table
  - one follow-up review bug was real and is now fixed in project commit `df5187b`:
    - `LiveCodeBench` prompt recovery must use the archive source formatter, not the steering checkpoint formatter
    - direct node-side precheck now confirms prompt recovery still succeeds even when the steering model ID is changed away from the archive source model
  - the reviewed-head two-condition rerun is also now on disk:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_smoke_t0p3_n8_seed0_20260421_fix3_sourcefmt/`
    - it matches the earlier repaired smoke exactly, so the durable smoke receipt now points at the final formatter-fixed code surface rather than the intermediate patch
  - the cheap four-condition control smoke is now finished too:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_smoke_t0p3_n8_seed0_20260421_fix4_controls/`
    - all four conditions stay at `0 / 8` `pass@1` with average length `1024`
    - loop fractions on the smoke slice are:
      - `0.0` for `no_steer`
      - `0.375` for `minus_v_spherical`
      - `0.125` for `plus_v_spherical`
      - `0.375` for `random_spherical`
    - so the cheap control read is negative:
      - `minus_v_spherical` does not beat random on this slice
      - both signed directions are worse than baseline on loop fraction
      - this remains an implementation/control receipt, not evidence for the steering story
  - the first larger held-out control table is now finished too:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_controls32_t0p3_seed0_20260421/`
    - all four conditions stay at `0 / 32` `pass@1`
    - the run used the same bounded pilot cap `max_new_tokens = 1024`
    - the run is still prefill-only spherical steering, not decode-step steering and not the parallel linear+spherical figure contract
    - loop fractions on the larger repaired held-out slice are:
      - `0.03125` for `no_steer`
      - `0.28125` for `minus_v_spherical`
      - `0.125` for `plus_v_spherical`
      - `0.34375` for `random_spherical`
    - the delayed `random_spherical` arm did finish; it was simply much slower than the first three conditions and ended as the worst loop-fraction control
    - so the first larger bounded prefill-only spherical table is also negative:
      - there is still no accuracy movement anywhere
      - both signed directions are worse than baseline on loop fraction
      - but this does not yet close the full stage under the figure contract, because linear steering and decode-step timing are still missing
  - the negative `32`-prompt table is now frozen into the report-style deliverable:
    - `docs/livecodebench-repaired-stage-report-2026-04-21.md`
    - `outputs/livecodebench_repaired_stage_report_apr21/`

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

### P0: Close The Repaired Detector Table

- Treat the older March comparison object under `.../livecodebench_majority_s0p5_seed0_20260421/` as superseded.
- Keep the repaired comparison surfaces as the current detector object:
  - repaired detector root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
  - repaired prompt-only baseline root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_prompt_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
  - repaired activation baseline root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
- Freeze the repaired detector writeup into the unified report so it replaces the superseded `54 / 128 / 160` object.
- Keep matching RFM multiseed / split-seed sweeps as a later detector-sensitivity check, not a blocker for the steering lane.

### P1: Finish The Direction-Quality Surface

- Keep the repaired LiveCodeBench vector bundle as the first real stage-2 artifact.
- Add the missing direction-coherence diagnostics:
  - cross-benchmark cosine alignment across the retained benchmark set
  - report-ready direction summary table
- Bootstrap cosine stability is now finished on the repaired LiveCodeBench bundle:
  - `100` replay fits per layer
  - all `28` layers clear mean cosine `>= 0.781`
  - the weakest `95%` low bound is `0.693`
  - late layers `23-26` are the most stable, while selected detector layer `27` remains the validation peak

### P2: First Steering Pass

- Keep the benchmark-local spherical steering runner on the fixed `t = 0.3` surface now that it exists in-repo.
- Add the missing benchmark-local linear steering runner on the same block-specific bundle rather than treating spherical as the whole steering family.
- Match the intervention surface to the probe surface through prompt-prefill residual hooks if feasible.
- Decide explicitly whether the intended contract is decode-step steering rather than prefill-once steering; if yes, add that controller before calling the steering stage complete.
- Use the exported signed per-layer bundle directly; do not add a top-`k` layer rule, probe gate, or controller in the first pass.
- Run paired benchmark-wise evaluation on held-out test prompts under `no_steer`, linear controls, spherical controls, and shuffled-label controls where feasible.
- Report length, loop, max-hit, `majority_s_0.5`, accuracy, and bootstrap deltas in one table.
- Report angular-move and norm-preservation diagnostics in a separate table.
- Keep the reviewed-head two-condition smoke plus the finished four-condition control smoke as implementation receipts only, then scale to the first larger held-out table before making any sign-sensitive steering claim.
- Do not treat the finished `32`-prompt spherical table as the whole stage verdict; it is only the bounded prefill-only spherical sub-lane.

### P3: Second-Pass Follow-Ups

- External-benchmark steering with the averaged "verbose" vector built from the retained four benchmark-local bundles.
- Stronger prompt-shape controls or residualized analysis on the repaired object.
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
- One LiveCodeBench report-style bundle now exists for the repaired stage object:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- After broader implementation beyond `LiveCodeBench`, one unified detector report first, then one steering report.
