# CoT Loop Detection Backlog

Last updated: 2026-04-21 14:00 UTC

Reference plan:
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Active TODOs

### P0: Close The Repaired Detector Table

- Treat the older March comparison object under `.../livecodebench_majority_s0p5_seed0_20260421/` as superseded.
  - That table trusted the archive's saved `tail_threshold = 0.9` label instead of recomputing the stage label `majority_s_0.5` from saved rollout lengths.
- Use the repaired `LiveCodeBench` detector surface as the current comparison object:
  - RFM detector root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
    - current row: layer `27`, bandwidth `100`, validation `PR-AUC 0.6555`, test `PR-AUC 0.7055`, test `ROC-AUC 0.8590`
  - prompt-only baseline root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_prompt_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
    - strongest cheap prompt-only row so far: `prompt_shape_linear` with test `PR-AUC 0.5871`, test `ROC-AUC 0.7290`
  - activation baseline root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
    - `best_rank` mean test `PR-AUC`: linear last-layer `0.4163`, linear ensemble `0.5698`, `mlp256d1` last-layer `0.7055`, `mlp256d1` ensemble `0.6637`
    - `best_loss` mean test `PR-AUC`: `mlp256d1` last-layer `0.7147`
- Freeze the repaired detector writeup into the unified report instead of the superseded `54 / 128 / 160` object.
- Treat matching RFM multiseed / split-seed sweeps as a detector-sensitivity follow-up, not as a steering blocker.

### P1: Finish The Direction-Quality Surface

- Keep the repaired LiveCodeBench vector bundle as the first real stage-2 artifact:
  - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/vector_exports/summary.json`
- Direction-bootstrap stability is now on disk for that bundle:
  - `100` replay fits per layer under the fixed selected hyperparameters
  - all `28` layers clear mean cosine `>= 0.781`
  - the weakest `95%` low bound is `0.693`
  - late layers `23-26` are the most direction-stable (`0.867` to `0.909` mean cosine)
- Remaining direction tasks before any transfer claim:
  - cross-benchmark cosine alignment across the retained benchmark set
  - concise direction table on the normal prompt-profile report surface
- Decide whether to export repaired bundles for the other retained benchmarks immediately or only after the detector lane is judged stable enough.

### P2: Benchmark-Local Spherical Steering

- The first spherical steering runner is now real:
  - repo surfaces:
    - `scripts/steer_prompt_profile_concept_vectors.py`
    - `slurm/run_prompt_profile_rfm_steering.sbatch`
  - first repaired `LiveCodeBench` smoke root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_smoke_t0p3_n8_seed0_20260421_fix2/`
  - smoke status:
    - both `no_steer` and `minus_v_spherical` ran end to end and wrote `config.json`, per-seed summaries, condition summaries, and `prompt_profile_rfm_steering_run.v1` ledgers
    - on this tiny `8`-prompt slice, both conditions stayed at `0 / 8` `pass@1` with mean length `1024`, so the smoke is an implementation receipt rather than a positive steering result
    - `minus_v_spherical` also raised loop fraction from `0.0` to `0.375` on the smoke slice, so the first real steering table still needs a larger held-out surface before any directional claim
- Keep the reviewed formatter fix from project commit `df5187b` on this lane:
  - `LiveCodeBench` prompt recovery now uses the archive source formatter (source model plus saved `lm_style_override` where available), not the steering checkpoint formatter
  - direct precheck on the node confirmed prompt recovery still works even when the steering model ID is changed
- Keep the first steering pass fixed to:
  - prompt-prefill residual hooks if feasible
  - full exported per-layer bundle
  - spherical steering with `t = 0.3`
  - controls `no_steer`, `minus_v_spherical`, `plus_v_spherical`, `random_spherical`, and `shuffled_label_spherical` if feasible
- Log steering ledgers with:
  - condition name
  - vector artifact hash
  - hook site
  - `t`
  - seeds
  - prompt IDs
  - generation config
  - grader version
  - output path
- Next steering TODOs:
  - rerun the smoke from reviewed head `df5187b` so the durable smoke receipt matches the final code surface
  - add `plus_v_spherical` and `random_spherical`
  - scale from `8` prompts to the first honest repaired held-out steering table

### P3: External Average-Vector Test

- Build one layerwise average "verbose" vector by sign-aligning and averaging the retained benchmark-local bundles.
- Pick one benchmark outside the retained four-benchmark training set.
- Apply the same fixed spherical protocol on that external benchmark and report the same paired metric table plus accuracy delta.

### Later Only If The First Steering Table Is Real

- Stronger prompt-shape or residualized controls on the repaired object.
- Layer-selection ablations, controller variants, or `t` sweeps only after the unconditional spherical pass shows real signal.
