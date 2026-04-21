# CoT Loop Detection Backlog

Last updated: 2026-04-21 12:34 UTC

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

- Implement `scripts/steer_prompt_profile_concept_vectors.py` and `slurm/run_prompt_profile_rfm_steering.sbatch`.
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

### P3: External Average-Vector Test

- Build one layerwise average "verbose" vector by sign-aligning and averaging the retained benchmark-local bundles.
- Pick one benchmark outside the retained four-benchmark training set.
- Apply the same fixed spherical protocol on that external benchmark and report the same paired metric table plus accuracy delta.

### Later Only If The First Steering Table Is Real

- Stronger prompt-shape or residualized controls on the repaired object.
- Layer-selection ablations, controller variants, or `t` sweeps only after the unconditional spherical pass shows real signal.
