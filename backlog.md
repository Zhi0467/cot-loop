# CoT Loop Detection Backlog

Last updated: 2026-04-21 18:42 UTC

Reference plan:
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Active TODOs

### P0: Add The Native RFM Detector

- Treat only `LiveCodeBench` as the meaningful held-out detector object on the frozen March `majority_s_0.5` surface until the label object or split policy changes; keep `GPQA`, `MATH-500`, and `MMLU-Pro` provenance-only for now.
- Extend the first matched `LiveCodeBench` baseline slice beyond seed `0`:
  - prompt-only baselines from `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_prompt_baselines/livecodebench_majority_s0p5_seed0_20260421/`
  - activation-side linear / MLP baselines from `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_baselines/livecodebench_majority_s0p5_seed0_20260421/`
- Build one detector comparison table on the exact March-reconstructed split with:
  - RFM layer `18`, bandwidth `100`, test `PR-AUC 0.1021`, test `ROC-AUC 0.7352`
  - prompt-only `prompt_length`, `prompt_shape_linear`, `prompt_shape_tree`
  - activation linear `last_layer` / `ensemble`
  - activation MLP `last_layer` / `ensemble`
  - `best_rank` as the primary activation checkpoint rule, with `best_loss` kept as a diagnostic secondary view
- Explain the first honest comparison plainly:
  - on the current seed-`0` matched split, RFM is ahead of prompt-only and activation-linear baselines;
  - activation MLP is still ahead of RFM on the same object.
- Add direction-coherence diagnostics for the exported RFM vectors:
  - bootstrap cosine stability
  - cross-layer cosine structure
  - cross-benchmark cosine alignment
  - held-out 1D projection separation
- Extend the unified prompt-profile report with the RFM rows and direction diagnostics before making any steering claim.

### P1: Benchmark-Local Spherical Steering

- Export signed per-layer benchmark-local vector bundles from the trained RFM surface.
- Add the benchmark-local steering runner with fixed angular strength `t = 0.3`.
- Match the intervention surface to the probe surface through prompt-prefill residual hooks if feasible.
- Normalize each exported vector and steer toward the anti-risk target `mu_l = -normalize(v_l)`.
- Use the exported signed per-layer bundle directly in the first pass; do not add a top-`k` layer rule, probe gate, or online controller yet.
- Run paired evaluation on each benchmark test set under:
  - `no_steer`
  - `minus_v_spherical`
  - `plus_v_spherical`
  - `random_spherical`
  - `shuffled_label_spherical`, if feasible
- Report one table with:
  - average completion length
  - median completion length
  - `>50%` budget fraction
  - loop fraction
  - max-length-hit fraction
  - `majority_s_0.5` fraction
  - accuracy
  - accuracy delta
  - bootstrap intervals where feasible
- Report one diagnostics table with:
  - mean starting angle
  - mean angular movement
  - pre/post norm
  - norm-preservation error
- Continue this steering pass even if RFM does not beat the current MLP detector on held-out `PR-AUC`.

### P2: External Average-Vector Test

- Build one average "verbose" vector by sign-aligning, normalizing, and averaging the retained four benchmark-local bundles layerwise.
- Pick one benchmark outside the retained four-benchmark training set.
- Apply the average vector to that external benchmark with the same fixed spherical strength `t = 0.3`.
- Report the same paired metric table plus accuracy delta.

### Later Only If The First Steering Table Is Real

- Prompt-shape controls or residualized analysis, because `majority_s_0.5` is still partly prompt-visible.
- Layer-selection ablations only if the first direct per-layer-bundle steering pass shows signal.
- `t` sweeps only if the fixed-`0.3` pass shows signal.
- Probe gating or online-controller ideas only after the unconditional spherical pass is on disk.
