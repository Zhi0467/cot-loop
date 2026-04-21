# CoT Loop Detection Backlog

Last updated: 2026-04-21 08:45 UTC

Reference plan:
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Active TODOs

### P0: Freeze The Detector Surface

- Add one committed retained-benchmark registry for the active four-benchmark stage:
  - `GPQA`
  - `MATH-500`
  - `MMLU-Pro`
  - `LiveCodeBench`
- Keep `AIME` out of the collaborator-facing detector / in-distribution steering stage.
- Reuse the saved March `2026-03-22` to `2026-03-23` prompt-profile archive roots already referenced by the trigger-attention replay.
- Fail closed on manifest/count/shape mismatches before training.
- Add a native layerwise RFM training path on the existing prompt-prefill stacked tensors for binary `majority_s_0.5`.
- Keep the initial RFM surface simple:
  - bandwidth in `{1, 10, 100}`
  - `lambda = 1e-3`
  - `T = 5` fixed
- Emit held-out detector tables against all existing baseline families on the same split:
  - prompt-only baselines
  - activation-side linear baselines
  - activation-side MLP baselines
- Add direction-coherence diagnostics for the exported RFM vectors:
  - bootstrap cosine stability
  - cross-layer cosine structure
  - cross-benchmark cosine alignment
  - held-out 1D projection separation
- Use bootstrap intervals in the detector report, especially on smaller held-out sets such as `GPQA`.
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
