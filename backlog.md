# CoT Loop Detection Backlog

Last updated: 2026-04-21 18:05 UTC

Reference plan:
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Active TODOs

### P0: Add The Native RFM Detector

- RFM trainer / Slurm wrapper now exist and run on-node.
- Keep the first detector comparison honest:
  - only `LiveCodeBench` currently has a non-degenerate held-out positive slice on the frozen March `majority_s_0.5` object;
  - `GPQA`, `MATH-500`, and `MMLU-Pro` currently have `0` held-out test positives on that object, so do not treat their first detector rows as meaningful held-out ranking evidence.
- The matched March-split baseline tooling now exists:
  - `scripts/materialize_prompt_profile_stage_binary_data.py`
  - `scripts/train_probe.py` with explicit `--train-split` / `--eval-split`
  - `scripts/eval_probe_checkpoint.py` with arbitrary split evaluation and split sample-ID hashes
- Rerun the baseline families on the exact March-reconstructed split before comparing RFM against prior April tables:
  - prompt-only baselines
  - activation-side linear baselines
  - activation-side MLP baselines
- Package the first viable detector result from `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_20260421/`.
- Add direction-coherence diagnostics for the exported RFM vectors:
  - bootstrap cosine stability
  - cross-layer cosine structure
  - cross-benchmark cosine alignment
  - held-out 1D projection separation
- Decide with the collaborator whether the retained four-benchmark detector table should stay as-is, or whether the detector surface should become explicitly `LiveCodeBench`-first until the label object is changed.
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
