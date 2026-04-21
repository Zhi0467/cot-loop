# CoT Loop Detection Backlog

Last updated: 2026-04-21 19:48 UTC

Reference plan:
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Active TODOs

### P0: Close The Repaired Detector Table

- The repaired `LiveCodeBench` detector object is now frozen in the report bundle:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- Remaining detector-side TODO:
  - decide whether detector ranking needs matching RFM multiseed or split-seed sensitivity before it is treated as the durable RFM-versus-MLP read.
- If that follow-up is skipped, propagate the repaired `LiveCodeBench` detector table into the next unified prompt-profile summary instead of reusing the superseded `54 / 128 / 160` March object.

### P1: Finish The Direction-Quality Surface

- The repaired `LiveCodeBench` vector bundle and its report-style direction summary are now frozen in:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- Remaining direction tasks before any transfer claim:
  - cross-benchmark cosine alignment across the retained benchmark set
  - explicit decision on whether the tiny-positive repaired non-`LiveCodeBench` objects count as real alignment inputs or stay provenance-only
- Repaired-materialization probecheck on the other retained benchmarks is now on disk:
  - `GPQA`: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/gpqa_majority_s0p5_rolloutrecompute_probecheck_20260421/`
    - natural prompt-majority counts: train `126/7`, val `32/2`, test `40/2`
  - `MATH-500`: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/math500_majority_s0p5_rolloutrecompute_probecheck_20260421/`
    - natural prompt-majority counts: train `320/18`, val `80/4`, test `100/4`
  - `MMLU-Pro`: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/mmlu_pro_majority_s0p5_rolloutrecompute_probecheck_20260421/`
    - natural prompt-majority counts: train `512/6`, val `128/1`, test `160/2`
- Earlier smaller train counts from the same materializer (`14/7`, `36/18`, `12/6`) were only the downsampled fit-train subsets, not the raw repaired prompt sets.
- Decide explicitly whether those tiny-positive repaired objects should count as real transfer / alignment inputs or stay provenance-only for now.

### P2: Benchmark-Local Spherical Steering

- The first larger benchmark-local steering table is now frozen in the report bundle and should be treated as a finished negative result:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- If the steering lane continues on `LiveCodeBench`, do not just scale the same table again.
- Open steering-side TODOs are now:
  - add `shuffled_label_spherical` if the benchmark-local lane is going to continue at all
  - pick a new benchmark-local steering hypothesis before launching more runs:
    - different hook surface
    - stability-informed layer restriction
    - or another control that is not already ruled out by the finished `32`-prompt table
  - if further steering jobs are launched, log per-condition wall time explicitly; `random_spherical` was much slower than the first three conditions

### P3: External Average-Vector Test

- Build one layerwise average "verbose" vector by sign-aligning and averaging the retained benchmark-local bundles.
- Pick one benchmark outside the retained four-benchmark training set.
- Apply the same fixed spherical protocol on that external benchmark and report the same paired metric table plus accuracy delta.

### Later Only If The First Steering Table Is Real

- Stronger prompt-shape or residualized controls on the repaired object.
- Layer-selection ablations, controller variants, or `t` sweeps only after the unconditional spherical pass shows real signal.
