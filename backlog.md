# CoT Loop Detection Backlog

Last updated: 2026-04-21 23:41 UTC

Reference plan:
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Active TODOs

### P0: Positive-Enrichment Screening Gate

- The active steering-trainable set is now gated by repaired train positive rate:
  - only promote datasets with train positive rate `>= 10%` after screening
  - on the current repaired surface, only `LiveCodeBench` passes (`140 / 420 = 33.3%`)
  - keep `GPQA`, `MATH-500`, and `MMLU-Pro` diagnostic-only for now (`7 / 133`, `18 / 338`, `6 / 518`)
- First screening queue:
  - `LiveCodeBench-extra`
  - `TACO-hard`
  - full `MATH` hard / level-5
  - `Omni-MATH` hard
- First `300`-prompt pass is now live under the direct node worktree:
  - output root:
    `/home/murphy/projects/worktrees/cot-loop-positive-screening/outputs/model_stats/positive_screen/`
  - log root:
    `/home/murphy/projects/worktrees/cot-loop-positive-screening/logs/positive_screen/`
  - launch order:
    - GPU `6`: `LiveCodeBench-extra` -> `MATH level-5`
    - GPU `7`: `TACO-hard` -> `Omni-MATH >= 7`
- The screening archive contract is now stricter than the older rollout-stats JSON:
  - save prompt text plus prompt token IDs
  - save dataset `record_id` plus dataset-side `record_metadata`
  - save per-rollout completion text plus exact `completion_token_ids`
  - save prompt-level `majority_s_0.5` summary fields in a separate prompt-profile sidecar
  - keep these sidecars next to the aggregate stats JSON so later activation replay and relabeling do not depend on Slack archaeology
- `LiveCodeBench-extra` must stay prompt-disjoint from the March stage object:
  - current screen uses exact prompt-text exclusion from
    `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_projection_livecodebench_majority05_seed0_20260323/data/diagnostics/prompt_rollout_archive.jsonl`
- Second screening queue:
  - `APPS` competition
  - `CodeContests` sample
  - optional `SuperGPQA`
  - optional `HLE`
- For each screened pool, report:
  - prompt count
  - repaired train positive count and positive rate under `majority_s_0.5`
  - completion-level `>50%` fraction
  - loop fraction
  - max-hit fraction
  - average / median completion length
  - accuracy if a grader exists
  - prompt-length stats
- Keep the “accuracy if a grader exists” caveat literal:
  - `LiveCodeBench-extra` and `MATH`-family screens have usable sanity anchors
  - `TACO-hard` is still ungraded in-repo on the first pass, so treat it as prevalence-first until an evaluator lands
- Do not revive cross-benchmark vector averaging or cosine-alignment claims until at least one more dataset clears the `10%` gate.

### P1: Close The Repaired Detector Table

- The repaired `LiveCodeBench` detector object is now frozen in the report bundle:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- Remaining detector-side TODO:
  - decide whether detector ranking needs matching RFM multiseed or split-seed sensitivity before it is treated as the durable RFM-versus-MLP read.
- If that follow-up is skipped, propagate the repaired `LiveCodeBench` detector table into the next unified prompt-profile summary instead of reusing the superseded `54 / 128 / 160` March object.

### P2: Finish The Direction-Quality Surface

- The repaired `LiveCodeBench` vector bundle and its report-style direction summary are now frozen in:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- Remaining direction tasks before any transfer claim:
  - cross-benchmark cosine alignment across screened-in datasets, not the old retained-four list
  - explicit decision on whether any future sub-`10%` repaired object should remain provenance-only even after materialization
- Repaired-materialization probecheck on the other retained benchmarks is now on disk:
  - `GPQA`: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/gpqa_majority_s0p5_rolloutrecompute_probecheck_20260421/`
    - natural prompt-majority counts: train `126/7`, val `32/2`, test `40/2`
  - `MATH-500`: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/math500_majority_s0p5_rolloutrecompute_probecheck_20260421/`
    - natural prompt-majority counts: train `320/18`, val `80/4`, test `100/4`
  - `MMLU-Pro`: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_binary/mmlu_pro_majority_s0p5_rolloutrecompute_probecheck_20260421/`
    - natural prompt-majority counts: train `512/6`, val `128/1`, test `160/2`
- Earlier smaller train counts from the same materializer (`14/7`, `36/18`, `12/6`) were only the downsampled fit-train subsets, not the raw repaired prompt sets.
- Under the new gate, those tiny-positive repaired objects stay provenance-only for now unless an enrichment pass creates a screened-in variant.

### P3: Benchmark-Local Block-Specific Steering

- The first larger benchmark-local steering table in the report bundle is only a bounded prefill-last-token spherical pilot and should be treated that way:
  - `outputs/livecodebench_repaired_stage_report_apr21/`
- If the steering lane continues on `LiveCodeBench`, the first required step is not a new hypothesis; it is the corrected same-slice rerun under the figure contract Wangzhi clarified on April 21.
- Open steering-side TODOs are now:
  - rerun the same `32` held-out `LiveCodeBench` prompt IDs with:
    - full decode budget (`30000`, not `1024`)
    - prefill-only timing
    - every prompt token steered at every selected block during prefill
    - block-specific linear conditions:
      - `minus_v_linear`
      - `plus_v_linear`
      - `random_linear`
    - block-specific spherical conditions:
      - `minus_v_spherical`
      - `plus_v_spherical`
      - `random_spherical`
  - record the rerun as superseding the old last-token `1024` pilot rather than as another comparable row in the same table
  - add `shuffled_label_spherical` if the benchmark-local lane is going to continue at all
  - pick a new benchmark-local steering hypothesis before launching more runs:
    - different hook surface
    - stability-informed layer restriction
    - or another control that is not already ruled out by the finished `32`-prompt table
  - if further steering jobs are launched, log per-condition wall time explicitly; `random_spherical` was much slower than the first three conditions
  - make the direction-similarity definition explicit in the next report:
    - it is the cosine between a bootstrap-refit signed normalized vector and the reference exported signed normalized vector for the same layer
    - the reference vector itself comes from the top eigenvector of that layer's symmetrized final `M`, after sign selection by fit-train `PR-AUC` then `ROC-AUC`

### P4: External Average-Vector Test

- This stays blocked until at least two screened-in benchmark-local bundles exist.
- Once that gate is met:
  - build one layerwise average "verbose" vector by sign-aligning and averaging the screened-in benchmark-local bundles
  - pick one benchmark outside that screened-in training set
  - apply the same fixed spherical protocol on that external benchmark and report the same paired metric table plus accuracy delta

### Later Only If The First Steering Table Is Real

- Stronger prompt-shape or residualized controls on the repaired object.
- Layer-selection ablations, controller variants, or `t` sweeps only after the unconditional spherical pass shows real signal.
