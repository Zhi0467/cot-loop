# CoT Loop Detection Backlog

Last updated: 2026-04-23 02:20 UTC

Reference docs:
- `docs/prompt-profile-rfm-steering-grounded-stage-2026-04-23.md`
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`

## Fixed current object

- Active benchmark-local stage:
  - repaired `LiveCodeBench` prompt-level `majority_s_0.5`
  - fit-train / val / test = `280 / 128 / 160`
  - positives = `140 / 35 / 54`
- Steering evaluation contract:
  - full test `160`, not the old `32`-prompt pilot
  - full decode budget `30000`
  - prefill-only
  - all prompt tokens steered at every selected block
  - block-specific linear and spherical conditions
- Published PR surface is still stale relative to local reality:
  - draft PR `#11` head at last check: `5a521d1`
  - local worktree head before this grounded-doc patch: `a255ff1`

## Finished evidence

- Repaired detector lane:
  - RFM layer `27`
  - validation `PR-AUC 0.6555`
  - test `PR-AUC 0.7055`
  - test `ROC-AUC 0.8590`
  - honest read: RFM > prompt-only, RFM > activation linear, RFM about tied with activation MLP last-layer
- Vector-quality lane:
  - all `28` layers have mean bootstrap cosine `>= 0.781`
  - late layers `23` to `26` are the most coherent
- Finished full-contract thinking-on steering rows:
  - `no_steer`: `2 / 160` `pass@1`, loop fraction `0.65625`, over-half-budget fraction `0.6375`
  - `plus_v_linear`: `2 / 160` `pass@1`, loop fraction `0.6125`, over-half-budget fraction `0.6000`

## Active TODOs

### P0: Finish the work already running

#### P0.1: Close the full thinking-on steering table

- Running jobs:
  - `2804` - `minus_v_linear`
  - `2810` - `random_linear`
  - `2811` - `minus_v_spherical`
  - `2815` - `plus_v_spherical`
  - `2816` - `random_spherical`
- Required deliverable:
  - one finished seven-row thinking-on table with:
    - `no_steer`
    - `minus_v_linear`
    - `plus_v_linear`
    - `random_linear`
    - `minus_v_spherical`
    - `plus_v_spherical`
    - `random_spherical`
  - report `pass@1`, loop fraction, over-half-budget fraction, avg/median generation length, and max-length-hit fraction
- Do not launch:
  - `t` sweeps
  - layer-restriction ablations
  - controller variants
  until this table exists.

#### P0.2: Finish the positive-enrichment screening gate

- Current sidecar checkpoints:
  - `LiveCodeBench-extra`: `255` profiled, `141` positives, positive rate `0.5529`, completion-tail fraction `0.5765`, loop fraction `0.3029`
  - `TACO-hard`: `213` profiled, `172` positives, positive rate `0.8075`, completion-tail fraction `0.8263`, loop fraction `0.3779`
  - `MATH level-5` parallel path: `180` profiled, `25` positives, positive rate `0.1389`, completion-tail fraction `0.2069`, loop fraction `0.0833`, success fraction `0.7750`
  - `Omni-MATH >= 7`: dependency-pending behind `2818`
- Finish the current `300`-prompt passes before opening another screening family.
- Promotion rule stays literal:
  - only promote datasets whose final repaired prompt-majority train positive rate stays `>= 10%`
- Per-dataset admission receipt must include:
  - profiled prompt count
  - prompt-majority positive count and rate
  - completion-tail fraction
  - loop fraction
  - max-length-hit fraction
  - accuracy if a grader exists
  - exact prompt/completion sidecar paths
- Keep `LiveCodeBench-extra` prompt-disjoint from the repaired March `LiveCodeBench` object by exact prompt-text exclusion.

#### P0.3: Let the March prompt-surface provenance pair run

- Queued jobs:
  - `2829` - `LiveCodeBench` `HFChatTemplate` `thinking-mode on`
  - `2830` - same surface with `thinking-mode off`
- Purpose:
  - settle the narrow provenance question around the March collector surface
- Priority rule:
  - this is real work, but it is below the full thinking-on steering table and below finishing the current screening gate

### P1: Unblock the non-thinking steering lane correctly

- Current status:
  - `2821`, `2822`, `2823`, `2825`, and `2826` all failed on dirty-slot CUDA OOM before first row
  - there is still no non-thinking condition summary on disk
- Interpretation:
  - this is currently a node-geometry blocker, not a science result
- Next action:
  - do not keep blind-retrying while the same contaminated slots are visible
  - wait for a clean slot or for the current thinking-on jobs to release capacity
  - rerun a narrow serial table first:
    - `no_steer`
    - `minus_v_linear`
    - `plus_v_linear`
    - `random_linear`
- Only extend non-thinking to spherical once at least one non-thinking row has landed.

### P2: Promote the first new screened-in dataset(s)

- Promotion order after the current screen finishes:
  - `LiveCodeBench-extra`
  - `TACO-hard`
  - `MATH level-5` if the final `300`-prompt pass stays above the gate
  - `Omni-MATH >= 7` after the dependency chain completes
- For each promoted dataset:
  - materialize the repaired `majority_s_0.5` object
  - train the benchmark-local RFM detector
  - export block-specific vectors
  - run the same bootstrap direction-stability diagnostics used for `LiveCodeBench`
  - write a report-style receipt before treating the dataset as part of the shared vector pool

### P3: Cross-benchmark vector pooling and transfer

- This stays blocked until at least two non-`LiveCodeBench` benchmark-local bundles:
  - pass the screen
  - are materialized
  - export stable directions
  - and have readable benchmark-local receipts
- Only after that:
  - build a sign-aligned average vector
  - test it on an external held-out benchmark

## Defer until the main stage is settled

- RFM multiseed / split-seed sensitivity on the repaired detector object:
  - still useful, but not the current critical path
- New steering hypotheses beyond the corrected figure contract
- Non-thinking spherical runs before the first non-thinking linear row exists
- Any reuse of the old `32`-prompt steering pilot as if it were current evidence

## Runtime and infra notes

- The positive-enrichment screen is currently using home-backed caches:
  - `/home/murphy/.cache/cot-loop-positive-screening/...`
  because `/data` is effectively full.
- Treat this as active execution context, not a footnote. It matters for reruns and for where the current sidecars live.
