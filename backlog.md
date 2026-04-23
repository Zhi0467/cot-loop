# CoT Loop Detection Backlog

Last updated: 2026-04-23 04:22 UTC

Reference docs:
- `docs/prompt-profile-rfm-mode-consistent-stage-2026-04-23.md`
- `docs/prompt-profile-rfm-steering-grounded-stage-2026-04-23.md` (superseded intermediate note)
- `docs/prompt-profile-rfm-steering-plan-2026-04-21.md` (historical plan)

## Fixed current object

- Active benchmark-local anchor:
  - repaired `LiveCodeBench` prompt-level `majority_s_0.5`
  - fit-train / val / test = `280 / 128 / 160`
  - positives = `140 / 35 / 54`
- New stage rule:
  - no cross-mode steering claim
  - each path must close `stats -> probe -> steer` inside one mode
- Repo surface is still ahead of the published PR:
  - local branch: `task/1776752262-rfm-stage0`
  - local head before this backlog rewrite: `dabe924`
  - draft PR `#11` published head still `5a521d1`

## Durable evidence that still stands

- Repaired predecessor detector/vector bundle:
  - root:
    - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
  - layer `27`
  - validation `PR-AUC 0.6555`
  - test `PR-AUC 0.7055`
  - test `ROC-AUC 0.8590`
  - all `28` layers have mean bootstrap cosine `>= 0.781`
  - late layers `23` to `26` are still the most coherent
- Thinking-on baseline receipt that may still be reusable:
  - `no_steer`: `2 / 160` `pass@1`, loop fraction `0.65625`, over-half-budget fraction `0.6375`
- Positive-enrichment pilot leaderboard:
  - `LiveCodeBench-extra`: `255` profiled, `141` positives, positive rate `0.5529`, completion-tail fraction `0.5765`, loop fraction `0.3029`
  - `TACO-hard`: `229` profiled, `186` positives, positive rate `0.8122`, completion-tail fraction `0.8308`, loop fraction `0.3723`
  - `MATH level-5` parallel path: `261` profiled, `40` positives, positive rate `0.1533`, completion-tail fraction `0.2126`, loop fraction `0.0738`, success fraction `0.7739`
  - `Omni-MATH >= 7`: dependency-pending behind `2818`

## Demoted evidence

- The finished steered thinking-on row
  - `plus_v_linear`: `2 / 160` `pass@1`, loop fraction `0.6125`, over-half-budget fraction `0.6000`
  is now implementation-only rather than valid stage evidence, because its config points at:
  - `vector_export_dir = /data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/vector_exports`
  and that vector bundle points back to:
  - `preprocessing.source_data_dir = /data/scratch/murphy/outputs/cot-loop-detection/livecodebench_mean_relative_from_archive_20260323`
- The live cross-mode steering jobs were canceled at `2026-04-23 04:22 UTC`:
  - `2804`
  - `2811`
  - `2815`
  - `2816`
- Treat the old thinking-on steered rows only as runner receipts:
  - full decode `30000`
  - hook site `prefill_layer_output_all_tokens`
  - all-prompt-token prefill intervention
  - ledger / diagnostics plumbing works

## Active TODOs

### P0: keep the queue on the right science surface

- running:
  - `2818` - `pp-screen-math5`
  - `2829` - `q3-lcb-thon`
- pending:
  - `2819` - `pp-screen-omni7`
  - `2830` - `q3-lcb-thoff`
- Do not relaunch any steered thinking-on job until a thinking-on-trained vector bundle exists.

### P1: close the `LiveCodeBench` thinking-on path

1. Finish `2829` and freeze the exact thinking-on stats receipt.
2. Materialize the prompt-level `majority_s_0.5` stage object on those thinking-on rollouts.
3. Train the thinking-on layerwise RFM and compare it against prompt-only / activation baselines on that same mode-local object.
4. Export the thinking-on vector bundle and rerun bootstrap direction-stability diagnostics.
5. Rerun the full seven-condition thinking-on steering table from those thinking-on vectors:
   - `no_steer`
   - `minus_v_linear`
   - `plus_v_linear`
   - `random_linear`
   - `minus_v_spherical`
   - `plus_v_spherical`
   - `random_spherical`

### P2: close the `LiveCodeBench` thinking-off path

1. Finish `2830` and freeze the exact thinking-off stats receipt.
2. Materialize the prompt-level `majority_s_0.5` stage object on those thinking-off rollouts.
3. Train the thinking-off RFM and export the thinking-off vector bundle.
4. Rerun the non-thinking steering table from that matched bundle.
5. If node geometry is still fragile, use a narrow linear-first order:
   - `no_steer`
   - `minus_v_linear`
   - `plus_v_linear`
   - `random_linear`
   then extend to spherical once the first mode-consistent non-thinking row lands.

### P3: keep screening and promotion mode-local

- The current screening pilot is still useful, but it is prevalence-first rather than admission-final.
- For any dataset promoted beyond `LiveCodeBench`, require a mode-tagged collector receipt on the chosen path before probe training.
- Promotion rule still stays literal inside each path:
  - only promote datasets whose repaired prompt-majority positive rate stays `>= 10%`
- Immediate candidate order from the current pilot:
  - `LiveCodeBench-extra`
  - `TACO-hard`
  - `MATH level-5`
  - `Omni-MATH >= 7`

### P4: keep vector pooling and transfer mode-local too

- Do not mix thinking-on and thinking-off bundles into one average vector.
- Any future average-vector transfer needs:
  - at least two readable benchmark-local vector bundles
  - in the same mode
  - with report-style receipts

## Defer until the two-path core is closed

- Reusing the old thinking-on `plus_v_linear` row as if it were valid causal evidence
- Any rerun of the canceled cross-mode steering jobs
- Averaged cross-benchmark transfer before two mode-local bundles exist
- Non-thinking spherical expansion before the first matched non-thinking linear row lands
- RFM multiseed / split-seed sensitivity as a blocker for the two-path LiveCodeBench core

## Runtime and infra notes

- The positive-enrichment pilot is still running from the home worktree with home-backed caches / outputs because `/data` is effectively full.
- Treat the current pilot outputs as candidate-discovery artifacts, not as silent mode-local admissions into the steering registry.
