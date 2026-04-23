# Prompt-Profile RFM Steering Mode-Consistent Stage Note

Last updated: 2026-04-23 04:22 UTC

## Exact correction

At 2026-04-23 03:50 UTC, Wangzhi tightened the stage contract again: stop steering the thinking model with vectors trained on the older non-thinking rollout surface. From here forward, each path has to be internally consistent:

- thinking-on path:
  - stats collection -> stage materialization -> probe training -> vector export -> steering
  - all on thinking-enabled rollouts
- thinking-off path:
  - the same chain
  - all on thinking-disabled rollouts

This note supersedes the earlier grounded note `docs/prompt-profile-rfm-steering-grounded-stage-2026-04-23.md` as the active stage definition.

## What remains valid

### LiveCodeBench stays the benchmark-local anchor

The benchmark-local core object is still the repaired `LiveCodeBench` prompt-level `majority_s_0.5` split:

- fit-train / val / test:
  - `280 / 128 / 160`
- positives:
  - `140 / 35 / 54`

That repaired object is still the right first benchmark for the two-path stage. The correction is about mode consistency, not about abandoning `LiveCodeBench`.

### The old detector/vector bundle still matters, but only as the non-thinking-side predecessor

The existing repaired detector/vector bundle under
`/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
is still a real asset. It gives:

- detector row:
  - layer `27`
  - validation `PR-AUC 0.6555`
  - test `PR-AUC 0.7055`
  - test `ROC-AUC 0.8590`
- direction quality:
  - all `28` layers clear mean bootstrap cosine `>= 0.781`
  - weakest `95%` low bound `0.693`
  - late layers `23` to `26` remain the most coherent

But its provenance is now important in a new way. The exported vector records point back to:

- `preprocessing.source_data_dir = /data/scratch/murphy/outputs/cot-loop-detection/livecodebench_mean_relative_from_archive_20260323`

and the March provenance audit already established that `LiveCodeBench` source as the old raw `CodeQwenInstruct` path rather than the newer `HFChatTemplate` thinking-on surface. So this bundle is usable as the predecessor for the non-thinking lane, not as the canonical probe bundle for thinking-on steering.

### The current screening pilot is still useful, but only as candidate discovery

The positive-enrichment pilot is materially farther along than the 02:20 UTC grounded note said:

| Candidate | Profiled prompts | Prompt-majority positives | Positive rate | Completion tail frac | Loop frac | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `LiveCodeBench-extra` | `255` | `141` | `0.5529` | `0.5765` | `0.3029` | running |
| `TACO-hard` | `229` | `186` | `0.8122` | `0.8308` | `0.3723` | running |
| `MATH level-5` parallel path | `261` | `40` | `0.1533` | `0.2126` | `0.0738` | running |
| `Omni-MATH >= 7` | `0` | `0` | `0.0000` | -- | -- | dependency-pending |

These pilot screens are still valuable for deciding which dataset families deserve follow-up. But they are not yet sufficient to admit a dataset into either steering lane, because the current pilot outputs do not yet serve as explicit mode-tagged stats receipts for both paths.

## What is now demoted

### The steered thinking-on rows are implementation receipts, not valid stage evidence

The old thinking-on steering table is no longer the main scientific object.

The finished `plus_v_linear` row under
`/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm_steering/livecodebench_test160_plus_v_linear_full30000_alltokens_seed0_20260421_split/`
still proves something useful:

- full decode budget `30000`
- hook site `prefill_layer_output_all_tokens`
- all-prompt-token prefill intervention
- linear steering ledgers and diagnostics all ran end to end

But its `config.json` points to:

- `vector_export_dir = /data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/vector_exports`

which is the non-thinking predecessor bundle above. Under the new rule, that means:

- the finished `plus_v_linear` thinking-on row is not valid stage evidence for thinking-on causal steering;
- the still-running steered thinking-on jobs were on the wrong science surface too.

At 2026-04-23 04:22 UTC I canceled the remaining live cross-mode steering jobs:

- `2804` - `minus_v_linear`
- `2811` - `minus_v_spherical`
- `2815` - `plus_v_spherical`
- `2816` - `random_spherical`

Use the old steered thinking-on rows only as implementation receipts for the runner. Do not use them as the main stage result.

### The `no_steer` thinking-on row survives only as a reusable baseline

The finished `no_steer` row on the thinking-on model surface does not violate the new rule, because it does not apply any vector. Its current receipt is:

- `pass@1 = 2 / 160 = 0.0125`
- loop fraction `0.65625`
- over-half-budget fraction `0.6375`
- average generation length `17921.375`
- median generation length `18199.5`

That row may be reusable as the baseline for the future thinking-on table if the prompt IDs and generation contract stay identical. But it is no longer enough to say the thinking-on steering lane is partially complete, because the matched steered rows now have to be redone from thinking-on-trained probes.

## Live queue after the correction

After canceling the cross-mode steering jobs, the queue looks like this:

- running:
  - `2818` - `pp-screen-math5`
  - `2829` - `q3-lcb-thon`
- pending:
  - `2819` - `pp-screen-omni7` (dependency on `2818`)
  - `2830` - `q3-lcb-thoff` (resources)

So the correction already changed the execution surface in one useful way: the explicit thinking-on `LiveCodeBench` collector `2829` started immediately once the invalid steering jobs were removed from the only GPU node.

## Repo reality

The repo state is split the same way the science state is split:

- local project branch:
  - `task/1776752262-rfm-stage0`
- local head before this note:
  - `dabe924`
- published GitHub surface:
  - draft PR `#11`
  - published head still `5a521d1`

So the collaborator-facing truth is again ahead of the published PR body. The docs need to carry the correction explicitly.

## Grand two-path schema

### Shared rule

For any mode-specific steering claim, the chain has to be closed inside that same mode:

1. collect rollouts on that mode;
2. materialize the prompt-level `majority_s_0.5` object from those rollouts;
3. train the benchmark-local RFM on that mode's prompt-prefill activations;
4. export the signed block-specific vectors from that mode's RFM;
5. steer the same mode with those vectors.

No cross-mode shortcut is now allowed for the main claim.

### Path A: thinking-on `LiveCodeBench`

Current state:

- explicit collector job `2829` is now running;
- no thinking-on-trained RFM bundle exists yet;
- therefore no valid thinking-on steered row exists yet.

Required backlog:

1. finish `2829` and freeze the exact thinking-on `LiveCodeBench` stats receipt;
2. materialize the prompt-level `majority_s_0.5` stage object on those thinking-on rollouts;
3. train the layerwise thinking-on RFM and compare it to prompt-only / activation baselines on that same mode-local object;
4. export the thinking-on block-specific vectors and rerun the bootstrap direction-stability diagnostics;
5. rerun the full seven-condition thinking-on steering table with those thinking-on vectors:
   - `no_steer`
   - `minus_v_linear`
   - `plus_v_linear`
   - `random_linear`
   - `minus_v_spherical`
   - `plus_v_spherical`
   - `random_spherical`

### Path B: thinking-off `LiveCodeBench`

Current state:

- explicit collector job `2830` is pending;
- the older March/raw repaired bundle still exists as a useful predecessor;
- but the mode-consistent stage should converge on the explicit thinking-off collector, not leave the old raw bundle as the only canonical source;
- actual non-thinking steering still has zero finished rows because every prior launch died on dirty-slot CUDA OOM before first row.

Required backlog:

1. finish `2830` and freeze the exact thinking-off `LiveCodeBench` stats receipt;
2. materialize the prompt-level `majority_s_0.5` object on those thinking-off rollouts;
3. train the thinking-off RFM and export the thinking-off vector bundle;
4. rerun the non-thinking steering table from that matched bundle;
5. if node geometry is still fragile, use a narrow linear-first launch order:
   - `no_steer`
   - `minus_v_linear`
   - `plus_v_linear`
   - `random_linear`
   then extend to spherical once the first mode-consistent non-thinking row lands cleanly.

### Stage-0.5 screening under the new schema

The current screening pilot should now be treated as a prevalence-first scout rather than an automatic admission ledger.

Use it to choose what to follow up, but do not let it silently define the steering registry. For any dataset promoted beyond `LiveCodeBench`:

1. rerun mode-tagged stats collection on the chosen path;
2. confirm that the repaired positive rate still clears the `>= 10%` gate on that mode-local receipt;
3. only then train the mode-local probe/vector bundle.

Practical implication:

- `LiveCodeBench-extra`, `TACO-hard`, and `MATH level-5` still look like the right next candidates;
- but none of them should enter the two-path steering registry directly from the current default-surface pilot alone.

### Cross-benchmark transfer also becomes mode-local

The older plan delayed average-vector transfer until at least two new benchmark-local bundles existed. That delay still stands, but now the averaging rule also has to respect mode:

- no mixed thinking-on / thinking-off average vector;
- build average vectors only within one mode-local pool.

## Mode-consistent execution order

### P0: keep the queue on the right science surface

1. keep `2829` running;
2. keep `2830` queued;
3. do not relaunch any steered thinking-on job until a thinking-on vector bundle exists.

### P1: finish the two `LiveCodeBench` collector receipts

1. `2829` - explicit thinking-on
2. `2830` - explicit thinking-off

These are now the real blockers for the two-path stage.

### P2: close the probe/vector loop separately inside each mode

For each mode, do:

1. stats collection
2. stage binary materialization
3. RFM training
4. vector export
5. bootstrap direction diagnostics

Do not jump directly from one lane's vectors to the other lane's steering job.

### P3: rerun steering separately inside each mode

1. thinking-on rerun from thinking-on vectors
2. thinking-off rerun from thinking-off vectors

Interpret old rows only as infrastructure receipts while these new tables are being built.

### P4: only then promote extra datasets path-by-path

Use the current screen leaderboard to prioritize the next collector reruns, but require explicit mode-local receipts before any new dataset gets a probe/vector bundle.

## Bottom line

The stage is no longer "finish the old thinking-on seven-row table and unblock non-thinking later." The real stage object is now two matched pipelines:

- thinking-on:
  - collect -> train -> steer
- thinking-off:
  - collect -> train -> steer

The old non-thinking-trained `LiveCodeBench` detector/vector bundle still matters, but it no longer licenses thinking-on steering claims. The valid next move is the one already reflected in the live queue: let the explicit thinking-on/off collectors land, then rebuild probe and steering receipts separately inside each path.
