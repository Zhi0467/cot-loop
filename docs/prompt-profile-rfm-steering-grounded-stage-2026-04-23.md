# Prompt-Profile RFM Steering Grounded Stage Note

Last updated: 2026-04-23 02:20 UTC

## Exact current object

The active benchmark-local stage is still the repaired `LiveCodeBench` prompt-level `majority_s_0.5` object, not the old stale March archive label and not a mixed cross-benchmark table.

- Source train pool: `640` prompts
- Fit-train: `280` prompts with `140` positives
- Validation: `128` prompts with `35` positives
- Test: `160` prompts with `54` positives
- Steering evaluation contract:
  - full test split, not the old `32`-prompt pilot
  - full decode budget `30000`
  - prefill-only
  - all prompt tokens steered at every selected block
  - block-specific linear and spherical conditions on the repaired exported vector bundle

This note supersedes the April 21 plan as the current status surface. The April 21 note is still useful as the stage design document, but it no longer matches the exact live repo/runtime state.

## What is already proved

### Detector and vector bundle

The detector-side object is already real enough to treat as finished stage evidence.

- Repaired detector row:
  - RFM layer `27`
  - validation `PR-AUC 0.6555`
  - test `PR-AUC 0.7055`
  - test `ROC-AUC 0.8590`
- Honest comparison on the same repaired object:
  - RFM is above prompt-only baselines
  - RFM is above activation linear baselines
  - activation `h256 d1` MLP last-layer is tied or slightly ahead depending on checkpoint rule
- Direction-quality read:
  - exported per-layer vectors are stable enough to use causally
  - all `28` layers clear mean bootstrap cosine `>= 0.781`
  - late layers `23` to `26` are the most coherent

So the detector/vector lane is no longer the blocker for the next stage. The real open work is steering and dataset expansion.

### Finished full-contract steering rows

Only two rows are scientifically finished on the current full-contract `160`-prompt thinking-on table:

| Condition | Pass@1 | Loop fraction | Over-half-budget fraction | Avg gen length | Median gen length |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_steer` | `0.0125` (`2 / 160`) | `0.65625` | `0.6375` | `17921.375` | `18199.5` |
| `plus_v_linear` | `0.0125` (`2 / 160`) | `0.6125` | `0.6000` | `17453.1875` | `17201.0` |

That is a real receipt, but it is not a finished steering table. The current evidence is only that `plus_v_linear` looks modestly better than the thinking-on baseline on loop fraction while leaving accuracy unchanged. It is not yet enough to claim a signed linear effect, because the paired negative, random, and spherical rows are still live rather than complete.

## What is live but unfinished

### Thinking-on steering table

The remaining five rows of the corrected full-contract thinking-on table are still running on `wth-gpu-01`:

- `2804` - `minus_v_linear`
- `2810` - `random_linear`
- `2811` - `minus_v_spherical`
- `2815` - `plus_v_spherical`
- `2816` - `random_spherical`

The right immediate deliverable is therefore one finished seven-row thinking-on table:

- `no_steer`
- `minus_v_linear`
- `plus_v_linear`
- `random_linear`
- `minus_v_spherical`
- `plus_v_spherical`
- `random_spherical`

Do not start `t` sweeps, layer-restriction ablations, or controller variants until that table exists.

### Stage-0.5 positive-enrichment screen

The screening gate is now materially beyond the speculative phase. Current progress sidecars on the node say:

| Candidate | Profiled prompts | Prompt-majority positives | Positive rate | Completion tail frac | Loop frac | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `LiveCodeBench-extra` | `255` | `141` | `0.5529` | `0.5765` | `0.3029` | running |
| `TACO-hard` | `213` | `172` | `0.8075` | `0.8263` | `0.3779` | running |
| `MATH level-5` (parallel path) | `180` | `25` | `0.1389` | `0.2069` | `0.0833` | running |
| `Omni-MATH >= 7` | `0` | `0` | `0.0000` | -- | -- | dependency-pending |

Current read:

- `LiveCodeBench-extra` is already far above the `>= 10%` gate and should be treated as a likely promote-on-finish candidate.
- `TACO-hard` is even more positive-rich, although it still lacks the same grader-backed sanity anchor as the `LiveCodeBench` and math-family lanes.
- `MATH level-5` is no longer a tiny-slice anecdote; at `180` profiled prompts it is still above the gate, so math-family promotion is now plausible rather than hypothetical.
- `Omni-MATH >= 7` is still blocked only by the dependency chain behind `2818`.

### March-prompt-surface provenance check

The explicit `LiveCodeBench` `HFChatTemplate` replay that isolates thinking-mode provenance is queued but not started:

- `2829` - `--thinking-mode on`
- `2830` - `--thinking-mode off`

That lane answers a real provenance question, but it is not the immediate blocker for the benchmark-local steering table. It should finish, but it should not be mistaken for the main steering result.

## What is blocked

### Non-thinking steering surface

Every attempted non-thinking `LiveCodeBench` launch so far died before first row:

- `2821` - `no_steer`
- `2822` - `minus_v_linear`
- `2823` - `plus_v_linear`
- `2825` - `no_steer` retry
- `2826` - `random_linear` retry

All five failed on the same dirty-slot CUDA OOM shape on the only GPU node, so there is still no non-thinking condition summary on disk. That means:

- there is no scientific non-thinking read yet;
- the blocker is slot geometry, not model behavior;
- repeated blind relaunches are wasted work until a clean slot opens or current jobs release enough space.

The next non-thinking action should therefore be:

1. wait for a genuinely clean slot or for the current thinking-on jobs to finish;
2. relaunch a narrow serial table first:
   - `no_steer`
   - `minus_v_linear`
   - `plus_v_linear`
   - `random_linear`
3. only extend non-thinking to spherical once at least one non-thinking row has landed.

## Repo reality

The repo surface is split too:

- local project branch: `task/1776752262-rfm-stage0`
- local worktree head before this grounded-note patch: `a255ff1`
- published GitHub surface: draft PR `#11`
- published PR head at last check: `5a521d1`
- unresolved non-outdated review threads on the published PR surface: `1`

So the published PR is behind the actual local stage state. Any collaborator-facing decision should use the local docs plus the live node receipts, not the older PR description alone.

There is also one runtime debt that now matters operationally:

- `/data` is effectively full, so the positive-enrichment screen is currently running with home-backed caches under `/home/murphy/.cache/cot-loop-positive-screening/...` rather than the normal scratch-backed path.

That is not yet a science blocker, but it is part of the real execution contract for the current screen.

## Grounded stage order

### P0: Finish what is already in flight

1. Finish the seven-row thinking-on `LiveCodeBench` steering table on the repaired `160 / 54` test object.
2. Finish the current `300`-prompt screening runs:
   - `LiveCodeBench-extra`
   - `TACO-hard`
   - `MATH level-5`
   - `Omni-MATH >= 7`
3. Let the queued `HFChatTemplate` `thinking on/off` provenance pair run when resources open.

### P1: Unblock the non-thinking comparison the right way

1. Stop burning retries while the node is still contaminated.
2. Once a clean slot exists, rerun the narrow non-thinking linear table first.
3. Treat "first row lands" as the actual gate. Before that, the non-thinking lane is infrastructure, not science.

### P2: Promote the first new screened-in dataset(s)

Once the current `300`-prompt screens finish:

- promote only candidates whose final repaired prompt-majority train positive rate stays `>= 10%`;
- keep exact prompt/provenance sidecars as part of the admission receipt;
- preserve exact prompt-text disjointness for `LiveCodeBench-extra`.

Planned promotion order after the current screen finishes:

1. `LiveCodeBench-extra`
   - same benchmark family as the current finished object
   - already clearly above the gate
   - has the cleanest evaluator/provenance story
2. `TACO-hard`
   - likely promote if the final rate holds
   - still needs explicit "prevalence-first" wording because the grader surface is weaker
3. `MATH level-5`
   - promote only if the final `300`-prompt pass stays above the gate
4. `Omni-MATH >= 7`
   - evaluate only after the dependency chain completes

### P3: Expand the vector pool only after promotion receipts exist

For each promoted dataset:

1. materialize the repaired `majority_s_0.5` object;
2. train the benchmark-local RFM detector;
3. export block-specific vectors;
4. run the same bootstrap direction-stability diagnostics;
5. then and only then add that dataset to any cross-benchmark vector pool.

### P4: Average-vector transfer stays later

Do not reopen the averaged external "verbose vector" test until at least two non-`LiveCodeBench` benchmark-local bundles have:

- passed the screen,
- been materialized,
- exported stable directions,
- and produced readable benchmark-local receipts.

## Backlog deltas from the April 21 plan

The plan itself was directionally right, but the grounded backlog now differs in five concrete ways:

1. The steering lane is split into three separate surfaces:
   - finished thinking-on rows,
   - live but unfinished thinking-on rows,
   - blocked non-thinking rows.
2. The screening lane is no longer just a first-launch TODO; it already has serious candidate-level evidence.
3. The March-prompt-surface provenance pair is real work, but it is not the main stage result.
4. The published PR surface is behind local reality, so the docs must carry that drift explicitly.
5. The old `32`-prompt steering pilot is obsolete as a scientific result; it survives only as implementation history.

## Artifact

Report-style PDF bundle:

- `outputs/prompt_profile_rfm_stage_grounded_plan_20260423/prompt_profile_rfm_stage_grounded_plan_20260423.pdf`

This note is the durable Markdown companion for that PDF.
