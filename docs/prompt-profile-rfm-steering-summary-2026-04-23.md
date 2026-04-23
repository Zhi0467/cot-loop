# Prompt-Profile RFM Steering Summary

Last updated: 2026-04-23 19:50 UTC

## Stage object

This thread settled the steering stage on one benchmark first: `LiveCodeBench`.

The repaired prompt-level anchor is still the same `majority_s_0.5` object:

- fit-train / val / test:
  - `280 / 128 / 160`
- positives:
  - `140 / 35 / 54`

The correction is about provenance and execution order, not about dropping `LiveCodeBench`.

## Locked rules

1. Keep each path mode-consistent end to end.
   - thinking `on`: stats collection -> prompt-level materialization -> RFM -> vector export -> steering, all on thinking-enabled rollouts
   - thinking `off`: the same chain, all on thinking-disabled rollouts
2. Do not use cross-mode steering rows as scientific evidence.
   - the older thinking-on steered rows used vectors from the older non-thinking/raw predecessor bundle
   - those rows now count only as runner receipts
3. Use the corrected steering surface.
   - prefill-only intervention
   - all prompt tokens are steered
   - each block gets its own block-specific direction
   - run both linear and spherical steering
4. Use the full decode budget, not the earlier `1024`-token pilot cap.
5. Keep transfer mode-local too.
   - no mixed thinking-on / thinking-off average vector
6. Treat the current screen lane as prevalence scouting only.
   - extra datasets only get promoted after a mode-tagged collector receipt on that same path still clears the `>= 10%` positive-rate gate

## Execution order

### Path A: thinking-on `LiveCodeBench`

1. Finish the mode-tagged thinking-on stats receipt.
2. Materialize the prompt-level `majority_s_0.5` object from those rollouts.
3. Train the thinking-on RFM on that object.
4. Export the block-specific thinking-on vectors.
5. Run the full seven-condition steering table:
   - `no_steer`
   - `minus_v_linear`
   - `plus_v_linear`
   - `random_linear`
   - `minus_v_spherical`
   - `plus_v_spherical`
   - `random_spherical`

### Path B: thinking-off `LiveCodeBench`

1. Finish the mode-tagged thinking-off stats receipt.
2. Materialize the prompt-level `majority_s_0.5` object from those rollouts.
3. Train the thinking-off RFM on that object.
4. Export the block-specific thinking-off vectors.
5. Run the thinking-off steering table from that matched bundle.

If the node is still fragile, do the thinking-off rerun linear-first:

- `no_steer`
- `minus_v_linear`
- `plus_v_linear`
- `random_linear`

Only extend to spherical after the first clean mode-consistent non-thinking row lands.

## What still counts from earlier work

- The existing repaired March/raw vector bundle is still useful as predecessor context for the thinking-off lane.
- The finished thinking-on `no_steer` row can still serve as a baseline receipt if the prompt IDs and decode contract stay matched.
- The old steered thinking-on rows should not be used as stage evidence.

## Promotion rule after `LiveCodeBench`

Only after the matched `LiveCodeBench` paths exist should new datasets move into steering. The current screen leaderboard can still prioritize what comes next, but every promoted dataset has to re-enter through the same mode-local chain:

1. mode-tagged stats collection
2. prompt-level materialization
3. mode-local RFM/vector export
4. matched steering

## Longer artifacts

For the longer grounded version of the same plan, see:

- `prompt-profile-rfm-mode-consistent-stage-2026-04-23.md`
- `../outputs/prompt_profile_rfm_mode_consistent_stage_20260423/prompt_profile_rfm_mode_consistent_stage_20260423.pdf`
