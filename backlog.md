# CoT Loop Detection Backlog

Last updated: 2026-04-23 19:50 UTC

Reference docs:
- `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`
- `docs/prompt-profile-rfm-steering-summary-2026-04-23.md`
- `docs/prompt-profile-rfm-artifact-schema-2026-04-21.md`
- `docs/understand-where-loop-and-max-length-come-from.md`

## Steering summary from the 2026-04-23 thread

- `LiveCodeBench` stays the first steering benchmark.
- The active steering claim is now two matched paths, not one mixed path:
  - thinking `on`: stats -> prompt-level materialization -> RFM -> vector export -> steering
  - thinking `off`: the same chain
- Do not use cross-mode rows as stage evidence.
  - the earlier thinking-on steered rows used the older non-thinking/raw vector bundle, so those rows are runner receipts only
- Keep the corrected steering contract:
  - prefill-only
  - all prompt tokens steered
  - block-specific direction at each block
  - both linear and spherical steering
  - full decode budget
- If the non-thinking path is still node-fragile, land the first clean mode-consistent row linear-first, then extend to spherical.
- Keep transfer mode-local too.
  - no mixed thinking-on / thinking-off average vector
- Treat the current screen lane as prevalence scouting only.
  - only promote a new dataset after a mode-tagged collector receipt on that same path still clears the `>= 10%` gate

## Fixed current object

- The active rollout-stat task is no longer "repair the March bundle" or "append Omni to the old `800`-prompt queue."
- The current canonical rebuild surface is:
  - `LiveCodeBench`
  - `TACO-hard`
  - `MATH level-5`
  - `Omni-MATH >= 7`
- Every dataset is being collected twice:
  - thinking `on`
  - thinking `off`
- Shared collection contract:
  - model `Qwen/Qwen3-1.7B`
  - `temperature=0.2`
  - `num_generations=10`
  - `max_tokens=81920`
  - `max_model_len=40960`
  - `tp=1`, `dp=1`
  - `max_num_seqs=10`
  - `max_num_batched_tokens=4096`
- Dataset-size contract:
  - `LiveCodeBench`: full dataset (`1055`)
  - `TACO-hard`: `1000` of `5536`
  - `MATH level-5`: `1000` of `2304`
  - `Omni-MATH >= 7`: full HF slice (`916`)
- Prompt/verifier contract:
  - `LiveCodeBench` uses `LM_STYLE_OVERRIDE=HFChatTemplate`
  - `TACO-hard`, `MATH level-5`, and `Omni-MATH >= 7` use `PROMPT_FORMAT=chat_template`
  - `TACO-hard` uses the native execution-based grader over saved `input_output`
  - `Omni-MATH >= 7` now uses `KbsdJames/Omni-MATH`, split `test`, with `difficulty >= 7`
  - `data/omni_math_ge7_screen_300.jsonl` remains a useful historical screen artifact but is no longer the active stats source
- Dataset correction:
  - `LiveCodeBench-extra` is dropped everywhere because it is a strict subset of `LiveCodeBench` on the same `release_v6` surface, not an independent benchmark lane
- Reuse contract:
  - finished archives must preserve `record_id`, prompt text, `prompt_token_ids`, rollout `completion_text`, `completion_token_ids`, and raw row metadata so the same rollouts can later drive prompt-profile relabeling, probe training, and steering

## Validated runtime surface

- Stale positive-screening jobs were canceled before relaunch:
  - `2843`
  - `2845`
- TACO-specific fixes that are now required knowledge:
  - the native grader must treat top-level functions as top-level callables rather than rebinding them as instance methods
  - `BAAI/TACO` must be loaded through the HF parquet surface because the old `TACO.py` dataset-script path is retired under the current `datasets` library
- Smoke receipts:
  - TACO GPU smoke:
    - `/data/scratch/murphy/outputs/cot-loop-detection/main_four_dataset_smoke/taco_hard_on_smoke.json`
  - that smoke proved the archive surface is sufficient for later reuse:
    - `record_id`
    - full prompt text
    - `prompt_token_ids`
    - rollout `completion_text`
    - rollout `completion_token_ids`
    - preserved `record_metadata`
- Chat-template control check:
  - thinking `on` leaves the plain assistant prefix
  - thinking `off` injects the empty `<think>\n\n</think>\n\n` block

## Live queue

- Fresh remote submission checkout:
  - `/data/scratch/murphy/projects/worktrees/cot-loop-main4-rebuild`
- Main output root:
  - `/data/scratch/murphy/outputs/cot-loop-detection/main_five_dataset_rebuild_full_or_1k_20260423`
- Queue history that matters:
  - the old slice-based jobs `2850` through `2857` and `2863` / `2864` were canceled after Wangzhi tightened the size contract
  - the first corrected relaunch `2865` through `2874` failed immediately because the sbatch wrapper dropped `CONDA_ENV`
  - `scripts/launch_main_rollout_stats_suite.py` now propagates `CONDA_ENV` / `CONDA_DEFAULT_ENV`
  - after Wangzhi pointed out that `LiveCodeBench-extra` is a strict subset of `LiveCodeBench`, I canceled its two live jobs (`2876`, `2881`) and removed it from the canonical suite definition
- Live submitted suite:
  - `2875` `q3-main5r2b-livecodebench-on`
  - `2877` `q3-main5r2b-taco_hard-on`
  - `2878` `q3-main5r2b-math_level5-on`
  - `2879` `q3-main5r2b-omni_math_ge7-on`
  - `2880` `q3-main5r2b-livecodebench-off`
  - `2882` `q3-main5r2b-taco_hard-off`
  - `2883` `q3-main5r2b-math_level5-off`
  - `2884` `q3-main5r2b-omni_math_ge7-off`
- Current queue state:
  - `2875` (`LiveCodeBench`, on) is running
  - `2877` (`TACO-hard`, on) is running
  - `2878` is pending on resources
  - `2879`, `2880`, `2882`, `2883`, and `2884` are pending on priority

## Active TODOs

### P0: keep the rebuild receipts clean

1. Monitor `2875`, `2877`, `2878`, `2879`, `2880`, `2882`, `2883`, and `2884` until all eight retained receipts land.
2. Treat any first-row failure as a launch/runtime bug, not as a scientific result.
3. Preserve the paired contract if repairs are needed:
   - same dataset
   - same sampling config
   - same thinking tag
4. Keep the suite manifest and final JSON/archive outputs together under the same output root.

### P1: materialize the next prompt-profile objects from these rebuilt archives

1. Recompute prompt-level labels from the new prompt-rollout archives instead of reusing March-era bundle assumptions.
2. Build the mode-tagged prompt-profile objects for all four retained datasets from the rebuilt archives.
3. Keep thinking `on` and `off` as separate prompt-profile objects all the way through detector training.

### P2: train mode-local probes and vectors from the rebuilt data

1. Train probe / RFM surfaces only on rebuilt mode-matched archives.
2. Export vector bundles only after the corresponding rebuilt prompt-profile objects exist.
3. Keep benchmark-local and mode-local provenance explicit in the vector metadata.

### P3: only restart steering after the rebuilt stats and detector surfaces exist

1. Do not reuse the old March-era or LiveCodeBench-only vector bundles for this rebuild.
2. Do not treat the canceled `2829` to `2838` rerun thread, the canceled `2850` to `2864` slice queue, the dropped `LiveCodeBench-extra` jobs `2876` / `2881`, or the failed `2865` to `2874` env-drop queue as valid scientific receipts.
3. Restart steering only after a rebuilt mode-local detector/vector object exists for the relevant dataset.

## Historical context

- The older March-provenance audit and the failed LiveCodeBench-only reruns are still useful debugging history, but they are no longer the active backlog surface.
- Keep those receipts in `roadmap.md` and the task report; do not let them redefine the current queue or the current project objective.

## Retained reference surfaces

- The earlier prompt-profile and trigger-attention surfaces from `main` still stand as reference objects even though they are not the active rollout-stat queue:
  - unified prompt-profile report: `docs/prompt-profile-unified-report-2026-04-09.md`
  - natural regression rerun: `docs/prompt-profile-natural-regression-rerun-2026-04-05.md`
  - corrected OLMo degeneration-origin audit: `docs/olmo-degeneration-origin-audit-2026-04-04.md`
  - binary capacity controls: `docs/prompt-profile-binary-capacity-controls-2026-04-04.md`
  - full Qwen3 trigger-attention rerun: `docs/qwen3-loop-trigger-attention-2026-04-14.md`
- Keep the earlier reporting caveats explicit when those older surfaces are cited again:
  - the trigger-attention replay still needs a matched non-loop control slice
  - the fixed-budget full-train prompt-profile surface still has constant `effective_budget`, so "metadata baseline" there effectively means prompt-length-only unless a stronger prompt-shape control is added
  - use `Spearman rank correlation` explicitly rather than shorthand `rank correlation`
  - do not describe the project goal as merely "ranking prompts"; the real object is target selection for prompt-prefill prediction
- Keep the earlier data gaps visible:
  - the original repaired `LiveCodeBench` row still lacks exact `avg_first_loop_prefix_length`
  - the old recovered `LiveCodeBench` projection artifact still lacks prompt-level correctness
