# CoT Loop Detection Backlog

Last updated: 2026-04-23 23:28 UTC

Reference docs:
- `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`
- `docs/prompt-profile-rfm-third-stage-steering-plan.md`
- `docs/prompt-profile-rfm-artifact-schema-2026-04-21.md`
- `docs/understand-where-loop-and-max-length-come-from.md`

## Third-stage steering backlog

The steering backlog is now governed by one canonical doc:
`docs/prompt-profile-rfm-third-stage-steering-plan.md`.

Core rules:

- Every scientific steering row must close one mode-local chain:
  `stats -> prompt-level materialization -> probe/RFM training -> vector export -> steering`.
- Thinking `on` and thinking `off` are separate objects. Do not use cross-mode
  rows or mixed-mode averaged vectors as stage evidence.
- The steering contract is fixed for the first real tables:
  - prefill-only intervention;
  - all prompt tokens steered;
  - every block steered by its own block-specific vector;
  - both linear and spherical conditions;
  - full source-manifest decode budget, not a reduced pilot cap.
- Dataset/mode admission into steering requires a rebuilt mode-local
  `majority_s_0.5` positive rate of at least `10%`.
- `LiveCodeBench` remains the first steering benchmark because it has the
  cleanest grader and the least ambiguous code-generation evaluation path. The
  new claim must use vectors trained from the rebuilt mode-local archives.

Immediate steering TODOs after the rebuilt stats archives land:

1. Materialize `LiveCodeBench` thinking-on and thinking-off prompt-profile
   objects from the rebuilt archives.
2. Train prompt-only, activation, and RFM detector tables on both mode-local
   `LiveCodeBench` objects.
3. Export thinking-on and thinking-off block-specific vector bundles plus
   direction diagnostics.
4. Run the seven-condition full-contract `LiveCodeBench` steering table in
   thinking `on`.
5. Run the seven-condition full-contract `LiveCodeBench` steering table in
   thinking `off`.
6. Promote `TACO-hard`, `MATH level-5`, and `Omni-MATH >= 7` path-by-path only
   after their rebuilt mode-local materializations pass the gate and have
   detector/vector receipts.

## Fixed current object

- The active rollout-stat task starts from the new rebuilt mode-local rollout
  archives, not from any earlier steering bundle.
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

1. Recompute prompt-level labels from the new prompt-rollout archives.
2. Build the mode-tagged prompt-profile objects for all four retained datasets from the rebuilt archives.
3. Keep thinking `on` and `off` as separate prompt-profile objects all the way through detector training.

### P2: train mode-local probes and vectors from the rebuilt data

1. Train probe / RFM surfaces only on rebuilt mode-matched archives.
2. Export vector bundles only after the corresponding rebuilt prompt-profile objects exist.
3. Keep benchmark-local and mode-local provenance explicit in the vector metadata.

### P3: only restart steering after the rebuilt stats and detector surfaces exist

1. Use only vector bundles exported from the rebuilt mode-local archive for the same dataset and thinking mode.
2. Do not treat the canceled `2829` to `2838` rerun thread, the canceled `2850` to `2864` slice queue, the dropped `LiveCodeBench-extra` jobs `2876` / `2881`, or the failed `2865` to `2874` env-drop queue as valid scientific receipts.
3. Restart steering only after a rebuilt mode-local detector/vector object exists for the relevant dataset.

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
