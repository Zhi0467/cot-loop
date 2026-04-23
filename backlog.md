# CoT Loop Detection Backlog

Last updated: 2026-04-23 05:55 UTC

Reference docs:
- `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`
- `docs/prompt-profile-rfm-artifact-schema-2026-04-21.md`
- `docs/understand-where-loop-and-max-length-come-from.md`

## Fixed current object

- The active rollout-stat task is no longer "repair the March bundle" or "finish a LiveCodeBench-only thinking comparison."
- The current canonical rebuild surface is:
  - `LiveCodeBench`
  - `LiveCodeBench-extra`
  - `TACO-hard`
  - `MATH level-5`
- Every dataset is being collected twice:
  - thinking `on`
  - thinking `off`
- Shared collection contract:
  - model `Qwen/Qwen3-1.7B`
  - `temperature=0.2`
  - `num_generations=10`
  - `max_samples=800`
  - `max_tokens=81920`
  - `max_model_len=40960`
  - `tp=1`, `dp=1`
  - `max_num_seqs=10`
  - `max_num_batched_tokens=4096`
- Prompt/verifier contract:
  - `LiveCodeBench` and `LiveCodeBench-extra` use `LM_STYLE_OVERRIDE=HFChatTemplate`
  - `TACO-hard` and `MATH level-5` use `PROMPT_FORMAT=chat_template`
  - `TACO-hard` uses the native execution-based grader over saved `input_output`
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
  - `/data/scratch/murphy/outputs/cot-loop-detection/main_four_dataset_rebuild_20260423`
- Submitted suite:
  - `2850` `q3-main4r1-livecodebench-on`
  - `2851` `q3-main4r1-livecodebench_extra-on`
  - `2852` `q3-main4r1-taco_hard-on`
  - `2853` `q3-main4r1-math_level5-on`
  - `2854` `q3-main4r1-livecodebench-off`
  - `2855` `q3-main4r1-livecodebench_extra-off`
  - `2856` `q3-main4r1-taco_hard-off`
  - `2857` `q3-main4r1-math_level5-off`
- Current queue state:
  - `2850` and `2851` are running
  - `2852` through `2857` are waiting behind them

## Active TODOs

### P0: keep the rebuild receipts clean

1. Monitor `2850` through `2857` until all eight receipts land.
2. Treat any first-row failure as a launch/runtime bug, not as a scientific result.
3. Preserve the paired contract if repairs are needed:
   - same dataset
   - same sampling config
   - same thinking tag
4. Keep the suite manifest and final JSON/archive outputs together under the same output root.

### P1: materialize the next prompt-profile objects from these rebuilt archives

1. Recompute prompt-level labels from the new prompt-rollout archives instead of reusing March-era bundle assumptions.
2. Build the mode-tagged prompt-profile objects for the four datasets from the rebuilt archives.
3. Keep thinking `on` and `off` as separate prompt-profile objects all the way through detector training.

### P2: train mode-local probes and vectors from the rebuilt data

1. Train probe / RFM surfaces only on rebuilt mode-matched archives.
2. Export vector bundles only after the corresponding rebuilt prompt-profile objects exist.
3. Keep benchmark-local and mode-local provenance explicit in the vector metadata.

### P3: only restart steering after the rebuilt stats and detector surfaces exist

1. Do not reuse the old March-era or LiveCodeBench-only vector bundles for this rebuild.
2. Do not treat the canceled `2829` to `2838` rerun thread as a valid scientific receipt.
3. Restart steering only after a rebuilt mode-local detector/vector object exists for the relevant dataset.

## Historical context

- The older March-provenance audit and the failed LiveCodeBench-only reruns are still useful debugging history, but they are no longer the active backlog surface.
- Keep those receipts in `roadmap.md` and the task report; do not let them redefine the current queue or the current project objective.
