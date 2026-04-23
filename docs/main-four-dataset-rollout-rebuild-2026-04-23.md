# Main Four-Dataset Rollout Rebuild — 2026-04-23

Last updated: 2026-04-23 05:55 UTC

## Scope

This note replaces the earlier "patch the March rows" framing for the current rollout-stat task. The new canonical rebuild surface is a paired thinking `on` / `off` collector run over four datasets:

- `LiveCodeBench`
- `LiveCodeBench-extra`
- `TACO-hard`
- `MATH level-5`

The point is not just to refresh summary stats. These runs also need to leave reusable prompt-rollout archives for later prompt-profile relabeling, probe training, and steering.

## Canonical run contract

Shared collector settings:

- model: `Qwen/Qwen3-1.7B`
- temperature: `0.2`
- num generations: `10`
- max samples: `800`
- max tokens: `81920`
- max model len: `40960`
- tp / dp: `1 / 1`
- dtype: `bfloat16`
- seed: `0`
- max num seqs: `10`
- max num batched tokens: `4096`

Dataset-specific contract:

- `LiveCodeBench`
  - task kind: `livecodebench_codegen`
  - dataset: `livecodebench/code_generation_lite`
  - split: `test`
  - release: `release_v6`
  - prompt surface: `LM_STYLE_OVERRIDE=HFChatTemplate`
- `LiveCodeBench-extra`
  - same as `LiveCodeBench`
  - disjointness enforced through `EXCLUDE_PROMPT_JSONL=/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_projection_livecodebench_majority05_seed0_20260323/data/diagnostics/prompt_rollout_archive.jsonl`
- `TACO-hard`
  - task kind: `taco_codegen`
  - dataset: `BAAI/TACO`
  - config: `ALL`
  - split: `train`
  - row filter: `difficulty in {HARD, VERY_HARD}`
  - prompt surface: `PROMPT_FORMAT=chat_template`
  - verifier: native execution-based TACO grader over saved `input_output`
- `MATH level-5`
  - task kind: `math_freeform`
  - dataset: `SuperSecureHuman/competition_math_hf_dataset`
  - split: `train`
  - row filter: `level == Level 5`
  - prompt surface: `PROMPT_FORMAT=chat_template`

## Runtime corrections landed during rebuild

Two real bugs had to be fixed before this suite was runnable:

1. The TACO native grader was incorrectly rebinding top-level functions as instance methods, which made even trivial call-based toy programs fail. The grader now returns top-level callables directly and still instantiates `Solution` when that class form is present.
2. `BAAI/TACO` can no longer be loaded through the old `TACO.py` dataset script under the current `datasets` library. The shared dataset loader now falls back to the Hugging Face parquet surface `hf://datasets/BAAI/TACO/ALL/<split>-*.parquet`, which restored stable ingest.

The shared collector wrapper also now carries an explicit `THINKING_MODE` environment knob instead of relying on ad hoc extra arguments.

## Validation receipts

### Stale job cleanup

Before launching the rebuild, stale Murphy-owned positive-screening jobs were canceled off the GPU node:

- `2843` (`math_level5_remaining24` resume lane)
- `2845` (`Omni-MATH >= 7` positive-screen lane)

That cleanup was necessary because they were stealing GPUs from the rebuild even though they were no longer the active task object.

### Chat-template check

Qwen's chat template was verified directly on the runtime env:

- `thinking_mode=on` yields the plain assistant prefix with no `<think>` block
- `thinking_mode=off` injects the empty `<think>\n\n</think>\n\n` block

So the new paired control surface is explicit rather than inferred.

### TACO smoke

One GPU smoke run landed at:

- `/data/scratch/murphy/outputs/cot-loop-detection/main_four_dataset_smoke/taco_hard_on_smoke.json`

Verified archive surface:

- prompt archive row carries `record_id`
- full `prompt` text is saved
- `prompt_token_ids` are saved
- rollout rows carry `completion_text`
- rollout rows carry `completion_token_ids`
- raw `record_metadata` is preserved, including `input_output`

This is the reuse surface needed for later probe training and steering.

## Live submission

Fresh clean checkout used for submission:

- `/data/scratch/murphy/projects/worktrees/cot-loop-main4-rebuild`

Output root:

- `/data/scratch/murphy/outputs/cot-loop-detection/main_four_dataset_rebuild_20260423`

Suite manifest:

- `/data/scratch/murphy/outputs/cot-loop-detection/main_four_dataset_rebuild_20260423/suite_manifest.json`

Submitted jobs:

- `2850` `q3-main4r1-livecodebench-on`
- `2851` `q3-main4r1-livecodebench_extra-on`
- `2852` `q3-main4r1-taco_hard-on`
- `2853` `q3-main4r1-math_level5-on`
- `2854` `q3-main4r1-livecodebench-off`
- `2855` `q3-main4r1-livecodebench_extra-off`
- `2856` `q3-main4r1-taco_hard-off`
- `2857` `q3-main4r1-math_level5-off`

Queue state at note time:

- `2850` and `2851` are already `RUNNING`
- `2852` through `2857` are queued behind them

## What this note demotes

The earlier March-provenance / narrow-`LiveCodeBench` repair thread is now historical debugging context. Do not treat the canceled or failed `2829`–`2838` reruns as the active science surface for this stage. The active object is the four-dataset paired rebuild described here.
