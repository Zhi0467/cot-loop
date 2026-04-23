# Main Five-Dataset Rollout Rebuild — 2026-04-23

Last updated: 2026-04-23 17:02 UTC

## Scope

This note replaces the earlier "patch the March rows" framing for the current rollout-stat task. The canonical rebuild surface is now a paired thinking `on` / `off` collector run over five datasets:

- `LiveCodeBench`
- `LiveCodeBench-extra`
- `TACO-hard`
- `MATH level-5`
- `Omni-MATH >= 7`

The point is not just to refresh summary stats. These runs also need to leave reusable prompt-rollout archives for later prompt-profile relabeling, probe training, and steering.

This note also supersedes the earlier slice-based queue. After Wangzhi asked for full datasets where possible, or at least `1000` rows on larger sets, the old `2850` through `2864` queue stopped being the right contract.

## Canonical run contract

Shared collector settings:

- model: `Qwen/Qwen3-1.7B`
- temperature: `0.2`
- num generations: `10`
- max samples: dataset-specific, not global
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
  - active size: full dataset (`1055`)
- `LiveCodeBench-extra`
  - same as `LiveCodeBench`
  - disjointness enforced through `EXCLUDE_PROMPT_JSONL=/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_projection_livecodebench_majority05_seed0_20260323/data/diagnostics/prompt_rollout_archive.jsonl`
  - active size: full disjoint remainder (`255`)
  - implementation note:
    - exclusion now matches archived prompt text **or** archived benchmark `sample_id`
    - this repair was required because prompt-text-only exclusion stopped working once the collector switched to the HF chat-template prompt surface
- `TACO-hard`
  - task kind: `taco_codegen`
  - dataset: `BAAI/TACO`
  - config: `ALL`
  - split: `train`
  - row filter: `difficulty in {HARD, VERY_HARD}`
  - prompt surface: `PROMPT_FORMAT=chat_template`
  - verifier: native execution-based TACO grader over saved `input_output`
  - active size: `1000` of `5536`
- `MATH level-5`
  - task kind: `math_freeform`
  - dataset: `SuperSecureHuman/competition_math_hf_dataset`
  - split: `train`
  - row filter: `level == Level 5`
  - prompt surface: `PROMPT_FORMAT=chat_template`
  - active size: `1000` of `2304`
- `Omni-MATH >= 7`
  - task kind: `math_freeform`
  - dataset: `KbsdJames/Omni-MATH`
  - split: `test`
  - row filter: `difficulty >= 7`
  - prompt surface: `PROMPT_FORMAT=chat_template`
  - grader: the same `math_freeform` / `math_verify` path as `MATH level-5`
  - active size: full HF slice (`916`)
  - provenance note:
    - `data/omni_math_ge7_screen_300.jsonl` remains useful as the earlier screening artifact, but it is no longer the active stats source

## Runtime corrections landed during rebuild

Four real runtime/path bugs had to be fixed before the current suite was honest:

1. The TACO native grader was incorrectly rebinding top-level functions as instance methods, which made even trivial call-based toy programs fail. The grader now returns top-level callables directly and still instantiates `Solution` when that class form is present.
2. `BAAI/TACO` can no longer be loaded through the old `TACO.py` dataset script under the current `datasets` library. The shared dataset loader now falls back to the Hugging Face parquet surface `hf://datasets/BAAI/TACO/ALL/<split>-*.parquet`, which restored stable ingest.
3. `LiveCodeBench-extra` exclusion had been silently broken on the HF chat-template surface. The earlier exclusion path only matched prompt text, but the archived March exclusion bundle was built from raw prompt strings. The collector now also excludes by archived benchmark `sample_id`, which restores the intended `255`-row disjoint remainder.
4. The first corrected relaunch still died immediately because the sbatch wrapper did not propagate the runtime env. `scripts/launch_main_rollout_stats_suite.py` now forwards `CONDA_ENV` / `CONDA_DEFAULT_ENV` into submitted jobs and fails fast if neither is present.

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

### Size and disjointness receipts

The active queue now matches Wangzhi's size rule exactly:

- `LiveCodeBench`: `1055`
- `LiveCodeBench-extra`: `255`
- `TACO-hard`: `5536`, so use `1000`
- `MATH level-5`: `2304`, so use `1000`
- `Omni-MATH >= 7`: `916`

The `LiveCodeBench-extra` sidecar now proves that the exclusion repair is active on the live HF-chat queue:

- `exclude_prompt_jsonl` points at the archived March bundle
- `excluded_prompt_count = 800`
- `excluded_archive_prompt_count = 800`
- `excluded_archive_sample_id_count = 800`

So the new `LiveCodeBench-extra` run is no longer silently overlapping the archived March prompt pool.

## Live submission

Fresh clean checkout used for submission:

- `/data/scratch/murphy/projects/worktrees/cot-loop-main4-rebuild`

Output root:

- `/data/scratch/murphy/outputs/cot-loop-detection/main_five_dataset_rebuild_full_or_1k_20260423`

Suite manifest:

- `/data/scratch/murphy/outputs/cot-loop-detection/main_five_dataset_rebuild_full_or_1k_20260423/suite_manifest.json`
- schema: `main_rollout_stats_suite.v2`
- suite-level `max_samples`: `null`

Queue history:

- canceled old slice queue:
  - `2850` through `2857`
  - `2863` and `2864`
- first corrected relaunch:
  - `2865` through `2874`
  - failed immediately before first row because the jobs started without `CONDA_ENV`
- live corrected relaunch:
  - `2875` `q3-main5r2b-livecodebench-on`
  - `2876` `q3-main5r2b-livecodebench_extra-on`
  - `2877` `q3-main5r2b-taco_hard-on`
  - `2878` `q3-main5r2b-math_level5-on`
  - `2879` `q3-main5r2b-omni_math_ge7-on`
  - `2880` `q3-main5r2b-livecodebench-off`
  - `2881` `q3-main5r2b-livecodebench_extra-off`
  - `2882` `q3-main5r2b-taco_hard-off`
  - `2883` `q3-main5r2b-math_level5-off`
  - `2884` `q3-main5r2b-omni_math_ge7-off`

Queue state at note time:

- `2875` and `2876` are `RUNNING` on `wth-gpu-01`
- `2877` is `PENDING (Resources)`
- `2878` through `2884` are `PENDING (Priority)`

## What this note demotes

The earlier March-provenance / narrow-`LiveCodeBench` repair thread is now historical debugging context. Do not treat the canceled or failed `2829`–`2838` reruns, the canceled slice queue `2850`–`2864`, or the env-dropped relaunch `2865`–`2874` as the active science surface for this stage. The active object is the five-dataset paired rebuild described here.
