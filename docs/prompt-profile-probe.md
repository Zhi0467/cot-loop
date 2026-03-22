# Prompt-Profile Probe Path

Last updated: 2026-03-22 07:30 UTC

## What Landed

The repo now has one runnable prompt-level repeated-rollout path for prefill probes:

- repeated rollouts per prompt via `--num-generations`;
- prompt-level binary-majority targets via `--target-kind binary --binary-target-mode prompt_majority_tail`;
- prompt-level probability targets via `--target-kind probability`;
- prompt-level regression targets via `--target-kind regression --profile-target mean_relative_length`;
- tail-probability targets `s_t = P(L / E >= t)` with `--profile-target s_tail` and `--profile-tail-threshold`;
- direct rate targets `p_loop` and `p_cap` from the same prompt-level rollout archive;
- dense realized-length regression via `mean_relative_length = E[L / E]`;
- task-aware prompt formatting via `--task-kind`, so `GPQA` / `MMLU-Pro` use multiple-choice prompt templates and `LiveCodeBench` can reuse its codegen prompt builder through `--livecodebench-repo`;
- prompt-profile diagnostics written to `diagnostics/train_prompt_profile.jsonl` and `diagnostics/test_prompt_profile.jsonl`;
- one combined repeated-rollout archive written to `diagnostics/prompt_rollout_archive.jsonl`, with prompt text, prompt token IDs, rollout texts, and per-rollout terminal stats so later relabels do not require rerollout;
- trainer/eval support for probability targets with Brier / MAE / Spearman / top-bucket capture metrics;
- trainer/eval support for regression targets with MSE / MAE / Spearman / top-bucket capture metrics;
- ensemble scoring with mean layer probability via `--score-rule mean_prob`.

## Current Recommendation

The next prompt-profile heads should be treated in two tiers:

- current shipped utility head: `mean_relative_length = E[L / E]`;
- cleaner loop-prox study head: `p_loop = E[1{rollout loops}]`;
- keep `p_cap` diagnostic-first, not the headline target;
- keep `majority_s_t` as a sparse pilot label, not the main objective.

Why this is the current recommendation:

- `s_0.9` already failed on the first real `GPQA` pilot because it collapsed to `p_cap` on that slice;
- `majority_s_0.5` does show real activation signal, but with `n = 4` it throws away most of the rollout-count information and on `AIME` is already mostly explained by prompt length;
- `p_loop` is already computed in the archive, stays closer to the failure mode we care about than raw length, and on both saved `GPQA` and `AIME` slices it is less prompt-length-correlated than `mean_relative_length`;
- the 2026-03-22 direct-head relabel check on the same `GPQA` archive showed that `p_loop` is not yet the most reliable *useful* head under the current training/selection rules: its ensemble run had a decent early ranking epoch (`eval Spearman 0.320`, top-20% capture `0.364`) but the default Brier-first checkpoint rule drifted toward near-constant predictions;
- the same relabel check showed that `mean_relative_length` is currently the stronger deployable head on this surface: the ensemble reached `eval Spearman 0.433` at the default MSE-selected checkpoint, and `0.658` at the best ranking epoch, both above the prompt-length-only baseline on that test split.

## Scope

This path is still intentionally narrow:

- prefill feature views only;
- one scalar head only;
- no completion-view repeated-rollout support yet;
- no joint multi-head trainer yet;
- balancing remains available only for binary targets; the prompt-profile probability and regression heads still run on the natural prompt-disjoint split.

## Recommended First ID Run

Target:

- ship `mean_relative_length = E[L / E]` first;
- run `p_loop = E[1{loop}]` in parallel when the goal is cleaner loop-prox supervision rather than immediate utility.

Decode policy:

- `temperature = 0.2`
- fixed `num_generations` per dataset (`4` for the current prompt-majority pilot surface, `10` for denser GPQA-style runs)
- fixed `max_tokens` / `max_model_len`

Feature views:

- one selected layer: `--classifier-mode last_layer --classifier-layer -1`
- per-layer ensemble: `--classifier-mode ensemble --score-rule mean_prob`

Evaluation boundary:

- keep the train/test split prompt-disjoint;
- always benchmark against prompt-token-count-only and effective-budget-only baselines;
- for `mean_relative_length`, prefer the ensemble view and do not rely only on MSE when the downstream goal is ranking or top-bucket capture;
- for `p_loop`, treat the current default Brier-first checkpoint rule as provisional because it can hide the better ranking epoch on small pilot splits;
- keep `p_cap`, correctness, and the prompt-majority controls as downstream diagnostics on the same prompts.

Example dataset build:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset <gpqa-source> \
  --train-config gpqa_diamond \
  --train-split train \
  --test-dataset <gpqa-source> \
  --test-config gpqa_diamond \
  --test-split train \
  --train-max-samples <train_n> \
  --test-max-samples <test_n> \
  --prompt-field Question \
  --task-kind multiple_choice_gpqa \
  --model-id Qwen/Qwen3-1.7B \
  --temperature 0.2 \
  --num-generations 10 \
  --max-tokens <cap> \
  --max-model-len <ctx> \
  --target-kind probability \
  --profile-target p_loop \
  --feature-pooling last_token_all_layers_stack \
  --feature-layer -1 \
  --out-dir outputs/gpqa_p_loop_prefill_dataset
```

Training:

```bash
python scripts/train_probe.py \
  --data-dir outputs/gpqa_p_loop_prefill_dataset \
  --out-dir outputs/gpqa_p_loop_prefill_run \
  --probe-preset mlp \
  --classifier-mode ensemble \
  --score-rule mean_prob \
  --wandb-project cot-loop-probe
```

## Backup Objective

The best no-reroll way to compare prompt-level heads now is:

- `python scripts/relabel_prompt_profile_dataset.py --source-dir <finished_prompt_profile_data_dir> --out-dir <new_data_dir> --target-kind regression --profile-target mean_relative_length`
- `python scripts/relabel_prompt_profile_dataset.py --source-dir <finished_prompt_profile_data_dir> --out-dir <new_data_dir> --target-kind probability --profile-target p_loop`

That helper reuses the saved prefill activations and `diagnostics/prompt_rollout_archive.jsonl`, so target swaps do not require a second rollout bundle or a second prefill pass.

`mean_relative_length` remains the best current shipped head because it is dense, stable, already implemented, and now has same-archive evidence that the ensemble readout can beat prompt length on `GPQA`. It remains a proxy rather than the cleanest main head because it mixes correct long reasoning, wrong long reasoning, and looped long reasoning.

Example dataset build:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset <gpqa-source> \
  --train-config gpqa_diamond \
  --train-split train \
  --test-dataset <gpqa-source> \
  --test-config gpqa_diamond \
  --test-split train \
  --prompt-field Question \
  --task-kind multiple_choice_gpqa \
  --model-id Qwen/Qwen3-1.7B \
  --temperature 0.2 \
  --num-generations 10 \
  --max-tokens <cap> \
  --max-model-len <ctx> \
  --target-kind regression \
  --profile-target mean_relative_length \
  --feature-pooling last_token_all_layers_stack \
  --feature-layer -1 \
  --out-dir outputs/gpqa_mean_rel_prefill_dataset
```

## Prompt-Majority Binary Pilot

The current all-dataset pilot head is still useful as a sparse control:

- `majority_s_0.5 = 1[\sum_r 1[L / E >= 0.5] > n / 2]`
- shared decode policy: `temperature = 0.2`, `num_generations = 4`
- feature surface: last prompt-token prefill activations only
- probe comparison: final-layer MLP vs per-layer ensemble MLP with vote aggregation

Keep it because it already showed that the prefill signal is above the prompt-length baseline on `GPQA`; do not treat it as the best next scalar objective.

## SLURM Launch Surface

`slurm/run_probe_train_e2e.sbatch` now accepts:

- `MODEL_ID=Qwen/Qwen3-1.7B` for non-preset models
- `TASK_KIND=multiple_choice_gpqa`
- `LIVECODEBENCH_REPO=/path/to/LiveCodeBench` plus `RELEASE_VERSION=release_v6` when `TASK_KIND=livecodebench_codegen`
- `TEMPERATURE=0.2`
- `MAX_MODEL_LEN=<ctx>`
- `TP=...`, `DP=...`, `MAX_NUM_BATCHED_TOKENS=...`
- `TARGET_KIND=binary|probability|regression`
- `BINARY_TARGET_MODE=rollout_label|prompt_majority_tail`
- `PROFILE_TAIL_THRESHOLD=<t>` for `s_t` heads
- `PROFILE_TARGET=s_tail|p_loop|p_cap|mean_relative_length|majority_tail`
- `NUM_GENERATIONS=...`
- `SCORE_RULE=mean_prob`

Example:

```bash
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TARGET_KIND=probability \
PROFILE_TARGET=p_loop \
NUM_GENERATIONS=10 \
TEMPERATURE=0.2 \
TRAIN_DATASET=<gpqa-source> \
TRAIN_CONFIG=gpqa_diamond \
TEST_DATASET=<gpqa-source> \
TEST_CONFIG=gpqa_diamond \
PROMPT_FIELD=Question \
MAX_TOKENS=<cap> \
MAX_MODEL_LEN=<ctx> \
TRAIN_EXTRA_ARGS="--classifier-mode ensemble" \
SCORE_RULE=mean_prob \
sbatch slurm/run_probe_train_e2e.sbatch
```

Repeated-rollout runs also leave one reusable archive at
`diagnostics/prompt_rollout_archive.jsonl`, so later prefill-activation plots
and relabels can reuse the same prompts without a second rollout bundle. The
new `scripts/relabel_prompt_profile_dataset.py` helper reuses that archive plus
the saved prefill shards directly, so prompt-level target swaps also avoid a
second feature-extraction pass.

## Validation Caveat

The pure-Python target aggregation path has been smoke-checked locally, and the archive-relabel path has now been exercised remotely on the saved `GPQA` prompt-profile dataset. The remaining open issue is not target math or relabel plumbing; it is which checkpoint-selection rule best matches the desired notion of usefulness for prompt-level heads on small pilot splits.
