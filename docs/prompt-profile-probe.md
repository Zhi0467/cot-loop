# Prompt-Profile Probe Path

Last updated: 2026-03-22 00:28 UTC

## What Landed

The repo now has a first runnable path for the current prompt-level target plan:

- repeated rollouts per prompt via `--num-generations`;
- prompt-level binary-majority targets via `--target-kind binary --binary-target-mode prompt_majority_tail`;
- prompt-level soft-target build via `--target-kind probability`;
- prompt-level regression build via `--target-kind regression --profile-target mean_relative_length`;
- first target family implemented as `s_t = P(L / E >= t)` with `--profile-tail-threshold` (use `0.9` for `s_0.9`);
- binary-majority head implemented as `majority_s_t = 1[\sum_r 1[L_r / E >= t] > n / 2]`;
- second single-head objective implemented as `mean_relative_length = E[L / E]` across repeated rollouts;
- task-aware prompt formatting via `--task-kind`, so `GPQA` and `MMLU-Pro` use their multiple-choice prompt templates, while `LiveCodeBench` can reuse its codegen prompt builder through `--livecodebench-repo`;
- prompt-profile diagnostics written to `diagnostics/train_prompt_profile.jsonl` and `diagnostics/test_prompt_profile.jsonl`;
- one combined repeated-rollout archive written to `diagnostics/prompt_rollout_archive.jsonl`, with prompt text, prompt token IDs, rollout texts, and the precomputed aggregate labels/metadata for each prompt;
- trainer/eval support for probability targets with Brier / MAE / Spearman / top-bucket capture metrics;
- trainer/eval support for regression targets with MSE / MAE / Spearman / top-bucket capture metrics;
- ensemble scoring can use mean layer probability (`--score-rule mean_prob`) instead of only hard-vote fraction.

## Current Scope

This v1 path is intentionally narrow:

- prefill feature views only;
- one scalar head only;
- no completion-view repeated-rollout support yet;
- no joint multi-head trainer yet;
- balancing remains available only for binary targets; the soft-target and regression prompt-profile heads still do not support downsampling.

## Prompt-Majority Binary Pilot

The current all-dataset pilot head is:

- `majority_s_0.5 = 1[\sum_r 1[L / E >= 0.5] > n / 2]`
- shared decode policy: `temperature=0.2`, `num_generations=4`
- feature surface: last prompt-token prefill activations only
- probe comparison: final-layer MLP vs per-layer ensemble MLP with majority vote

Example dataset build:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset <dataset-source> \
  --train-split <split> \
  --test-dataset <dataset-source> \
  --test-split <split> \
  --train-max-samples <train_n> \
  --test-max-samples <test_n> \
  --prompt-field <field> \
  --task-kind <task-kind> \
  --model-id Qwen/Qwen3-1.7B \
  --temperature 0.2 \
  --num-generations 4 \
  --max-tokens 30000 \
  --max-model-len 40960 \
  --target-kind binary \
  --binary-target-mode prompt_majority_tail \
  --profile-target majority_tail \
  --profile-tail-threshold 0.5 \
  --feature-pooling last_token_all_layers_stack \
  --feature-layer -1 \
  --out-dir outputs/prompt_majority_binary_dataset
```

Training examples:

```bash
python scripts/train_probe.py \
  --data-dir outputs/prompt_majority_binary_dataset \
  --out-dir outputs/prompt_majority_last_layer \
  --probe-preset mlp \
  --classifier-mode last_layer \
  --classifier-layer -1 \
  --wandb-project cot-loop-probe

python scripts/train_probe.py \
  --data-dir outputs/prompt_majority_binary_dataset \
  --out-dir outputs/prompt_majority_layerwise_vote \
  --probe-preset mlp \
  --classifier-mode ensemble \
  --score-rule vote_fraction \
  --wandb-project cot-loop-probe
```

## Recommended First GPQA Run

Target:
- `s_0.9 = P(L / E >= 0.9)`

Decode policy:
- `temperature=0.2`
- `num_generations=10`

Feature views:
- one selected layer: `--classifier-mode last_layer --classifier-layer -1`
- per-layer ensemble: `--classifier-mode ensemble --score-rule mean_prob`

Evaluation boundary:
- train and evaluate `s_0.9` directly;
- keep `p(max_length_hit)` as diagnostic-only if desired, not the headline target;
- benchmark against prompt-token-count and `E = max_model_len - prompt_len` leakage baselines outside the probe.

Dataset build:

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
  --profile-tail-threshold 0.9 \
  --feature-pooling last_token_all_layers_stack \
  --feature-layer -1 \
  --out-dir outputs/gpqa_s09_prefill_dataset
```

Training:

```bash
python scripts/train_probe.py \
  --data-dir outputs/gpqa_s09_prefill_dataset \
  --out-dir outputs/gpqa_s09_prefill_run \
  --probe-preset mlp \
  --classifier-mode ensemble \
  --score-rule mean_prob \
  --wandb-project cot-loop-probe
```

## Second Objective

The second single-head path is dense regression on the mean realized fraction:

- target: `mean_relative_length = E[L / E]`;
- CLI: `--target-kind regression --profile-target mean_relative_length`;
- trainer loss: sigmoid-MSE on the repeated-rollout aggregate.

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
- `PROFILE_TAIL_THRESHOLD=0.9`
- `PROFILE_TARGET=mean_relative_length` for the regression head
- `NUM_GENERATIONS=10`
- `SCORE_RULE=mean_prob`

Example:

```bash
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TARGET_KIND=probability \
PROFILE_TAIL_THRESHOLD=0.9 \
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

Prompt-majority binary example:

```bash
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TARGET_KIND=binary \
BINARY_TARGET_MODE=prompt_majority_tail \
PROFILE_TARGET=majority_tail \
PROFILE_TAIL_THRESHOLD=0.5 \
NUM_GENERATIONS=4 \
TEMPERATURE=0.2 \
TRAIN_CONFIG=gpqa_diamond \
TEST_CONFIG=gpqa_diamond \
PROMPT_FIELD=Question \
TRAIN_EXTRA_ARGS="--classifier-mode ensemble" \
SCORE_RULE=vote_fraction \
sbatch slurm/run_probe_train_e2e.sbatch
```

Repeated-rollout runs now also leave a single reusable archive at
`diagnostics/prompt_rollout_archive.jsonl`, so later prefill-activation plots can reuse the same prompts and prompt-level labels without generating a second rollout bundle.

## Validation Caveat

The code compiles cleanly, and the pure-Python target aggregation path was smoke-checked locally, but this workspace did not have a local Torch runtime or project virtualenv at hand. That means the remaining unverified piece in this session is the actual Torch-backed train/build execution path, not the CLI wiring or the target math itself.
