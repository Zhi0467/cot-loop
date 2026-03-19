# Prompt-Profile Probe Path

Last updated: 2026-03-19 22:35 UTC

## What Landed

The repo now has a first runnable path for the Athena-backed single-head prompt-level target:

- repeated rollouts per prompt via `--num-generations`;
- prompt-level soft target build via `--target-kind probability`;
- first target family implemented as `s_t = P(L / E >= t)` with `--profile-tail-threshold` (use `0.9` for `s_0.9`);
- prompt-profile diagnostics written to `diagnostics/train_prompt_profile.jsonl` and `diagnostics/test_prompt_profile.jsonl`;
- trainer/eval support for soft targets with Brier / MAE / Spearman / top-bucket capture metrics;
- ensemble scoring can use mean layer probability (`--score-rule mean_prob`) instead of only hard-vote fraction.

## Current Scope

This v1 path is intentionally narrow:

- prefill feature views only;
- one scalar head only;
- no completion-view repeated-rollout support yet;
- no multi-head `s_0.9 + mu_log_rel` trainer yet;
- binary downsampling is disabled for prompt-profile targets.

## Recommended First GPQA Run

Target:
- `s_0.9 = P(L / E >= 0.9)`

Decode policy:
- `temperature=0.2`
- `num_generations=10`

Feature views:
- one selected layer: `--classifier-mode last_layer --classifier-layer <L>`
- per-layer ensemble: `--classifier-mode ensemble --score-rule mean_prob`

Dataset build:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset <gpqa-source> \
  --train-split train \
  --test-dataset <gpqa-source> \
  --test-split train \
  --train-max-samples <train_n> \
  --test-max-samples <test_n> \
  --prompt-field <prompt_field> \
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

## SLURM Launch Surface

`slurm/run_probe_train_e2e.sbatch` now accepts:

- `TARGET_KIND=probability`
- `PROFILE_TAIL_THRESHOLD=0.9`
- `NUM_GENERATIONS=10`
- `SCORE_RULE=mean_prob`

Example:

```bash
TARGET_KIND=probability \
PROFILE_TAIL_THRESHOLD=0.9 \
NUM_GENERATIONS=10 \
TRAIN_DATASET=<gpqa-source> \
TEST_DATASET=<gpqa-source> \
PROMPT_FIELD=<prompt_field> \
MAX_TOKENS=<cap> \
TRAIN_EXTRA_ARGS="--classifier-mode ensemble" \
SCORE_RULE=mean_prob \
sbatch slurm/run_probe_train_e2e.sbatch
```

## Validation Caveat

The code compiles cleanly, and the pure-Python target aggregation path was smoke-checked locally, but this workspace did not have a local Torch runtime or project virtualenv at hand. That means the remaining unverified piece in this session is the actual Torch-backed train/build execution path, not the CLI wiring or the target math itself.
