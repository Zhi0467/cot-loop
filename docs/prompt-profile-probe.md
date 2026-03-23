# Prompt-Profile Probe Path

Last updated: 2026-03-23 03:57 UTC

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
- trainer checkpoint emission for both `best_loss` and ranking-oriented `best_rank`, with backward-compatible `best.pt -> best_loss`;
- ensemble scoring with mean layer probability via `--score-rule mean_prob`.

## Current Recommendation

The next prompt-profile heads should be treated as a two-head default:

- current shipped utility head: `mean_relative_length = E[L / E]`;
- default loop-prox companion head: `p_loop = E[1{rollout loops}]`;
- ship `best_rank` for utility-facing prompt ranking and keep `best_loss` for calibration-style reporting;
- keep `p_cap` diagnostic-first, not the headline target;
- keep `majority_s_t` as a sparse pilot label, not the main objective.

Why this is the current recommendation:

- `s_0.9` already failed on the first real `GPQA` pilot because it collapsed to `p_cap` on that slice;
- `majority_s_0.5` does show real activation signal, but with `n = 4` it throws away most of the rollout-count information and on `AIME` is already mostly explained by prompt length;
- `p_loop` is already computed in the archive, stays closer to the failure mode we care about than raw length, and on both saved `GPQA` and `AIME` slices it is less prompt-length-correlated than `mean_relative_length`;
- the first same-archive `GPQA` / `AIME` relabel checks showed that loss-best selection was hiding the useful epochs on small pilots, especially for `p_loop`;
- the trainer now writes both `best_loss` and `best_rank`, so the shipped checkpoint can be aligned with ranking utility without dropping the calibration-first view;
- under that updated read, `mean_relative_length` remains the safer shipped utility head, while `p_loop` stays in the default bundle because it is cleaner and materially less prompt-length-shaped on the heavier-tail slices.

## Scope

This path is still intentionally narrow:

- prefill feature views only;
- one scalar head per fit;
- default workflow is now two separate fits from one repeated-rollout archive rather than one joint multi-head run;
- no completion-view repeated-rollout support yet;
- no joint multi-head trainer yet;
- balancing remains available only for binary targets; the prompt-profile probability and regression heads still run on the natural prompt-disjoint split.

## Recommended First ID Run

Target:

- train `mean_relative_length = E[L / E]` and `p_loop = E[1{loop}]` from the same repeated-rollout archive by default;
- use `mean_relative_length` as the shipped utility score;
- keep `p_loop` as the default failure-prox companion score rather than a dropped ablation.

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
- ship `best_rank` for utility-facing ranking and keep `best_loss` beside it for calibration-style reporting;
- keep `p_cap`, correctness, and the prompt-majority controls as downstream diagnostics on the same prompts.

Checkpoint selection:

- `train_probe.py` now writes `best_loss.pt`, `best_rank.pt`, and backward-compatible `best.pt` (aliasing `best_loss`);
- for `mean_relative_length`, `best_rank` picks the max-Spearman epoch among checkpoints within `10%` of best MSE, then tie-breaks by top-20% capture and lower MSE;
- for `p_loop`, `best_rank` picks the max top-20% capture epoch among checkpoints within `10%` of best Brier, then tie-breaks by Spearman and lower Brier;
- use `best_rank` as the default shipped checkpoint and keep `best_loss` as the calibration/control checkpoint;
- for serious sweeps, still compare the chosen checkpoint against the stronger non-degenerate metadata baseline on a validation slice before treating the head as shipped-useful.

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

## Archive Relabel Path

The best no-reroll way to fit both prompt-level heads now is:

- `python scripts/relabel_prompt_profile_dataset.py --source-dir <finished_prompt_profile_data_dir> --out-dir <new_data_dir> --target-kind regression --profile-target mean_relative_length`
- `python scripts/relabel_prompt_profile_dataset.py --source-dir <finished_prompt_profile_data_dir> --out-dir <new_data_dir> --target-kind probability --profile-target p_loop`

That helper reuses the saved prefill activations and `diagnostics/prompt_rollout_archive.jsonl`, so target swaps do not require a second rollout bundle or a second prefill pass.

`mean_relative_length` remains the shipped utility head because it is dense, stable, already implemented, and now has same-archive evidence that the ensemble readout can beat prompt length on `GPQA`, `AIME`, and the lower-tail controls. It remains a proxy rather than the cleanest main head because it mixes correct long reasoning, wrong long reasoning, and looped long reasoning. `p_loop` remains the cleaner companion head for failure proximity, especially on the heavier-tail datasets.

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

The pure-Python target aggregation path has been smoke-checked locally, and the archive-relabel path has now been exercised remotely on the saved `GPQA` and `AIME` prompt-profile datasets. The remaining open issue is no longer target math or relabel plumbing; it is making the shipped checkpoint rule match the intended downstream use. Small pilot splits can let `min(Brier)` / `min(MSE)` choose nearly constant checkpoints even when an earlier epoch has much stronger ranking utility.
