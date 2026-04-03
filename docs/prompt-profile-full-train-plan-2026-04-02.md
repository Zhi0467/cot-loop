# Prompt-Profile Full Train Plan

Last updated: 2026-04-02 23:04 UTC

## Bottom line

- The next full run is locked to two train targets:
  - regression: `mean_relative_length`
  - binary: `majority_s_0.5`
- `p_loop` stays in the repo, but only as a downstream diagnostic target for this run.
- The purpose of this run is no longer to choose the target. That part is closed. The purpose is to measure how well the locked pair trains at the full saved prompt-profile surface under one frozen contract.

## Why this pair is locked

- `mean_relative_length` is the strongest current regression target by held-out fit on the saved five-dataset bundle.
  - Best activation `Spearman` is higher on `4 / 5` datasets.
  - Average best held-out `Spearman` is about `0.661` for `mean_relative_length` versus `0.372` for `p_loop`.
- `majority_s_0.5` is the strongest finished binary label surface that already has a full held-out classification read.
  - It is not a pure loop label.
  - It is still the cleanest binary thing we can train today without reopening the objective question.
- This run therefore keeps one dense continuous target and one mature binary target.

## Frozen run contract

- Model and policy:
  - `MODEL_ID=Qwen/Qwen3-1.7B`
  - `TEMPERATURE=0.2`
  - `NUM_GENERATIONS=4`
  - `MAX_TOKENS=30000`
  - `MAX_MODEL_LEN=40960`
- Loop detector:
  - `LOOP_N=30`
  - `LOOP_K=20`
- Feature surface:
  - prompt-prefill only
  - `FEATURE_POOLING=last_token_all_layers_stack`
  - `FEATURE_LAYER=-1`
  - default feature key remains `last_token_all_layers_stack_final`
- Split contract:
  - prompt-disjoint only
  - reuse the same saved count splits that produced the current decision surface
  - keep dataset seed fixed; vary optimizer seed only
- Probe surface:
  - main view: per-layer ensemble
  - cheap control: last-layer only
  - do not reopen stacked-all-layer single-MLP as the main surface
- Checkpoint rule:
  - use `best_loss` as the main fit checkpoint for the locked targets
  - keep `best_rank` only as a downstream diagnostic checkpoint
  - do not let a ranking-oriented checkpoint re-open target choice
- Non-activation baselines:
  - regression must keep the full metadata suite: `prompt_length`, `effective_budget`, and `prompt_length + effective_budget`
  - binary must keep the existing 1D controls already emitted by the prompt-profile surface: `prompt_length` and `effective_budget`

Why `NUM_GENERATIONS=4` stays fixed here:

- the locked target decision was made on the saved `num_generations=4` prompt-profile bundles
- changing archive density and target choice in the same run would move the target itself
- if a denser archive is wanted later, that should be a separate run after the locked full pass lands

## Dataset surface

These are the current saved full-surface prompt-profile splits. The full train run should reuse them exactly rather than changing prompt counts and objectives at the same time.

| Dataset | Task kind | Train prompts | Test prompts | Runtime notes |
| --- | --- | ---: | ---: | --- |
| `GPQA` | `multiple_choice_gpqa` | `158` | `40` | `DP=1`, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=1024` |
| `AIME` | `math_freeform` | `48` | `12` | `DP=1`, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=1024` |
| `MATH-500` | `math_freeform` | `400` | `100` | `DP=1`, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=1024` |
| `MMLU-Pro` | `multiple_choice_mmlupro` | `640` | `160` | `DP=1`, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=1024` |
| `LiveCodeBench` | `livecodebench_codegen` | `640` | `160` | keep the heavier saved path: `DP=2`, `MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=4096` |

Prompt fields remain:

- `GPQA`: `Question`
- `AIME`: `question`
- `MATH-500`: `problem`
- `MMLU-Pro`: `problem`
- `LiveCodeBench`: `problem`

## Exact TO DOs

1. Build one shared repeated-rollout archive per dataset under the locked regression surface.
   - Use `mean_relative_length` as the build target because it is dense and already emits the shared prompt archive and prefill shards we want to reuse.
   - Save one archive root per dataset under a stable `outputs/full_train/<dataset>/shared_archive/` path.
2. Train the regression head from that shared archive.
   - Main run: ensemble.
   - Cheap control: last-layer.
   - Optimizer seeds: `0 1 2`.
3. Relabel the same archive to the locked binary head.
   - Use `scripts/relabel_prompt_profile_dataset.py`.
   - `--target-kind binary`
   - `--profile-target majority_tail`
   - `--profile-tail-threshold 0.5`
   - `--balance-train downsample`
   - `--balance-test none`
   - No rerollout and no second prefill pass.
4. Train the binary head from the relabeled archive.
   - Main run: ensemble.
   - Cheap control: last-layer.
   - Optimizer seeds: `0 1 2`.
5. Score both locked heads against the required metadata controls on the same held-out prompts.
   - Regression: compare against `prompt_length`, `effective_budget`, and the joint baseline.
   - Binary: compare against the existing 1D prompt-geometry controls.
6. Write one cross-dataset run note that reports only the locked pair.
   - `p_loop`, `p_cap`, bucket concentration, and accuracy slices stay in the diagnostic lane.
   - They are allowed in the appendix, not as the selector.

## Main command pattern

The clean path is:

1. one build to create the shared archive
2. one regression train from that archive
3. one relabel to `majority_s_0.5`
4. one binary train from the relabeled archive

Example build-and-train launch for `GPQA` regression:

```bash
CONDA_ENV=swe311 \
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TARGET_KIND=regression \
PROFILE_TARGET=mean_relative_length \
NUM_GENERATIONS=4 \
TEMPERATURE=0.2 \
LOOP_N=30 \
LOOP_K=20 \
MAX_TOKENS=30000 \
MAX_MODEL_LEN=40960 \
MAX_NUM_SEQS=4 \
MAX_NUM_BATCHED_TOKENS=1024 \
TRAIN_DATASET=data/gpqa_diamond.csv \
TEST_DATASET=data/gpqa_diamond.csv \
TRAIN_SPLIT=train \
TEST_SPLIT=train \
TRAIN_MAX_SAMPLES=158 \
TEST_MAX_SAMPLES=40 \
PROMPT_FIELD=Question \
SEEDS="0 1 2" \
TRAIN_EXTRA_ARGS="--classifier-mode ensemble" \
SCORE_RULE=mean_prob \
OUT_DATA_DIR=outputs/full_train/gpqa/shared_archive \
OUT_RUN_DIR=outputs/full_train/gpqa/mean_relative_length/ensemble \
sbatch slurm/run_probe_train_e2e.sbatch
```

Example relabel for the binary head:

```bash
python scripts/relabel_prompt_profile_dataset.py \
  --source-dir outputs/full_train/gpqa/shared_archive \
  --out-dir outputs/full_train/gpqa/majority_s_0.5/data \
  --target-kind binary \
  --profile-target majority_tail \
  --profile-tail-threshold 0.5 \
  --balance-train downsample \
  --balance-test none
```

Example binary training command after relabel:

```bash
python scripts/train_probe.py \
  --data-dir outputs/full_train/gpqa/majority_s_0.5/data \
  --out-dir outputs/full_train/gpqa/majority_s_0.5/ensemble/seed_0 \
  --probe-preset mlp \
  --epochs 10 \
  --batch-size 256 \
  --seed 0 \
  --classifier-mode ensemble \
  --score-rule vote_fraction \
  --wandb-project cot-loop-probe
```

The same pattern should be repeated for the other four datasets with only the dataset-specific source fields and runtime knobs changed.

## Reporting rule for this run

- Regression should be judged by held-out fit on `mean_relative_length` itself plus metadata lift.
- Binary should be judged by held-out classification quality on `majority_s_0.5` itself.
- The old `top 20%` bucket test can still be reported, but only after the main target-fit ledger.
- Do not use downstream loop concentration to argue that `p_loop` should have been the target after the fact.

## Done criteria

This full run is complete only when all of the following exist for every dataset:

- one shared archive with `prompt_rollout_archive.jsonl`
- one regression dataset and one binary relabel from that same archive
- ensemble and last-layer runs for each locked target
- `best_loss` and `best_rank` metrics saved for each run
- one metadata-baseline comparison on the same held-out prompts
- one cross-dataset summary that stays on the locked pair

## Explicit non-goals

- do not reopen `p_loop` as a train target inside this run
- do not reopen `p_cap`
- do not change to OOD evaluation
- do not change prompt counts, rollout count, or decode policy mid-run
- do not mix target selection and downstream bucket concentration again
