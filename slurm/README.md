# SLURM Scripts

This directory contains SLURM workflows for the CoT loop detector project.

## Scripts

- `run_vllm_generate.sbatch`: Generate trajectories used for loop-label collection and detector analysis.
- `analyze_prefill_stability.sbatch`: Prefill-loop sanity check and stability checks with greedy rollouts.
- `run_probe_train_e2e.sbatch`: End-to-end probe pipeline for the canonical stacked prefill dataset (build + probe training, including optional multi-seed runs).
- `run_prompt_profile_projection.sbatch`: One-GPU prompt-profile visualization path (build + prompt-level projection export, with optional render when `matplotlib` is available).
- `run_k5_threeview_dataset.sbatch`: Historical multi-view dataset build for the k=5 / max_tokens=15000 ablation study.
- `run_k5_threeview_ablation.sbatch`: Historical MLP sweep over the k=5 three-view ablation dataset.

## Detector E2E Defaults

`run_probe_train_e2e.sbatch` defaults to:
- `MODEL_PRESET=openthinker3_1p5b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- explicit rollout/runtime overrides: `MODEL_ID=...`, `TEMPERATURE=...`, `TP=...`, `DP=...`, `MAX_MODEL_LEN=...`, `MAX_NUM_BATCHED_TOKENS=...`
- LiveCodeBench prompt inputs: `LIVECODEBENCH_REPO=...`, `RELEASE_VERSION=release_v6`, optional `LM_STYLE_OVERRIDE=...`
- optional repeated-rollout override for prompt-level targets: `NUM_GENERATIONS=...`
- optional prefill throughput override (single GPU): `PREFILL_BATCH_SIZE=...` (default: `32`)
- optional rollout-completion feature throughput override: `COMPLETION_BATCH_SIZE=...` (default: `1`)
- task-aware prompt formatting: `TASK_KIND=math_freeform|multiple_choice_gpqa|multiple_choice_mmlupro`
- prompt-level target controls:
  - `TARGET_KIND=binary` (legacy loop bit by default, or prompt-majority tail labels with `BINARY_TARGET_MODE=prompt_majority_tail`), `TARGET_KIND=probability` (prompt-level `s_t` soft target), or `TARGET_KIND=regression` (prompt-level continuous target)
  - `BINARY_TARGET_MODE=rollout_label|prompt_majority_tail`
  - `PROFILE_TAIL_THRESHOLD=0.9` for `s_0.9`
  - `PROFILE_TARGET=majority_tail` for prompt-majority binary, or `PROFILE_TARGET=mean_relative_length` for the dense realized-length regression head
  - `SCORE_RULE=mean_prob` for non-binary ensembles
- canonical prefill dataset controls:
  - `FEATURE_POOLING=last_token_all_layers_stack`
  - `FEATURE_LAYER=-1`
  - `FEATURE_KEY=<name>` (optional manifest key for the stacked view)
  - `TRAIN_EXTRA_ARGS="--classifier-mode ensemble"` to train one voting probe per layer instead of the default final-layer slice
- `TRAIN_DATASET=HuggingFaceH4/MATH-500`
- `TRAIN_SPLIT=test`
- `TEST_DATASET` omitted by default (falls back to `data/aime_2024_2025.jsonl`)
- `TEST_SPLIT=test`
- `PROMPT_FIELD=problem`
- `PROBE_PRESET=mlp`

Submit with defaults:
```bash
sbatch slurm/run_probe_train_e2e.sbatch
```

When multiple seeds are used (default: `0 1 2`), the script also writes:
- `${OUT_RUN_DIR}/probe_multiseed_curves.png` (aggregated train/eval curves from `seed_*/metrics.jsonl`)

Override values with exported environment variables or inline `VAR=... sbatch ...`.

Example: build the default stacked dataset and train an ensemble over all layers:
```bash
FEATURE_POOLING=last_token_all_layers_stack \
TRAIN_EXTRA_ARGS="--classifier-mode ensemble" \
sbatch slurm/run_probe_train_e2e.sbatch
```

Example: run the prompt-level `s_0.9` GPQA-style path with per-layer ensemble averaging:
```bash
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TARGET_KIND=probability \
PROFILE_TAIL_THRESHOLD=0.9 \
NUM_GENERATIONS=10 \
TEMPERATURE=0.2 \
TRAIN_CONFIG=gpqa_diamond \
TEST_CONFIG=gpqa_diamond \
PROMPT_FIELD=Question \
TRAIN_EXTRA_ARGS="--classifier-mode ensemble" \
SCORE_RULE=mean_prob \
sbatch slurm/run_probe_train_e2e.sbatch
```

Example: run the prompt-majority `majority_s_0.5` GPQA-style path with per-layer majority vote:
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

Example: run the prompt-level mean-length-fraction regression path:
```bash
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TARGET_KIND=regression \
PROFILE_TARGET=mean_relative_length \
NUM_GENERATIONS=10 \
TEMPERATURE=0.2 \
TRAIN_CONFIG=gpqa_diamond \
TEST_CONFIG=gpqa_diamond \
PROMPT_FIELD=Question \
TRAIN_EXTRA_ARGS="--classifier-mode last_layer --classifier-layer -1" \
SCORE_RULE=mean_prob \
sbatch slurm/run_probe_train_e2e.sbatch
```

## Prompt-Level Projection Defaults

`run_prompt_profile_projection.sbatch` defaults to:

- `#SBATCH --gres=gpu:1`
- prompt-profile build only (no probe training)
- `PROFILE_TAIL_THRESHOLD=0.5` as the primary saved majority label
- additional export thresholds `0.5 0.6 0.9`
- prompt-level projection outputs under `OUT_PROJECTION_DIR/export` and figures under `OUT_PROJECTION_DIR/figures`

Example: build and export the one-dot-per-prompt GPQA view:

```bash
MODEL_ID=Qwen/Qwen3-1.7B \
TASK_KIND=multiple_choice_gpqa \
TRAIN_DATASET=data/gpqa_diamond.csv \
TEST_DATASET=data/gpqa_diamond.csv \
TRAIN_SPLIT=train \
TEST_SPLIT=train \
TRAIN_MAX_SAMPLES=158 \
TEST_MAX_SAMPLES=40 \
PROMPT_FIELD=Question \
NUM_GENERATIONS=4 \
TEMPERATURE=0.2 \
MAX_TOKENS=30000 \
MAX_MODEL_LEN=40960 \
TP=1 \
DP=1 \
PREFILL_BATCH_SIZE=8 \
MAX_NUM_SEQS=4 \
MAX_NUM_BATCHED_TOKENS=1024 \
OUT_DATA_DIR=outputs/prompt_profile_projection_gpqa/data \
OUT_PROJECTION_DIR=outputs/prompt_profile_projection_gpqa \
FIGURE_LABEL=GPQA \
sbatch slurm/run_prompt_profile_projection.sbatch
```

## Optional Trajectory Generation

Use `run_vllm_generate.sbatch` to produce labeled rollouts for detector benchmarking:
```bash
sbatch --export=ALL,MODEL_ID=open-thoughts/OpenThinker3-7B,TP=1,DP=8,NUM_REPETITION=1,METRICS_OUT=outputs/openthinker3_7b_metrics.rep1.csv \
    slurm/run_vllm_generate.sbatch
```

To force a specific vLLM cache root:
```bash
VLLM_CACHE_ROOT=/data/users/zhiwang/cache/vllm \
sbatch slurm/run_vllm_generate.sbatch
```

For same-source train/test splitting in `run_probe_train_e2e.sbatch`, when both `TRAIN_*` and `TEST_*` refer to the same dataset and split:
- if both `TRAIN_MAX_SAMPLES` and `TEST_MAX_SAMPLES` are set, the builder creates exact disjoint subsets of those sizes;
- otherwise it uses a random ratio split (`SPLIT_RATIO`) and then applies any provided caps.
