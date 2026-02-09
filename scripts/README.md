# Scripts: Loop Detection Analysis

This directory contains scripts for analyzing repetitive loops in LLM generation and reproducing Figure 1 from the paper.

## Overview

These scripts analyze chain-of-thought trajectories for repetitive patterns (n-gram loops) and prefill activation stability. They are distinct from the probe training pipeline (see main [README.md](../README.md)).

SLURM submission scripts are in `../slurm/`.

## Core Analysis Scripts

### 1. Loop Detection with vLLM

#### `run_vllm_generate.py`

Generate completions with vLLM and compute loop metrics on-the-fly.

**Features:**
- Tensor parallelism (TP) and data parallelism (DP) support
- Samples multiple temperatures: 0, 0.2, 0.4, 0.6, 0.8, 1.0
- Detects loops using n-gram repetition (default: 30-gram ≥20 times)
- Outputs per-model metrics CSV with loop fraction and average token count

**Usage (single GPU):**

```bash
python scripts/run_vllm_generate.py \
  --model-id Qwen/QwQ-32B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/qwq32b_metrics.csv \
  --tp 8
```

**Usage (multi-GPU with data parallelism):**

```bash
python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-7B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_7b_metrics.csv \
  --dp 8 \
  --tp 1
```

**SLURM batch submission:**

```bash
MODEL_ID=Qwen/QwQ-32B \
TP=8 \
DP=1 \
METRICS_OUT=outputs/qwq32b_metrics.csv \
sbatch slurm/run_vllm_generate.sbatch
```

**Key arguments:**
- `--data`: Input JSONL with `question` and `answer` fields (see `data/README.md`)
- `--metrics-out`: Output CSV with aggregated metrics
- `--generations-out`: Optional JSONL with raw generation text
- `--tp`: Tensor parallel size (for large models)
- `--dp`: Data parallel workers (splits dataset across multiple processes)
- `--max-tokens`: Max generation length (default: 30000)
- `--n`, `--k`: Loop detection parameters (n-gram size, repetition threshold)

**Output CSV columns:**
- `model_id`: Model identifier
- `temperature`: Sampling temperature
- `num_samples`: Total samples at this temperature
- `looped`: Count of samples with detected loops
- `not_looped`: Count of samples without loops
- `loop_fraction`: Proportion of looped samples
- `avg_tokens`: Mean generation length
- `accuracy`: Math problem accuracy (if answers provided)

#### `slurm/run_vllm_generate.sbatch`

SLURM script for batch job submission. Configure via environment variables:

```bash
export MODEL_ID="Qwen/QwQ-32B"
export DATA="data/aime_2024_2025.jsonl"
export METRICS_OUT="outputs/qwq32b_metrics.csv"
export TP=8
export DP=1
sbatch slurm/run_vllm_generate.sbatch
```

### 2. Post-Generation Metrics

#### `compute_metrics.py`

Compute loop metrics from existing generation JSONL files (when not computed during generation).

**Usage:**

```bash
python scripts/compute_metrics.py \
  --generations outputs/qwq_rollouts.jsonl \
  --out outputs/qwq_metrics.csv \
  --n 30 \
  --k 20 \
  --data data/aime_2024_2025.jsonl  # Optional, for accuracy
```

**When to use:**
- You have raw generation files without metrics
- You want to recompute metrics with different loop detection parameters
- You need to add accuracy grading after the fact

### 3. Visualization

#### `plot_fig1.py`

Generate Figure 1: loop fraction vs temperature across models.

**Usage:**

```bash
# From individual metrics files
python scripts/plot_fig1.py \
  --metrics outputs/qwq32b_metrics.csv \
  --metrics outputs/openthinker3_7b_metrics.csv \
  --metrics outputs/openthinker3_1p5b_metrics.csv \
  --out outputs/fig1.png

# From glob pattern
python scripts/plot_fig1.py \
  --metrics "outputs/*_metrics*.csv" \
  --out outputs/fig1.png

# From directory (auto-discovers metrics files)
python scripts/plot_fig1.py \
  --metrics outputs \
  --out outputs/fig1.png
```

**Output:**
- Two-panel figure showing loop fraction and average CoT length vs temperature
- Supports multiple repetitions (`.rep1`, `.rep2`, etc.) with error bars

### 4. Prefill Activation Analysis

#### `analyze_prefill_stability.py`

**Current implementation:** Runs greedy rollouts (temperature=0) and counts loop occurrences across multiple runs.

**Purpose:** Analyze whether models deterministically loop under greedy decoding.

**Usage:**

```bash
python scripts/analyze_prefill_stability.py \
  --model-id open-thoughts/OpenThinker3-7B \
  --data data/aime_2024_2025.jsonl \
  --out-rollout-csv outputs/prefill_rollout_counts.csv \
  --rollouts 10 \
  --tp 1 \
  --max-tokens 30000
```

**SLURM submission:**

```bash
MODEL_ID=open-thoughts/OpenThinker3-7B \
TP=8 \
sbatch slurm/analyze_prefill_stability.sbatch
```

**Output:**
- CSV with columns: `model_id`, `temperature`, `num_prompts`, `rollouts_per_prompt`, `num_samples`, `looped`, `not_looped`, `loop_fraction`, `max_tokens`, `seed`

**Note:** This script was recently refactored. The original prefill cosine similarity analysis (comparing hidden states across repeated forward passes) was removed. If that analysis is needed, it should be restored from git history.

#### `slurm/analyze_prefill_stability.sbatch`

SLURM submission script for prefill stability analysis.

```bash
export MODEL_ID="open-thoughts/OpenThinker3-7B"
export TP=8
export OUT_ROLLOUT_CSV="outputs/prefill_rollout_counts.csv"
sbatch slurm/analyze_prefill_stability.sbatch
```

### 5. Probe End-to-End (SLURM)

#### `slurm/run_probe_train_e2e.sbatch`

Builds probe dataset and trains the linear probe in one job.

Default config:
- `MODEL_PRESET=openthinker3_7b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- `TRAIN_DATASET=HuggingFaceH4/MATH-500`, `TRAIN_SPLIT=test`
- `TEST_DATASET=math-ai/aime25`, `TEST_SPLIT=test`
- `PROMPT_FIELD=problem`
- `SEEDS=$SEED` (single training seed by default)
- `REUSE_DATASET=1` (reuses `OUT_DATA_DIR` when manifest+shards match)
- `LR_SCHEDULER=cosine`, `WARMUP_RATIO=0.1`, `MIN_LR_RATIO=0.2`

Usage:

```bash
sbatch slurm/run_probe_train_e2e.sbatch
```

Override defaults via environment variables:

```bash
MODEL_PRESET=openthinker3_7b \
MAX_NUM_SEQS=16 \
TRAIN_DATASET=HuggingFaceH4/MATH-500 \
TRAIN_SPLIT=test \
TEST_DATASET=math-ai/aime25 \
TEST_SPLIT=test \
PROMPT_FIELD=problem \
WANDB_PROJECT=cot-loop-probe \
sbatch slurm/run_probe_train_e2e.sbatch
```

Run multiple training seeds and aggregate mean/std:

```bash
SEEDS=0,1,2 \
DATASET_SEED=0 \
sbatch slurm/run_probe_train_e2e.sbatch
```

Multi-seed outputs:
- per-seed run dirs: `${OUT_RUN_DIR}/seed_<seed>/`
- aggregate JSON summary: `${OUT_RUN_DIR}/seed_summary.json`
- aggregate CSV summary: `${OUT_RUN_DIR}/seed_summary.csv`

## Dataset Preparation

### `build_aime_jsonl.py`

Build the AIME 2024/2025 dataset for Figure 1 reproduction.

**Usage:**

```bash
python scripts/build_aime_jsonl.py \
  --out data/aime_2024_2025.jsonl
```

See `data/README.md` for expected format (60 problems: AIME 2024 I/II + 2025 I/II).

## Figure 1 Reproduction

Complete workflow to reproduce Figure 1:

```bash
# 1. Build dataset
python scripts/build_aime_jsonl.py --out data/aime_2024_2025.jsonl

# 2. Generate with each model
python scripts/run_vllm_generate.py \
  --model-id Qwen/QwQ-32B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/qwq32b_metrics.rep1.csv \
  --tp 8

python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-7B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_7b_metrics.rep1.csv \
  --tp 1

python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-1.5B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_1p5b_metrics.rep1.csv \
  --tp 1

# 3. Plot
python scripts/plot_fig1.py \
  --metrics outputs/*_metrics.rep1.csv \
  --out outputs/fig1.png
```

See [FIG1_REPRO.md](../FIG1_REPRO.md) for detailed reproduction notes.

## Utility Modules

### `utils.py`

Shared utilities used across scripts:

- `load_jsonl()` - Load JSONL datasets
- `build_prompt()` - Format prompts with chat templates
- `has_ngram_loop()` - Detect n-gram repetition in token sequences
- `_math_verify()` - Grade math problem answers
- Loop metrics aggregation and CSV I/O helpers

## Loop Detection Algorithm

The default loop detector identifies when any n-gram (contiguous token sequence) appears k or more times:

- **n** (default: 30): N-gram size in tokens
- **k** (default: 20): Repetition threshold

Example: "The quick brown fox" repeated 25 times → detected as loop (4-gram appears 25 times).

This is computed in **model token space**, not character space, to accurately reflect what the model generates.

## Notes

### Data Parallelism (DP)

`run_vllm_generate.py` supports splitting the dataset across multiple GPU workers:

```bash
python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-1.5B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/metrics.csv \
  --dp 8 \
  --tp 1
```

Each worker:
1. Gets a disjoint shard of the dataset
2. Runs vLLM generation independently
3. Reports metrics back to the main process
4. May hang during vLLM teardown (handled gracefully)

### vLLM Configuration

The generation scripts:
- Pull `top_p` and `top_k` from each model's `GenerationConfig`
- Set `repetition_penalty=1.0` (no repetition penalty)
- Use `apply_chat_template()` for prompt formatting
- Default prompt: `{question}\nPlease reason step by step, and put your final answer within \boxed{}.`

### Hardware Requirements

Recommended GPU configurations:
- **Qwen/QwQ-32B**: 8×A100 (80GB) with `--tp 8`
- **OpenThinker3-7B**: 1×A100 with `--tp 1`
- **OpenThinker3-1.5B**: 1×A100 with `--tp 1`

For multi-node setups, use SLURM scripts with appropriate `#SBATCH` directives.

## Troubleshooting

**vLLM workers hang during teardown:**
The updated `run_vllm_generate.py` gracefully handles this by terminating workers after metrics are received.

**Out of memory:**
Reduce `--max-num-seqs` or `--max-num-batched-tokens` to limit vLLM's memory usage.

**Missing loop detection:**
Adjust `--n` and `--k` parameters. Lower values detect shorter loops; higher values catch only extreme repetition.

## Relation to Probe Training

These scripts focus on **analyzing** loop behavior in generated text. For **predicting** loops from prefill activations, see the probe training pipeline in the main [README.md](../README.md).
