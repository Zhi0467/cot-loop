# SLURM Scripts

This directory contains all SLURM batch scripts for this repository.

## Scripts

- `run_vllm_generate.sbatch`: Figure 1 vLLM generation + metrics aggregation.
- `analyze_prefill_stability.sbatch`: Greedy rollout loop-rate analysis.
- `run_probe_train_e2e.sbatch`: End-to-end probe pipeline (dataset build + probe training).

## Probe E2E Defaults

`run_probe_train_e2e.sbatch` defaults to:
- `MODEL_PRESET=openthinker3_1p5b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- optional prefill throughput override (single GPU): `PREFILL_BATCH_SIZE=...` (default: `8`)
- `TRAIN_DATASET=HuggingFaceH4/MATH-500`
- `TRAIN_SPLIT=test`
- `TEST_DATASET` omitted by default (falls back to local `data/aime_2024_2025.jsonl`)
- `TEST_SPLIT=test`
- `PROMPT_FIELD=problem`
- `PROBE_PRESET=mlp` (passed to `scripts/train_probe.py --probe-preset`)
- `VLLM_CACHE_ROOT=/data/users/$USER/cache/vllm` when `/data/users/$USER/cache` exists (else falls back to `$HOME/.cache/vllm`)

Submit with defaults:
loop fraction and accuracy ablation studies:
```bash 
sbatch --export=ALL,MODEL_ID=open-thoughts/OpenThinker3-7B,TP=1,DP=8,NUM_REPETITION=1,METRICS_OUT=outputs/openthinker3_7b_metrics.rep1.csv \
    slurm/run_vllm_generate.sbatch
```

train probe with defaults:
```bash
sbatch slurm/run_probe_train_e2e.sbatch
```

When multiple seeds are used (default: `0 1 2`), the script also writes:
- `${OUT_RUN_DIR}/probe_multiseed_curves.png` (aggregated train/eval curves from `seed_*/metrics.jsonl`)

Override values with exported environment variables or inline `VAR=... sbatch ...`.

To force a specific vLLM cache root:
```bash
VLLM_CACHE_ROOT=/data/users/zhiwang/cache/vllm \
sbatch slurm/run_probe_train_e2e.sbatch
```

to train the probe with train/test from the same dataset, use:
```bash
MODEL_PRESET=openthinker3_1p5b \
MAX_TOKENS=10000 \
PROMPT_FIELD=problem \
TRAIN_DATASET=AI-MO/NuminaMath-CoT \
TRAIN_SPLIT=train \
TRAIN_MAX_SAMPLES=50000 \
TEST_DATASET=AI-MO/NuminaMath-CoT \
TEST_SPLIT=train \
TEST_MAX_SAMPLES=10000 \
SPLIT_RATIO=0.1 \
DATASET_SEED=0 \
SEEDS="0 1 2" \
sbatch slurm/run_probe_train_e2e.sbatch
```

When `TRAIN_*` and `TEST_*` point to the same dataset/config/split:
- if both `TRAIN_MAX_SAMPLES` and `TEST_MAX_SAMPLES` are set, the builder now creates disjoint subsets with exactly those sizes;
- otherwise it uses a random ratio split (controlled by `SPLIT_RATIO`) and then applies any provided per-split caps.
