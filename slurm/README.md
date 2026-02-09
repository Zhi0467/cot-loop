# SLURM Scripts

This directory contains all SLURM batch scripts for this repository.

## Scripts

- `run_vllm_generate.sbatch`: Figure 1 vLLM generation + metrics aggregation.
- `analyze_prefill_stability.sbatch`: Greedy rollout loop-rate analysis.
- `run_probe_train_e2e.sbatch`: End-to-end probe pipeline (dataset build + probe training).

## Probe E2E Defaults

`run_probe_train_e2e.sbatch` defaults to:
- `MODEL_PRESET=openthinker3_7b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- `TRAIN_DATASET=HuggingFaceH4/MATH-500`
- `TRAIN_SPLIT=test`
- `TEST_DATASET=math-ai/aime25`
- `TEST_SPLIT=test`
- `PROMPT_FIELD=problem`
- `SEEDS=$SEED` (single seed by default)
- `REUSE_DATASET=1` (skip rebuild when `OUT_DATA_DIR/manifest.json` is compatible)
- `LR_SCHEDULER=cosine`, `WARMUP_RATIO=0.1`, `MIN_LR_RATIO=0.2`

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

Run 3 seeds (shared dataset build, then aggregate mean/std):
```bash
SEEDS=0,1,2 \
DATASET_SEED=0 \
sbatch slurm/run_probe_train_e2e.sbatch
```

When `SEEDS` has multiple values, outputs are:
- per-seed run dirs: `${OUT_RUN_DIR}/seed_<seed>/`
- aggregate summary JSON: `${OUT_RUN_DIR}/seed_summary.json`
- aggregate summary CSV: `${OUT_RUN_DIR}/seed_summary.csv`

Override values with exported environment variables or inline `VAR=... sbatch ...`.
