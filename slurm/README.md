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

Override values with exported environment variables or inline `VAR=... sbatch ...`.
