# SLURM Scripts

This directory contains SLURM workflows for the CoT loop detector project.

## Scripts

- `run_vllm_generate.sbatch`: Generate trajectories used for loop-label collection and detector analysis.
- `analyze_prefill_stability.sbatch`: Prefill-loop sanity check and stability checks with greedy rollouts.
- `run_probe_train_e2e.sbatch`: End-to-end probe pipeline for the canonical stacked prefill dataset (build + probe training, including optional multi-seed runs).
- `run_k5_threeview_dataset.sbatch`: Historical multi-view dataset build for the k=5 / max_tokens=15000 ablation study.
- `run_k5_threeview_ablation.sbatch`: Historical MLP sweep over the k=5 three-view ablation dataset.

## Detector E2E Defaults

`run_probe_train_e2e.sbatch` defaults to:
- `MODEL_PRESET=openthinker3_1p5b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- optional prefill throughput override (single GPU): `PREFILL_BATCH_SIZE=...` (default: `32`)
- optional rollout-completion feature throughput override: `COMPLETION_BATCH_SIZE=...` (default: `1`)
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
