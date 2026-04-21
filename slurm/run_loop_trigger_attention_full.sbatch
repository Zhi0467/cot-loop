#!/bin/bash
#SBATCH --job-name=trigger-attn-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

export HF_HOME="${HF_HOME:-/data/scratch/${USER}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/scratch/${USER}/cache/vllm}"
export TORCH_HOME="${TORCH_HOME:-/data/scratch/${USER}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/scratch/${USER}/cache}"

source /data/scratch/murphy/venvs/cot-loop-detection-runtime-py310/bin/activate

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
OUT_DIR="${OUT_DIR:-outputs/qwen3_loop_trigger_attention_full_20260413}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-1000}"
MAX_TRIGGER_PREFIX="${MAX_TRIGGER_PREFIX:-40960}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
QUERY_POSITION_MODE="${QUERY_POSITION_MODE:-trigger_end}"

cd "${REPO_ROOT}"

mkdir -p "${OUT_DIR}"
SHARD_ROOT="${OUT_DIR}/shards"
mkdir -p "${SHARD_ROOT}"

if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi

if [[ "${NUM_SHARDS}" -gt 8 ]]; then
  echo "NUM_SHARDS=${NUM_SHARDS} exceeds the 8 GPUs requested by this job" >&2
  exit 1
fi

pids=()
for (( shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++ )); do
  shard_out="${SHARD_ROOT}/shard_${shard_idx}"
  mkdir -p "${shard_out}"
  CUDA_VISIBLE_DEVICES="${shard_idx}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  python3 scripts/analyze_loop_trigger_attention.py \
    --out-dir "${shard_out}" \
    --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
    --max-trigger-prefix "${MAX_TRIGGER_PREFIX}" \
    --query-position-mode "${QUERY_POSITION_MODE}" \
    --device cuda:0 \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard_idx}" \
    > "${OUT_DIR}/shard_${shard_idx}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

python3 scripts/merge_loop_trigger_attention_shards.py \
  --shard-root "${SHARD_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --num-shards "${NUM_SHARDS}"
