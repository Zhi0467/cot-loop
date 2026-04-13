#!/bin/bash
#SBATCH --job-name=trigger-attn-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
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
DEVICE="${DEVICE:-cuda}"

cd "${REPO_ROOT}"

python3 scripts/analyze_loop_trigger_attention.py \
  --out-dir "${OUT_DIR}" \
  --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
  --max-trigger-prefix "${MAX_TRIGGER_PREFIX}" \
  --device "${DEVICE}"
