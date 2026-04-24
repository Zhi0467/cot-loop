#!/bin/bash
#SBATCH --job-name=prefill-stability
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# SLURM starts jobs in the submit directory, so we're already in the right place
mkdir -p logs outputs

# Optional: activate your environment here (e.g., conda/venv/module load).
source .venv/bin/activate
export HF_HUB_CONNECT_TIMEOUT=30
export HF_HUB_READ_TIMEOUT=120

if [[ -d "/data/users/${USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${USER}/cache"
else
  echo "[prefill-stability] no supported non-home cache root found" >&2
  exit 1
fi
mkdir -p "${DEFAULT_CACHE_BASE}"

# Keep model and dataset downloads in the shared Hugging Face cache, not a
# user-specific scratch cache.
# shellcheck source=slurm/cache_env.sh
source "${COT_LOOP_CACHE_ENV:-slurm/cache_env.sh}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEFAULT_CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${DEFAULT_CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DEFAULT_CACHE_BASE}}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

MODEL_ID="${MODEL_ID:-open-thoughts/OpenThinker3-7B}"
DATA="${DATA:-data/aime_2024_2025.jsonl}"
OUT_ROLLOUT_CSV="${OUT_ROLLOUT_CSV:-outputs/prefill_rollout_counts.csv}"
TP="${TP:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python scripts/analyze_prefill_stability.py \
  --model-id "$MODEL_ID" \
  --data "$DATA" \
  --tp "$TP" \
  --out-rollout-csv "$OUT_ROLLOUT_CSV" \
  ${EXTRA_ARGS}
