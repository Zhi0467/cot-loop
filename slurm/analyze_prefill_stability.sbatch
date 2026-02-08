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
