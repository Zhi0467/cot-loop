#!/bin/bash
#SBATCH --job-name=prefill-stability
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

# Optional: activate your environment here (e.g., conda/venv/module load).
source .venv/bin/activate
export HF_HUB_CONNECT_TIMEOUT=30
export HF_HUB_READ_TIMEOUT=120

MODEL_ID="${MODEL_ID:-open-thoughts/OpenThinker3-7B}"
DATA="${DATA:-data/aime_2024_2025.jsonl}"
OUT_CSV="${OUT_CSV:-outputs/prefill_stability.csv}"
OUT_PLOT="${OUT_PLOT:-outputs/prefill_stability.png}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python scripts/analyze_prefill_stability.py \
  --model-id "$MODEL_ID" \
  --data "$DATA" \
  --out-csv "$OUT_CSV" \
  --out-plot "$OUT_PLOT" \
  ${EXTRA_ARGS}
