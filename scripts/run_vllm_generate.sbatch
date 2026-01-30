#!/bin/bash
#SBATCH --job-name=vllm-generate-loop-cot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

# Optional: activate your environment here (e.g., conda/venv/module load).
source .venv/bin/activate

MODEL_ID="${MODEL_ID:-open-thoughts/OpenThinker3-7B}"
DATA="${DATA:-data/aime_2024_2025.jsonl}"
METRICS_OUT="${METRICS_OUT:-outputs/openthinker3_7b_metrics.csv}"
TP="${TP:-1}"
DP="${DP:-1}"
TEMPS="${TEMPS:-0,0.2,0.4,0.6,0.8,1.0}"
N="${N:-20}"
NUM_REPETITION="${NUM_REPETITION:-1}"
MAX_TOKENS="${MAX_TOKENS:-30000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${MAX_NUM_SEQS}" ]]; then
  case "${MODEL_ID}" in
    *QwQ-32B*)
      MAX_NUM_SEQS=32
      ;;
    *OpenThinker3-7B*)
      MAX_NUM_SEQS=16
      ;;
    *OpenThinker3-1.5B*)
      MAX_NUM_SEQS=32
      ;;
    *)
      MAX_NUM_SEQS=16
      ;;
  esac
fi

ARGS=(
  --model-id "$MODEL_ID"
  --data "$DATA"
  --metrics-out "$METRICS_OUT"
  --dp "$DP"
  --tp "$TP"
  --temps "$TEMPS"
  --n "$N"
  --num-repetition "$NUM_REPETITION"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --seed "$SEED"
)

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

python scripts/run_vllm_generate.py "${ARGS[@]}" ${EXTRA_ARGS}
