#!/bin/bash
#SBATCH --job-name=collect-model-stats
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs/model_stats

source .venv/bin/activate
export VLLM_SUPPRESS_SEM_UNLINK_ERRORS=1
export HF_HUB_CONNECT_TIMEOUT=30
export HF_HUB_READ_TIMEOUT=120

TASK_KIND="${TASK_KIND:-math_freeform}"
DATASET="${DATASET:-data/aime_2024_2025.jsonl}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
SPLIT="${SPLIT:-test}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
STATISTICS="${STATISTICS:-success_fraction,loop_fraction,avg_generation_length,avg_loop_generation_length,avg_first_loop_prefix_length,max_length_hit_fraction,generation_length_variance,max_length_hit_success_fraction,loop_success_fraction,avg_correct_generation_length,avg_wrong_generation_length}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TP="${TP:-1}"
DP="${DP:-1}"
MAX_TOKENS="${MAX_TOKENS:-81920}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-0}"
LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-20}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
LIVECODEBENCH_REPO="${LIVECODEBENCH_REPO:-}"
LM_STYLE_OVERRIDE="${LM_STYLE_OVERRIDE:-}"
OUT="${OUT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${TASK_KIND}" == "multiple_choice_gpqa" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for TASK_KIND=multiple_choice_gpqa." >&2
  exit 1
fi

if [[ "${TASK_KIND}" == "livecodebench_codegen" ]]; then
  if [[ -z "${HF_DATASETS_CACHE:-}" ]]; then
    echo "HF_DATASETS_CACHE is required for TASK_KIND=livecodebench_codegen." >&2
    exit 1
  fi
  if [[ -z "${LIVECODEBENCH_REPO}" ]]; then
    echo "LIVECODEBENCH_REPO is required for TASK_KIND=livecodebench_codegen." >&2
    exit 1
  fi
  if [[ ! -d "${LIVECODEBENCH_REPO}" ]]; then
    echo "LIVECODEBENCH_REPO does not exist: ${LIVECODEBENCH_REPO}" >&2
    exit 1
  fi
fi

ARGS=(
  --task-kind "$TASK_KIND"
  --dataset "$DATASET"
  --split "$SPLIT"
  --model-id "$MODEL_ID"
  --statistics "$STATISTICS"
  --tp "$TP"
  --dp "$DP"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --seed "$SEED"
  --loop-n "$LOOP_N"
  --loop-k "$LOOP_K"
  --release-version "$RELEASE_VERSION"
)

if [[ -n "${DATASET_CONFIG}" ]]; then
  ARGS+=(--dataset-config "$DATASET_CONFIG")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ -n "${LIVECODEBENCH_REPO}" ]]; then
  ARGS+=(--livecodebench-repo "$LIVECODEBENCH_REPO")
fi

if [[ -n "${LM_STYLE_OVERRIDE}" ]]; then
  ARGS+=(--lm-style-override "$LM_STYLE_OVERRIDE")
fi

if [[ -n "${OUT}" ]]; then
  ARGS+=(--out "$OUT")
fi

python scripts/collect_model_stats.py "${ARGS[@]}" ${EXTRA_ARGS}
