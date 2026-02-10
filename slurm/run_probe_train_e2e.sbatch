#!/bin/bash
#SBATCH --job-name=probe-train-e2e
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# SLURM starts jobs in the submit directory, so we're already in the right place
mkdir -p logs outputs

# Optional: activate your environment here (e.g., conda/venv/module load).
source .venv/bin/activate
export HF_HUB_CONNECT_TIMEOUT="${HF_HUB_CONNECT_TIMEOUT:-30}"
export HF_HUB_READ_TIMEOUT="${HF_HUB_READ_TIMEOUT:-120}"

MODEL_PRESET="${MODEL_PRESET:-openthinker3_1p5b}"
PROMPT_FIELD="${PROMPT_FIELD:-problem}"

TRAIN_DATASET="${TRAIN_DATASET:-HuggingFaceH4/MATH-500}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"

TEST_DATASET="${TEST_DATASET:-}"
TEST_CONFIG="${TEST_CONFIG:-}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:-}"

SPLIT_RATIO="${SPLIT_RATIO:-0.1}"
SEED="${SEED:-}"
SEEDS="${SEEDS:-}"
DATASET_SEED="${DATASET_SEED:-}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-20}"
SHARD_SIZE="${SHARD_SIZE:-2048}"

OUT_DATA_DIR="${OUT_DATA_DIR:-outputs/probe_data/${MODEL_PRESET}}"
RUN_NAME="${RUN_NAME:-${MODEL_PRESET}-${SLURM_JOB_ID}}"
OUT_RUN_DIR="${OUT_RUN_DIR:-outputs/probe_runs/${RUN_NAME}}"

WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
PROBE_PRESET="${PROBE_PRESET:-mlp}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-5}"

TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

dataset_cached() {
  local data_dir="$1"
  [[ -f "${data_dir}/manifest.json" ]] || return 1
  compgen -G "${data_dir}/train/shard-*.pt" > /dev/null || return 1
  compgen -G "${data_dir}/test/shard-*.pt" > /dev/null || return 1
  return 0
}

declare -a SEED_LIST=()
if [[ -n "${SEEDS}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
elif [[ -n "${SEED}" ]]; then
  SEED_LIST=("${SEED}")
else
  SEED_LIST=(0 1 2)
fi

if [[ "${#SEED_LIST[@]}" -eq 0 ]]; then
  echo "[probe-e2e] no seeds resolved from SEEDS/SEED" >&2
  exit 1
fi

if [[ -z "${DATASET_SEED}" ]]; then
  DATASET_SEED="${SEED_LIST[0]}"
fi

echo "[probe-e2e] model_preset=${MODEL_PRESET} train=${TRAIN_DATASET}:${TRAIN_SPLIT} test=${TEST_DATASET}:${TEST_SPLIT}"
echo "[probe-e2e] probe_preset=${PROBE_PRESET}"
echo "[probe-e2e] resolved seeds: ${SEED_LIST[*]}"

BUILD_ARGS=(
  --train-dataset "${TRAIN_DATASET}"
  --train-split "${TRAIN_SPLIT}"
  --prompt-field "${PROMPT_FIELD}"
  --split-ratio "${SPLIT_RATIO}"
  --seed "${DATASET_SEED}"
  --loop-n "${LOOP_N}"
  --loop-k "${LOOP_K}"
  --shard-size "${SHARD_SIZE}"
  --out-dir "${OUT_DATA_DIR}"
)

if [[ -n "${MODEL_PRESET}" ]]; then
  BUILD_ARGS+=(--model-preset "${MODEL_PRESET}")
fi
if [[ -n "${TEST_DATASET}" ]]; then
  BUILD_ARGS+=(--test-dataset "${TEST_DATASET}" --test-split "${TEST_SPLIT}")
fi

if [[ -n "${TRAIN_CONFIG}" ]]; then
  BUILD_ARGS+=(--train-config "${TRAIN_CONFIG}")
fi
if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
  BUILD_ARGS+=(--train-max-samples "${TRAIN_MAX_SAMPLES}")
fi
if [[ -n "${TEST_DATASET}" && -n "${TEST_CONFIG}" ]]; then
  BUILD_ARGS+=(--test-config "${TEST_CONFIG}")
fi
if [[ -n "${TEST_DATASET}" && -n "${TEST_MAX_SAMPLES}" ]]; then
  BUILD_ARGS+=(--test-max-samples "${TEST_MAX_SAMPLES}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  BUILD_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

if dataset_cached "${OUT_DATA_DIR}"; then
  echo "[probe-e2e] dataset cache hit at ${OUT_DATA_DIR}; skipping build"
else
  echo "[probe-e2e] dataset cache miss; building at ${OUT_DATA_DIR} with dataset_seed=${DATASET_SEED}"
  python scripts/build_probe_dataset.py "${BUILD_ARGS[@]}"
fi

for RUN_SEED in "${SEED_LIST[@]}"; do
  if [[ "${#SEED_LIST[@]}" -gt 1 ]]; then
    SEED_RUN_DIR="${OUT_RUN_DIR}/seed_${RUN_SEED}"
    SEED_WANDB_RUN_NAME="${WANDB_RUN_NAME}-seed${RUN_SEED}"
  else
    SEED_RUN_DIR="${OUT_RUN_DIR}"
    SEED_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
  fi

  TRAIN_ARGS=(
    --data-dir "${OUT_DATA_DIR}"
    --out-dir "${SEED_RUN_DIR}"
    --probe-preset "${PROBE_PRESET}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --seed "${RUN_SEED}"
    --eval-every "${EVAL_EVERY}"
    --log-every "${LOG_EVERY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${SEED_WANDB_RUN_NAME}"
  )

  echo "[probe-e2e] seed=${RUN_SEED}: train probe data_dir=${OUT_DATA_DIR} out_dir=${SEED_RUN_DIR} probe_preset=${PROBE_PRESET}"
  python scripts/train_probe.py "${TRAIN_ARGS[@]}" ${TRAIN_EXTRA_ARGS}
done

if [[ "${#SEED_LIST[@]}" -gt 1 ]]; then
  PLOT_OUT="${OUT_RUN_DIR}/probe_multiseed_curves.png"
  echo "[probe-e2e] plotting multi-seed curves from ${OUT_RUN_DIR} -> ${PLOT_OUT}"
  python scripts/plot_probe_multiseed.py \
    --run-dir "${OUT_RUN_DIR}" \
    --out "${PLOT_OUT}"
fi
