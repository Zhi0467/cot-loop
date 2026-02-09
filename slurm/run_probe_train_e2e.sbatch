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

MODEL_PRESET="${MODEL_PRESET:-openthinker3_7b}"
PROMPT_FIELD="${PROMPT_FIELD:-problem}"

TRAIN_DATASET="${TRAIN_DATASET:-HuggingFaceH4/MATH-500}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"

TEST_DATASET="${TEST_DATASET:-math-ai/aime25}"
TEST_CONFIG="${TEST_CONFIG:-}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:-}"

SPLIT_RATIO="${SPLIT_RATIO:-0.1}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-${SEED}}"
DATASET_SEED="${DATASET_SEED:-}"
REUSE_DATASET="${REUSE_DATASET:-1}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-20}"
SHARD_SIZE="${SHARD_SIZE:-2048}"

OUT_DATA_DIR="${OUT_DATA_DIR:-outputs/probe_data/${MODEL_PRESET}}"
RUN_NAME="${RUN_NAME:-${MODEL_PRESET}-${SLURM_JOB_ID}}"
OUT_RUN_DIR="${OUT_RUN_DIR:-outputs/probe_runs/${RUN_NAME}}"

WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.2}"

TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
PROBE_SUMMARY_JSON="${PROBE_SUMMARY_JSON:-${OUT_RUN_DIR}/seed_summary.json}"
PROBE_SUMMARY_CSV="${PROBE_SUMMARY_CSV:-${OUT_RUN_DIR}/seed_summary.csv}"

RAW_SEEDS=()
IFS=',' read -r -a RAW_SEEDS <<< "${SEEDS}"
SEED_LIST=()
for raw_seed in "${RAW_SEEDS[@]}"; do
  seed_no_ws="${raw_seed//[[:space:]]/}"
  if [[ -n "${seed_no_ws}" ]]; then
    SEED_LIST+=("${seed_no_ws}")
  fi
done
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "[probe-e2e] error: SEEDS resolved to an empty list" >&2
  exit 1
fi
for seed in "${SEED_LIST[@]}"; do
  if [[ ! "${seed}" =~ ^-?[0-9]+$ ]]; then
    echo "[probe-e2e] error: seed '${seed}' is not an integer" >&2
    exit 1
  fi
done
if [[ -z "${DATASET_SEED}" ]]; then
  DATASET_SEED="${SEED_LIST[0]}"
fi
if [[ ! "${DATASET_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "[probe-e2e] error: DATASET_SEED '${DATASET_SEED}' is not an integer" >&2
  exit 1
fi

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
if [[ "${REUSE_DATASET}" == "1" ]]; then
  BUILD_ARGS+=(--reuse-if-compatible)
fi

BASE_TRAIN_ARGS=(
  --data-dir "${OUT_DATA_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --lr-scheduler "${LR_SCHEDULER}"
  --warmup-ratio "${WARMUP_RATIO}"
  --min-lr-ratio "${MIN_LR_RATIO}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --eval-every "${EVAL_EVERY}"
  --log-every "${LOG_EVERY}"
  --wandb-project "${WANDB_PROJECT}"
)

echo "[probe-e2e] build dataset: model_preset=${MODEL_PRESET} train=${TRAIN_DATASET}:${TRAIN_SPLIT} test=${TEST_DATASET}:${TEST_SPLIT} dataset_seed=${DATASET_SEED} reuse_dataset=${REUSE_DATASET}"
python scripts/build_probe_dataset.py "${BUILD_ARGS[@]}"

RUN_DIRS=()
for seed in "${SEED_LIST[@]}"; do
  run_dir="${OUT_RUN_DIR}"
  wandb_run_name="${WANDB_RUN_NAME}"
  if [[ ${#SEED_LIST[@]} -gt 1 ]]; then
    run_dir="${OUT_RUN_DIR}/seed_${seed}"
    wandb_run_name="${WANDB_RUN_NAME}-seed${seed}"
  fi

  echo "[probe-e2e] train probe: seed=${seed} data_dir=${OUT_DATA_DIR} out_dir=${run_dir}"
  python scripts/train_probe.py \
    "${BASE_TRAIN_ARGS[@]}" \
    --out-dir "${run_dir}" \
    --seed "${seed}" \
    --wandb-run-name "${wandb_run_name}" \
    ${TRAIN_EXTRA_ARGS}
  RUN_DIRS+=("${run_dir}")
done

if [[ ${#RUN_DIRS[@]} -gt 1 ]]; then
  echo "[probe-e2e] aggregate seeds: ${SEEDS}"
  python scripts/aggregate_probe_runs.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --out-json "${PROBE_SUMMARY_JSON}" \
    --out-csv "${PROBE_SUMMARY_CSV}"
fi
