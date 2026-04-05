#!/bin/bash
#SBATCH --job-name=prompt-profile-regression-natural
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

# Prefer an explicit conda env when requested; otherwise fall back to local venv.
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[prompt-profile-regression-natural] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "[prompt-profile-regression-natural] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi

if [[ -d "/data/users/${USER}" ]]; then
  CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch/${USER}" ]]; then
  CACHE_BASE="/data/scratch/${USER}/cache"
else
  echo "[prompt-profile-regression-natural] expected a data-volume cache root outside HOME" >&2
  exit 1
fi

mkdir -p "${CACHE_BASE}"
export HF_HOME="${HF_HOME:-${CACHE_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

SOURCE_ROOT="${SOURCE_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection/full_train_locked_pair_20260404}"
OUT_ROOT="${OUT_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection/prompt_profile_regression_natural_rerun_20260405}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUT_ROOT}/summary}"
DATASET_LIST="${DATASET_LIST:-gpqa aime math500 mmlu_pro livecodebench}"
SEEDS="${SEEDS:-0 1 2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
PROBE_PRESET="${PROBE_PRESET:-mlp}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"
MLP_DEPTH="${MLP_DEPTH:-}"
MLP_DROPOUT="${MLP_DROPOUT:-}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"

read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
RESOLVED_DATASETS=("${DATASETS[@]}")
if [[ "${DATASETS[0]}" == "all" ]]; then
  RESOLVED_DATASETS=(gpqa aime math500 mmlu_pro livecodebench)
fi

mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"

echo "[prompt-profile-regression-natural] SOURCE_ROOT=${SOURCE_ROOT}"
echo "[prompt-profile-regression-natural] OUT_ROOT=${OUT_ROOT}"
echo "[prompt-profile-regression-natural] SUMMARY_DIR=${SUMMARY_DIR}"
echo "[prompt-profile-regression-natural] DATASETS=${DATASET_LIST}"
echo "[prompt-profile-regression-natural] SEEDS=${SEEDS}"
echo "[prompt-profile-regression-natural] probe=${PROBE_PRESET} hidden=${MLP_HIDDEN_DIM:-default} depth=${MLP_DEPTH:-default} dropout=${MLP_DROPOUT:-default}"
echo "[prompt-profile-regression-natural] epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR} weight_decay=${WEIGHT_DECAY}"
echo "[prompt-profile-regression-natural] HF_HOME=${HF_HOME}"
echo "[prompt-profile-regression-natural] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

check_dataset_inputs() {
  local dataset="$1"
  local archive_manifest="${SOURCE_ROOT}/${dataset}/shared_archive/manifest.json"
  if [[ ! -f "${archive_manifest}" ]]; then
    echo "[prompt-profile-regression-natural] missing shared archive manifest: ${archive_manifest}" >&2
    exit 1
  fi
}

for dataset in "${RESOLVED_DATASETS[@]}"; do
  check_dataset_inputs "${dataset}"
done

REGRESSION_CMD=(
  "${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py
  --stage regression
  --out-root "${OUT_ROOT}"
  --summary-dir "${SUMMARY_DIR}"
  --source-out-root "${SOURCE_ROOT}"
  --regression-data-mode natural
  --regression-train-balance-mode none
  --python-bin "${PYTHON_BIN}"
  --wandb-project "${WANDB_PROJECT}"
  --probe-preset "${PROBE_PRESET}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seeds
)

for seed in "${SEED_ARRAY[@]}"; do
  REGRESSION_CMD+=("${seed}")
done

if [[ -n "${MLP_HIDDEN_DIM}" ]]; then
  REGRESSION_CMD+=(--mlp-hidden-dim "${MLP_HIDDEN_DIM}")
fi
if [[ -n "${MLP_DEPTH}" ]]; then
  REGRESSION_CMD+=(--mlp-depth "${MLP_DEPTH}")
fi
if [[ -n "${MLP_DROPOUT}" ]]; then
  REGRESSION_CMD+=(--mlp-dropout "${MLP_DROPOUT}")
fi

for dataset in "${DATASETS[@]}"; do
  if [[ "${dataset}" == "all" ]]; then
    REGRESSION_CMD+=(--dataset all)
    break
  fi
done

if [[ "${DATASETS[0]}" != "all" ]]; then
  REGRESSION_CMD+=(--dataset)
  for dataset in "${DATASETS[@]}"; do
    REGRESSION_CMD+=("${dataset}")
  done
fi

printf '$'
for arg in "${REGRESSION_CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${REGRESSION_CMD[@]}"

SUMMARY_CMD=(
  "${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py
  --stage summary
  --out-root "${OUT_ROOT}"
  --summary-dir "${SUMMARY_DIR}"
)

for dataset in "${DATASETS[@]}"; do
  if [[ "${dataset}" == "all" ]]; then
    SUMMARY_CMD+=(--dataset all)
    break
  fi
done

if [[ "${DATASETS[0]}" != "all" ]]; then
  SUMMARY_CMD+=(--dataset)
  for dataset in "${DATASETS[@]}"; do
    SUMMARY_CMD+=("${dataset}")
  done
fi

printf '$'
for arg in "${SUMMARY_CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${SUMMARY_CMD[@]}"
