#!/bin/bash
#SBATCH --job-name=prompt-profile-binary-retrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  CONDA_ENV="${CONDA_ENV:-swe311}"
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[prompt-profile-binary-retrain] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
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
  echo "[prompt-profile-binary-retrain] expected a data-volume cache root outside HOME" >&2
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
OUT_ROOT="${OUT_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection/full_train_locked_pair_20260404_binary_h256d2}"
DATASET_LIST="${DATASET_LIST:-gpqa aime math500 mmlu_pro livecodebench}"
SEEDS="${SEEDS:-0 1 2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
PROBE_PRESET="${PROBE_PRESET:-mlp}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-256}"
MLP_DEPTH="${MLP_DEPTH:-2}"
MLP_DROPOUT="${MLP_DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"

read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a SEED_ARRAY <<< "${SEEDS}"

mkdir -p "${OUT_ROOT}"

echo "[prompt-profile-binary-retrain] SOURCE_ROOT=${SOURCE_ROOT}"
echo "[prompt-profile-binary-retrain] OUT_ROOT=${OUT_ROOT}"
echo "[prompt-profile-binary-retrain] DATASETS=${DATASET_LIST}"
echo "[prompt-profile-binary-retrain] SEEDS=${SEEDS}"
echo "[prompt-profile-binary-retrain] probe=${PROBE_PRESET} hidden=${MLP_HIDDEN_DIM} depth=${MLP_DEPTH} dropout=${MLP_DROPOUT}"
echo "[prompt-profile-binary-retrain] epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR} weight_decay=${WEIGHT_DECAY}"
echo "[prompt-profile-binary-retrain] HF_HOME=${HF_HOME}"
echo "[prompt-profile-binary-retrain] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

link_binary_data() {
  local dataset="$1"
  local source_dir="${SOURCE_ROOT}/${dataset}/majority_s_0.5/data"
  local target_dir="${OUT_ROOT}/${dataset}/majority_s_0.5/data"
  if [[ ! -f "${source_dir}/manifest.json" ]]; then
    echo "[prompt-profile-binary-retrain] missing relabeled manifest: ${source_dir}/manifest.json" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_dir}")"
  rm -rf "${target_dir}"
  ln -s "${source_dir}" "${target_dir}"
  echo "[prompt-profile-binary-retrain] linked ${dataset} -> ${source_dir}"
}

for dataset in "${DATASETS[@]}"; do
  link_binary_data "${dataset}"
done

CMD=(
  "${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py
  --stage binary
  --out-root "${OUT_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --wandb-project "${WANDB_PROJECT}"
  --probe-preset "${PROBE_PRESET}"
  --mlp-hidden-dim "${MLP_HIDDEN_DIM}"
  --mlp-depth "${MLP_DEPTH}"
  --mlp-dropout "${MLP_DROPOUT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seeds
)

for seed in "${SEED_ARRAY[@]}"; do
  CMD+=("${seed}")
done

for dataset in "${DATASETS[@]}"; do
  if [[ "${dataset}" == "all" ]]; then
    CMD+=(--dataset all)
    break
  fi
done

if [[ "${DATASETS[0]}" != "all" ]]; then
  CMD+=(--dataset)
  for dataset in "${DATASETS[@]}"; do
    CMD+=("${dataset}")
  done
fi

printf '$'
for arg in "${CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${CMD[@]}"
