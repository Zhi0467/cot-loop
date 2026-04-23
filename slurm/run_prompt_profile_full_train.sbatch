#!/bin/bash
#SBATCH --job-name=prompt-profile-full-train
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
    echo "[prompt-profile-full-train] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "[prompt-profile-full-train] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
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
  echo "[prompt-profile-full-train] expected a data-volume cache root outside HOME" >&2
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

OUT_ROOT="${OUT_ROOT:-outputs/full_train}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUT_ROOT}/summary}"
ARCHIVE_SOURCE_ROOT="${ARCHIVE_SOURCE_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection}"
WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[prompt-profile-full-train] OUT_ROOT=${OUT_ROOT}"
echo "[prompt-profile-full-train] SUMMARY_DIR=${SUMMARY_DIR}"
echo "[prompt-profile-full-train] ARCHIVE_SOURCE_ROOT=${ARCHIVE_SOURCE_ROOT}"
echo "[prompt-profile-full-train] HF_HOME=${HF_HOME}"
echo "[prompt-profile-full-train] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

mkdir -p "${OUT_ROOT}"

link_archive() {
  local dataset="$1"
  local source_dir="$2"
  local target_dir="${OUT_ROOT}/${dataset}/shared_archive"
  if [[ ! -e "${source_dir}/manifest.json" ]]; then
    echo "[prompt-profile-full-train] missing archive manifest: ${source_dir}/manifest.json" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_dir}")"
  rm -rf "${target_dir}"
  ln -s "${source_dir}" "${target_dir}"
  echo "[prompt-profile-full-train] linked ${dataset} -> ${source_dir}"
}

while IFS=$'\t' read -r dataset_key archive_dir _active_stage; do
  link_archive "${dataset_key}" "${archive_dir}"
done < <(
  "${PYTHON_BIN}" scripts/emit_prompt_profile_rfm_stage_registry.py \
    --archive-source-root "${ARCHIVE_SOURCE_ROOT}" \
    --include-excluded \
    --format tsv
)

"${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py \
  --stage regression \
  --out-root "${OUT_ROOT}" \
  --summary-dir "${SUMMARY_DIR}" \
  --wandb-project "${WANDB_PROJECT}"

"${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py \
  --stage relabel \
  --out-root "${OUT_ROOT}" \
  --summary-dir "${SUMMARY_DIR}" \
  --wandb-project "${WANDB_PROJECT}"

"${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py \
  --stage binary \
  --out-root "${OUT_ROOT}" \
  --summary-dir "${SUMMARY_DIR}" \
  --wandb-project "${WANDB_PROJECT}"

"${PYTHON_BIN}" scripts/run_prompt_profile_full_train.py \
  --stage summary \
  --out-root "${OUT_ROOT}" \
  --summary-dir "${SUMMARY_DIR}" \
  --wandb-project "${WANDB_PROJECT}"
