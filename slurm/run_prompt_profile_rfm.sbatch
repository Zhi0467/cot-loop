#!/bin/bash
#SBATCH --job-name=prompt-profile-rfm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${WORKDIR}"

mkdir -p logs outputs

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  :
elif [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[prompt-profile-rfm] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[prompt-profile-rfm] no runtime env found (PYTHON_BIN not executable, .venv missing, and CONDA_ENV not set)" >&2
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
  echo "[prompt-profile-rfm] expected a data-volume cache root outside HOME" >&2
  exit 1
fi

mkdir -p "${CACHE_BASE}"
# Keep model and dataset downloads in the shared Hugging Face cache, not a
# user-specific scratch cache.
# shellcheck source=slurm/cache_env.sh
source "${COT_LOOP_CACHE_ENV:-slurm/cache_env.sh}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

BENCHMARK="${BENCHMARK:?set BENCHMARK=gpqa|math500|mmlu_pro|livecodebench}"
ARCHIVE_SOURCE_ROOT="${ARCHIVE_SOURCE_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_rfm_seed${SEED:-0}}"
OUT_DIR="${OUT_DIR:-outputs/prompt_profile_rfm/${RUN_NAME}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
MODEL_REVISION="${MODEL_REVISION:-}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
SEED="${SEED:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
BALANCE_TRAIN="${BALANCE_TRAIN:-downsample}"
REG="${REG:-1e-3}"
ITERS="${ITERS:-5}"
BANDWIDTHS="${BANDWIDTHS:-1 10 100}"
SOLVER="${SOLVER:-solve}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-0}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-}"
CENTER_BATCH_SIZE="${CENTER_BATCH_SIZE:-}"
MAX_AGOP_SAMPLES="${MAX_AGOP_SAMPLES:-}"
LAYERS="${LAYERS:-}"

echo "[prompt-profile-rfm] BENCHMARK=${BENCHMARK}"
echo "[prompt-profile-rfm] OUT_DIR=${OUT_DIR}"
echo "[prompt-profile-rfm] ARCHIVE_SOURCE_ROOT=${ARCHIVE_SOURCE_ROOT}"
echo "[prompt-profile-rfm] HF_HOME=${HF_HOME}"
echo "[prompt-profile-rfm] TORCH_HOME=${TORCH_HOME}"

mkdir -p "${OUT_DIR}"

cmd=(
  "${PYTHON_BIN}" scripts/train_prompt_profile_rfm.py
  --benchmark "${BENCHMARK}"
  --archive-source-root "${ARCHIVE_SOURCE_ROOT}"
  --out-dir "${OUT_DIR}"
  --val-fraction "${VAL_FRACTION}"
  --balance-train "${BALANCE_TRAIN}"
  --reg "${REG}"
  --iters "${ITERS}"
  --seed "${SEED}"
  --solver "${SOLVER}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --model-id "${MODEL_ID}"
)

cmd+=(--bandwidths)
for bandwidth in ${BANDWIDTHS}; do
  cmd+=("${bandwidth}")
done

if [[ -n "${MODEL_REVISION}" ]]; then
  cmd+=(--model-revision "${MODEL_REVISION}")
fi
if [[ -n "${TOKENIZER_REVISION}" ]]; then
  cmd+=(--tokenizer-revision "${TOKENIZER_REVISION}")
fi
if [[ -n "${SAMPLE_BATCH_SIZE}" ]]; then
  cmd+=(--sample-batch-size "${SAMPLE_BATCH_SIZE}")
fi
if [[ -n "${CENTER_BATCH_SIZE}" ]]; then
  cmd+=(--center-batch-size "${CENTER_BATCH_SIZE}")
fi
if [[ -n "${MAX_AGOP_SAMPLES}" ]]; then
  cmd+=(--max-agop-samples "${MAX_AGOP_SAMPLES}")
fi
if [[ -n "${LAYERS}" ]]; then
  cmd+=(--layers)
  for layer in ${LAYERS}; do
    cmd+=("${layer}")
  done
fi
if [[ "${DIAG:-0}" == "1" ]]; then
  cmd+=(--diag)
fi
if [[ "${CENTERING:-0}" == "1" ]]; then
  cmd+=(--centering)
fi

printf '[prompt-profile-rfm] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
