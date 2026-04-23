#!/bin/bash
#SBATCH --job-name=prompt-profile-steer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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
    echo "[prompt-profile-steer] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[prompt-profile-steer] no runtime env found (PYTHON_BIN not executable, .venv missing, and CONDA_ENV not set)" >&2
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
  echo "[prompt-profile-steer] expected a data-volume cache root outside HOME" >&2
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

PYTHON_BIN="${PYTHON_BIN:-python3}"
VECTOR_EXPORT_DIR="${VECTOR_EXPORT_DIR:?set VECTOR_EXPORT_DIR=/abs/path/to/vector_exports}"
BENCHMARK="${BENCHMARK:-livecodebench}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_steer_smoke_seed${SEEDS:-0}}"
OUT_DIR="${OUT_DIR:-outputs/prompt_profile_rfm_steering/${RUN_NAME}}"
SPLIT="${SPLIT:-test}"
CONDITIONS="${CONDITIONS:-no_steer minus_v_spherical}"
SEEDS="${SEEDS:-0}"
T="${T:-0.3}"
EPSILON="${EPSILON:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
LAYERS="${LAYERS:-}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
THINKING_MODE="${THINKING_MODE:-on}"
HOOK_SITE="${HOOK_SITE:-prefill_layer_output_all_tokens}"
LIVECODEBENCH_REPO="${LIVECODEBENCH_REPO:-}"
RELEASE_VERSION="${RELEASE_VERSION:-}"

echo "[prompt-profile-steer] VECTOR_EXPORT_DIR=${VECTOR_EXPORT_DIR}"
echo "[prompt-profile-steer] OUT_DIR=${OUT_DIR}"
echo "[prompt-profile-steer] HF_HOME=${HF_HOME}"
echo "[prompt-profile-steer] TORCH_HOME=${TORCH_HOME}"

mkdir -p "${OUT_DIR}"

cmd=(
  "${PYTHON_BIN}" scripts/steer_prompt_profile_concept_vectors.py
  --vector-export-dir "${VECTOR_EXPORT_DIR}"
  --out-dir "${OUT_DIR}"
  --benchmark "${BENCHMARK}"
  --split "${SPLIT}"
  --t "${T}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --hook-site "${HOOK_SITE}"
  --thinking-mode "${THINKING_MODE}"
)

if [[ -n "${EPSILON}" ]]; then
  cmd+=(--epsilon "${EPSILON}")
fi

cmd+=(--conditions)
for condition in ${CONDITIONS}; do
  cmd+=("${condition}")
done

cmd+=(--seed)
for seed in ${SEEDS}; do
  cmd+=("${seed}")
done

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  cmd+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ -n "${TEMPERATURE}" ]]; then
  cmd+=(--temperature "${TEMPERATURE}")
fi
if [[ -n "${TOP_P}" ]]; then
  cmd+=(--top-p "${TOP_P}")
fi
if [[ -n "${TOP_K}" ]]; then
  cmd+=(--top-k "${TOP_K}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  cmd+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${LAYERS}" ]]; then
  cmd+=(--layers)
  for layer in ${LAYERS}; do
    cmd+=("${layer}")
  done
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ -n "${MODEL_ID}" ]]; then
  cmd+=(--model-id "${MODEL_ID}")
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  cmd+=(--model-revision "${MODEL_REVISION}")
fi
if [[ -n "${TOKENIZER_REVISION}" ]]; then
  cmd+=(--tokenizer-revision "${TOKENIZER_REVISION}")
fi
if [[ -n "${LIVECODEBENCH_REPO}" ]]; then
  cmd+=(--livecodebench-repo "${LIVECODEBENCH_REPO}")
fi
if [[ -n "${RELEASE_VERSION}" ]]; then
  cmd+=(--release-version "${RELEASE_VERSION}")
fi

printf '[prompt-profile-steer] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
