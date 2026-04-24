#!/bin/bash
#SBATCH --job-name=prompt-profile-proj
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

# Prefer an explicit conda env when requested; otherwise fall back to local venv.
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[prompt-profile-proj] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "[prompt-profile-proj] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi
export HF_HUB_CONNECT_TIMEOUT="${HF_HUB_CONNECT_TIMEOUT:-30}"
export HF_HUB_READ_TIMEOUT="${HF_HUB_READ_TIMEOUT:-120}"

DEFAULT_CACHE_BASE="${HOME}/.cache"
if [[ -d "/data/users/${USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${USER}/cache"
fi
mkdir -p "${DEFAULT_CACHE_BASE}"

# Keep model and dataset downloads in the shared Hugging Face cache, not a
# user-specific scratch cache.
# shellcheck source=slurm/cache_env.sh
source "${COT_LOOP_CACHE_ENV:-slurm/cache_env.sh}"
export TORCH_HOME="${TORCH_HOME:-${DEFAULT_CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DEFAULT_CACHE_BASE}}"
if [[ -z "${VLLM_CACHE_ROOT:-}" ]]; then
  export VLLM_CACHE_ROOT="${DEFAULT_CACHE_BASE}/vllm"
fi
if [[ -n "${PINNED_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${PINNED_VISIBLE_DEVICES}"
fi
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

echo "[prompt-profile-proj] HF_HOME=${HF_HOME}"
echo "[prompt-profile-proj] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[prompt-profile-proj] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[prompt-profile-proj] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "[prompt-profile-proj] TORCH_HOME=${TORCH_HOME}"
echo "[prompt-profile-proj] XDG_CACHE_HOME=${XDG_CACHE_HOME}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[prompt-profile-proj] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

MODEL_PRESET="${MODEL_PRESET-}"
MODEL_ID="${MODEL_ID:-}"
if [[ -z "${MODEL_PRESET}" && -z "${MODEL_ID}" ]]; then
  MODEL_ID="Qwen/Qwen3-1.7B"
fi
PROMPT_FIELD="${PROMPT_FIELD:-problem}"
TASK_KIND="${TASK_KIND:-math_freeform}"
LIVECODEBENCH_REPO="${LIVECODEBENCH_REPO:-}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
LM_STYLE_OVERRIDE="${LM_STYLE_OVERRIDE:-}"

TRAIN_DATASET="${TRAIN_DATASET:-HuggingFaceH4/MATH-500}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
TEST_DATASET="${TEST_DATASET:-}"
TEST_CONFIG="${TEST_CONFIG:-}"
TEST_SPLIT="${TEST_SPLIT:-}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:-}"
TEST_SPLIT_DISPLAY="${TEST_SPLIT:-test}"
SPLIT_RATIO="${SPLIT_RATIO:-0.1}"
SEED="${SEED:-0}"

TEMPERATURE="${TEMPERATURE:-0.2}"
TP="${TP:-1}"
DP="${DP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_TOKENS="${MAX_TOKENS:-30000}"
PREFILL_BATCH_SIZE="${PREFILL_BATCH_SIZE:-16}"
COMPLETION_BATCH_SIZE="${COMPLETION_BATCH_SIZE:-1}"
FEATURE_POOLING="${FEATURE_POOLING:-last_token_all_layers_stack}"
FEATURE_LAYER="${FEATURE_LAYER:--1}"
FEATURE_KEY="${FEATURE_KEY:-}"
PROFILE_TAIL_THRESHOLD="${PROFILE_TAIL_THRESHOLD:-0.5}"
TAIL_THRESHOLDS="${TAIL_THRESHOLDS:-0.5 0.6 0.9}"
PROJECTION_VIEW="${PROJECTION_VIEW:-last_layer}"
MAX_CLUSTERS="${MAX_CLUSTERS:-6}"
FIGURE_LABEL="${FIGURE_LABEL:-}"

LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-20}"
SHARD_SIZE="${SHARD_SIZE:-2048}"

OUT_DATA_DIR="${OUT_DATA_DIR:-outputs/prompt_profile_projection_data}"
OUT_PROJECTION_DIR="${OUT_PROJECTION_DIR:-outputs/prompt_profile_projection}"

echo "[prompt-profile-proj] train=${TRAIN_DATASET}:${TRAIN_SPLIT} test=${TEST_DATASET}:${TEST_SPLIT_DISPLAY}"
echo "[prompt-profile-proj] model_id=${MODEL_ID} task_kind=${TASK_KIND}"
echo "[prompt-profile-proj] livecodebench_repo=${LIVECODEBENCH_REPO} release_version=${RELEASE_VERSION} lm_style_override=${LM_STYLE_OVERRIDE}"
echo "[prompt-profile-proj] tail_threshold=${PROFILE_TAIL_THRESHOLD} export_thresholds=${TAIL_THRESHOLDS}"
echo "[prompt-profile-proj] tp=${TP} dp=${DP} max_model_len=${MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[prompt-profile-proj] out_data_dir=${OUT_DATA_DIR}"
echo "[prompt-profile-proj] out_projection_dir=${OUT_PROJECTION_DIR}"

BUILD_ARGS=(
  --train-dataset "${TRAIN_DATASET}"
  --train-split "${TRAIN_SPLIT}"
  --prompt-field "${PROMPT_FIELD}"
  --task-kind "${TASK_KIND}"
  --split-ratio "${SPLIT_RATIO}"
  --seed "${SEED}"
  --loop-n "${LOOP_N}"
  --loop-k "${LOOP_K}"
  --target-kind binary
  --binary-target-mode prompt_majority_tail
  --profile-target majority_tail
  --profile-tail-threshold "${PROFILE_TAIL_THRESHOLD}"
  --shard-size "${SHARD_SIZE}"
  --prefill-batch-size "${PREFILL_BATCH_SIZE}"
  --completion-batch-size "${COMPLETION_BATCH_SIZE}"
  --feature-pooling "${FEATURE_POOLING}"
  --feature-layer "${FEATURE_LAYER}"
  --out-dir "${OUT_DATA_DIR}"
  --temperature "${TEMPERATURE}"
  --tp "${TP}"
  --dp "${DP}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --num-generations "${NUM_GENERATIONS}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ -n "${MODEL_PRESET}" ]]; then
  BUILD_ARGS+=(--model-preset "${MODEL_PRESET}")
fi
if [[ -n "${MODEL_ID}" ]]; then
  BUILD_ARGS+=(--model-id "${MODEL_ID}")
fi
if [[ -n "${TEST_DATASET}" ]]; then
  BUILD_ARGS+=(--test-dataset "${TEST_DATASET}")
fi
if [[ -n "${TEST_SPLIT}" ]]; then
  BUILD_ARGS+=(--test-split "${TEST_SPLIT}")
fi
if [[ "${TASK_KIND}" == "livecodebench_codegen" ]]; then
  BUILD_ARGS+=(--livecodebench-repo "${LIVECODEBENCH_REPO}" --release-version "${RELEASE_VERSION}")
  if [[ -n "${LM_STYLE_OVERRIDE}" ]]; then
    BUILD_ARGS+=(--lm-style-override "${LM_STYLE_OVERRIDE}")
  fi
fi
if [[ -n "${TRAIN_CONFIG}" ]]; then
  BUILD_ARGS+=(--train-config "${TRAIN_CONFIG}")
fi
if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
  BUILD_ARGS+=(--train-max-samples "${TRAIN_MAX_SAMPLES}")
fi
if [[ -n "${TEST_CONFIG}" ]]; then
  BUILD_ARGS+=(--test-config "${TEST_CONFIG}")
fi
if [[ -n "${TEST_MAX_SAMPLES}" ]]; then
  BUILD_ARGS+=(--test-max-samples "${TEST_MAX_SAMPLES}")
fi
if [[ -n "${FEATURE_KEY}" ]]; then
  BUILD_ARGS+=(--feature-key "${FEATURE_KEY}")
fi

python scripts/build_probe_dataset.py "${BUILD_ARGS[@]}"

read -r -a TAIL_THRESHOLD_ARRAY <<< "${TAIL_THRESHOLDS}"

EXPORT_DIR="${OUT_PROJECTION_DIR}/export"
FIGURE_DIR="${OUT_PROJECTION_DIR}/figures"
mkdir -p "${EXPORT_DIR}" "${FIGURE_DIR}"

python scripts/export_prompt_profile_projection.py \
  --data-dir "${OUT_DATA_DIR}" \
  --out-dir "${EXPORT_DIR}" \
  --projection-view "${PROJECTION_VIEW}" \
  --max-clusters "${MAX_CLUSTERS}" \
  --tail-thresholds "${TAIL_THRESHOLD_ARRAY[@]}"

if python -c "import matplotlib" >/dev/null 2>&1; then
  RENDER_ARGS=(
    --summary-json "${EXPORT_DIR}/projection_summary.json"
    --prompt-csv "${EXPORT_DIR}/prompt_projection.csv"
    --out-dir "${FIGURE_DIR}"
  )
  if [[ -n "${FIGURE_LABEL}" ]]; then
    RENDER_ARGS+=(--figure-label "${FIGURE_LABEL}")
  fi
  python scripts/render_prompt_profile_projection.py "${RENDER_ARGS[@]}"
else
  echo "[prompt-profile-proj] matplotlib not available in runtime env; export finished, render skipped."
fi
