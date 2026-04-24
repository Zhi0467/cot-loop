#!/bin/bash
# Shared cache defaults for Slurm entrypoints. Source this file; do not execute it.

if [[ -z "${USER:-}" ]]; then
  echo "[cache-env] USER is not set; refusing to resolve cache paths" >&2
  exit 1
fi
HOME_DIR="${HOME:-}"

COT_LOOP_HF_CACHE_ROOT="${COT_LOOP_HF_CACHE_ROOT:-/data/shared/huggingface}"
if [[ "${COT_LOOP_HF_CACHE_ROOT}" != /* ]]; then
  echo "[cache-env] COT_LOOP_HF_CACHE_ROOT must be absolute: ${COT_LOOP_HF_CACHE_ROOT}" >&2
  exit 1
fi
if [[ -n "${HOME_DIR}" && ( "${COT_LOOP_HF_CACHE_ROOT}" == "${HOME_DIR}" || "${COT_LOOP_HF_CACHE_ROOT}" == "${HOME_DIR}/"* ) ]]; then
  echo "[cache-env] Hugging Face caches must not live under HOME: ${COT_LOOP_HF_CACHE_ROOT}" >&2
  exit 1
fi
if [[ "${COT_LOOP_HF_CACHE_ROOT}" == "/data/scratch/${USER}" || "${COT_LOOP_HF_CACHE_ROOT}" == "/data/scratch/${USER}/"* ]]; then
  echo "[cache-env] Hugging Face caches must not be user-specific scratch: ${COT_LOOP_HF_CACHE_ROOT}" >&2
  exit 1
fi
if [[ "${COT_LOOP_HF_CACHE_ROOT}" == "/data/users/${USER}" || "${COT_LOOP_HF_CACHE_ROOT}" == "/data/users/${USER}/"* ]]; then
  echo "[cache-env] Hugging Face caches must not be user-specific data: ${COT_LOOP_HF_CACHE_ROOT}" >&2
  exit 1
fi

export HF_HOME="${COT_LOOP_HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${COT_LOOP_HF_CACHE_ROOT}/hub"
export HF_HUB_CACHE="${COT_LOOP_HF_CACHE_ROOT}/hub"
export TRANSFORMERS_CACHE="${COT_LOOP_HF_CACHE_ROOT}/transformers"
export HF_DATASETS_CACHE="${COT_LOOP_HF_CACHE_ROOT}/datasets"
export HF_ASSETS_CACHE="${COT_LOOP_HF_CACHE_ROOT}/assets"

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}"
