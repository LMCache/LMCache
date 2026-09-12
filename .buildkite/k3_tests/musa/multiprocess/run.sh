#!/usr/bin/env bash
# MUSA MP lane: adapt the vendor image to the shared multiprocess runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODE="${1:-${MUSA_MP_TEST_MODE:-lm_eval}}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../unittests/ci-common.sh"
if [ "${MODE}" = "smoke" ]; then
    MODE="lm_eval"
fi

export TORCH_DEVICE_TYPE="musa"
export VLLM_TARGET_DEVICE="musa"
export DEVICE_AFFINITY_VAR="MUSA_VISIBLE_DEVICES"
export GPU_MEMORY_PROBE_ENABLED="${GPU_MEMORY_PROBE_ENABLED:-0}"
export BATCH_INVARIANT_DEFAULT="${BATCH_INVARIANT_DEFAULT:-0}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-/models/Qwen3-0.6B}"
export MODEL="${MODEL:-${DEFAULT_MODEL}}"
export LM_EVAL_NUM_CONCURRENT_DEFAULT="${LM_EVAL_NUM_CONCURRENT_DEFAULT:-4}"
export LM_EVAL_VERIFY_MODE_DEFAULT="${LM_EVAL_VERIFY_MODE_DEFAULT:-samples}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
export LMCACHE_DEVICE_BACKEND="musa"
export LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"
export MUSA_CI_PREPROVISIONED="${MUSA_CI_PREPROVISIONED:-1}"
export RESULTS_DIR="${RESULTS_DIR:-${MUSA_CI_ARTIFACT_DIR:-/tmp}/multiprocess}"
export LIMIT="${LIMIT:-4}"
export NUM_CONCURRENT="${NUM_CONCURRENT:-2}"
unset CUDA_VISIBLE_DEVICES || true

bootstrap_musa_ci 0

if [ "${MUSA_CI_PREPROVISIONED}" = "1" ]; then
    # The container wrapper already validated and installed LMCache.
    exec bash "${REPO_ROOT}/.buildkite/k3_tests/multiprocess/run.sh" "${MODE}"
fi

exec bash "${REPO_ROOT}/.buildkite/k3_tests/multiprocess/run.sh" "${MODE}"
