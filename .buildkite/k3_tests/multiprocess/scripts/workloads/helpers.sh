#!/usr/bin/env bash
# Shared setup for OpenAI-compatible multiprocess workloads.

WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP_SCRIPT_DIR="$(cd "${WORKLOAD_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WORKLOAD_DIR}/../../../../.." && pwd)"
INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
ENGINE_ADAPTER="${MP_SCRIPT_DIR}/engines/${INFERENCE_ENGINE}.sh"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

if [[ ! -f "$ENGINE_ADAPTER" ]]; then
    echo "Unsupported inference engine '${INFERENCE_ENGINE}': $ENGINE_ADAPTER not found" >&2
    exit 1
fi
source "$ENGINE_ADAPTER"
engine_configure_defaults

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_DIR="${LMCACHE_DIR:-$REPO_ROOT}"
