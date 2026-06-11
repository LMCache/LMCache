#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Generic CPU vLLM e2e wrapper for both macOS and Ubuntu.
# Assumes vLLM CPU build and facebook/opt-125m are already installed/
# downloaded by the CI workflow steps before this script is invoked.
#
# Delegates all logic to the shared run-cpu-e2e-validation.sh with:
#   SKIP_INSTALL=1              (install done by CI workflow steps)
#   SKIP_CACHE_HIT_VALIDATION=1 (skip Phase 3 to keep CI time reasonable)
#
# Transport mode is selected via LMCACHE_TRANSPORT_MODE:
#   pickle  -> LMCACHE_SHM_NAME=""  (pickle transport)
#   shm     -> LMCACHE_SHM_NAME=__default__  (shared memory, Linux only)
#   handle  -> LMCACHE_MP_TRANSFER_MODE=handle (POSIX SHM server-side copy)
#
# Environment variables (all optional, defaults shown):
#   LMCACHE_TRANSPORT_MODE   Transport mode: pickle|shm|handle (default: handle)
#   LMCACHE_HTTP_PORT        HTTP port for LMCache server  (default: 8080)
#   VLLM_PORT                HTTP port for vLLM server     (default: 8000)
#   LMCACHE_L1_SIZE_GB       LMCache L1 cache size in GB   (default: 2)
#   VLLM_READY_TIMEOUT       Seconds to wait for vLLM      (default: 300)
#   LMCACHE_HEALTHCHECK_TIMEOUT  Seconds to wait for LMCache (default: 60)

set -euo pipefail

OS="$(uname -s)"
echo "==> CPU vLLM e2e test (OS: ${OS})"
echo "    Python: $(python3 --version 2>&1 || true)"
echo "    uname:  $(uname -a)"
if [ "${OS}" = "Darwin" ]; then
    sw_vers 2>/dev/null || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHARED_SCRIPT="${REPO_ROOT}/.buildkite/k3_tests/multiprocess/scripts/run-cpu-e2e-validation.sh"

if [ ! -f "${SHARED_SCRIPT}" ]; then
    echo "!! Shared script not found: ${SHARED_SCRIPT}"
    exit 1
fi

LMCACHE_TRANSPORT_MODE="${LMCACHE_TRANSPORT_MODE:-handle}"

# Map LMCACHE_TRANSPORT_MODE to the vars expected by the shared script.
# pickle/shm explicitly set LMCACHE_MP_TRANSFER_MODE=data because
# leaving it at the script default (`auto`) would make Step 5.5's
# transport-mode verification expect ``auto`` instead of pickle/shm.
case "${LMCACHE_TRANSPORT_MODE}" in
  pickle)
    export LMCACHE_SHM_NAME=""
    export LMCACHE_MP_TRANSFER_MODE="data"
    ;;
  shm)
    export LMCACHE_SHM_NAME="__default__"
    export LMCACHE_MP_TRANSFER_MODE="data"
    ;;
  handle)
    export LMCACHE_MP_TRANSFER_MODE="handle"
    # LMCACHE_SHM_NAME is not used in handle mode; leave at default
    ;;
  *)
    echo "!! Unknown LMCACHE_TRANSPORT_MODE='${LMCACHE_TRANSPORT_MODE}'"
    echo "   Valid values: pickle, shm, handle"
    exit 1
    ;;
esac

export SKIP_INSTALL="${SKIP_INSTALL:-1}"
export SKIP_CACHE_HIT_VALIDATION="${SKIP_CACHE_HIT_VALIDATION:-0}"
export LMCACHE_HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-60}"
export VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-300}"
export LMCACHE_LOG_FILE="${LMCACHE_LOG_FILE:-/tmp/cpu_e2e_lmcache.log}"
export VLLM_LOG_FILE="${VLLM_LOG_FILE:-/tmp/cpu_e2e_vllm.log}"

echo "    LMCACHE_TRANSPORT_MODE=${LMCACHE_TRANSPORT_MODE}"
echo "    SKIP_INSTALL=${SKIP_INSTALL}"
echo "    SKIP_CACHE_HIT_VALIDATION=${SKIP_CACHE_HIT_VALIDATION}"
echo "    Delegating to: ${SHARED_SCRIPT}"

exec bash "${SHARED_SCRIPT}"
