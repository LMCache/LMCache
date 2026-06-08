#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# macOS vLLM CPU end-to-end smoke test for lmcache multiprocess server.
# Installs vLLM CPU build, starts LMCache server + vLLM, and sends a
# completion request to verify the full stack works on macOS.
#
# Key differences from the Linux CPU e2e validation:
#   - No /dev/shm on macOS -> use pickle transport (LMCACHE_SHM_NAME="")
#   - No apt-get / libnuma1 install step
#   - Phase 3 (cache-hit metrics) is skipped to keep CI time reasonable
#
# Environment variables (all optional, defaults shown):
#   LMCACHE_HTTP_PORT   HTTP port for LMCache server  (default: 8080)
#   VLLM_PORT           HTTP port for vLLM server     (default: 8000)
#   LMCACHE_L1_SIZE_GB  LMCache L1 cache size in GB   (default: 2)
#   VLLM_READY_TIMEOUT  Seconds to wait for vLLM      (default: 300)
#   LMCACHE_HEALTHCHECK_TIMEOUT  Seconds to wait for LMCache (default: 60)

set -euo pipefail

echo "==> macOS vLLM CPU e2e test"
echo "    Python: $(python3 --version 2>&1 || true)"
echo "    uname:  $(uname -a)"
sw_vers 2>/dev/null || true

LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-2}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
LMCACHE_HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-60}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-300}"

LMCACHE_LOG="${LMCACHE_LOG_FILE:-/tmp/macos_e2e_lmcache.log}"
VLLM_LOG="/tmp/macos_e2e_vllm.log"
LMCACHE_PID=""
VLLM_PID=""

# macOS has no /dev/shm -> always use pickle transport
LMCACHE_SHM_NAME=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wait_for_endpoint() {
    local url="$1"
    local timeout="$2"
    local expected="$3"
    local label="$4"
    local response

    for _ in $(seq 1 "${timeout}"); do
        if response="$(curl -fsS "${url}" 2>/dev/null)"; then
            if [ -z "${expected}" ] || \
               echo "${response}" | grep -q "${expected}"; then
                return 0
            fi
        fi
        sleep 1
    done
    echo "!! ${label} did not become ready within ${timeout}s"
    return 1
}

print_logs() {
    echo "=== LMCache log (${LMCACHE_LOG}) ==="
    tail -n 100 "${LMCACHE_LOG}" 2>/dev/null || echo "(not found)"
    echo "=== vLLM log (${VLLM_LOG}) ==="
    tail -n 100 "${VLLM_LOG}" 2>/dev/null || echo "(not found)"
}

cleanup() {
    set +e
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "==> Stopping vLLM (PID=${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    if [ -n "${LMCACHE_PID}" ] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
        echo "==> Stopping LMCache (PID=${LMCACHE_PID})"
        kill "${LMCACHE_PID}" 2>/dev/null || true
        wait "${LMCACHE_PID}" 2>/dev/null || true
    fi
    set -e
}

on_error() {
    local exit_code=$?
    trap - ERR
    echo "!! macOS vLLM CPU e2e test FAILED (exit code: ${exit_code})"
    print_logs
    cleanup
    exit "${exit_code}"
}

trap on_error ERR
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Phase 1: Install vLLM CPU build
# ---------------------------------------------------------------------------

echo ""
echo "=== Phase 1: Install vLLM CPU build ==="

echo "==> Installing numpy<2 for scipy/vLLM compatibility"
pip install "numpy<2"
echo "    numpy<2 installed"

echo "==> Installing vLLM CPU build"
pip install vllm \
    --extra-index-url \
    https://wheels.vllm.ai/71df063c494c111ab60f6a33c54aafe7b9ae1d02/cpu
echo "    vLLM CPU install completed"

echo "==> Validating imports"
python3 -c "import lmcache; import vllm; print('imports OK')"
python3 -c "import vllm; print('vllm:', vllm.__version__)"
python3 -c "import lmcache; print('lmcache:', lmcache.__version__)"

echo "==> Downloading facebook/opt-125m (cache-aware)"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/opt-125m')
print('model ready')
"

echo "=== Phase 1 passed ==="

# ---------------------------------------------------------------------------
# Phase 2: Start LMCache server
# ---------------------------------------------------------------------------

echo ""
echo "=== Phase 2: Start LMCache server ==="
echo "    Transport mode: pickle (macOS has no /dev/shm)"
echo "    LMCache log: ${LMCACHE_LOG}"

lmcache server \
    --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
    --eviction-policy "${LMCACHE_EVICTION_POLICY}" \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    --shm-name "${LMCACHE_SHM_NAME}" \
    >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
echo "==> LMCache server started (PID=${LMCACHE_PID})"

sleep 2
if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
    echo "!! LMCache server exited immediately"
    print_logs
    exit 1
fi

echo "==> Waiting for LMCache healthcheck (timeout: ${LMCACHE_HEALTHCHECK_TIMEOUT}s)"
if ! wait_for_endpoint \
    "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" \
    "${LMCACHE_HEALTHCHECK_TIMEOUT}" \
    "" \
    "LMCache server"; then
    print_logs
    exit 1
fi
echo "    LMCache server is healthy"

# ---------------------------------------------------------------------------
# Phase 3: Start vLLM CPU server
# ---------------------------------------------------------------------------

echo ""
echo "=== Phase 3: Start vLLM CPU server ==="
echo "    vLLM log: ${VLLM_LOG}"

VLLM_TARGET_DEVICE=cpu vllm serve facebook/opt-125m \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
    >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "==> vLLM server started (PID=${VLLM_PID})"

sleep 2
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "!! vLLM server exited immediately"
    print_logs
    exit 1
fi

echo "==> Waiting for vLLM readiness (timeout: ${VLLM_READY_TIMEOUT}s)"
if ! wait_for_endpoint \
    "http://localhost:${VLLM_PORT}/v1/models" \
    "${VLLM_READY_TIMEOUT}" \
    "facebook/opt-125m" \
    "vLLM server"; then
    print_logs
    exit 1
fi
echo "    vLLM server is ready"

# ---------------------------------------------------------------------------
# Phase 4: E2E completion request
# ---------------------------------------------------------------------------

echo ""
echo "=== Phase 4: E2E completion request ==="

COMPLETION_RESPONSE="$(curl -fsS \
    "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"facebook/opt-125m","prompt":"Hello","max_tokens":5}')"
echo "    Response: ${COMPLETION_RESPONSE}"

if ! echo "${COMPLETION_RESPONSE}" | grep -q '"choices"'; then
    echo "!! E2E response missing 'choices' field"
    exit 1
fi
if ! echo "${COMPLETION_RESPONSE}" | grep -q "facebook/opt-125m"; then
    echo "!! E2E response missing expected model name"
    exit 1
fi
echo "    E2E request validation passed"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "==> macOS vLLM CPU e2e test passed"
echo "=========================================="
