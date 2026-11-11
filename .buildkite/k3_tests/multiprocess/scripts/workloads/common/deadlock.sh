#!/usr/bin/env bash
# Deadlock regression workload.
#
# The multiprocess harness launches the selected inference engine with the
# `deadlock` profile. This script sends 50 requests with ~30K token prefixes
# and verifies they all complete within three minutes. A CUDA-driver/GIL
# deadlock would cause requests to hang indefinitely, failing the timeout.
set -o pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"

# ── Configuration ───────────────────────────────────────────
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
TIMEOUT_SECONDS=180   # 3 minutes
ENGINE_NAME="${ENGINE_NAME:-inference engine}"
ENGINE_PORT="${ENGINE_PORT:-8000}"
ENGINE_LOG_FILE="${ENGINE_LOG_FILE:-/tmp/build_${BUILD_ID}_engine.log}"

LMCACHE_PID="$(sed -n '1p' "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$LMCACHE_PID" ]] || ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
    echo "LMCache server PID is missing or not running: ${LMCACHE_PID:-unknown}" >&2
    exit 1
fi

# ── Install py-spy for deadlock diagnosis ──────────────────
echo "=== Installing py-spy ==="
uv pip install py-spy
PY_SPY="$(which py-spy)"
echo "py-spy installed at: $PY_SPY"

PYSPY_LOG="/tmp/build_${BUILD_ID}_pyspy.log"

# ── Helper: dump stacks of server processes via py-spy ─────
dump_stacks() {
    echo "" | tee -a "$PYSPY_LOG"
    echo "=== py-spy stack dump (native + Python) ===" | tee -a "$PYSPY_LOG"

    if kill -0 "$LMCACHE_PID" 2>/dev/null; then
        echo "" | tee -a "$PYSPY_LOG"
        echo "--- LMCache server (PID=$LMCACHE_PID) ---" | tee -a "$PYSPY_LOG"
        sudo "$PY_SPY" dump --pid "$LMCACHE_PID" --native 2>&1 | tee -a "$PYSPY_LOG" || true
    fi

    # Copy to repo root so cleanup.sh collects it as a Buildkite artifact
    cp "$PYSPY_LOG" "${REPO_ROOT}/build_${BUILD_ID}_pyspy.log" 2>/dev/null || true
}

# Run benchmark with timeout.
echo "=== Running lmcache bench engine (random-prefill, 50 reqs, ~30K tokens) ==="
echo "Timeout: ${TIMEOUT_SECONDS}s"

if ! timeout "$TIMEOUT_SECONDS" lmcache bench engine \
        --engine-url "http://localhost:${ENGINE_PORT}" \
        --workload random-prefill \
        --tokens-per-gb-kvcache 6000 \
        --rp-request-length 30000 \
        --rp-num-requests 50 \
        --no-interactive \
        --no-csv \
        -q; then
    echo "FAIL: Benchmark failed or timed out (possible deadlock)"
    echo ""
    echo "=== LMCache log (last 50 lines) ==="
    tail -50 "/tmp/build_${BUILD_ID}_lmcache.log" 2>/dev/null || true
    echo ""
    echo "=== ${ENGINE_NAME} log (last 50 lines) ==="
    tail -50 "$ENGINE_LOG_FILE" 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== Benchmark completed within ${TIMEOUT_SECONDS}s ==="
echo "PASS: No deadlock detected"
