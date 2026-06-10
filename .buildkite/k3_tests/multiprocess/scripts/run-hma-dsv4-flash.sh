#!/usr/bin/env bash
# Self-contained DeepSeek-V4-Flash HMA registration test.
#
# Launches an LMCache MP server, dummy-loads a 4-layer DeepSeek-V4-Flash
# through vLLM with the LMCacheMPConnector, and verifies via the server's
# HTTP /status that every KV cache group registered with the per-group
# geometry the vLLM specs declare (tokens_per_chunk 256/64/8/4, compress
# ratios 4/128/1). See hma-dsv4-flash-check.py for the full contract.
#
# No generation is attempted, so the test does not require a GPU that can
# run FlashMLA-Sparse kernels.
#
# This test is self-contained: it handles its own server lifecycle
# instead of using the standard launch-processes.sh / wait-for-servers.sh
# (no vLLM API server is needed).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ── Configuration ───────────────────────────────────────────
export LMCACHE_PORT="${LMCACHE_PORT:-6701}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8701}"
BUILD_ID="${BUILD_ID:-local_$$}"
GPU_FOR_VLLM="${GPU_FOR_VLLM:-0}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache_dsv4.log"

# ── 1. Launch LMCache server ───────────────────────────────
echo "=== Launching LMCache server ==="
echo "ZMQ port: $LMCACHE_PORT, HTTP port: $LMCACHE_HTTP_PORT"

CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --http-port "$LMCACHE_HTTP_PORT" \
    --chunk-size 256 \
    --l1-size-gb 4 \
    --eviction-policy LRU \
    --max-workers 2 \
    > "$LMCACHE_LOG" 2>&1 &

LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"

cleanup() {
    kill "$LMCACHE_PID" 2>/dev/null
}
trap cleanup EXIT

echo "Waiting for LMCache to initialize..."
for _ in $(seq 1 30); do
    if grep -q "ZMQ cache server is running" "$LMCACHE_LOG" 2>/dev/null; then
        break
    fi
    sleep 2
done
if ! grep -q "ZMQ cache server is running" "$LMCACHE_LOG" 2>/dev/null; then
    echo "LMCache server failed to start; log tail:"
    tail -30 "$LMCACHE_LOG"
    exit 1
fi

# ── 2. Run the registration check ───────────────────────────
echo "=== Running DeepSeek-V4-Flash registration check ==="
if ! CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python "${SCRIPT_DIR}/hma-dsv4-flash-check.py"; then
    echo "DSV4 registration check failed; LMCache server log tail:"
    tail -30 "$LMCACHE_LOG"
    exit 1
fi

echo "DSV4 flash HMA registration test passed"
