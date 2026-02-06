#!/bin/bash
# Run LMCache multiprocessing tests locally (no Docker).
# Starts the 3 processes directly, then calls the existing CI test scripts.
#
# Usage: bash run_mp_test_local.sh

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_SCRIPT_DIR="$SCRIPT_DIR/.buildkite/scripts/multiprocessing-test"

##############################
# Configuration (same env vars the CI scripts expect)
##############################
export MODEL="${MODEL:-Qwen/Qwen3-14B}"
export LMCACHE_PORT="${LMCACHE_PORT:-6555}"
export VLLM_PORT="${VLLM_PORT:-8199}"
export VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9199}"
export BUILD_ID="${BUILD_ID:-local_$$}"
export RESULTS_DIR="${RESULTS_DIR:-$HOME/lmcache_ci_results_${BUILD_ID}}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"

GPU_LMCACHE="${GPU_LMCACHE:-2}"
GPU_BASELINE="${GPU_BASELINE:-3}"

# Track overall result
TEST_RESULT=0

##############################
# PIDs to clean up
##############################
LMCACHE_PID=""
VLLM_PID=""
VLLM_BASELINE_PID=""

cleanup() {
    echo ""
    echo "============================================"
    echo "=== Cleaning up local processes ==="
    echo "============================================"
    for pid_var in VLLM_BASELINE_PID VLLM_PID LMCACHE_PID; do
        pid="${!pid_var}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $pid_var (pid=$pid)"
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Cleanup done."
}
trap cleanup EXIT

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "=== LMCache Multiprocessing Test (local) ==="
echo "============================================"
echo "Build ID: $BUILD_ID"
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT"
echo "vLLM port: $VLLM_PORT"
echo "vLLM baseline port: $VLLM_BASELINE_PORT"
echo "Results: $RESULTS_DIR"
echo ""

##############################
# Step 1: Start LMCache server
##############################
echo "============================================"
echo "=== Step 1: Starting LMCache server ==="
echo "============================================"

python -m lmcache.v1.multiprocess.server \
    --port "$LMCACHE_PORT" \
    --l1-size-gb 80 \
    --eviction-policy LRU \
    --max-workers 4 \
    > "$RESULTS_DIR/lmcache_server.log" 2>&1 &
LMCACHE_PID=$!
echo "LMCache server PID: $LMCACHE_PID"
sleep 5

if ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
    echo "LMCache server failed to start. Log:"
    cat "$RESULTS_DIR/lmcache_server.log"
    exit 1
fi
echo ""

##############################
# Step 2: Start vLLM + LMCache
##############################
echo "============================================"
echo "=== Step 2: Starting vLLM + LMCache (GPU $GPU_LMCACHE) ==="
echo "============================================"

CUDA_VISIBLE_DEVICES="$GPU_LMCACHE" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_BATCH_INVARIANT=1 \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT}}" \
    --port "$VLLM_PORT" \
    --no-async-scheduling \
    > "$RESULTS_DIR/vllm_lmcache.log" 2>&1 &
VLLM_PID=$!
echo "vLLM+LMCache PID: $VLLM_PID"

##############################
# Step 3: Wait for vLLM+LMCache before starting baseline
##############################
echo ""
echo "============================================"
echo "=== Step 3: Waiting for vLLM+LMCache to be ready ==="
echo "============================================"

wait_for_server() {
    local port="$1"
    local name="$2"
    local pid="$3"
    local start_time=$(date +%s)

    while true; do
        local now=$(date +%s)
        local elapsed=$((now - start_time))

        if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
            echo "TIMEOUT: $name did not start within ${MAX_WAIT_SECONDS}s"
            return 1
        fi

        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name process died. Check $RESULTS_DIR for logs."
            return 1
        fi

        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "$name is ready! (took ${elapsed}s)"
            return 0
        fi

        echo "  Waiting for $name... (${elapsed}s elapsed)"
        sleep 10
    done
}

if ! wait_for_server "$VLLM_PORT" "vLLM+LMCache" "$VLLM_PID"; then
    echo "--- vLLM+LMCache logs ---"
    tail -30 "$RESULTS_DIR/vllm_lmcache.log"
    TEST_RESULT=1; exit 1
fi

##############################
# Step 4: Start baseline vLLM (after vLLM+LMCache is ready to avoid port conflicts)
##############################
echo ""
echo "============================================"
echo "=== Step 4: Starting baseline vLLM (GPU $GPU_BASELINE) ==="
echo "============================================"

CUDA_VISIBLE_DEVICES="$GPU_BASELINE" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_BATCH_INVARIANT=1 \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --port "$VLLM_BASELINE_PORT" \
    --no-async-scheduling \
    > "$RESULTS_DIR/vllm_baseline.log" 2>&1 &
VLLM_BASELINE_PID=$!
echo "Baseline vLLM PID: $VLLM_BASELINE_PID"

echo ""
echo "============================================"
echo "=== Step 5: Waiting for baseline vLLM to be ready ==="
echo "============================================"

if ! wait_for_server "$VLLM_BASELINE_PORT" "Baseline vLLM" "$VLLM_BASELINE_PID"; then
    echo "--- Baseline vLLM logs ---"
    tail -30 "$RESULTS_DIR/vllm_baseline.log"
    TEST_RESULT=1; exit 1
fi
echo ""
echo "All servers ready."

##############################
# Step 6: Run lm_eval workload
##############################
echo "============================================"
echo "=== Step 5: Running lm_eval workload ==="
echo "============================================"
if ! "$CI_SCRIPT_DIR/run-lm-eval.sh"; then
    echo "lm_eval workload test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

##############################
# Step 7: Run vllm bench serve
##############################
echo "============================================"
echo "=== Step 7: Running vllm bench serve ==="
echo "============================================"
if ! "$CI_SCRIPT_DIR/run-vllm-bench.sh"; then
    echo "vllm bench serve test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

##############################
# Step 8: Run long doc QA
##############################
echo "============================================"
echo "=== Step 8: Running long doc QA ==="
echo "============================================"
if ! "$CI_SCRIPT_DIR/run-long-doc-qa.sh"; then
    echo "long doc QA test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

echo "============================================"
echo "=== All tests passed! ==="
echo "============================================"

# Step 8: Cleanup runs automatically via trap
