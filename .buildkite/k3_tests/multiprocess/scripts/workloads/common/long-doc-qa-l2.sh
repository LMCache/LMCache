#!/usr/bin/env bash
# Run long_doc_qa for L2 skip_l1 mode with mock L2 adapter.
#
# This script:
#   1. Kills the existing LMCache MP server
#   2. Relaunches it with L2 config (skip_l1 + mock L2 at 2 GB/s)
#   3. Restarts the selected inference engine against the L2 server
#   4. Runs long_doc_qa against the baseline and L2-enabled engine
#   5. Verifies L2 query is faster than baseline and warmup overhead is bounded,
#      then the L2 data flow and observability metrics (long-doc-qa-l2-verify.sh)
#
# Expects the following env vars from run-mp-test.sh:
#   ENGINE_PORT, ENGINE_BASELINE_PORT, MODEL, BUILD_ID, RESULTS_DIR, LMCACHE_DIR,
#   LMCACHE_PORT, CPU_BUFFER_SIZE, MAX_WORKERS, GPU_FOR_ENGINE (optional)
set -e
set -o pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"

# Configuration
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
ENGINE_L2_LOG_FILE="/tmp/build_${BUILD_ID}_${INFERENCE_ENGINE}_l2.log"

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-30}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-5}"

# Mock L2 config
L2_MAX_SIZE_GB="${L2_MAX_SIZE_GB:-80}"
L2_BANDWIDTH_GB="${L2_BANDWIDTH_GB:-4}"

# L2 performance thresholds
# Recent CI runs show ~1.51-1.67x query speedup, ~1.77-2.02x TTFT speedup,
# and ~0.87-0.99x warmup overhead. Tighten from the previous pass-anything
# thresholds (1.0x/1.0x/2.0x) while leaving headroom for variance.
MIN_L2_SPEEDUP="${MIN_L2_SPEEDUP:-1.3}"
MIN_L2_TTFT_SPEEDUP="${MIN_L2_TTFT_SPEEDUP:-1.5}"
MAX_WARMUP_OVERHEAD="${MAX_WARMUP_OVERHEAD:-1.2}"

L2_RESULTS_DIR="$RESULTS_DIR/long_doc_qa_l2"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
# /metrics is now served by the LMCache FastAPI HTTP server (port 8080
# by default) — the legacy ``--prometheus-port`` standalone server was
# disabled for the ``lmcache server`` entrypoint by #3164.  Defined here
# (not just in Step 4) so the relaunch and the curl scrape agree.
METRICS_HTTP_PORT="${METRICS_HTTP_PORT:-8080}"

echo "=== Long Doc QA L2 Performance Test ==="
echo "Model: $MODEL"
echo "L2 adapter: mock (${L2_MAX_SIZE_GB}GB, ${L2_BANDWIDTH_GB}GB/s)"
echo "Store policy: skip_l1 | Eviction: noop"
echo "Thresholds: speedup>=${MIN_L2_SPEEDUP}x, TTFT speedup>=${MIN_L2_TTFT_SPEEDUP}x, overhead<=${MAX_WARMUP_OVERHEAD}x"
echo "Results: $L2_RESULTS_DIR"
echo ""

mkdir -p "$L2_RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Kill existing LMCache + engine, relaunch both with L2 config
# ---------------------------------------------------------------------------

echo "--- Stopping existing LMCache MP server and ${ENGINE_NAME} ---"
# PID file layout: line1=LMCache, line2=engine w/ LMCache, line3=baseline.
# These processes were launched by an earlier script (launch-processes.sh)
# and are not children of this shell, so ``wait $pid`` is a no-op here.
# We instead poll until each PID actually exits, then poll until the
# Prometheus port is free, otherwise the LMCache relaunch below would
# fail to bind /metrics and the metrics check would fail spuriously.
if [ -f "$PID_FILE" ]; then
    LMCACHE_PID=$(sed -n '1p' "$PID_FILE")
    OLD_ENGINE_PID=$(sed -n '2p' "$PID_FILE")
    for pid in $LMCACHE_PID $OLD_ENGINE_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            for _ in $(seq 1 60); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            # Last resort: SIGKILL if SIGTERM didn't take after 30s.
            if kill -0 "$pid" 2>/dev/null; then
                echo "PID $pid still alive after SIGTERM; sending SIGKILL"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done
    # Poll until the Prometheus port is fully released so the new server
    # below can bind it cleanly.
    for _ in $(seq 1 30); do
        if ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) \
                | awk '{print $4}' | grep -qE ":${METRICS_HTTP_PORT}$"; then
            break
        fi
        sleep 0.5
    done
fi

echo "--- Launching LMCache MP server with L2 config ---"
L2_ADAPTER_JSON="{\"type\":\"mock\",\"max_size_gb\":${L2_MAX_SIZE_GB},\"mock_bandwidth_gb\":${L2_BANDWIDTH_GB}}"

# Determine GPU to use
GPU_DEVICE="${GPU_FOR_ENGINE:-0}"

server_environment=("${DEVICE_AFFINITY_VAR}=${GPU_DEVICE}")
engine_add_lmcache_server_environment server_environment
env "${server_environment[@]}" lmcache server \
    --transport "$LMCACHE_REQUEST_TRANSPORT" \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy noop \
    --l2-store-policy skip_l1 \
    --l2-prefetch-policy default \
    --l2-adapter "$L2_ADAPTER_JSON" \
    --max-workers "$MAX_WORKERS" \
    --metrics-sample-rate 1.0 \
    --http-port "$METRICS_HTTP_PORT" \
    --port "$LMCACHE_PORT" \
    > "/tmp/build_${BUILD_ID}_lmcache_l2.log" 2>&1 &

NEW_LMCACHE_PID=$!
echo "LMCache L2 server started (PID=$NEW_LMCACHE_PID)"

echo "Waiting for LMCache L2 to initialize..."
sleep 10

echo "--- Launching ${ENGINE_NAME} with LMCache ---"
engine_prepare_launch "$GPU_DEVICE"
engine_launch lmcache "$ENGINE_PORT" "$GPU_DEVICE" "$ENGINE_L2_LOG_FILE"
NEW_ENGINE_PID="$ENGINE_PID"
echo "${ENGINE_NAME} started (PID=$NEW_ENGINE_PID)"

# Update PID file (replace lines 1 and 2, keep baseline on line 3)
if [ -f "$PID_FILE" ]; then
    sed -i "1s/.*/$NEW_LMCACHE_PID/" "$PID_FILE"
    sed -i "2s/.*/$NEW_ENGINE_PID/" "$PID_FILE"
else
    echo "$NEW_LMCACHE_PID" > "$PID_FILE"
    echo "$NEW_ENGINE_PID" >> "$PID_FILE"
fi

# Wait for the engine to be ready (needs time to load model)
echo "--- Waiting for ${ENGINE_NAME} to be ready ---"
if ! wait_for_server "$ENGINE_PORT" 300; then
    echo "${ENGINE_NAME} failed to start after restart"
    echo "LMCache L2 log (last 50 lines):"
    tail -50 "/tmp/build_${BUILD_ID}_lmcache_l2.log" || true
    echo "${ENGINE_NAME} log (last 50 lines):"
    tail -50 "$ENGINE_L2_LOG_FILE" || true
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_long_doc_qa() {
    local port="$1"
    local result_file="$2"
    local description="$3"

    echo "--- Running long_doc_qa ($description) on port $port ---"
    local output_file="$L2_RESULTS_DIR/${description}_output.txt"

    python3 "$LMCACHE_DIR/benchmarks/long_doc_qa/long_doc_qa.py" \
        --port "$port" \
        --model "$MODEL" \
        --document-length "$DOCUMENT_LENGTH" \
        --num-documents "$NUM_DOCUMENTS" \
        --output-len "$OUTPUT_LEN" \
        --repeat-count "$REPEAT_COUNT" \
        --repeat-mode "$REPEAT_MODE" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --max-inflight-requests "$MAX_INFLIGHT_REQUESTS" \
        --output "$output_file" \
        --json-output \
        2>>"$output_file" | tee "$result_file"

    echo "Completed: $description"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 2: Run benchmarks
# ---------------------------------------------------------------------------

# Phase 1: Baseline -- reuse results from step 5 (same port, same params)
STEP5_BASELINE="$RESULTS_DIR/long_doc_qa/baseline_result.json"
if [ -f "$STEP5_BASELINE" ]; then
    echo "============================================"
    echo "=== Phase 1: Reusing baseline from step 5 ==="
    echo "============================================"
    cp "$STEP5_BASELINE" "$L2_RESULTS_DIR/baseline_result.json"
    echo "Copied baseline results from $STEP5_BASELINE"
    echo ""
else
    echo "============================================"
    echo "=== Phase 1: ${ENGINE_NAME} baseline (no LMCache) ==="
    echo "============================================"
    run_long_doc_qa "$ENGINE_BASELINE_PORT" "$L2_RESULTS_DIR/baseline_result.json" "baseline"
fi

# Phase 2+3: L2 warmup + query (repeat_count=2, tile mode)
#   Round 1 (warmup): prompts -> L1 write buffer -> L2 store -> L1 delete
#   Round 2 (query):  prompts -> L1 miss -> L2 prefetch -> L1 load -> serve
echo "============================================"
echo "=== Phase 2+3: ${ENGINE_NAME} + LMCache L2 ==="
echo "============================================"
run_long_doc_qa "$ENGINE_PORT" "$L2_RESULTS_DIR/l2_result.json" "l2"

# ---------------------------------------------------------------------------
# Steps 3-5: Verify thresholds, L2 data flow and observability metrics
# ---------------------------------------------------------------------------
# The verifier runs every check and reports all of them before exiting, so a
# threshold failure still leaves the /metrics snapshot and the data-flow
# verdict in $L2_RESULTS_DIR while the L2 server is alive to be scraped.
# Its non-zero exit fails this script under ``set -e`` as before.

export L2_RESULTS_DIR METRICS_HTTP_PORT
export MIN_L2_SPEEDUP MIN_L2_TTFT_SPEEDUP MAX_WARMUP_OVERHEAD
LMCACHE_L2_LOG="/tmp/build_${BUILD_ID}_lmcache_l2.log" \
    "${COMMON_WORKLOAD_DIR}/long-doc-qa-l2-verify.sh"

echo "============================================"
echo "=== L2 Long Doc QA test completed ==="
echo "============================================"
echo "Results: $L2_RESULTS_DIR"
