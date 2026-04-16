#!/usr/bin/env bash
# Run fp8 serde end-to-end test with disk (fs) L2 adapter.
#
# This script:
#   1. Kills the existing LMCache MP server + vLLM
#   2. Relaunches LMCache server with disk L2 + fp8 serde
#   3. Relaunches vLLM connected via LMCacheMPConnector
#   4. Sends a long inference prompt — KV serialized (fp8) and stored to disk
#   5. Force-clears L1 cache via HTTP API
#   6. Re-sends the same prompt — KV deserialized from disk, vLLM resumes from cache
#   7. Verifies serde events in the lmcache log + identical responses
#
# Expects the following env vars from run-mp-test.sh:
#   VLLM_PORT, MODEL, BUILD_ID, RESULTS_DIR, LMCACHE_DIR,
#   LMCACHE_PORT, CPU_BUFFER_SIZE, MAX_WORKERS, GPU_FOR_VLLM (optional)
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"

FP8_RESULTS_DIR="$RESULTS_DIR/fp8_serde"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
DISK_PATH="/tmp/lmcache_fp8_serde_disk_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache_fp8.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm_fp8.log"

echo "=== Fp8 Serde End-to-End Test ==="
echo "Model: $MODEL"
echo "L2 adapter: fs (disk) at ${DISK_PATH}"
echo "Serde: fp8 (float8_e4m3fn)"
echo "Results: $FP8_RESULTS_DIR"
echo ""

mkdir -p "$FP8_RESULTS_DIR"
rm -rf "$DISK_PATH"
mkdir -p "$DISK_PATH"

# ---------------------------------------------------------------------------
# Step 1: Kill existing LMCache + vLLM, relaunch with fp8 serde config
# ---------------------------------------------------------------------------

echo "--- Stopping existing LMCache MP server and vLLM ---"
if [ -f "$PID_FILE" ]; then
    LMCACHE_PID=$(sed -n '1p' "$PID_FILE")
    VLLM_PID=$(sed -n '2p' "$PID_FILE")
    for pid in $LMCACHE_PID $VLLM_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
fi

echo "--- Launching LMCache MP server with disk L2 + fp8 serde ---"
L2_ADAPTER_JSON="{\"type\":\"fs\",\"base_path\":\"${DISK_PATH}\",\"serde\":{\"type\":\"fp8\",\"fp8_dtype\":\"float8_e4m3fn\"}}"

GPU_DEVICE="${GPU_FOR_VLLM:-0}"

CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
LMCACHE_LOG_LEVEL=DEBUG \
lmcache server \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy LRU \
    --l2-store-policy default \
    --l2-prefetch-policy default \
    --l2-adapter "$L2_ADAPTER_JSON" \
    --max-workers "$MAX_WORKERS" \
    --port "$LMCACHE_PORT" \
    --http-port "$LMCACHE_HTTP_PORT" \
    > "$LMCACHE_LOG" 2>&1 &

NEW_LMCACHE_PID=$!
echo "LMCache fp8 serde server started (PID=$NEW_LMCACHE_PID)"

echo "Waiting for LMCache HTTP API to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${LMCACHE_HTTP_PORT}/api/healthcheck" > /dev/null 2>&1; then
        echo "LMCache ready after ${i}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo "LMCache failed to start (HTTP API never came up)"
        echo "LMCache log (last 50 lines):"
        tail -50 "$LMCACHE_LOG" || true
        exit 1
    fi
done

echo "--- Launching vLLM with LMCacheMPConnector ---"
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_DEVICE}" | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
if [ "$GPU_MEMORY_GB" -gt 90 ]; then
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

env -u VLLM_PORT \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 10}}" \
    --port "$VLLM_PORT" \
    --no-enable-prefix-caching \
    --enforce-eager \
    $GPU_MEMORY_UTIL_ARG \
    > "$VLLM_LOG" 2>&1 &

NEW_VLLM_PID=$!
echo "vLLM started (PID=$NEW_VLLM_PID)"

# Update PID file
if [ -f "$PID_FILE" ]; then
    sed -i "1s/.*/$NEW_LMCACHE_PID/" "$PID_FILE"
    sed -i "2s/.*/$NEW_VLLM_PID/" "$PID_FILE"
else
    echo "$NEW_LMCACHE_PID" > "$PID_FILE"
    echo "$NEW_VLLM_PID" >> "$PID_FILE"
fi

echo "--- Waiting for vLLM to be ready ---"
if ! wait_for_server "$VLLM_PORT" 600; then
    echo "vLLM failed to start"
    echo "LMCache log (last 50 lines):"
    tail -50 "$LMCACHE_LOG" || true
    echo "vLLM log (last 50 lines):"
    tail -50 "$VLLM_LOG" || true
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Build a long enough prompt to fill at least one 256-token chunk
# ---------------------------------------------------------------------------
PROMPT=""
for i in $(seq 1 8); do
    PROMPT+="The history and significance of the Roman empire spans more than a thousand years and profoundly shaped Western civilization. "
    PROMPT+="Its legal, architectural, linguistic, and political legacies persist to this day, influencing modern governments, languages, art, engineering, and law. "
    PROMPT+="The empire's trajectory from the founding of Rome through the Republic, the transition to the Principate under Augustus, the Pax Romana, the crisis of the third century, "
    PROMPT+="the Dominate under Diocletian, the adoption of Christianity under Constantine, the splitting into Western and Eastern halves, and the eventual collapse of the West "
    PROMPT+="is one of history's great narratives. Key figures include Julius Caesar, Augustus, Marcus Aurelius, Diocletian, Constantine, Justinian, and many others. "
done
PROMPT+="Tell me a long, detailed story about the rise, peak, and eventual fall of Rome, naming important figures and events."

# ---------------------------------------------------------------------------
# Step 3: First inference (cold path -> fp8 serialize + L2 store)
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 3: First inference (cold) ==="
echo "============================================"
RESPONSE_1_FILE="$FP8_RESULTS_DIR/response_1.json"
curl -sf -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"$PROMPT\",
        \"max_tokens\": 32,
        \"temperature\": 0
    }" > "$RESPONSE_1_FILE"
RESPONSE_1=$(python3 -c "import json; print(json.load(open('$RESPONSE_1_FILE'))['choices'][0]['text'])")
echo "Response 1: ${RESPONSE_1:0:100}..."

# Wait for L2 store to flush serialized data to disk
echo "Waiting 5s for L2 store flush..."
sleep 5

# ---------------------------------------------------------------------------
# Step 4: Force-clear L1 cache (simulates eviction / restart)
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 4: Force-clearing L1 cache ==="
echo "============================================"
curl -sf -X POST "http://localhost:${LMCACHE_HTTP_PORT}/api/clear-cache" \
    | python3 -m json.tool

# ---------------------------------------------------------------------------
# Step 5: Second inference (L1 miss -> L2 prefetch + fp8 deserialize)
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 5: Second inference (L1 miss -> L2 prefetch) ==="
echo "============================================"
RESPONSE_2_FILE="$FP8_RESULTS_DIR/response_2.json"
curl -sf -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"$PROMPT\",
        \"max_tokens\": 32,
        \"temperature\": 0
    }" > "$RESPONSE_2_FILE"
RESPONSE_2=$(python3 -c "import json; print(json.load(open('$RESPONSE_2_FILE'))['choices'][0]['text'])")
echo "Response 2: ${RESPONSE_2:0:100}..."

# ---------------------------------------------------------------------------
# Step 6: Verify the serde + L2 round-trip happened
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 6: Verifying serde + L2 round-trip ==="
echo "============================================"

failed=0

# Check 1: fp8 serialized data appeared on disk
disk_files=$(find "$DISK_PATH" -type f | wc -l)
if [ "$disk_files" -gt 0 ]; then
    echo "[PASS] L2 disk: $disk_files serialized files written"
else
    echo "[FAIL] L2 disk: no files written to $DISK_PATH"
    failed=1
fi

# Check 2: serde serialize fired
if grep -qE "Serde: serialize task .* completed successfully" "$LMCACHE_LOG"; then
    serialize_count=$(grep -cE "Serde: serialize task .* completed successfully" "$LMCACHE_LOG")
    echo "[PASS] Serde serialize: $serialize_count successful tasks logged"
else
    echo "[FAIL] No 'Serde: serialize task ... completed successfully' in lmcache log"
    failed=1
fi

# Check 3: serde deserialize fired
if grep -qE "Serde: deserialize task .* completed successfully" "$LMCACHE_LOG"; then
    deserialize_count=$(grep -cE "Serde: deserialize task .* completed successfully" "$LMCACHE_LOG")
    echo "[PASS] Serde deserialize: $deserialize_count successful tasks logged"
else
    echo "[FAIL] No 'Serde: deserialize task ... completed successfully' in lmcache log"
    failed=1
fi

# Check 4: prefetch hit L2 (request 2 should have non-zero L2 prefix hits)
if grep -qE "Prefetch request completed \(L1\+L2\): [1-9][0-9]*/.* prefix hits \(0 L1, [1-9][0-9]* L2\)" "$LMCACHE_LOG"; then
    l2_hit_line=$(grep -E "Prefetch request completed \(L1\+L2\): [1-9][0-9]*/.* prefix hits \(0 L1, [1-9][0-9]* L2\)" "$LMCACHE_LOG" | tail -1)
    echo "[PASS] L2 cache hit: $l2_hit_line"
else
    echo "[FAIL] No L2 prefix hits found in lmcache log"
    failed=1
fi

# Check 5: both responses are identical (deserialized KV produced same generation)
if [ "$RESPONSE_1" = "$RESPONSE_2" ]; then
    echo "[PASS] Inference outputs identical across cold + cached requests"
else
    echo "[FAIL] Inference outputs differ:"
    echo "  Response 1: $RESPONSE_1"
    echo "  Response 2: $RESPONSE_2"
    failed=1
fi

echo ""
if [ "$failed" -ne 0 ]; then
    echo "============================================"
    echo "=== fp8 serde test FAILED ==="
    echo "============================================"
    echo "LMCache log (last 50 lines):"
    tail -50 "$LMCACHE_LOG" || true
    exit 1
fi

echo "============================================"
echo "=== fp8 serde test PASSED ==="
echo "============================================"
echo "Results: $FP8_RESULTS_DIR"

# Clean up disk artifacts (logs are kept for inspection)
rm -rf "$DISK_PATH"
