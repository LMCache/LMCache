#!/usr/bin/env bash
# LMCache-driven TP=2 correctness smoke test for DeepSeek-V4-Flash.
#
# The test stores a deterministic long prompt, restarts vLLM while retaining
# LMCache, then verifies the same completion is produced from restored KV.
# The model's native FP4 indexer cache is enabled to exercise compressed DSV4
# KV groups; this is distinct from loading a separately quantized checkpoint.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1200}"
GPU_RELEASE_TIMEOUT="${GPU_RELEASE_TIMEOUT:-180}"
MAX_TOKENS="${MAX_TOKENS:-64}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-60}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TEST_DIR="$RESULTS_DIR/deepseek_v4_flash_tp"
PROMPT_FILE="$TEST_DIR/prompt.txt"
OUT_A="$TEST_DIR/output_cold.txt"
OUT_B="$TEST_DIR/output_retrieve.txt"
VLLM_PID=""

mkdir -p "$TEST_DIR"

echo "=== DeepSeek-V4-Flash LMCache-driven TP=2 smoke test ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | vLLM port: $VLLM_PORT | TP=2"
echo "Chunk size: $CHUNK_SIZE | max model length: $MAX_MODEL_LEN"
echo "Native FP4 indexer cache: enabled"
echo ""

launch_vllm() {
    local log_file="$1"
    local saved_port="$VLLM_PORT"
    unset VLLM_PORT

    vllm serve "$MODEL" \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --attention_config.use_fp4_indexer_cache=True \
        --moe-backend deep_gemm_mega_moe \
        --tokenizer-mode deepseek_v4 \
        --port "$saved_port" \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\":\"recompute\", \"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT, \"lmcache.mp.mq_timeout\":300}}" \
        > "$log_file" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" >> "$PID_FILE"
    export VLLM_PORT="$saved_port"

    if ! wait_for_server "$VLLM_PORT" "$VLLM_READY_TIMEOUT" "$log_file"; then
        echo "vLLM failed to start."
        return 1
    fi
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        local deadline=$(( $(date +%s) + 60 ))
        while [ "$(date +%s)" -lt "$deadline" ] && kill -0 "$VLLM_PID" 2>/dev/null; do
            sleep 2
        done
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true

    local deadline=$(( $(date +%s) + GPU_RELEASE_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i 1 2>/dev/null | tr -d ' ' || echo 999999)
        if [ -n "$used" ] && [ "$used" -lt 2000 ]; then
            return 0
        fi
        sleep 3
    done
    echo "WARNING: GPU 1 is still busy after ${GPU_RELEASE_TIMEOUT}s."
}

send_completion() {
    local out_file="$1"
    python3 - "$VLLM_PORT" "$MODEL" "$PROMPT_FILE" "$MAX_TOKENS" "$out_file" <<'PYEOF'
import json
import sys
import urllib.request

port, model, prompt_file, max_tokens, out_file = sys.argv[1:6]
with open(prompt_file) as prompt_handle:
    prompt = prompt_handle.read()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "seed": 0,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
with open(out_file, "w") as output_handle:
    output_handle.write(result["choices"][0]["text"])
PYEOF
}

count_retrieves() {
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb 80 \
    --eviction-policy LRU \
    --max-workers 4 \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
sleep 10

python3 - "$PROMPT_FILE" <<'PYEOF'
import sys

paragraph = (
    "A compressed key-value cache stores a reusable representation of a "
    "prefix so a second request can skip the same attention computation. "
    "The representation must preserve logical token positions even when "
    "physical cache slots use a different packing ratio. "
)
with open(sys.argv[1], "w") as prompt_handle:
    prompt_handle.write(paragraph * 160)
    prompt_handle.write("\n\nSummarize the cache invariants above:")
PYEOF

launch_vllm "/tmp/build_${BUILD_ID}_vllm.log"
send_completion "$OUT_A"
sleep "$STORE_DRAIN_SECONDS"
retrieves_before=$(count_retrieves)

stop_vllm
launch_vllm "/tmp/build_${BUILD_ID}_vllm_restart.log"
send_completion "$OUT_B"
retrieves_after=$(count_retrieves)

if ! cmp -s "$OUT_A" "$OUT_B"; then
    echo "FAILED: cold and LMCache-served completions differ."
    exit 1
fi
if [ "$retrieves_after" -le "$retrieves_before" ]; then
    echo "FAILED: LMCache served no retrieves (before=${retrieves_before}, after=${retrieves_after})."
    exit 1
fi

echo "PASS: identical completion restored from LMCache ($((retrieves_after - retrieves_before)) retrieves)."
