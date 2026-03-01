#!/usr/bin/env bash
# Integration test: starts vLLM with LMCache (CPU and disk backends),
# sends requests via the OpenAI API, and verifies responses.
# Runs directly in the K8s pod — no Docker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh
uv pip install aiohttp

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT=8000
VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "--- Shutting down vLLM (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_server() {
    local port=$1 timeout=${2:-180}
    echo "Waiting for vLLM on port $port (timeout=${timeout}s)..."
    for ((i=0; i<timeout; i++)); do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "vLLM ready on port $port (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "vLLM failed to start within ${timeout}s"
    return 1
}

send_request() {
    local port=$1 prompt=$2
    curl -sf "http://localhost:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"${prompt}\",
            \"max_tokens\": 64,
            \"temperature\": 0
        }"
}

run_test() {
    local test_name=$1
    shift
    local env_vars=("$@")

    echo "--- :test_tube: ${test_name}"

    # Start vLLM with LMCache
    env "${env_vars[@]}" \
        vllm serve "$MODEL" \
            --port "$PORT" \
            --load-format dummy \
            --enforce-eager \
            --no-enable-prefix-caching \
            --gpu-memory-utilization 0.8 \
            --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
            > "${test_name}-vllm.log" 2>&1 &
    VLLM_PID=$!

    wait_for_server "$PORT" 180

    # Send requests — same prompt twice to exercise the cache (second hit should be cached)
    local prompt="The quick brown fox jumps over the lazy dog. Explain this sentence in detail."

    echo "Sending request 1 (cold)..."
    local resp1
    resp1=$(send_request "$PORT" "$prompt")
    echo "Response 1: $(echo "$resp1" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["text"][:100])' 2>/dev/null || echo "PARSE_ERROR")"

    echo "Sending request 2 (should hit cache)..."
    local resp2
    resp2=$(send_request "$PORT" "$prompt")
    echo "Response 2: $(echo "$resp2" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["text"][:100])' 2>/dev/null || echo "PARSE_ERROR")"

    # Verify we got valid responses
    for resp_name in resp1 resp2; do
        local resp_val="${!resp_name}"
        if ! echo "$resp_val" | python3 -c 'import sys,json; r=json.load(sys.stdin); assert r["choices"][0]["text"]' 2>/dev/null; then
            echo "FAIL: ${test_name} — invalid response in ${resp_name}"
            echo "$resp_val"
            kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true; VLLM_PID=""
            return 1
        fi
    done

    echo "PASS: ${test_name}"

    # Shut down for next test
    kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true; VLLM_PID=""
    sleep 2
}

# ── Test 1: CPU backend ───────────────────────────────────────
run_test "local_cpu" \
    "LMCACHE_CHUNK_SIZE=256" \
    "LMCACHE_LOCAL_CPU=True" \
    "LMCACHE_MAX_LOCAL_CPU_SIZE=5"

# ── Test 2: Disk backend ─────────────────────────────────────
DISK_DIR=$(mktemp -d)
trap "cleanup; rm -rf $DISK_DIR" EXIT

run_test "local_disk" \
    "LMCACHE_CHUNK_SIZE=256" \
    "LMCACHE_LOCAL_CPU=True" \
    "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
    "LMCACHE_LOCAL_DISK=file://${DISK_DIR}"

echo "--- :white_check_mark: All integration tests passed"
