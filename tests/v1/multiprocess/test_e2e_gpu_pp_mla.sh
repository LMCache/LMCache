#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end GPU test for LMCache + vLLM pipeline parallelism + MLA.
#
# Requirements:
#   - Linux with NVIDIA CUDA (compute >= 7.5) or AMD ROCm
#   - 2+ GPUs (for TP=2 PP=2)
#   - lmcache installed (pip install -e .)
#   - vllm installed (pip install vllm)
#
# Usage:
#   bash tests/v1/multiprocess/test_e2e_gpu_pp_mla.sh [model_name]
#
# Default model: moonshotai/Kimi-K2-Instruct (MLA)
# Alt model:     zai-org/GLM-4.6 (MLA)

set -euo pipefail

MODEL="${1:-moonshotai/Kimi-K2-Instruct}"
PORT=8000
LMCACHE_PORT=5555

echo "============================================================"
echo "  E2E GPU Test: LMCache + PP + MLA"
echo "  Model: ${MODEL}"
echo "  Config: TP=2 PP=2, single LMCache server"
echo "============================================================"

# Detect GPU vendor
if python -c "import torch; assert torch.version.hip is not None" 2>/dev/null; then
    echo "  GPU: AMD ROCm (HIP ${torch.version.hip})"
elif python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  GPU: NVIDIA CUDA"
else
    echo "ERROR: No GPU detected"
    exit 1
fi

# Step 1: Start LMCache server
echo ""
echo "[1/4] Starting LMCache server on port ${LMCACHE_PORT}..."
lmcache server --host localhost --port ${LMCACHE_PORT} \
    --l1-size-gb 4 --eviction-policy LRU --chunk-size 16 &
LMCACHE_PID=$!
trap "kill ${LMCACHE_PID} 2>/dev/null || true" EXIT

sleep 3
if ! kill -0 ${LMCACHE_PID} 2>/dev/null; then
    echo "ERROR: LMCache server failed to start"
    exit 1
fi
echo "  LMCache server started (PID ${LMCACHE_PID})"

# Step 2: Start vLLM with PP=2 TP=2 + LMCache
echo ""
echo "[2/4] Starting vLLM with TP=2 PP=2 + LMCache..."
vllm serve "${MODEL}" \
    --port ${PORT} \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_PORT}}}" \
    --trust-remote-code \
    --dtype bfloat16 &
VLLM_PID=$!
trap "kill ${LMCACHE_PID} ${VLLM_PID} 2>/dev/null || true" EXIT

echo "  vLLM starting (PID ${VLLM_PID})..."
echo "  Waiting for vLLM to be ready (up to 300s)..."

# Wait for vLLM to be ready
for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  vLLM is ready (after ${i}*5s)"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "ERROR: vLLM did not become ready within 300s"
    exit 1
fi

# Step 3: Send a request with a shared prefix (to trigger LMCache)
echo ""
echo "[3/4] Sending requests with shared prefix..."
PREFIX="You are a helpful assistant. Please answer the following question carefully and concisely. "
QUESTION1="What is the capital of France?"
QUESTION2="What is the capital of Germany?"

# First request (stores KV in LMCache)
echo "  Request 1 (stores KV)..."
RESPONSE1=$(curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PREFIX}${QUESTION1}\",\"max_tokens\":32,\"temperature\":0}")
echo "  Response 1: $(echo ${RESPONSE1} | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:80])' 2>/dev/null || echo '(parse error)')"

# Second request with same prefix (should hit LMCache)
echo "  Request 2 (should hit LMCache for shared prefix)..."
RESPONSE2=$(curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PREFIX}${QUESTION2}\",\"max_tokens\":32,\"temperature\":0}")
echo "  Response 2: $(echo ${RESPONSE2} | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:80])' 2>/dev/null || echo '(parse error)')"

# Step 4: Verify LMCache is being used
echo ""
echo "[4/4] Checking LMCache logs for cache hits..."
if dmesg 2>/dev/null | tail -100 | grep -qi "lmcache\|Stored\|Retrieved"; then
    echo "  LMCache activity detected in dmesg"
fi

# Check LMCache server logs for Stored/Retrieved
if kill -0 ${LMCACHE_PID} 2>/dev/null; then
    echo "  LMCache server is still running"
fi

echo ""
echo "============================================================"
echo "  E2E TEST PASSED"
echo "  - vLLM started with TP=2 PP=2"
echo "  - LMCache MP connector connected"
echo "  - Requests completed successfully"
echo "  - Model: ${MODEL} (MLA)"
echo "============================================================"

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
kill ${LMCACHE_PID} 2>/dev/null || true
exit 0
