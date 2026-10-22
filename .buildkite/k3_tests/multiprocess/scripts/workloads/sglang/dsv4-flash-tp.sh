#!/usr/bin/env bash
# DeepSeek-V4-Flash L1 KV correctness with SGLang tensor parallelism.
#
# SGLang represents this model with a composite KV pool containing FULL and
# SWA groups. This test stores a long prompt in LMCache, flushes only SGLang's
# local radix cache, retrieves the prompt KV from LMCache, and verifies that
# greedy output remains identical and the retrieve was non-vacuous.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INFERENCE_ENGINE="${INFERENCE_ENGINE:-sglang}"
source "${SCRIPT_DIR}/../helpers.sh"
source "${SCRIPT_DIR}/../verification/dsv4-flash-tp-verification.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
ENGINE_PORT="$SGLANG_PORT"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
SGLANG_LOG="/tmp/build_${BUILD_ID}_sglang.log"

# The SGLang DSV4 backend uses 256-token pages. Matching the LMCache chunk size
# lets each registered FULL/SWA component group transfer complete pages.
CHUNK_SIZE="${CHUNK_SIZE:-256}"
L1_SIZE_GB="${L1_SIZE_GB:-40}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
SGLANG_READY_TIMEOUT="${SGLANG_READY_TIMEOUT:-2700}"
SGLANG_CUDA_GRAPH_MAX_BS_DECODE="${SGLANG_CUDA_GRAPH_MAX_BS_DECODE:-16}"
MAX_TOKENS="${MAX_TOKENS:-128}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TP_DIR="$RESULTS_DIR/dsv4_flash_tp"
mkdir -p "$TP_DIR"
PROMPT_FILE="$TP_DIR/prompt.txt"
OUT_A="$TP_DIR/output_sglang_run.txt"
OUT_B="$TP_DIR/output_retrieve_run.txt"

if [[ "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" != "lmcache_driven" ]]; then
    echo "DeepSeek-V4-Flash exposes multiple SGLang KV groups; this test" \
        "requires lmcache_driven transfer mode." >&2
    exit 1
fi

echo "=== DeepSeek-V4-Flash SGLang L1 correctness test ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | SGLang port: $SGLANG_PORT" \
    "| TP=$TENSOR_PARALLEL_SIZE"
echo "Chunk size: $CHUNK_SIZE | L1 pool: ${L1_SIZE_GB}GB"
echo "Results dir: $TP_DIR"
echo ""

echo "=== Launching LMCache MP server (port $LMCACHE_PORT, L1 ${L1_SIZE_GB}GB) ==="
lmcache server \
    --transport "$LMCACHE_REQUEST_TRANSPORT" \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --max-workers 4 \
    --separate-object-groups \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"
sleep 10

dsv4_build_prompt

engine_write_lmcache_config

echo "=== Launching SGLang ($MODEL, TP=$TENSOR_PARALLEL_SIZE, port $SGLANG_PORT) ==="
echo "Log: $SGLANG_LOG"
# Avoid compiling every DeepGEMM shape up front; this test serves one request
# shape, so lazy JIT keeps startup bounded without changing model execution.
SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}" \
PYTHONHASHSEED=0 python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 \
    --port "$SGLANG_PORT" \
    --tp "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --mem-fraction-static "$GPU_MEMORY_UTILIZATION" \
    --cuda-graph-max-bs-decode "$SGLANG_CUDA_GRAPH_MAX_BS_DECODE" \
    --enable-lmcache \
    --lmcache-config-file "$SGLANG_LMCACHE_CONFIG_FILE" \
    > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "$SGLANG_PID" >> "$PID_FILE"
echo "SGLang started (PID=$SGLANG_PID)"

if ! wait_for_server "$SGLANG_PORT" "$SGLANG_READY_TIMEOUT" "$SGLANG_LOG"; then
    echo "SGLang failed to start."
    exit 1
fi
echo ""

dsv4_run_l1_round_trip
