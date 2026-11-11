#!/usr/bin/env bash
# Cross-TP-rank KV correctness for Kimi-Linear served by SGLang.
#
# The MLA state is replicated across TP ranks, while KDA recurrent state is
# sharded. Flushing SGLang's local radix cache forces both kinds of state to be
# restored from the still-running LMCache server.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"

export INFERENCE_ENGINE="${INFERENCE_ENGINE:-sglang}"
source "${SCRIPT_DIR}/../helpers.sh"
source "${SCRIPT_DIR}/../verification/l1-round-trip-verification.sh"

LMCACHE_PORT="${LMCACHE_PORT:-6555}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
ENGINE_PORT="$SGLANG_PORT"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"

# SGLang checkpoints Kimi's recurrent state on a 64-token grid. A 64-token
# page keeps the full-attention and recurrent groups aligned; the LMCache
# chunk must be divisible by both group sizes.
PAGE_SIZE="${PAGE_SIZE:-64}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
SGLANG_READY_TIMEOUT="${SGLANG_READY_TIMEOUT:-900}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"
MAX_TOKENS="${MAX_TOKENS:-128}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TP_DIR="$RESULTS_DIR/kimi_linear_tp"
mkdir -p "$TP_DIR"
PROMPT_FILE="$TP_DIR/prompt.txt"
OUT_A="$TP_DIR/output_sglang_run.txt"
OUT_B="$TP_DIR/output_retrieve_run.txt"

SGLANG_PID=""

if [[ "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" != "lmcache_driven" ]]; then
    echo "Kimi-Linear SGLang coverage currently requires lmcache_driven mode." >&2
    exit 1
fi
if ((CHUNK_SIZE % PAGE_SIZE != 0)); then
    echo "CHUNK_SIZE ($CHUNK_SIZE) must be divisible by PAGE_SIZE ($PAGE_SIZE)." >&2
    exit 1
fi
if ((MAMBA_TRACK_INTERVAL % PAGE_SIZE != 0)); then
    echo "MAMBA_TRACK_INTERVAL ($MAMBA_TRACK_INTERVAL) must be divisible by PAGE_SIZE ($PAGE_SIZE)." >&2
    exit 1
fi

echo "=== Kimi-Linear cross-TP-rank KV correctness test (SGLang cache flush) ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | SGLang port: $SGLANG_PORT | TP=2"
echo "Chunk size: $CHUNK_SIZE | page size: $PAGE_SIZE"
echo "Mamba track interval: $MAMBA_TRACK_INTERVAL"
echo "Results dir: $TP_DIR"
echo ""

launch_sglang() {
    local log_file="$1"

    echo "=== Launching SGLang (Kimi-Linear, TP=2, port $SGLANG_PORT) ==="
    echo "Log: $log_file"
    engine_write_lmcache_config
    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --host 127.0.0.1 \
        --port "$SGLANG_PORT" \
        --tp-size 2 \
        --trust-remote-code \
        --mem-fraction-static "$GPU_MEMORY_UTILIZATION" \
        --max-running-requests 1 \
        --attention-backend triton \
        --linear-attn-backend triton \
        --page-size "$PAGE_SIZE" \
        --mamba-radix-cache-strategy extra_buffer \
        --mamba-track-interval "$MAMBA_TRACK_INTERVAL" \
        --enable-lmcache \
        --lmcache-config-file "$SGLANG_LMCACHE_CONFIG_FILE" \
        > "$log_file" 2>&1 &
    SGLANG_PID=$!
    echo "$SGLANG_PID" >> "$PID_FILE"
    echo "SGLang started (PID=$SGLANG_PID)"

    if ! wait_for_server "$SGLANG_PORT" "$SGLANG_READY_TIMEOUT" "$log_file"; then
        echo "SGLang failed to start."
        return 1
    fi
    echo ""
}

echo "=== Launching LMCache MP server (port $LMCACHE_PORT) ==="
lmcache server \
    --transport "$LMCACHE_REQUEST_TRANSPORT" \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb 80 \
    --eviction-policy LRU \
    --max-workers 4 \
    --separate-object-groups \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"
sleep 10

l1_round_trip_build_prompt \
    "$PROMPT_FILE" \
    $'\n\nSummarize the manual page text above:'

launch_sglang "/tmp/build_${BUILD_ID}_sglang.log"
l1_round_trip_run "Kimi-Linear cross-TP-rank KV"
