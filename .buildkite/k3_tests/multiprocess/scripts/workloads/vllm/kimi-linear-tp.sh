#!/usr/bin/env bash
# Cross-TP-rank KV correctness for a hybrid MLA + linear-attention model
# (Kimi-Linear) served with tensor parallelism after a local-cache reset.
#
# Why this test exists:
#   Kimi-Linear is hybrid -- MLA full-attention layers (whose KV latent is
#   REPLICATED across TP ranks) plus Kimi-Delta-Attention linear-attention
#   layers (whose recurrent/conv "mamba" state is SHARDED across TP ranks).
#   LMCache's MLA optimization ("store KV once on rank 0, share it with every
#   TP rank") is only valid when all cached state is replicated. Applying it to
#   this hybrid model made every TP rank load rank 0's mamba shard on a cache
#   hit, so a request served from LMCache produced a different (wrong) output
#   than the same request computed from scratch. See
#   lmcache/integration/vllm/utils.py::mla_only.
#
# This test is self-contained: it launches its own LMCache server + a TP=2 vLLM
# (both GPUs) instead of using launch-processes.sh / wait-for-servers.sh, since
# it needs tensor parallelism and trust-remote-code. PIDs are written to the
# shared PID_FILE so the dispatcher's cleanup.sh trap still tears everything down.
#
# Flow (2 GPUs, TP=2):
#   1. Launch the LMCache server (kept alive for the whole test) + vLLM.
#   2. vLLM run: send one long deterministic (greedy) completion request; vLLM
#      computes it from scratch and populates LMCache. Capture output A.
#   3. Reset vLLM's local prefix cache, keeping the LMCache server running, so
#      the request's prefix KV -- including the per-rank linear-attention state
#      -- must be served by LMCache.
#   4. LMCache retrieve run: send the identical request. Capture output B.
#   5. Assert A == B. A broken cross-rank restore hands the non-zero ranks the
#      wrong state shard, which diverges the greedy decode -- so the outputs
#      differ. The stored state is byte-reloaded (not recomputed), so a correct
#      run is deterministic and the two outputs match exactly.
#   6. Assert LMCache actually served retrieves in run 2 (non-vacuous).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"

export INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
source "${SCRIPT_DIR}/../helpers.sh"
source "${SCRIPT_DIR}/../verification/l1-round-trip-verification.sh"

# ── Configuration ───────────────────────────────────────────
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
ENGINE_PORT="$VLLM_PORT"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"

# vLLM block size == LMCache chunk size for this model. 'align' requires
# block_size <= max_num_batched_tokens < 2*block_size so every prefill step
# advances exactly one block (one reusable linear-attention snapshot per chunk).
CHUNK_SIZE="${CHUNK_SIZE:-944}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1500}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# Readiness timeout for the vLLM launch. This is owned by the test (a 48B TP-shard
# load is slow) and deliberately does NOT reuse MAX_WAIT_SECONDS, which
# run-single-test.sh pre-exports to 600s -- that would shadow the value here.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"
# Kimi-Linear mixes MLA and linear-attention groups. Keep the LMCache server's
# hybrid-group registration explicit instead of relying on the default, which
# flipped off in #3869/#4437.
SEPARATE_OBJECT_GROUPS="${SEPARATE_OBJECT_GROUPS:-1}"
SEPARATE_OBJECT_GROUPS_ARG=""
if [ "$SEPARATE_OBJECT_GROUPS" = "1" ] || [ "$SEPARATE_OBJECT_GROUPS" = "true" ]; then
    SEPARATE_OBJECT_GROUPS_ARG="--separate-object-groups"
fi

# Tokens to generate per request. Greedy (temperature 0); a divergence in the
# resumed state shows up within the first few tokens, but a longer generation
# makes an accidental match astronomically unlikely.
MAX_TOKENS="${MAX_TOKENS:-128}"
# Seconds to let async LMCache stores drain before clearing vLLM's local cache.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TP_DIR="$RESULTS_DIR/kimi_linear_tp"
mkdir -p "$TP_DIR"
PROMPT_FILE="$TP_DIR/prompt.txt"
OUT_A="$TP_DIR/output_vllm_run.txt"
OUT_B="$TP_DIR/output_retrieve_run.txt"

VLLM_PID=""

echo "=== Kimi-Linear cross-TP-rank KV correctness test (vLLM cache reset) ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | vLLM port: $VLLM_PORT | TP=2"
echo "Chunk size: $CHUNK_SIZE | max_num_batched_tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Results dir: $TP_DIR"
echo ""

# Launch a TP=2 vLLM+LMCache instance and wait until it serves.
# Arguments:
#   $1 log_file - where to redirect vLLM stdout/stderr.
# Sets the global VLLM_PID and appends it to PID_FILE for cleanup.
launch_vllm() {
    local log_file="$1"
    echo "=== Launching vLLM (Kimi-Linear, TP=2, port $VLLM_PORT) ==="
    echo "Log: $log_file"
    # Save and unset VLLM_PORT: vLLM's internal get_open_port() would otherwise
    # collide with the serving port for torch.distributed.
    local saved_port="$VLLM_PORT"
    unset VLLM_PORT

    VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
        --tensor-parallel-size 2 \
        --trust-remote-code \
        --enforce-eager \
        --no-enable-flashinfer-autotune \
        --enable-prefix-caching \
        --moe-backend triton \
        --mamba-cache-mode align \
        --max-model-len auto \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --port "$saved_port" \
        --block-size "$CHUNK_SIZE" \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.host\": \"$LMCACHE_REQUEST_SCHEME://localhost\", \"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 120}}" \
        > "$log_file" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" >> "$PID_FILE"
    export VLLM_PORT="$saved_port"
    echo "vLLM started (PID=$VLLM_PID)"

    if ! wait_for_server "$VLLM_PORT" "$VLLM_READY_TIMEOUT" "$log_file"; then
        echo "vLLM failed to start."
        return 1
    fi
    echo ""
}

# ── 1. Launch LMCache MP server ─────────────────────────────
echo "=== Launching LMCache MP server (port $LMCACHE_PORT) ==="
lmcache server \
    --transport "$LMCACHE_REQUEST_TRANSPORT" \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb 80 \
    --eviction-policy LRU \
    --max-workers 4 \
    ${SEPARATE_OBJECT_GROUPS_ARG} \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"
sleep 10

# ── 2. Build a long, deterministic prompt, then ask for a summary ──
# A ~7-8k word document is long enough to store multiple recurrent-state
# checkpoints. The shared helper keeps both engines on byte-identical input.
l1_round_trip_build_prompt \
    "$PROMPT_FILE" \
    $'\n\nSummarize the manual page text above:'

# ── 3. vLLM run: compute from scratch, populating LMCache ───
launch_vllm "/tmp/build_${BUILD_ID}_vllm.log"

# ── 4-6. Clear local APC, retrieve from LMCache, and verify ──
l1_round_trip_run "Kimi-Linear cross-TP-rank KV"
