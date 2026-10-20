#!/usr/bin/env bash
# L1 + slot-compression KV correctness for DeepSeek-V4-Flash (sparse-MLA +
# fp8) served with tensor parallelism.
#
# Why this test exists:
#   DeepSeek-V4-Flash interleaves several KV cache groups with different block
#   geometries (tokens_per_block 256/64/8/4): compressed fp8 MLA latents and
#   float32 sparse-attention indexer caches that pack multiple logical tokens
#   into one physical slot (MLAAttentionSpec.compress_ratio > 1). LMCache's
#   hybrid-group registration + slot-compression store/retrieve path
#   (docs/design/integration/vllm/hybrid-kv-cache-groups.md) serves these
#   groups per-block-size; this test pins that path end-to-end against the
#   real model with the L1 (CPU pool) tier enabled. L1 is the only storage
#   tier configured here, so a served retrieve proves the L1 path.
#
# Hardware requirement:
#   Hopper (SM90), datacenter Blackwell (SM100), or workstation Blackwell
#   (SM120, e.g. RTX PRO 6000). DeepSeek-V4-Flash routes its fp8 block-scaled
#   linears, MoE experts, hyper-connection prenorm GEMM and attention o_proj
#   einsum through DeepGEMM, so the arch must have DeepGEMM kernels.
#
#   Use the DeepGEMM bundled with vLLM so its kernels and bindings match the
#   selected vLLM/PyTorch stack. The startup probe and completion requests
#   below exercise the model's kernels rather than just checking imports.
#
# This test is self-contained: it launches its own LMCache server + a TP=N
# vLLM instead of using launch-processes.sh / wait-for-servers.sh, since it
# needs tensor parallelism and the model's dedicated launch flags
# (docs/source/recipes/deepseek_v4_flash.rst). PIDs are written to the shared
# PID_FILE so the dispatcher's cleanup.sh trap still tears everything down.
#
# Flow (TP=4 by default):
#   1. Launch the LMCache server with an explicit L1 pool + vLLM (dev mode,
#      so /reset_prefix_cache is available).
#   2. vLLM run: send one long deterministic (greedy) completion request; vLLM
#      computes it from scratch and populates LMCache through the
#      slot-compression store path. Capture output A.
#   3. Reset vLLM's *local* prefix cache (APC) via POST /reset_prefix_cache,
#      leaving LMCache intact. Unlike the Kimi-Linear test, no vLLM restart is
#      needed: DeepSeek-V4-Flash keeps no Mamba state, all its cached state is
#      paged KV that the APC reset evicts. This saves a second multi-minute
#      160GB weight load.
#   4. LMCache retrieve run: send the identical request; vLLM's APC misses, so
#      the prefix KV -- MLA latents and indexer caches -- must be served by
#      LMCache. Capture output B.
#   5. Assert A == B. A broken slot-compression restore corrupts the restored
#      latents/indexer state and diverges the greedy decode. Both runs happen
#      in one vLLM process (same tuned kernels), and the retrieved KV is
#      byte-reloaded rather than recomputed, so a correct run is deterministic
#      and the outputs match exactly.
#   6. Assert LMCache actually served retrieves in run 2 (non-vacuous).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
source "${SCRIPT_DIR}/../helpers.sh"
source "${SCRIPT_DIR}/../verification/dsv4-flash-tp-verification.sh"

# ── Configuration ───────────────────────────────────────────
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
ENGINE_PORT="$VLLM_PORT"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"

# LMCache chunk size must be a multiple of every KV cache group's
# tokens_per_block; DeepSeek-V4-Flash's groups use 256/64/8/4, so 256 is the
# smallest valid chunk size.
CHUNK_SIZE="${CHUNK_SIZE:-256}"
# One long prompt's KV is small (compressed MLA latents), but the pool is
# lazily allocated, so a generous default costs nothing up front.
L1_SIZE_GB="${L1_SIZE_GB:-40}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# Readiness timeout for the vLLM launch. Owned by the test (a 160GB fp8
# TP-shard load is slow, and the first CI run also downloads the weights);
# deliberately does NOT reuse MAX_WAIT_SECONDS, which run-single-test.sh
# pre-exports to 600s -- that would shadow the value here.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-2700}"
# DeepSeek-V4-Flash has multiple KV cache groups with different block
# geometries. Keep per-group registration explicit instead of depending on the
# LMCache server default, which flipped off in #3869/#4437.
SEPARATE_OBJECT_GROUPS="${SEPARATE_OBJECT_GROUPS:-1}"
SEPARATE_OBJECT_GROUPS_ARG=""
if [ "$SEPARATE_OBJECT_GROUPS" = "1" ] || [ "$SEPARATE_OBJECT_GROUPS" = "true" ]; then
    SEPARATE_OBJECT_GROUPS_ARG="--separate-object-groups"
fi

# Tokens to generate per request. Greedy (temperature 0); a divergence in the
# restored KV shows up within the first few tokens, but a longer generation
# makes an accidental match astronomically unlikely.
MAX_TOKENS="${MAX_TOKENS:-128}"
# Seconds to let async LMCache stores drain before the retrieve run.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TP_DIR="$RESULTS_DIR/dsv4_flash_tp"
mkdir -p "$TP_DIR"
PROMPT_FILE="$TP_DIR/prompt.txt"
OUT_A="$TP_DIR/output_vllm_run.txt"
OUT_B="$TP_DIR/output_retrieve_run.txt"

echo "=== DeepSeek-V4-Flash L1 slot-compression correctness test ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | vLLM port: $VLLM_PORT | TP=$TENSOR_PARALLEL_SIZE"
echo "Chunk size: $CHUNK_SIZE | L1 pool: ${L1_SIZE_GB}GB"
echo "Results dir: $TP_DIR"
echo ""

# ── 0a. Probe the ambient vLLM, downgrade only if it is broken ──
# The 0.28.1rc1 nightlies break DeepSeek-V4-Flash sparse decode on SM120
# ("eidx must be contiguous", flashinfer _sparse_mla_sm120) during engine
# start. That is a pure-vLLM fault -- it reproduces with no connector
# configured -- so probe for it instead of hardcoding "nightlies are bad":
# boot vLLM alone with dummy weights and see whether the engine comes up.
# Healthy keeps the ambient (nightly) vLLM, so this test returns to nightly
# coverage on its own once upstream fixes the kernel, with no code change.
# Broken installs the last release that passes. Same torch (2.13.0+cu130)
# either way, so the LMCache native build stays valid.
#
# Dummy weights keep the probe cheap: the crash is a stride/contiguity check
# in the warmup decode, which runs regardless of weight *values*. Every
# ambiguous outcome (timeout, launch failure) is treated as broken, since
# falling back to the pinned version still produces a valid test run.
DSV4_FALLBACK_VLLM_VERSION="${DSV4_FALLBACK_VLLM_VERSION-0.27.1}"
DSV4_PROBE_PORT="${DSV4_PROBE_PORT:-8971}"
DSV4_PROBE_TIMEOUT="${DSV4_PROBE_TIMEOUT:-900}"

# Wait for every process in $1's process group to disappear, so the probe
# cannot leave workers holding GPU memory when the real run starts.
wait_for_pgid_exit() {
    local pgid="$1" deadline=$((SECONDS + 120))
    [ -n "$pgid" ] || return 0
    kill -TERM -"$pgid" 2>/dev/null || true
    while kill -0 -"$pgid" 2>/dev/null; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            kill -KILL -"$pgid" 2>/dev/null || true
            break
        fi
        sleep 2
    done
    sleep 10  # let the driver release device memory
}

# Boot vLLM alone (no LMCache connector) on dummy weights.
# Returns 0 if the engine serves, 1 if it crashes / never comes up.
probe_ambient_vllm() {
    local probe_log="$TP_DIR/vllm_probe.log"
    echo "=== Probing ambient vLLM $(python3 -c 'import vllm; print(vllm.__version__)') ==="
    echo "Probe log: $probe_log"
    # Model-shaping flags mirror the real launch below (they select the
    # kernels); the connector, prefix caching and dev mode are omitted.
    setsid env -u VLLM_PORT vllm serve "$MODEL" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --enable-expert-parallel \
        --kv-cache-dtype fp8_ds_mla \
        --tokenizer-mode deepseek_v4 \
        --trust-remote-code \
        --enforce-eager \
        --load-format dummy \
        --max-model-len auto \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --port "$DSV4_PROBE_PORT" \
        > "$probe_log" 2>&1 &
    local probe_pid=$! probe_pgid rc=1 verdict="timed out after ${DSV4_PROBE_TIMEOUT}s"
    probe_pgid="$(ps -o pgid= -p "$probe_pid" | tr -d ' ')"
    echo "$probe_pid" >> "$PID_FILE"

    local deadline=$((SECONDS + DSV4_PROBE_TIMEOUT))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -sf "http://localhost:${DSV4_PROBE_PORT}/v1/models" >/dev/null 2>&1; then
            rc=0 verdict="serves"
            break
        fi
        if ! kill -0 "$probe_pid" 2>/dev/null; then
            verdict="died during startup"
            tail -n 25 "$probe_log"
            break
        fi
        sleep 5
    done
    echo "Probe verdict: ambient vLLM ${verdict}."

    wait_for_pgid_exit "$probe_pgid"
    return "$rc"
}

if [ -z "$DSV4_FALLBACK_VLLM_VERSION" ]; then
    echo "=== DSV4_FALLBACK_VLLM_VERSION empty: using ambient vLLM unprobed ==="
elif ! probe_ambient_vllm; then
    echo "=== Installing vLLM ${DSV4_FALLBACK_VLLM_VERSION} for this test ==="
    uv pip install "vllm[runai,tensorizer,flashinfer]==${DSV4_FALLBACK_VLLM_VERSION}" \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
    python3 -c "import vllm; print('vLLM for this test:', vllm.__version__)"
fi

# ── 1. Launch LMCache MP server with an explicit L1 pool ────
echo "=== Launching LMCache MP server (port $LMCACHE_PORT, L1 ${L1_SIZE_GB}GB) ==="
lmcache server \
    --transport "$LMCACHE_REQUEST_TRANSPORT" \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --max-workers 4 \
    ${SEPARATE_OBJECT_GROUPS_ARG} \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"
sleep 10

# ── 2. Build a long, deterministic prompt ───────────────────
# A ~7-8k word document (spanning many 256-token LMCache chunks, so several
# slot-compressed blocks per group are stored) built by repeating a fixed
# paragraph, so the input is identical on every run.
#
dsv4_build_prompt

# ── 3. Launch vLLM (dev mode for /reset_prefix_cache) ───────
echo "=== Launching vLLM ($MODEL, TP=$TENSOR_PARALLEL_SIZE, port $VLLM_PORT) ==="
echo "Log: $VLLM_LOG"
# Save and unset VLLM_PORT: vLLM's internal get_open_port() would otherwise
# collide with the serving port for torch.distributed.
saved_port="$VLLM_PORT"
unset VLLM_PORT

# Launch flags per docs/source/recipes/deepseek_v4_flash.rst:
# --kv-cache-dtype fp8_ds_mla and --tokenizer-mode deepseek_v4 are required
# for this model; --enable-expert-parallel distributes the MoE experts across
# the TP ranks.
# --enforce-eager: capturing FULL decode CUDA graphs crashes in vLLM's
# custom all-reduce kernel at TP=4 ("illegal memory access",
# custom_all_reduce.cuh) on current nightlies; a correctness test needs no
# CUDA graphs, and eager also skips several minutes of capture/compile.
VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-expert-parallel \
    --kv-cache-dtype fp8_ds_mla \
    --tokenizer-mode deepseek_v4 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --max-model-len auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$saved_port" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.host\": \"$LMCACHE_REQUEST_SCHEME://localhost\", \"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 120}}" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
export VLLM_PORT="$saved_port"
echo "vLLM started (PID=$VLLM_PID)"

if ! wait_for_server "$VLLM_PORT" "$VLLM_READY_TIMEOUT" "$VLLM_LOG"; then
    echo "vLLM failed to start."
    exit 1
fi
echo ""

# ── 4. Store, clear vLLM APC, retrieve, and verify ──────────
dsv4_run_l1_round_trip
