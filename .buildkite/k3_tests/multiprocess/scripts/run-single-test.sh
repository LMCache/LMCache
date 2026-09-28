#!/usr/bin/env bash
# Orchestrator for a single multiprocessing test (native, no Docker).
# Usage: run-single-test.sh <test_name>
#   test_name: lm_eval | lm_eval_preemption | hma_lm_eval_gemma4 | vllm_bench
#              | long_doc_qa | long_doc_qa_l2 | fault_tolerance | deadlock
#              | restart_recovery | lazy_offload | gds_smoke
#
# Each invocation is self-contained: launches servers, runs one test, cleans up.
# This mirrors the comprehensive tests' run-single-config.sh pattern.
set -o pipefail

TEST_NAME="${1:?Usage: $0 <test_name>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

# ── Configuration ────────────────────────────────────────────
export LMCACHE_PORT="${LMCACHE_PORT:-6555}"
export INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
ENGINE_ADAPTER="${SCRIPT_DIR}/engines/${INFERENCE_ENGINE}.sh"
if [[ ! -f "$ENGINE_ADAPTER" ]]; then
    echo "Unsupported inference engine '${INFERENCE_ENGINE}': $ENGINE_ADAPTER not found" >&2
    exit 1
fi
source "$ENGINE_ADAPTER"
source "${SCRIPT_DIR}/workload-discovery.sh"

# Keep direct run-single-test.sh callers consistent with run.sh. The latter
# performs this check before dependency installation; this second check keeps
# the lower-level entry point safe for AMD and local callers.
if resolve_engine_workload "$TEST_NAME" "$SCRIPT_DIR" "$INFERENCE_ENGINE"; then
    :
else
    status=$?
    if [[ "$status" -eq "$WORKLOAD_UNSUPPORTED_STATUS" ]]; then
        exit 0
    fi
    exit "$status"
fi
exec_script="$RESOLVED_WORKLOAD_SCRIPT"

if [[ "$TEST_NAME" == "lm_eval_preemption" ]]; then
    export LM_EVAL_VERIFY_MODE=preemption
fi
# Keep this aligned with wait-for-servers.sh. Large-model startup can exceed
# five minutes on cold or contended CI nodes before the service is unhealthy.
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-${ENGINE_DEFAULT_MODEL:-Qwen/Qwen3-14B}}"
export LMCACHE_REQUEST_TRANSPORT="${LMCACHE_REQUEST_TRANSPORT:-zmq}"
export LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"
engine_configure_defaults

case "${LMCACHE_REQUEST_TRANSPORT}" in
    zmq) export LMCACHE_REQUEST_SCHEME="tcp" ;;
    grpc) export LMCACHE_REQUEST_SCHEME="grpc" ;;
    *)
        echo "Unknown LMCACHE_REQUEST_TRANSPORT='${LMCACHE_REQUEST_TRANSPORT}'"
        echo "Valid values: zmq, grpc"
        exit 1
        ;;
esac

# gds_smoke enables the GDS L1 NVMe-slab tier
GDS_SCRATCH="${GDS_SCRATCH:-/scratch}"
if [ "$TEST_NAME" = "gds_smoke" ]; then
    export GDS_L1_PATH="${GDS_SCRATCH}/lmcache-gds-${BUILD_ID}-${TEST_NAME}"
    echo "GDS L1 tier enabled (slab dir: $GDS_L1_PATH)"
fi

# Per-test default model (overridable via the MODEL env var). The HMA test needs
# a hybrid model whose KV cache groups have different block sizes, so the
# connector exercises the per-group hybrid-memory-allocator path.
if [ "$TEST_NAME" = "hma_lm_eval_gemma4" ]; then
    # gemma-4-31B-it is public (no gating, so no HF token check) and has
    # heterogeneous head dims (head_dim 256 / global_head_dim 512), so vLLM
    # gives its KV cache groups different block sizes -- this is what exercises
    # LMCache's per-group block-size handling. It forces TRITON_ATTN, so the
    # pipeline sets ATTENTION_BACKEND=auto; its ~63GB of weights also need a
    # higher GPU_MEMORY_UTILIZATION than the default (all set in pipeline.yml).
    export MODEL="${MODEL:-google/gemma-4-31B-it}"
elif [ "$TEST_NAME" = "hma_lm_eval_qwen3_5" ]; then
    # Qwen3.5-0.8B is a Mamba/GDN + full-attention hybrid (caches re-viewed at
    # registration; see lmcache/integration/vllm/kv_cache_group_edits.py).
    export MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
    export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
    # LMCache chunk size must be a multiple of the unified vLLM block size (544).
    export CHUNK_SIZE="${CHUNK_SIZE:-544}"
    # GDN supports only the 'align' Mamba cache mode.
    export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
    # 'align' snapshots the Mamba state only at scheduler-step boundaries; cap
    # the step at the chunk size for one reusable snapshot per chunk.
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-544}"
    # GDN has no batch-invariant mode, so runs are not bit-exact; compare within
    # a score tolerance and use enough samples to shrink run-to-run drift
    # (~1/sqrt(LIMIT)) well inside it.
    export BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
    export SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.05}"
    export LIMIT="${LIMIT:-300}"
elif [ "$TEST_NAME" = "kimi_linear_tp" ]; then
    # Self-contained test: kimi-linear-tp.sh owns the server lifecycle and
    # all launch flags (TP=2, trust-remote-code, align, chunk/batch sizes). Only
    # the model name is declared here so the banner and the script's ${MODEL:-}
    # fallback both resolve to Kimi-Linear rather than the generic default below.
    export MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
elif [ "$TEST_NAME" = "dsv4_flash_tp" ]; then
    # Self-contained test: dsv4-flash-tp.sh owns the server lifecycle and
    # all launch flags (TP=4, fp8_ds_mla, deepseek_v4 tokenizer). Only the
    # model name is declared here so the banner and the script's ${MODEL:-}
    # fallback both resolve to DeepSeek-V4-Flash.
    export MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
elif [ "$TEST_NAME" = "deadlock" ]; then
    # The workload is engine-neutral, while each adapter owns the launch flags
    # needed to reproduce the high-concurrency TP=2 scenario.
    export MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
    export ENGINE_LAUNCH_PROFILE=deadlock
    export ENGINE_USE_ALL_DEVICES=true
    export LMCACHE_HOST=localhost
    export CHUNK_SIZE=256
    export CPU_BUFFER_SIZE=50
    export MAX_WORKERS=2
elif [ "$TEST_NAME" = "lazy_offload" ]; then
    # The shared GPU launcher includes these values in the real vLLM
    # kv-transfer configuration only for this integration test.
    export LMCACHE_MP_LAZY_OFFLOAD=true
    # vLLM's default paged-block size is 16 tokens. Matching it keeps this
    # test's expected LMCache chunk counts exact and small.
    export CHUNK_SIZE="${CHUNK_SIZE:-16}"
    export MODEL="${MODEL:-$DEFAULT_MODEL}"
else
    export MODEL="${MODEL:-$DEFAULT_MODEL}"
fi
export CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export LMCACHE_DIR="$REPO_ROOT"
export RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

if declare -F engine_configure_workload > /dev/null; then
    engine_configure_workload "$TEST_NAME"
fi

mkdir -p "$RESULTS_DIR"

# Cleanup: always kill background processes on exit
trap '"${SCRIPT_DIR}/cleanup.sh"' EXIT

echo "============================================"
echo "=== LMCache Multiprocessing Test: ${TEST_NAME} ==="
echo "============================================"
echo "Build ID: $BUILD_ID"
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT"
echo "Request transport: $LMCACHE_REQUEST_TRANSPORT"
echo "Inference engine: $ENGINE_NAME"
echo "Engine port: $ENGINE_PORT"
echo "Engine baseline port: $ENGINE_BASELINE_PORT"
echo "Results dir: $RESULTS_DIR"
echo ""

# Tests that still handle their own server lifecycle.
SELF_CONTAINED_TESTS=" mp_autostart_tp2 p2p kimi_linear_tp dsv4_flash_tp "

# Tests that compare against a baseline engine (no LMCache) on a second GPU.
# Only these need the baseline server (and thus a 2-GPU pod); everything
# else runs on GPU 0 alone, so launch-processes.sh skips the baseline.
# Respect an explicit environment override from a device-specific wrapper
# before applying the default baseline heuristic below.
BASELINE_TESTS=" vllm_bench long_doc_qa long_doc_qa_l2 preemption_correctness "
if [ -n "${LAUNCH_BASELINE:-}" ]; then
    export LAUNCH_BASELINE
elif [[ "$BASELINE_TESTS" == *" $TEST_NAME "* ]]; then
    export LAUNCH_BASELINE=true
else
    export LAUNCH_BASELINE=false
fi

if [[ "$SELF_CONTAINED_TESTS" != *" $TEST_NAME "* ]]; then
    # ── Step 1: Launch native processes ──────────────────────────
    echo "============================================"
    echo "=== Launching native processes ==="
    echo "============================================"
    if ! "${SCRIPT_DIR}/launch-processes.sh"; then
        echo "Failed to launch processes"
        exit 1
    fi
    echo ""

    # ── Step 2: Wait for the inference engine to be ready ───────
    echo "============================================"
    echo "=== Waiting for ${ENGINE_NAME} to be ready ==="
    echo "============================================"
    if ! "${SCRIPT_DIR}/wait-for-servers.sh"; then
        echo "${ENGINE_NAME} failed to become ready"
        exit 1
    fi
    echo ""
fi

# ── Step 3: Run the requested test ──────────────────────────
echo "============================================"
echo "=== Running test: ${TEST_NAME} ==="
echo "============================================"

if ! "$exec_script"; then
    echo "${TEST_NAME} test failed"
    exit 1
fi

echo ""
echo "============================================"
echo "=== Test ${TEST_NAME} passed! ==="
echo "============================================"
