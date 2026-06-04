#!/usr/bin/env bash
# Orchestrator for a single multiprocessing test (native, no Docker).
# Usage: run-single-test.sh <test_name>
#   test_name: lm_eval | hma_lm_eval | vllm_bench | long_doc_qa | long_doc_qa_l2
#              | fault_tolerance | deadlock | restart_recovery
#
# Each invocation is self-contained: launches servers, runs one test, cleans up.
# This mirrors the comprehensive tests' run-single-config.sh pattern.
set -o pipefail

TEST_NAME="${1:?Usage: $0 <test_name>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

# Preflight for gated HuggingFace models (e.g. gemma-3 used by hma_lm_eval).
# Fails fast with an actionable message when HF_TOKEN is missing or cannot
# access the model, instead of letting vLLM fail later with a confusing
# model-download / startup timeout.
# Arguments:
#   $1 model - HuggingFace model id to check access for.
# Exits:
#   1 if HF_TOKEN is unset/empty, or the HuggingFace API denies access (401/403).
check_hf_token_access() {
    local model="$1"
    local token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    if [ -z "$token" ]; then
        echo "ERROR: '$model' is a gated model but HF_TOKEN is not set."
        echo "       Provide HF_TOKEN from an account that has accepted the license at"
        echo "       https://huggingface.co/${model}"
        exit 1
    fi
    echo "Checking HuggingFace access to gated model '$model' (token: ${#token} chars)..."
    # Follow redirects (-L): the resolve endpoint 302-redirects to the CDN, and
    # we want the final status. --max-time bounds transient network hangs.
    local code
    code=$(curl -s -L --max-time 15 -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${token}" \
        "https://huggingface.co/${model}/resolve/main/config.json")
    case "$code" in
        200)
            echo "HF token OK: '$model' is accessible."
            ;;
        401 | 403)
            echo "ERROR: HF_TOKEN cannot access '$model' (HTTP $code)."
            echo "       The token is invalid, or the account has not accepted the license at"
            echo "       https://huggingface.co/${model}"
            exit 1
            ;;
        *)
            echo "WARNING: could not verify '$model' access (HTTP $code); continuing."
            ;;
    esac
}

# ── Configuration ────────────────────────────────────────────
export LMCACHE_PORT="${LMCACHE_PORT:-6555}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
# Per-test default model (overridable via the MODEL env var). The HMA test needs
# a hybrid (sliding-window + full-attention) model so vLLM exposes multiple KV
# cache groups and the connector exercises the hybrid-memory-allocator path.
# gemma-3 interleaves local (sliding-window) and global (full) attention layers,
# so it is hybrid, and -- unlike gpt-oss's MXFP4 MoE -- it runs under vLLM's
# batch-invariant mode. That lets the hma test assert the gsm8k score is
# *identical* with and without LMCache (see run-hma-lm-eval.sh) rather than only
# within a tolerance. gemma-3 is gated, so CI must provide HF_TOKEN. The
# batch-invariant / FLASH_ATTN / non-Marlin settings come from
# launch-processes.sh defaults, which are exactly what gemma-3 needs.
if [ "$TEST_NAME" = "hma_lm_eval" ]; then
    export MODEL="${MODEL:-google/gemma-3-4b-it}"
    check_hf_token_access "$MODEL"
else
    export MODEL="${MODEL:-Qwen/Qwen3-14B}"
fi
export CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export LMCACHE_DIR="$REPO_ROOT"
export RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

mkdir -p "$RESULTS_DIR"

# Cleanup: always kill background processes on exit
trap '"${SCRIPT_DIR}/cleanup.sh"' EXIT

echo "============================================"
echo "=== LMCache Multiprocessing Test: ${TEST_NAME} ==="
echo "============================================"
echo "Build ID: $BUILD_ID"
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT"
echo "vLLM port: $VLLM_PORT"
echo "vLLM baseline port: $VLLM_BASELINE_PORT"
echo "Results dir: $RESULTS_DIR"
echo ""

# Tests that handle their own server lifecycle (different GPU/model config)
SELF_CONTAINED_TESTS=" deadlock "

# Tests that compare against a baseline vLLM (no LMCache) on a second GPU.
# Only these need the baseline server (and thus a 2-GPU pod); everything
# else runs on GPU 0 alone, so launch-processes.sh skips the baseline.
BASELINE_TESTS=" vllm_bench long_doc_qa long_doc_qa_l2 hma_lm_eval "
if [[ "$BASELINE_TESTS" == *" $TEST_NAME "* ]]; then
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

    # ── Step 2: Wait for vLLM to be ready ───────────────────────
    echo "============================================"
    echo "=== Waiting for vLLM to be ready ==="
    echo "============================================"
    if ! "${SCRIPT_DIR}/wait-for-servers.sh"; then
        echo "vLLM failed to become ready"
        exit 1
    fi
    echo ""
fi

# ── Step 3: Run the requested test ──────────────────────────
echo "============================================"
echo "=== Running test: ${TEST_NAME} ==="
echo "============================================"

case "$TEST_NAME" in
    lm_eval)
        exec_script="${SCRIPT_DIR}/run-lm-eval.sh"
        ;;
    hma_lm_eval)
        exec_script="${SCRIPT_DIR}/run-hma-lm-eval.sh"
        ;;
    vllm_bench)
        exec_script="${SCRIPT_DIR}/run-vllm-bench.sh"
        ;;
    long_doc_qa)
        exec_script="${SCRIPT_DIR}/run-long-doc-qa.sh"
        ;;
    long_doc_qa_l2)
        exec_script="${SCRIPT_DIR}/run-long-doc-qa-l2.sh"
        ;;
    fault_tolerance)
        exec_script="${SCRIPT_DIR}/run-fault-tolerance.sh"
        ;;
    deadlock)
        exec_script="${SCRIPT_DIR}/run-deadlock.sh"
        ;;
    restart_recovery)
        exec_script="${SCRIPT_DIR}/run-restart-recovery.sh"
        ;;
    cache_stats)
        exec_script="${SCRIPT_DIR}/run-cache-stats.sh"
        ;;
    http_api)
        exec_script="${SCRIPT_DIR}/run-http-api.sh"
        ;;
    *)
        echo "Unknown test: $TEST_NAME"
        echo "Valid tests: lm_eval, hma_lm_eval, vllm_bench, long_doc_qa, long_doc_qa_l2, fault_tolerance, deadlock, restart_recovery, cache_stats, http_api"
        exit 1
        ;;
esac

if ! "$exec_script"; then
    echo "${TEST_NAME} test failed"
    exit 1
fi

echo ""
echo "============================================"
echo "=== Test ${TEST_NAME} passed! ==="
echo "============================================"
