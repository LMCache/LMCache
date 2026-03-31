#!/usr/bin/env bash
# Orchestrator for a single multiprocessing test (native, no Docker).
# Usage: run-single-test.sh <test_name>
#   test_name: lm_eval | vllm_bench | long_doc_qa | long_doc_qa_l2 | fault_tolerance
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
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
export MODEL="${MODEL:-Qwen/Qwen3-14B}"
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

# ── Step 3: Run the requested test ──────────────────────────
echo "============================================"
echo "=== Running test: ${TEST_NAME} ==="
echo "============================================"

case "$TEST_NAME" in
    lm_eval)
        exec_script="${SCRIPT_DIR}/run-lm-eval.sh"
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
    *)
        echo "Unknown test: $TEST_NAME"
        echo "Valid tests: lm_eval, vllm_bench, long_doc_qa, long_doc_qa_l2, fault_tolerance"
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
