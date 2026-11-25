#!/usr/bin/env bash
# An LMCache server that sees no GPU and a vLLM worker on GPU 0, both left on
# their default transfer mode (run-single-test.sh hides the devices from the
# server and sets no mode on either side).
#
# A server without an accelerator cannot open the worker's CUDA IPC handles,
# so it defaults to --supported-transfer-mode auto, and the worker's
# mp_transfer_mode=auto must resolve to engine_driven through get_server_info.
# The KV must still round-trip: a cold request stores it, the vLLM prefix
# cache is reset, and the same request must be served by LMCache with
# identical output.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../helpers.sh"
source "${SCRIPT_DIR}/../verification/l1-round-trip-verification.sh"

LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
MAX_TOKENS="${MAX_TOKENS:-64}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"
TEST_DIR="$RESULTS_DIR/cpu_server_auto"
mkdir -p "$TEST_DIR"
PROMPT_FILE="$TEST_DIR/prompt.txt"
OUT_A="$TEST_DIR/output_cold_run.txt"
OUT_B="$TEST_DIR/output_retrieve_run.txt"

echo "=== LMCache server without a GPU, vLLM on mp_transfer_mode=auto ==="
echo "Model: $MODEL"
echo "vLLM log: $ENGINE_LOG_FILE"
echo ""

if ! grep -q "LMCache server runs on cpu and loads .*engine_driven" \
        "$ENGINE_LOG_FILE"; then
    echo "FAILED: vLLM did not see a CPU-only server that loads engine_driven."
    grep -E "LMCache server|get_server_info|Creating transfer context" \
        "$ENGINE_LOG_FILE" | tail -n 5 || true
    exit 1
fi
if ! grep -q "Creating transfer context (device_type=cuda, mode=engine_driven)" \
        "$ENGINE_LOG_FILE"; then
    echo "FAILED: vLLM did not build an engine-driven transfer context."
    grep "Creating transfer context" "$ENGINE_LOG_FILE" | tail -n 5 || true
    exit 1
fi
echo "PASS: mp_transfer_mode=auto resolved to engine_driven."
echo ""

l1_round_trip_build_prompt "$PROMPT_FILE" \
    " Summarize the text above in one sentence."
l1_round_trip_run cpu_server_auto
