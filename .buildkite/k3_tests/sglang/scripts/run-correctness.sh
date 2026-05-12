#!/usr/bin/env bash
# Correctness: LMCache does not change the output, AND is actually exercised.
#
# Pattern: A → B → A, with two distinct ~2500-token prompts.
#   - Step 1 (A) populates LMCache and SGLang radix for A.
#   - Step 2 (B) populates LMCache for B; B's allocation evicts A from the
#     SGLang radix because their combined KV (~5000 tokens) exceeds
#     --max-total-tokens 4096.
#   - Step 3 (A again) is a radix miss → LMCache hit; the daemon's RETRIEVE
#     handler fires and logs "Retrieved N tokens".
# We compare the step-3 output against a reference produced by an
# LMCache-disabled SGLang server. With temperature=0 greedy decode the
# outputs must match byte-for-byte.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DAEMON_LOG="correctness-daemon.log"
LMC_SGL_LOG="correctness-sgl-lmc.log"
NO_SGL_LOG="correctness-sgl-no.log"
PORT_LMC=30200
PORT_NO=30201

LMC_PID=""
NO_PID=""
DAEMON_PID_LOCAL=""

cleanup_all() {
    for pid in "${LMC_PID}" "${NO_PID}" "${DAEMON_PID_LOCAL}"; do
        if [[ -n "${pid}" ]]; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done
    pkill -9 -f "sglang::scheduler" 2>/dev/null || true
    sleep 2
}
trap cleanup_all EXIT

# ── Launch daemon + both SGLang servers ─────────────────────
launch_daemon "${DAEMON_LOG}"
DAEMON_PID_LOCAL="${DAEMON_PID}"

launch_sglang "${PORT_LMC}" "${LMC_SGL_LOG}" "lmcache"
LMC_PID="${SGLANG_PID}"

SGLANG_PID=""  # so the next launch_sglang doesn't clobber LMC_PID
launch_sglang "${PORT_NO}" "${NO_SGL_LOG}" "no-lmcache"
NO_PID="${SGLANG_PID}"

# ── Prepare prompts ─────────────────────────────────────────
PROMPT_A="$(generate_prompt a)"
PROMPT_B="$(generate_prompt b)"
echo "Prompt A size: $(printf '%s' "${PROMPT_A}" | wc -c) chars"
echo "Prompt B size: $(printf '%s' "${PROMPT_B}" | wc -c) chars"

# ── A → B → A on the LMCache-enabled server ────────────────
echo "--- :one: Populate LMCache with A (LMCache server)"
chat_completion "${PORT_LMC}" "${PROMPT_A}" 64 > /tmp/lmc_run1_A.txt

echo "--- :two: Send B to evict A from SGLang radix"
chat_completion "${PORT_LMC}" "${PROMPT_B}" 64 > /tmp/lmc_run2_B.txt

# Snapshot the daemon's retrieve count before the cache-hit call.
RETRIEVALS_BEFORE=$(count_retrievals "${DAEMON_LOG}")
echo "  daemon retrievals before cache-hit call: ${RETRIEVALS_BEFORE}"

echo "--- :three: Send A again — must miss radix, hit LMCache"
chat_completion "${PORT_LMC}" "${PROMPT_A}" 64 > /tmp/lmc_out.txt

RETRIEVALS_AFTER=$(count_retrievals "${DAEMON_LOG}")
echo "  daemon retrievals after cache-hit call: ${RETRIEVALS_AFTER}"

# ── Reference call against the no-LMCache server ──────────
echo "--- :mag: Reference call (no LMCache)"
chat_completion "${PORT_NO}" "${PROMPT_A}" 64 > /tmp/no_out.txt

# ── Assertions ────────────────────────────────────────────
RETRIEVAL_DELTA=$((RETRIEVALS_AFTER - RETRIEVALS_BEFORE))
echo "  retrieval delta = ${RETRIEVAL_DELTA}"
if [[ "${RETRIEVAL_DELTA}" -lt 1 ]]; then
    echo "FAIL: LMCache was not exercised (retrieval delta ${RETRIEVAL_DELTA} < 1)" >&2
    echo "--- daemon log tail ---" >&2
    tail -n 60 "${DAEMON_LOG}" >&2
    exit 1
fi

if ! diff -u /tmp/no_out.txt /tmp/lmc_out.txt; then
    echo "FAIL: LMCache changed the output" >&2
    exit 1
fi

echo "PASS: outputs match AND LMCache served ${RETRIEVAL_DELTA} retrieval(s) on the third call."
