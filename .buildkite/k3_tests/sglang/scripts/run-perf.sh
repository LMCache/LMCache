#!/usr/bin/env bash
# Performance: LMCache makes TTFT smaller, AND is actually exercised.
#
# Symmetric A → B → A pattern on each phase so the "timed A" is always a
# radix miss in both cases. Without LMCache the radix miss forces a full
# prefill; with LMCache the radix miss is served by an LMCache RETRIEVE.
#
# Phase A — with LMCache:
#   1. Launch daemon + SGLang with --enable-lmcache.
#   2. Send prompt A, then prompt B (evicts A from radix).
#   3. Time prompt A again (max_tokens=1 ≈ TTFT). Verify RETRIEVE fired.
# Phase B — without LMCache:
#   4. Restart SGLang without --enable-lmcache.
#   5. Same A → B → A pattern. Time prompt A again. Plain prefill.
# Pass: latency_with_lmcache < latency_without_lmcache (strict).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

trap cleanup_all EXIT

DAEMON_LOG="perf-daemon.log"
LMC_LOG="perf-sgl-lmc.log"
NO_LOG="perf-sgl-no.log"
PORT_LMC=30200
PORT_NO=30201

# ── Prepare prompts ─────────────────────────────────────────
PROMPT_A="$(generate_prompt a)"
PROMPT_B="$(generate_prompt b)"
echo "Prompt A size: $(printf '%s' "${PROMPT_A}" | wc -c) chars"
echo "Prompt B size: $(printf '%s' "${PROMPT_B}" | wc -c) chars"

# ── Phase A — with LMCache ──────────────────────────────────
launch_daemon "${DAEMON_LOG}"
launch_sglang "${PORT_LMC}" "${LMC_LOG}" "lmcache"

echo "--- :fire: A → B (LMCache: populates, evicts A from radix)"
chat_completion "${PORT_LMC}" "${PROMPT_A}" 1 > /dev/null
chat_completion "${PORT_LMC}" "${PROMPT_B}" 1 > /dev/null

RETRIEVALS_BEFORE=$(count_retrievals "${DAEMON_LOG}")
echo "  daemon retrievals before measured call: ${RETRIEVALS_BEFORE}"

echo "--- :stopwatch: Measured call: A again (LMCache hit expected)"
LAT_LMC=$(measure_latency_seconds "${PORT_LMC}" "${PROMPT_A}")
echo "  latency_with_lmcache = ${LAT_LMC}s"

RETRIEVALS_AFTER=$(count_retrievals "${DAEMON_LOG}")
RETRIEVAL_DELTA=$((RETRIEVALS_AFTER - RETRIEVALS_BEFORE))
echo "  retrieval delta = ${RETRIEVAL_DELTA}"

echo "--- :stop_sign: Stopping LMCache-enabled SGLang"
kill -9 "${SGLANG_PID}" 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
wait "${SGLANG_PID}" 2>/dev/null || true
SGLANG_PID=""
sleep 3

# ── Phase B — without LMCache ───────────────────────────────
launch_sglang "${PORT_NO}" "${NO_LOG}" "no-lmcache"

echo "--- :fire: A → B (no LMCache: warms radix, then evicts A)"
chat_completion "${PORT_NO}" "${PROMPT_A}" 1 > /dev/null
chat_completion "${PORT_NO}" "${PROMPT_B}" 1 > /dev/null

echo "--- :stopwatch: Measured call: A again (full prefill expected)"
LAT_NO=$(measure_latency_seconds "${PORT_NO}" "${PROMPT_A}")
echo "  latency_without_lmcache = ${LAT_NO}s"

# ── Assertions ──────────────────────────────────────────────
echo ""
echo "  latency_with_lmcache    = ${LAT_LMC}s"
echo "  latency_without_lmcache = ${LAT_NO}s"
echo "  retrieval delta         = ${RETRIEVAL_DELTA}"

if [[ "${RETRIEVAL_DELTA}" -lt 1 ]]; then
    echo "FAIL: LMCache was not exercised (retrieval delta ${RETRIEVAL_DELTA} < 1)" >&2
    echo "--- daemon log tail ---" >&2
    tail -n 60 "${DAEMON_LOG}" >&2
    exit 1
fi

if ! python3 -c "import sys; sys.exit(0 if ${LAT_LMC} < ${LAT_NO} else 1)"; then
    echo "FAIL: LMCache did not improve TTFT (${LAT_LMC}s >= ${LAT_NO}s)" >&2
    exit 1
fi

echo "PASS: LMCache improved TTFT (${LAT_LMC}s < ${LAT_NO}s) with ${RETRIEVAL_DELTA} retrieval(s)."
