#!/usr/bin/env bash
# Performance: LMCache makes the bench's query-round mean TTFT smaller,
# AND is actually exercised.
#
# Workload: 4 distinct ~1500-token docs × 2 tiles via the existing
# long_doc_qa.py bench. Combined KV (~6000 tokens) exceeds the
# --max-total-tokens 4096 radix pool, so during the warmup tile SGLang
# evicts older entries; the query tile re-queries them — radix misses
# become LMCache hits (with --enable-lmcache) or full prefills (without).
#
# Phase A — with LMCache: launch + bench + record query TTFT + verify
#   the daemon log shows RETRIEVE traffic during the bench.
# Phase B — without LMCache: restart sglang without --enable-lmcache,
#   bench again, record query TTFT.
# Pass: ttft_with_lmcache < ttft_without_lmcache (strict).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

trap cleanup_all EXIT

DAEMON_LOG="perf-daemon.log"
LMC_LOG="perf-sgl-lmc.log"
NO_LOG="perf-sgl-no.log"
PORT_LMC=30200
PORT_NO=30201

# ── Phase A — with LMCache ──────────────────────────────────
launch_daemon "${DAEMON_LOG}"
launch_sglang "${PORT_LMC}" "${LMC_LOG}" "lmcache"

RETRIEVALS_BEFORE=$(count_retrievals "${DAEMON_LOG}")
echo "--- :stopwatch: Bench against LMCache-enabled SGLang"
TTFT_LMC=$(run_long_doc_qa_query_ttft "${PORT_LMC}")
echo "  query_round_mean_ttft_with_lmcache = ${TTFT_LMC}s"

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

echo "--- :stopwatch: Bench against LMCache-disabled SGLang"
TTFT_NO=$(run_long_doc_qa_query_ttft "${PORT_NO}")
echo "  query_round_mean_ttft_without_lmcache = ${TTFT_NO}s"

# ── Assertions ──────────────────────────────────────────────
echo ""
echo "  query_round_mean_ttft_with_lmcache    = ${TTFT_LMC}s"
echo "  query_round_mean_ttft_without_lmcache = ${TTFT_NO}s"
echo "  retrieval delta                       = ${RETRIEVAL_DELTA}"

if [[ "${RETRIEVAL_DELTA}" -lt 1 ]]; then
    echo "FAIL: LMCache was not exercised (retrieval delta ${RETRIEVAL_DELTA} < 1)" >&2
    echo "--- daemon log tail ---" >&2
    tail -n 60 "${DAEMON_LOG}" >&2
    exit 1
fi

if ! python3 -c "import sys; sys.exit(0 if ${TTFT_LMC} < ${TTFT_NO} else 1)"; then
    echo "FAIL: LMCache did not improve query-round TTFT (${TTFT_LMC}s >= ${TTFT_NO}s)" >&2
    exit 1
fi

echo "PASS: LMCache improved query-round TTFT (${TTFT_LMC}s < ${TTFT_NO}s) with ${RETRIEVAL_DELTA} retrieval(s)."
