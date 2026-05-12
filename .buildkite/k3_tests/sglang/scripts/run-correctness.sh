#!/usr/bin/env bash
# Correctness: LMCache must not change the output, and must be exercised.
# A → B → A across the LMCache-enabled server: A populates LMCache, B
# evicts A from SGLang's radix (their KV sum exceeds --max-total-tokens),
# the second A is a radix miss / LMCache hit. The cache-hit output is
# diffed against a no-LMCache reference run.
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
        [[ -n "${pid}" ]] && kill -9 "${pid}" 2>/dev/null || true
    done
    pkill -9 -f "sglang::scheduler" 2>/dev/null || true
    sleep 2
}
trap cleanup_all EXIT

echo "--- :rocket: Start daemon + 2 SGLang servers (with/without LMCache)"
launch_daemon "${DAEMON_LOG}"
DAEMON_PID_LOCAL="${DAEMON_PID}"

launch_sglang "${PORT_LMC}" "${LMC_SGL_LOG}" "lmcache"
LMC_PID="${SGLANG_PID}"

SGLANG_PID=""  # don't clobber LMC_PID on the next launch
launch_sglang "${PORT_NO}" "${NO_SGL_LOG}" "no-lmcache"
NO_PID="${SGLANG_PID}"

PROMPT_A="$(generate_prompt a)"
PROMPT_B="$(generate_prompt b)"

echo "--- :test_tube: Drive A → B → A on the LMCache server"
chat_completion "${PORT_LMC}" "${PROMPT_A}" 64 > /tmp/lmc_run1_A.txt
chat_completion "${PORT_LMC}" "${PROMPT_B}" 64 > /tmp/lmc_run2_B.txt
RETRIEVALS_BEFORE=$(count_retrievals "${DAEMON_LOG}")
chat_completion "${PORT_LMC}" "${PROMPT_A}" 64 > /tmp/lmc_out.txt
RETRIEVALS_AFTER=$(count_retrievals "${DAEMON_LOG}")
RETRIEVAL_DELTA=$((RETRIEVALS_AFTER - RETRIEVALS_BEFORE))
echo "  daemon retrieval delta on the cache-hit call: ${RETRIEVAL_DELTA}"

echo "--- :mag: Reference call on the no-LMCache server"
chat_completion "${PORT_NO}" "${PROMPT_A}" 64 > /tmp/no_out.txt

echo "+++ :scales: Verdict"
if [[ "${RETRIEVAL_DELTA}" -lt 1 ]]; then
    echo "FAIL: LMCache was not exercised (retrieval delta ${RETRIEVAL_DELTA} < 1)" >&2
    tail -n 60 "${DAEMON_LOG}" >&2
    exit 1
fi
if ! diff -u /tmp/no_out.txt /tmp/lmc_out.txt; then
    echo "FAIL: LMCache changed the output" >&2
    exit 1
fi
echo "PASS — outputs match AND LMCache served ${RETRIEVAL_DELTA} retrieval(s) on the cache-hit call."
