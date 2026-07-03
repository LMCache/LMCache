#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Tier 0 -- LMCache storage smoke test on CPU. No model, no GPU, no vLLM.
#
# Starts an `lmcache server` with a 1 GiB CPU (L1) cache and runs
# `lmcache bench server --mode cpu`, which stores KV chunks, retrieves them,
# and byte-compares the result. Seeing "CHECKSUM MATCH OK" proves LMCache's
# store -> retrieve path works end-to-end on plain host memory.
#
# This is the fastest possible confirmation that the CPU install works and
# that LMCache can cache and return data without any accelerator.
#
# Environment (all optional, defaults shown):
#   LMCACHE_ZMQ_PORT      RPC port                (default: 15555)
#   LMCACHE_HTTP_PORT     HTTP/metrics port       (default: 18080)
#   BENCH_NUM_REQUESTS    requests to run         (default: 3)
#   BENCH_NUM_TOKENS      tokens per request      (default: 512)
#   TRANSFER_MODE         engine_driven|lmcache_driven (default: engine_driven)
#   HEALTHCHECK_TIMEOUT   seconds to wait for server (default: 60)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

ZMQ_PORT="${LMCACHE_ZMQ_PORT:-15555}"
HTTP_PORT="${LMCACHE_HTTP_PORT:-18080}"
BENCH_NUM_REQUESTS="${BENCH_NUM_REQUESTS:-3}"
BENCH_NUM_TOKENS="${BENCH_NUM_TOKENS:-512}"
TRANSFER_MODE="${TRANSFER_MODE:-engine_driven}"
HEALTHCHECK_TIMEOUT="${HEALTHCHECK_TIMEOUT:-60}"

LOG_FILE="$(mktemp -t lmcache_tier0_XXXX.log)"
BENCH_LOG="$(mktemp -t lmcache_tier0_bench_XXXX.log)"
SERVER_PID=""

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "==> Stopping lmcache server (pid=${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> Tier 0: LMCache storage smoke test (CPU, no model)"
echo "    OS:            $(uname -s)"
echo "    ZMQ/HTTP port: ${ZMQ_PORT}/${HTTP_PORT}"
echo "    transfer mode: ${TRANSFER_MODE}"
echo "    server log:    ${LOG_FILE}"

echo ""
echo "==> Starting lmcache server (1 GiB CPU cache, LRU eviction)"
lmcache server \
  --port "${ZMQ_PORT}" \
  --http-port "${HTTP_PORT}" \
  --l1-size-gb 1 \
  --eviction-policy LRU \
  >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

sleep 1
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "!! lmcache server exited immediately. Last 40 log lines:"
  tail -n 40 "${LOG_FILE}" || true
  exit 1
fi

echo "==> Waiting for healthcheck (timeout ${HEALTHCHECK_TIMEOUT}s)"
if ! wait_for_endpoint_contains \
    "http://127.0.0.1:${HTTP_PORT}/healthcheck" \
    "${HEALTHCHECK_TIMEOUT}" "" "lmcache server"; then
  echo "!! Last 40 log lines:"
  tail -n 40 "${LOG_FILE}" || true
  exit 1
fi
echo "    server healthy"

echo ""
echo "==> Running: lmcache bench server --mode cpu (${BENCH_NUM_REQUESTS} requests)"
lmcache bench server \
  --rpc-url "tcp://127.0.0.1:${ZMQ_PORT}" \
  --url "http://127.0.0.1:${HTTP_PORT}" \
  --mode cpu \
  --transfer-mode "${TRANSFER_MODE}" \
  --num-tokens "${BENCH_NUM_TOKENS}" \
  --end "${BENCH_NUM_REQUESTS}" \
  2>&1 | tee "${BENCH_LOG}"

echo ""
echo "==> Validating store/retrieve integrity"
if grep -q "CHECKSUM MISMATCH" "${BENCH_LOG}"; then
  echo "!! CHECKSUM MISMATCH -- store/retrieve corrupted data. FAIL."
  exit 1
fi
MATCH_COUNT="$(grep -c "CHECKSUM MATCH OK" "${BENCH_LOG}" || true)"
if [ "${MATCH_COUNT}" -lt "${BENCH_NUM_REQUESTS}" ]; then
  echo "!! Only ${MATCH_COUNT}/${BENCH_NUM_REQUESTS} requests verified. FAIL."
  exit 1
fi

echo ""
echo "==> PASS: ${MATCH_COUNT}/${BENCH_NUM_REQUESTS} requests stored and retrieved"
echo "    with matching checksums on CPU memory. LMCache works without a GPU."
