#!/usr/bin/env bash
# MUSA hardware smoke: focused transfer tests plus MP server health.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../unittests/ci-common.sh"

SERVER_LOG="${ARTIFACT_DIR}/lmcache-server.log"

bootstrap_musa_ci 1
pytest_base_args

log "Running focused MUSA connector and transfer tests"
run_pytest "${PYTEST_ARGS[@]}" \
    tests/v1/test_musa_support.py \
    tests/v1/test_musa_connector.py \
    tests/v1/test_musa_native.py \
    tests/v1/platform/musa/test_musa_pin_memory.py \
    tests/v1/platform/musa/test_musa_staging_copy.py \
    tests/v1/platform/musa/test_musa_mp_block_transfer.py::test_musa_block_transfer_device_non_mla_d2h_and_h2d \
    tests/v1/platform/musa/test_musa_mp_block_transfer.py::test_musa_block_transfer_device_mla_d2h_and_h2d \
    tests/v1/multiprocess/test_engine_driven_transfer.py::test_musa_data_context_keeps_layout_validation_device_agnostic \
    tests/v1/multiprocess/test_engine_driven_transfer.py::test_musa_data_context_store_uses_device_agnostic_gather \
    tests/v1/multiprocess/test_engine_driven_transfer.py::test_musa_data_context_retrieve_uses_device_agnostic_scatter \
    2>&1 | tee "${ARTIFACT_DIR}/pytest.log"

ZMQ_PORT="${MUSA_CI_ZMQ_PORT:-6555}"
HTTP_PORT="${MUSA_CI_HTTP_PORT:-7555}"

log "Starting the LMCache multiprocess server smoke test"
LMCACHE_DEVICE_BACKEND=musa lmcache server \
    --host 127.0.0.1 \
    --port "${ZMQ_PORT}" \
    --http-host 127.0.0.1 \
    --http-port "${HTTP_PORT}" \
    --l1-size-gb 0.25 \
    --no-l1-use-lazy \
    --eviction-policy LRU \
    --chunk-size 128 \
    --disable-metrics \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -n 200 "${SERVER_LOG}" >&2
        fail "LMCache server exited before becoming healthy"
    fi

    if curl -fsS "http://127.0.0.1:${HTTP_PORT}/healthcheck" >/dev/null; then
        log "LMCache server is healthy"
        break
    fi

    sleep 1
done

if ! curl -fsS "http://127.0.0.1:${HTTP_PORT}/healthcheck" >/dev/null; then
    tail -n 200 "${SERVER_LOG}" >&2
    fail "LMCache server did not become healthy within 60 seconds"
fi

kill "${SERVER_PID}"
set +e
wait_for_process_exit "${SERVER_PID}" 30
SERVER_EXIT_CODE=$?
set -e

if [[ "${SERVER_EXIT_CODE}" -eq 124 ]]; then
    tail -n 200 "${SERVER_LOG}" >&2
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    fail "LMCache server did not stop within 30 seconds"
fi

if [[ "${SERVER_EXIT_CODE}" -ne 0 && "${SERVER_EXIT_CODE}" -ne 143 ]]; then
    tail -n 200 "${SERVER_LOG}" >&2
    SERVER_PID=""
    fail "LMCache server exited with status ${SERVER_EXIT_CODE} during shutdown"
fi

if ! grep -q "LMCache HTTP server stopped" "${SERVER_LOG}"; then
    tail -n 200 "${SERVER_LOG}" >&2
    SERVER_PID=""
    fail "LMCache server did not report a clean HTTP shutdown"
fi

SERVER_PID=""
log "MUSA hardware smoke test finished successfully"
