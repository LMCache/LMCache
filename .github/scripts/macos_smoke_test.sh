#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# macOS basic-compatibility smoke test for the lmcache multiprocess
# server. Verifies that:
#   1) `lmcache --help` works (CLI entry point is importable)
#   2) common C++ extensions (native_storage_ops / lmcache_redis /
#      lmcache_fs) load on macOS, exercising the PipeNotifier fallback
#      in csrc/storage_backends/event_notifier.h
#   3) `lmcache server` can launch the ZMQ + HTTP server on CPU
#   4) the HTTP server answers GET / and GET /healthcheck
#
# This script is intentionally minimal — it does not exercise any
# GPU / CUDA / vLLM code paths. It is meant to be a fast regression
# signal against accidental Linux-only imports or filesystem usage
# (e.g. /dev/shm, librt, eventfd, fcntl at import time).

set -euo pipefail

HTTP_HOST="127.0.0.1"
HTTP_PORT="${LMCACHE_HTTP_PORT:-18080}"
ZMQ_PORT="${LMCACHE_ZMQ_PORT:-15555}"
LOG_FILE="${LMCACHE_LOG_FILE:-/tmp/lmcache_server.log}"
# GitHub macOS runners are noticeably slower than local macs on cold
# `import torch / fastapi / opentelemetry` chains, so budget enough
# wall-clock for the first HTTP hit after `lmcache server` starts.
STARTUP_TIMEOUT="${LMCACHE_STARTUP_TIMEOUT:-180}"

echo "==> Environment"
uname -a || true
sw_vers || true
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> Step 1: lmcache CLI help"
lmcache --help >/dev/null

echo "==> Step 1.5: import common C++ extensions (CPU-only build)"
# Mirrors the verify step in .github/workflows/build_cpu_artifacts.yml
# on the macOS axis: makes sure NO_GPU_EXT=1 produced loadable .so's
# and that c_ops resolves to the python_ops_fallback shim.
python -c "
import sys
import lmcache
import lmcache.native_storage_ops  # noqa: F401
import lmcache.lmcache_redis  # noqa: F401
import lmcache.lmcache_fs  # noqa: F401
import lmcache.c_ops  # noqa: F401
assert lmcache.torch_device_type == 'cpu', lmcache.torch_device_type
assert sys.modules['lmcache.c_ops'].__name__ == 'lmcache.python_ops_fallback'
"

echo "==> Step 2: launch 'lmcache server' on ${HTTP_HOST}:${HTTP_PORT} (zmq ${ZMQ_PORT})"
rm -f "${LOG_FILE}"
# Run the server in the background. Using `setsid`-like behavior via
# a subshell so we can kill the whole process group cleanly.
(
  lmcache server \
    --host "${HTTP_HOST}" \
    --port "${ZMQ_PORT}" \
    --http-host "${HTTP_HOST}" \
    --http-port "${HTTP_PORT}" \
    --l1-size-gb "${LMCACHE_L1_SIZE_GB:-1}" \
    --eviction-policy "${LMCACHE_EVICTION_POLICY:-LRU}" \
    --no-l1-use-lazy \
    >"${LOG_FILE}" 2>&1
) &
SERVER_PID=$!

cleanup() {
  echo "==> Cleanup: stopping server (pid=${SERVER_PID})"
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    # Give it a moment to exit; escalate if needed.
    for _ in $(seq 1 10); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -f "${LOG_FILE}" ]]; then
    echo "==> Last 100 lines of server log:"
    tail -n 100 "${LOG_FILE}" || true
  fi
}
trap cleanup EXIT

echo "==> Step 3: wait for HTTP endpoint (timeout=${STARTUP_TIMEOUT}s)"
READY=0
for i in $(seq 1 "${STARTUP_TIMEOUT}"); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "!! lmcache server exited prematurely after ${i}s"
    break
  fi
  if curl -fsS --max-time 2 "http://${HTTP_HOST}:${HTTP_PORT}/" >/dev/null 2>&1; then
    READY=1
    echo "==> Server reachable after ${i}s"
    break
  fi
  sleep 1
done

if [[ "${READY}" != "1" ]]; then
  echo "!! lmcache server did not become ready within ${STARTUP_TIMEOUT}s"
  exit 1
fi

echo "==> Step 4: curl GET /"
ROOT_BODY="$(curl -fsS "http://${HTTP_HOST}:${HTTP_PORT}/")"
echo "    body: ${ROOT_BODY}"
echo "${ROOT_BODY}" | grep -q '"status"' || {
  echo "!! GET / did not return expected status field"
  exit 1
}

echo "==> Step 5: curl GET /healthcheck"
HEALTH_BODY="$(curl -fsS "http://${HTTP_HOST}:${HTTP_PORT}/healthcheck")"
echo "    body: ${HEALTH_BODY}"
echo "${HEALTH_BODY}" | grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"' || {
  echo "!! GET /healthcheck did not report healthy"
  exit 1
}

echo "==> macOS smoke test passed."
