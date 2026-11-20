#!/usr/bin/env bash
# LMCache RBLN multiprocess smoke: RBLN device tests on a real NPU, then the
# LMCache MP server up to /healthcheck.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

PYTHON=python3
HTTP_PORT=7555
SERVER_LOG="$(mktemp)"

echo "--- :package: Install LMCache"
"${PYTHON}" -m pip install --no-cache-dir \
    -r requirements/build.txt \
    -r requirements/common.txt \
    -r requirements/test.txt
# --no-deps keeps the image's torch-rbln / rebel-compiler pair untouched.
BUILD_WITH_RBLN=1 \
BUILD_MOONCAKE=0 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0+ci \
    "${PYTHON}" -m pip install --no-cache-dir --no-deps --no-build-isolation -e .

echo "--- :mag: RBLN preflight"
"${PYTHON}" "${SCRIPT_DIR}/preflight.py"

echo "--- :test_tube: RBLN device tests"
"${PYTHON}" -m pytest -q -rs tests/v1/platform/devices/rbln

echo "--- :rocket: LMCache MP server smoke"
LMCACHE_DEVICE_BACKEND=rbln lmcache server \
    --host 127.0.0.1 \
    --port 6555 \
    --http-host 127.0.0.1 \
    --http-port "${HTTP_PORT}" \
    --l1-size-gb 0.25 \
    --no-l1-use-lazy \
    --eviction-policy LRU \
    --chunk-size 128 \
    --disable-metrics \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

if ! "${PYTHON}" "${SCRIPT_DIR}/wait_for_health.py" \
    "http://127.0.0.1:${HTTP_PORT}/healthcheck" "${SERVER_PID}"; then
    tail -n 200 "${SERVER_LOG}"
    exit 1
fi
