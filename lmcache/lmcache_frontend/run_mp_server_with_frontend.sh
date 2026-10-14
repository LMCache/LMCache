#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Launch the LMCache MP HTTP server with the frontend plugin wired in.
# The plugin reports heartbeats to a (remote or local) discovery service
# whose URL is passed through ``--runtime-plugin-config``.
#
# All tunables below can be overridden via environment variables, e.g.:
#   HTTP_PORT=9090 REPORT_HOST=127.0.0.1 bash run_mp_server_with_frontend.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLUGIN="${REPO_ROOT}/lmcache/lmcache_frontend/lmcache_mp_plugin/lmcache_mp_frontend_plugin.py"

# --- Discovery service --------------------------------------------------------
HEARTBEAT_URL="${HEARTBEAT_URL:-http://localhost:5000/lmcache_heartbeat}"
# Host to report in heartbeat api_address. Defaults to 127.0.0.1 to match the
# loopback HTTP_HOST below; set it to "" to let get_local_ip() pick a NIC
# address when HTTP_HOST is non-loopback.
REPORT_HOST="${REPORT_HOST-127.0.0.1}"

# --- MP server internal control channel --------------------------------------
MP_HOST="${MP_HOST:-localhost}"
MP_PORT="${MP_PORT:-5555}"

# --- MP server outward-facing HTTP endpoint ----------------------------------
# A non-loopback HTTP_HOST (e.g. 0.0.0.0) requires LMCACHE_HTTP_API_KEY to be
# exported; the server refuses to start without it.
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8085}"

# --- Cache tuning ------------------------------------------------------------
L1_SIZE_GB="${L1_SIZE_GB:-2}"
EVICTION_POLICY="${EVICTION_POLICY:-LRU}"

PLUGIN_CFG="{\"plugin.frontend.heartbeat-url\": \"${HEARTBEAT_URL}\""
if [[ -n "${REPORT_HOST}" ]]; then
    PLUGIN_CFG="${PLUGIN_CFG}, \"plugin.frontend.report-host\": \"${REPORT_HOST}\""
fi
PLUGIN_CFG="${PLUGIN_CFG}}"

python3 -m lmcache.v1.multiprocess.http_server \
    --host "${MP_HOST}" --port "${MP_PORT}" \
    --http-host "${HTTP_HOST}" --http-port "${HTTP_PORT}" \
    --l1-size-gb "${L1_SIZE_GB}" \
    --eviction-policy "${EVICTION_POLICY}" \
    --runtime-plugin-locations "${PLUGIN}" \
    --runtime-plugin-config "${PLUGIN_CFG}"
