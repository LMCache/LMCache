#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Launch the LMCache MP HTTP server with the frontend plugin wired in.
# The plugin reports heartbeats to a (remote or local) discovery service
# whose URL is passed through ``--runtime-plugin-config``.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLUGIN="${REPO_ROOT}/lmcache/lmcache_frontend/lmcache_mp_plugin/lmcache_mp_frontend_plugin.py"
HEARTBEAT_URL="${HEARTBEAT_URL:-http://localhost:5000/lmcache_heartbeat}"

python3 -m lmcache.v1.multiprocess.http_server \
    --host localhost --port 5555 \
    --http-host 0.0.0.0 --http-port 8085 \
    --l1-size-gb 2 \
    --eviction-policy LRU \
    --runtime-plugin-locations "${PLUGIN}" \
    --runtime-plugin-config \
        "{\"plugin.frontend.heartbeat-url\": \"${HEARTBEAT_URL}\"}"
