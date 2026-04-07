#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Example MP runtime plugin (Bash) for LMCache multiprocess server.
#
# In MP mode the config JSON contains aggregated sections:
#   mp_config, storage_manager_config, obs_config
#
# This script demonstrates:
#   1. Reading the MP-mode environment variables.
#   2. Extracting fields from the aggregated JSON via jq.
#   3. Running a periodic heartbeat loop.
#
# Requires: jq (optional, falls back to raw echo)

# Graceful shutdown
trap "echo '[mp_heartbeat] Received termination signal, exiting...'; exit 0" SIGTERM SIGINT

config="${LMCACHE_RUNTIME_PLUGIN_CONFIG}"

echo "[mp_heartbeat] Started"

# Try to extract key fields with jq if available
if command -v jq &>/dev/null && [ -n "${config}" ]; then
    host=$(echo "${config}" | jq -r '.mp_config.host // "?"')
    port=$(echo "${config}" | jq -r '.mp_config.port // "?"')
    chunk_size=$(echo "${config}" | jq -r '.mp_config.chunk_size // "?"')
    eviction=$(echo "${config}" | jq -r '.storage_manager_config.eviction_config.eviction_policy // "?"')
    echo "[mp_heartbeat] MP server: host=${host}  port=${port}  chunk_size=${chunk_size}  eviction=${eviction}"
else
    echo "[mp_heartbeat] jq not found or config empty; raw config:"
    echo "${config}"
fi

# Heartbeat loop
loop_count=0
while true; do
    echo "[mp_heartbeat] heartbeat #${loop_count}"
    loop_count=$((loop_count + 1))
    sleep 30
done
