#!/bin/bash

role="$LMCACHE_PLUGIN_ROLE"
worker_id="$LMCACHE_PLUGIN_WORKER_ID"  # Assuming WORKER_ID is passed as an environment variable

echo "Worker plugin started for role: $role, worker ID: $worker_id"

while true; do
    echo "Worker plugin is running..."
    sleep 5
done