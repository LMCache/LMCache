#!/bin/bash
# Example plugin for LMCache system
# This plugin runs continuously and exits when parent process terminates

# Handle termination signal
trap "echo 'Received termination signal, exiting...'; exit 0" SIGTERM

role="$LMCACHE_PLUGIN_ROLE"
config="$LMCACHE_PLUGIN_CONFIG"

echo "Shell plugin running with role: $role"
echo "Config: $config"

# Main loop
while true; do
    echo "Plugin is running..."
    sleep 10
done

echo "Shell plugin finished"
