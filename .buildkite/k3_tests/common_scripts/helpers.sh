#!/usr/bin/env bash
# Shared helper functions for K3s test scripts.
# Source this file from every scripts/*.sh.

# Track background PIDs for cleanup
TRACKED_PIDS=()

# Find an available TCP port starting from a given port number.
# Usage: find_free_port [start_port]
find_free_port() {
    local port="${1:-8000}"
    while [ "$port" -lt 65536 ]; do
        if ! lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 &&
           ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            echo "$port"
            return 0
        fi
        ((port++))
    done
    echo "ERROR: No available port found starting from ${1:-8000}" >&2
    return 1
}

# Wait for a vLLM server to become ready by polling /v1/models.
# Usage: wait_for_server <port> [timeout_secs]
wait_for_server() {
    local port="$1"
    local timeout="${2:-180}"
    echo "Waiting for vLLM on port $port (timeout=${timeout}s)..."
    for ((i = 0; i < timeout; i++)); do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "vLLM ready on port $port (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "vLLM failed to start on port $port within ${timeout}s" >&2
    return 1
}

# Kill all tracked background PIDs and wait for them.
# Call this in a trap handler.
cleanup_pids() {
    echo "--- Cleaning up background processes..."
    for pid in "${TRACKED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    TRACKED_PIDS=()
    # Give GPU memory a moment to release
    sleep 2
}
