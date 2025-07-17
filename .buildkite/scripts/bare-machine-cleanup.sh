#!/usr/bin/env bash
set -eu

# Utility: kill and cleanup a PID stored in Buildkite meta‑data
cleanup_meta_pid() {
  local key="$1"
  local label="$2"

  # Try to fetch the PID from meta‑data; if the key doesn't exist, `get` will exit non‑zero
  if pid=$(buildkite-agent meta-data get "$key" 2>/dev/null); then
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "🧹 Killing ${label} process ${pid}"
      kill "$pid" || true
      wait "$pid" 2>/dev/null || true
    else
      echo "No running ${label} process with PID ${pid}"
    fi
    # Remove the meta‑data key now that we’ve cleaned up
    buildkite-agent meta-data delete "$key"
  fi
}

# Clean up the bare‑machine CPU test
cleanup_meta_pid "cpu-CID"  "bare‑machine CPU test"

# Clean up the bare‑machine disk test
cleanup_meta_pid "disk-CID" "bare‑machine disk test"
