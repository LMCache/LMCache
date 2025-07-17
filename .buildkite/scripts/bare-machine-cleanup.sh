#!/usr/bin/env bash
set -eu

# Utility: kill and cleanup a PID file
cleanup_pid_file() {
  local file="$1"
  local label="$2"

  if [[ -f "$file" ]]; then
    local pid
    pid=$(< "$file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "🧹 Killing ${label} process ${pid}"
      kill "$pid" || true
      # wait for it to actually exit (ignore errors)
      wait "$pid" 2>/dev/null || true
    else
      echo "No running ${label} process with PID ${pid}"
    fi
    rm -f "$file"
  fi
}

# Clean up the bare-machine test processes
cleanup_pid_file ".buildkite/bare_cpu_pid"  "bare‑machine CPU test"
cleanup_pid_file ".buildkite/bare_disk_pid" "bare‑machine disk test"
