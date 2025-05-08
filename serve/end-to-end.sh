#!/usr/bin/env bash
set -euo pipefail

# Cleanup function: kill all engine sessions on SIGINT/SIGTERM
cleanup() {
  echo
  echo "Caught interrupt signal! Cleaning up…"
  for pgid in "${PIDS[@]}"; do
    if kill -0 -- -"${pgid}" 2>/dev/null; then
      echo "  Killing engine session PGID ${pgid}"
      kill -TERM -- -"${pgid}" 2>/dev/null || true
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

# --- Configuration arrays (must all be the same length) ---
ports=(8000 8001 8002 8003 8004 8005)
configs=(
  ../config/qmsum7.yaml
  ../config/qmsum8.yaml
  ../config/qmsum3.yaml
  ../config/qmsum5.yaml
  ../config/qmsum.yaml
  ../config/qmsum2.yaml
)
logs=(
  results/May_7_6/ours/tokens/1.log
  results/May_7_6/ours/tokens/01.log
  results/May_7_6/baseline/tokens/03.log
  results/May_7_6/baseline/tokens/02.log
  results/May_7_6/ours/tokens/001.log
  results/May_7_6/ours/tokens/10.log
)
outputs=(
  results/May_7_6/ours/1.csv
  results/May_7_6/ours/01.csv
  results/May_7_6/baseline/03.csv
  results/May_7_6/baseline/02.csv
  results/May_7_6/ours/001.csv
  results/May_7_6/ours/10.csv
)

# Array to track running PGIDs
PIDS=()

# Verify arrays are aligned
if [ "${#ports[@]}" -ne "${#configs[@]}" ] || \
   [ "${#ports[@]}" -ne "${#logs[@]}" ] || \
   [ "${#ports[@]}" -ne "${#outputs[@]}" ]; then
  echo "ERROR: ports, configs, logs and outputs must all have the same number of entries."
  exit 1
fi

for i in "${!ports[@]}"; do
  port=${ports[i]}
  cfg=${configs[i]}
  logf=${logs[i]}
  out=${outputs[i]}

  echo "=== Instance $((i+1)) on port $port ==="

  # Start engine in its own session/process-group (PGID = PID)
  setsid bash start_engine.sh -p "$port" -c "$cfg" -l "$logf" >"$logf" 2>&1 &
  engine_pid=$!
  PIDS+=("$engine_pid")

  # Wait for health endpoint
  echo -n "Waiting for http://localhost:$port/health … "
  until curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" | grep -q '^2'; do
    sleep 1
    echo -n "."
  done
  echo " OK"

  # Run the test
  echo "Running test: python3 online_test.py --output $out --port $port"
  python3 online_test.py --output "$out" --port "$port" >>"$logf" 2>&1

  # Tear down this engine session
  echo "Test finished; tearing down engine session PGID $engine_pid"
  kill -TERM -- -"${engine_pid}"
  # Wait until no processes remain in that group
  while kill -0 -- -"${engine_pid}" 2>/dev/null; do
    sleep 0.1
  done
  echo "Engine session $engine_pid has exited."

  # Remove this PGID from the list
  PIDS=( "${PIDS[@]/$engine_pid}" )
done

echo "All done!"
