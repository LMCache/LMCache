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
ports=(8000 8001 8002 8003)
configs=(
  ../config/Jul_8_ablation/rate/01.yaml
  ../config/Jul_8_ablation/rate/05.yaml
  ../config/Jul_8_ablation/rate/1.yaml
  ../config/Jul_8_ablation/rate/10.yaml
)
logs=(
  results/Jul_8_ablation/rate/tokens/01.log
  results/Jul_8_ablation/rate/tokens/05.log
  results/Jul_8_ablation/rate/tokens/1.log
  results/Jul_8_ablation/rate/tokens/10.log
)
outputs=(
  results/Jul_8_ablation/rate/01.csv
  results/Jul_8_ablation/rate/05.csv
  results/Jul_8_ablation/rate/1.csv
  results/Jul_8_ablation/rate/10.csv
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
  python3 coding.py --output "$out" --port "$port" >>"$logf" 2>&1

  # Tear down this engine session
  echo "Test finished; tearing down engine session PGID $engine_pid"
  kill -TERM -- -"${engine_pid}"
  # Wait until no processes remain in that group
  while kill -0 -- -"${engine_pid}" 2>/dev/null; do
    sleep 0.1
  done
  echo "Engine session $engine_pid has exited."

  rm -rf /home/ubuntu/kvcache/*

  # Remove this PGID from the list
  PIDS=( "${PIDS[@]/$engine_pid}" )
done

echo "All done!"
