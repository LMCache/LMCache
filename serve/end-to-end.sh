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
ports=(8000)
configs=(
  ../config/May_13_2_triviaqa_rr/ours/kivi_1.yaml
  # ../config/May_13_2_triviaqa_rr/ours/kivi_01.yaml
  # ../config/May_13_2_triviaqa_rr/ours/kivi_10.yaml
  # ../config/May_13_2_triviaqa_rr/ours/streaming_1.yaml
  # ../config/May_13_2_triviaqa_rr/ours/streaming_04.yaml
  # ../config/May_13_2_triviaqa_rr/ours/streaming_10.yaml
)
logs=(
  results/May_13_2_triviaqa_rr/ours/tokens/kivi_1.log
  # results/May_13_2_triviaqa_rr/ours/tokens/kivi_01.log
  # results/May_13_2_triviaqa_rr/ours/tokens/kivi_10.log
  # results/May_13_2_triviaqa_rr/ours/tokens/streaming_1.log
  # results/May_13_2_triviaqa_rr/ours/tokens/streaming_04.log
  # results/May_13_2_triviaqa_rr/ours/tokens/streaming_10.log
)
outputs=(
  results/May_13_2_triviaqa_rr/ours/kivi_1.csv
  # results/May_13_2_triviaqa_rr/ours/kivi_01.csv
  # results/May_13_2_triviaqa_rr/ours/kivi_10.csv
  # results/May_13_2_triviaqa_rr/ours/streaming_1.csv
  # results/May_13_2_triviaqa_rr/ours/streaming_04.csv
  # results/May_13_2_triviaqa_rr/ours/streaming_10.csv
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
  python3 triviaqa.py --output "$out" --port "$port" >>"$logf" 2>&1

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
