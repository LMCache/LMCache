#!/usr/bin/env bash
set -euo pipefail

# --- Configuration arrays (must all be the same length) ---
ports=(8000 8001 8002 8003 8004 8005 8006 8007)
configs=(
  ../config/qmsum.yaml
  ../config/qmsum2.yaml
  ../config/qmsum3.yaml
  ../config/qmsum4.yaml
  ../config/qmsum5.yaml
  ../config/qmsum6.yaml
  ../config/qmsum7.yaml
  ../config/qmsum8.yaml
)
logs=(
  results/May_7_2/ours/tokens/04.log
  results/May_7_2/ours/tokens/07.log
  results/May_7_2/baseline/tokens/03.log
  results/May_7_2/baseline/tokens/06.log
  results/May_7_2/baseline/tokens/02.log
  results/May_7_2/prefill/tokens/0.log
  results/May_7_2/ours/tokens/1.log
  results/May_7_2/ours/tokens/01.log
)
outputs=(
  results/May_7_2/ours/04.csv
  results/May_7_2/ours/07.csv
  results/May_7_2/baseline/03.csv
  results/May_7_2/baseline/06.csv
  results/May_7_2/baseline/02.csv
  results/May_7_2/prefill/0.csv
  results/May_7_2/ours/1.csv
  results/May_7_2/ours/01.csv
)

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

  # Start the engine, redirect all its stdout+stderr into the log file
  bash start_engine.sh -p "$port" -c "$cfg" -l "$logf" >"$logf" 2>&1 &
  engine_pid=$!

  # Wait until the health endpoint returns HTTP 2xx
  echo -n "Waiting for http://localhost:$port/health … "
  until curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" | grep -q '^2'; do
    sleep 1
    echo -n "."
  done
  echo " OK"

  # Run the test; append its stdout+stderr into the same log file
  echo "Running test: python3 online_test.py --output $out --port $port"
  python3 online_test.py --output "$out" --port "$port" >>"$logf" 2>&1

  # Tear down
  echo "Test finished; killing engine (PID $engine_pid)"
  kill "$engine_pid"
  echo
done

echo "All done!"
