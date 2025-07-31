#!/usr/bin/env bash

# Usage: source pick-free-gpu.sh <MIN_FREE_MEM_MB>
MIN_FREE_MEM="${1:-10000}"    # in MiB (default: 10 GB)
MAX_UTIL=20                   # hardcoded utilization threshold (%)
TIMEOUT_SECONDS=180
INTERVAL=10

echo "🔍 Waiting up to ${TIMEOUT_SECONDS}s for a GPU with >= ${MIN_FREE_MEM} MiB free and <= ${MAX_UTIL}% utilization..."

start_time=$(date +%s)

while true; do
  now=$(date +%s)
  elapsed=$((now - start_time))

  if (( elapsed >= TIMEOUT_SECONDS )); then
    echo "❌ Timeout: No suitable GPU found within ${TIMEOUT_SECONDS}s"
    return 1
  fi

  mapfile -t candidates < <(
    nvidia-smi --query-gpu=memory.free,utilization.gpu,index \
      --format=csv,noheader,nounits \
    | awk -F',' -v min_mem="$MIN_FREE_MEM" -v max_util="$MAX_UTIL" '{
        gsub(/ /, "", $0);
        if ($1 >= min_mem && $2 <= max_util) {
          print $1 "," $2 "," $3;
        }
      }'
  )

  if [ "${#candidates[@]}" -gt 0 ]; then
    IFS=',' read -r _ _ chosen_gpu <<< "$(printf "%s\n" "${candidates[@]}" | sort -t',' -k1 -nr | head -n1)"
    export CUDA_VISIBLE_DEVICES="${chosen_gpu}"
    echo "✅ Selected GPU #${chosen_gpu} (CUDA_VISIBLE_DEVICES=${chosen_gpu})"
    break
  fi

  echo "⏳ Still waiting... ${elapsed}s elapsed"
  sleep $INTERVAL
done
