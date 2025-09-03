#!/usr/bin/env bash

# Usage: source pick-free-gpu.sh <MIN_FREE_MEM_MB>
MIN_FREE_MEM="${1:-10000}"    # in MiB (default: 10 GB)
MAX_UTIL=20                   # hardcoded utilization threshold (%)
GPU_LIMIT=4                   # reserves GPU 0-3 for CI/Build
# 30 minutes
TIMEOUT_SECONDS=1800
INTERVAL=10

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
    | awk -F',' -v min_mem="$MIN_FREE_MEM" -v max_util="$MAX_UTIL" -v gpu_limit="$GPU_LIMIT" '{
        mem = $1; util = $2; idx = $3;
        gsub(/^[ \t]+|[ \t]+$/, "", mem);
        gsub(/^[ \t]+|[ \t]+$/, "", util);
        gsub(/^[ \t]+|[ \t]+$/, "", idx);
        if (mem >= min_mem && util <= max_util && idx < gpu_limit) {
          print mem "," util "," idx;
        }
      }'
  )

  if [ "${#candidates[@]}" -gt 0 ]; then
    # select the GPU with the maximum free memory
    IFS=',' read -r _ _ chosen_gpu <<< "$(
      printf "%s\n" "${candidates[@]}" \
        | sort -t',' -k1,1 -nr \
        | head -n1
    )"
    export CUDA_VISIBLE_DEVICES="${chosen_gpu}"
    echo "✅ Selected GPU #${chosen_gpu} (CUDA_VISIBLE_DEVICES=${chosen_gpu})"
    break
  fi

  sleep $INTERVAL
done
