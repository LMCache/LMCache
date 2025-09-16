#!/usr/bin/env bash

# Usage: source pick-free-gpu.sh <MIN_FREE_MEM_MB>
# Selects the best 2 available GPUs (or 1 if only 1 is available)
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
    # select the top 2 GPUs with the maximum free memory
    mapfile -t top_gpus < <(
      printf "%s\n" "${candidates[@]}" \
        | sort -t',' -k1,1 -nr \
        | head -n2 \
        | awk -F',' '{print $3}'
    )
    
    if [ "${#top_gpus[@]}" -eq 1 ]; then
      # Only one suitable GPU found
      export CUDA_VISIBLE_DEVICES="${top_gpus[0]}"
      echo "✅ Selected GPU #${top_gpus[0]} (CUDA_VISIBLE_DEVICES=${top_gpus[0]})"
    else
      # Two or more suitable GPUs found, use top 2
      chosen_gpus=$(IFS=','; echo "${top_gpus[*]}")
      export CUDA_VISIBLE_DEVICES="${chosen_gpus}"
      echo "✅ Selected GPUs #${top_gpus[0]},#${top_gpus[1]} (CUDA_VISIBLE_DEVICES=${chosen_gpus})"
    fi
    break
  fi

  sleep $INTERVAL
done
