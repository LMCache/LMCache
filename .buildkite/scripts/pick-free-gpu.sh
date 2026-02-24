#!/usr/bin/env bash

# Usage: source pick-free-gpu.sh <MIN_FREE_MEM_MB> <GPU_COUNT>
# Selects the specified number of best available GPUs
MIN_FREE_MEM="${1:-10000}"    # in MiB (default: 10 GB)
REQUESTED_GPU_COUNT="${2:-1}" # number of GPUs to select (default: 1)
MAX_UTIL=20                   # hardcoded utilization threshold (%)
GPU_LIMIT=8                   # reserves GPU 0-7 for CI/Build
# 60 minutes
TIMEOUT_SECONDS=3600
INTERVAL=10

echo "[pick-free-gpu] Starting GPU selection: MIN_FREE_MEM=${MIN_FREE_MEM}MiB, REQUESTED_GPU_COUNT=${REQUESTED_GPU_COUNT}, MAX_UTIL=${MAX_UTIL}%, GPU_LIMIT=${GPU_LIMIT}" >&2

start_time=$(date +%s)

while true; do
  now=$(date +%s)
  elapsed=$((now - start_time))

  echo "[pick-free-gpu $(date '+%H:%M:%S')] Elapsed: ${elapsed}s / ${TIMEOUT_SECONDS}s - Checking for available GPUs..." >&2
  echo "[pick-free-gpu] nvidia-smi output:" >&2
  nvidia-smi --query-gpu=memory.free,utilization.gpu,index --format=csv,noheader,nounits >&2 || echo "[pick-free-gpu] nvidia-smi failed!" >&2

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
    echo "[pick-free-gpu] Found ${#candidates[@]} candidate GPU(s): ${candidates[*]}" >&2
    # select the top N GPUs with the maximum free memory
    mapfile -t top_gpus < <(
      printf "%s\n" "${candidates[@]}" \
        | sort -t',' -k1,1 -nr \
        | head -n"${REQUESTED_GPU_COUNT}" \
        | awk -F',' '{print $3}'
    )
    
    # Validate we found the requested number of GPUs
    if [ "${#top_gpus[@]}" -lt "$REQUESTED_GPU_COUNT" ]; then
      echo "⚠️  Warning: Requested $REQUESTED_GPU_COUNT GPUs but only found ${#top_gpus[@]} available. Waiting..."
      sleep $INTERVAL
      continue
    fi
    
    if [ "${#top_gpus[@]}" -eq 1 ]; then
      # Only one GPU found/requested
      export CUDA_VISIBLE_DEVICES="${top_gpus[0]}"
      echo "✅ Selected GPU #${top_gpus[0]} (CUDA_VISIBLE_DEVICES=${top_gpus[0]})"
    else
      # Multiple GPUs found/requested
      chosen_gpus=$(IFS=','; echo "${top_gpus[*]}")
      export CUDA_VISIBLE_DEVICES="${chosen_gpus}"
      gpu_list=$(printf "#%s," "${top_gpus[@]}")
      echo "✅ Selected GPUs ${gpu_list%,} (CUDA_VISIBLE_DEVICES=${chosen_gpus})"
    fi
    break
  fi

  echo "[pick-free-gpu] No suitable GPUs found yet, sleeping ${INTERVAL}s..." >&2
  sleep $INTERVAL
done
