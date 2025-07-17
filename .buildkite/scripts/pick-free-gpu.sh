#!/usr/bin/env bash
# Query each GPU’s free memory and pick the index with the highest free value
# Format of each line: "<free-MiB>, <index>"
chosen_gpu=$(
  nvidia-smi \
    --query-gpu=memory.free,index \
    --format=csv,noheader,nounits \
  | sort -t',' -k1 -nr \
  | head -n1 \
  | cut -d',' -f2
)

export CUDA_VISIBLE_DEVICES=$chosen_gpu
echo "--- 🚀 Selected GPU #${chosen_gpu} (CUDA_VISIBLE_DEVICES=${chosen_gpu})"
