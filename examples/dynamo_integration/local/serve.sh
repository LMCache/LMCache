#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Container entry point used by the local launch script.
set -euo pipefail

PIDS=()

# Stop all processes started by this script and preserve its exit status.
cleanup() {
  local status=$?
  trap - EXIT
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "$status"
}

# Wait for a health endpoint, stopping early if any started process exits.
wait_for_http() {
  local service=$1
  local endpoint=$2
  local max_attempts=${3:-60}
  local attempt pid
  for ((attempt=0; attempt<max_attempts; attempt++)); do
    if ((${#PIDS[@]})); then
      for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          echo "A serving process exited while waiting for $service" >&2
          return 1
        fi
      done
    fi
    if curl -fsS --connect-timeout 1 --max-time 2 "$endpoint" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$service did not become ready at $endpoint" >&2
  return 1
}

if [[ $# -ne 1 ]]; then
  echo "Usage: serve.sh aggregated|disaggregated" >&2
  exit 2
fi
case "$1" in
  aggregated|disaggregated) MODE=$1 ;;
  *) echo "Unknown serving mode: $1" >&2; exit 2 ;;
esac

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Let Dynamo manage its Prometheus multiprocess directory.
unset PROMETHEUS_MULTIPROC_DIR

# Wait for the NATS and etcd services started by the host launcher.
wait_for_http NATS http://localhost:8222/healthz
wait_for_http etcd http://localhost:2379/health

# Start the LMCache MP server with 16 GiB of CPU cache.
lmcache server --l1-size-gb 16 --eviction-policy LRU \
  --port 5555 --http-port 8080 &
PIDS+=("$!")
wait_for_http LMCache http://localhost:8080/healthcheck

# Connect vLLM workers to LMCache on port 5555 through LMCacheMPConnector.
if [[ "$MODE" == aggregated ]]; then
  # Run prefill and decode in one vLLM worker on GPU 0.
  DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"lmcache.mp.port": 5555}
    }' &
  PIDS+=("$!")
  wait_for_http "aggregated worker" http://localhost:8081/health 600
else
  # Start the vLLM decode worker on GPU 0.
  DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --disaggregation-mode decode \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"lmcache.mp.port": 5555}
    }' &
  PIDS+=("$!")

  # Wait for decode to finish downloading and loading the model before prefill.
  wait_for_http "decode worker" http://localhost:8081/health 600

  # Start the vLLM prefill worker on GPU 1, using the cached model files.
  DYN_SYSTEM_PORT=8082 CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --disaggregation-mode prefill \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"lmcache.mp.port": 5555}
    }' &
  PIDS+=("$!")
  wait_for_http "prefill worker" http://localhost:8082/health 600
fi

# Start the HTTP frontend on port 8000 once all workers are ready.
python3 -m dynamo.frontend &
PIDS+=("$!")

# Stop the whole demo if any serving process exits, preserving its failure code.
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      echo "A serving process exited; stopping the demo" >&2
      if [[ $status -eq 0 ]]; then
        status=1
      fi
      exit "$status"
    fi
  done
  sleep 1
done
