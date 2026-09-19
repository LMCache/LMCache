#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE=(docker compose -f "$SCRIPT_DIR/docker-compose.yml")
CONTAINER_NAME="dynamo-lmcache-disaggregated-$$"
DOCKER_PID=

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$DOCKER_PID" ]]; then
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    kill "$DOCKER_PID" 2>/dev/null || true
    wait "$DOCKER_PID" 2>/dev/null || true
  fi
  "${COMPOSE[@]}" stop || true
  exit "$status"
}

# Stop this demo's services on failure or Ctrl+C as well as normal exit.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${COMPOSE[@]}" up -d

docker run --rm --init --name "$CONTAINER_NAME" \
  --gpus all --network host --ipc host --ulimit memlock=-1 \
  -v "$SCRIPT_DIR:/opt/dynamo-lmcache:ro" \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2 \
  bash /opt/dynamo-lmcache/serve.sh disaggregated &
DOCKER_PID=$!
wait "$DOCKER_PID"
