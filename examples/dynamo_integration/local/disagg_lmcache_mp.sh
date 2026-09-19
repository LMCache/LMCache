#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DYNAMO_MODE=disaggregated
COMPOSE=(docker compose
  -f "$SCRIPT_DIR/docker-compose.yml"
  -f "$SCRIPT_DIR/docker-compose.dynamo.yml")

# Stop this demo's services on failure or Ctrl+C as well as normal exit.
trap '"${COMPOSE[@]}" stop' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${COMPOSE[@]}" up --abort-on-container-exit --exit-code-from dynamo
