#!/usr/bin/env bash
# Run the broad MUSA-compatible unit suite on a self-hosted agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MUSA_CI_UNIT_ONLY=1

exec bash "${SCRIPT_DIR}/run.sh"
