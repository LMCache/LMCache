#!/usr/bin/env bash
# Blend test entrypoint for K8s pods.
# Thin wrapper: sets up shared env, then delegates to scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# Shared PR setup: GPU health check, vLLM nightly (uv), LMCache build from source.
source .buildkite/k3_harness/setup-blend-env.sh

# Run blend-specific logic.
exec bash "${SCRIPT_DIR}/scripts/run-blend-test.sh" "$@"
