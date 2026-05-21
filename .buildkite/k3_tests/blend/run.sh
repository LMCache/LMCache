#!/usr/bin/env bash
# Blend test entrypoint for K8s pods.
# Thin wrapper: sets up shared env, then delegates to scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# Shared PR setup: GPU health check, vLLM nightly (uv), LMCache build from source.
# Modal production E2E can prebuild these layers into the image; skip runtime
# installs there so H100 time is spent only on service startup + validation.
if [[ "${LMCACHE_SKIP_RUNTIME_INSTALL:-0}" == "1" ]]; then
  echo "--- 📦 Skipping runtime installs; using prebuilt Modal image"
else
  source .buildkite/k3_harness/setup-blend-env.sh
fi

# Run blend-specific logic.
exec bash "${SCRIPT_DIR}/scripts/run-blend-test.sh" "$@"
