#!/usr/bin/env bash
# Run a single comprehensive integration test config.
# Usage: run.sh <config.yaml>
#
# This replaces the old vllm-integration-tests.sh which ran all configs
# sequentially in one job. Now each config runs in its own K8s pod.
#
# The old script built a Docker image and ran vLLM inside Docker containers.
# Here we run vLLM directly in the pod — the environment is set up by setup-env.sh.
set -euo pipefail

CFG_NAME="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/.buildkite/configs"
LOGFILE="${CFG_NAME%.yaml}.log"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh

# Install test utilities
uv pip install yq jq 2>/dev/null || true
pip install yq 2>/dev/null || true

# ── Parse config ─────────────────────────────────────────────
cfg_file="${CONFIG_DIR}/${CFG_NAME}"
if [[ ! -f "$cfg_file" ]]; then
    echo "Config not found: ${cfg_file}"
    exit 1
fi

feature_type=$(yq -r '.feature.type // ""' "$cfg_file")
echo "===== Testing LMCache with ${CFG_NAME} (type=${feature_type:-standard}) ====="

# ── Delegate to the existing test script ─────────────────────
# The existing vllm-integration-tests.sh handles all the complexity of
# starting vLLM servers, running workloads, and checking memory leaks.
# We just need to run it with a single config instead of the full list.
#
# Create a temp file with just this one config name.
CONFIGS_FILE=$(mktemp)
echo "${CFG_NAME}" > "${CONFIGS_FILE}"
trap "rm -f ${CONFIGS_FILE}" EXIT

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}" \
    .buildkite/scripts/vllm-integration-tests.sh \
    --hf-token="${HF_TOKEN:-}" \
    --server-wait-timeout=240 \
    --configs="${CONFIGS_FILE}" \
    2>&1 | tee "${LOGFILE}"
