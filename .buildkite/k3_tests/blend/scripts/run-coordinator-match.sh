#!/usr/bin/env bash
# CacheBlend coordinator global-match smoke.
#
# Runs the cacheblend-plugin's coordinator_match harness against this PR's
# LMCache: a coordinator with the fleet blend index plus two blend server +
# vLLM stacks (gpt-oss-20b, dummy weights) sharing an fs L2. Stack B must
# blend a body that only stack A stored, found through the coordinator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

# Fail fast if the GPUs are occupied by stale host processes.
check_gpu_health 80

: "${CB_PLUGIN_DIR:?CB_PLUGIN_DIR not set — source k3_harness/setup-blend-env.sh first}"

CARD="${CB_COORD_CARD:-gpt_oss_20b}"
LOG_DIR="${REPO_ROOT}/cb_coord_logs"      # workspace-relative so artifact_paths can collect it
mkdir -p "${LOG_DIR}"

# Same venv resolution as run-compat.sh: the image's /opt/venv in CI,
# CB_VENV for local runs.
if [ -n "${CB_VENV:-}" ]; then
    VENV="${CB_VENV}"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV="${VIRTUAL_ENV}"
elif command -v vllm >/dev/null 2>&1; then
    VENV="$(cd "$(dirname "$(command -v vllm)")/.." && pwd)"
else
    VENV="/opt/venv"
fi

echo "+++ :satellite: coordinator_match  card=${CARD}  venv=${VENV}  plugin=${CB_PLUGIN_DIR}"
python "${CB_PLUGIN_DIR}/.buildkite/harness/coordinator_match.py" \
    --models "${CARD}" \
    --gpus 0,1 \
    --venv "${VENV}" \
    --load-format dummy \
    --log-dir "${LOG_DIR}" \
    --out "${LOG_DIR}/result.json"
