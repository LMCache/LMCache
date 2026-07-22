#!/usr/bin/env bash
# XPU smoke test entrypoint.
# Runs directly inside the agent-stack-k8s job pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

log() {
  echo "[intel-xpu-unit-tests] $*"
}

fail() {
  echo "[intel-xpu-unit-tests] ERROR: $*" >&2
  exit 1
}

[ -f /opt/intel/oneapi/setvars.sh ] || fail "/opt/intel/oneapi/setvars.sh not found"
# shellcheck disable=SC1091
log "enable Intel XPU runtime environment"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

log "checking torch.xpu and xpu h/w availability"
python - <<'PY'
import torch

assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU not available in pod"
print("torch.xpu.is_available() = True")
PY

export TEST_SELECTOR="${TEST_SELECTOR:-calculate_cdf or get_gpu_pci_bus_id or load_and_reshape_flash}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.buildkite/k3_harness/setup-lmcache-only-env.sh"

log "installing job dependencies"
uv pip install -r requirements/common.txt -r requirements/test.txt

log "running XPU smoke tests"
pytest -q tests/v1/test_torch_ops.py \
  -k "${TEST_SELECTOR}" --maxfail=1
log "xpu smoke test finished successfully"
