#!/usr/bin/env bash
# XPU smoke test entrypoint.
# Runs directly inside the agent-stack-k8s job pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

log() {
  echo "[xpu-test] $*"
}

fail() {
  echo "[xpu-test] ERROR: $*" >&2
  exit 1
}

export TEST_SELECTOR="${TEST_SELECTOR:-calculate_cdf or get_gpu_pci_bus_id or load_and_reshape_flash}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/.buildkite/k3_harness/setup-lmcache-only-env.sh"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
  # Intel XPU runtime environment.
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

log "installing job dependencies"
uv pip install -r requirements/common.txt -r requirements/test.txt

log "checking torch.xpu availability"
python - <<'PY'
import torch

assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU not available in pod"
print("torch.xpu.is_available() = True")
PY

log "running XPU smoke tests"
pytest -q tests/test_serde.py tests/v1/test_python_ops_fallback.py \
  -k "${TEST_SELECTOR}" --maxfail=1

log "xpu smoke test finished successfully"