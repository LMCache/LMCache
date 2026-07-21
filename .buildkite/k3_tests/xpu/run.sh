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

if [ -f /opt/intel/oneapi/setvars.sh ]; then
  # Intel XPU runtime environment.
  # shellcheck disable=SC1091
  log "enable Intel XPU runtime environment"
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.buildkite/k3_harness/setup-lmcache-only-env.sh"

log "installing job dependencies"
uv pip install -r requirements/common.txt -r requirements/test.txt

log "checking torch.xpu availability"
python - <<'PY'
import torch

assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU not available in pod"
print("torch.xpu.is_available() = True")
PY

discover_xpu_tests() {
  python - <<'PY'
import os
from fnmatch import fnmatch
from pathlib import Path
import re

root = Path("tests")
name_pattern = re.compile(r"(^|/)test_.*xpu.*\.py$")
xpu_only_skipif_pattern = re.compile(
    r'pytestmark\s*=\s*pytest\.mark\.skipif\(\s*torch_device_type\s*!=\s*"xpu".*?reason\s*=\s*"XPU-only tests"',
    re.S,
)
xpu_only_reason_pattern = re.compile(r'reason\s*=\s*"XPU-only tests"')
xpu_only_signal_patterns = [
    r'torch_device_type\s*==\s*"xpu"',
    r'torch\.xpu',
    r'xpu_connectors',
    r'xpu_ops',
    r'XPU-only tests',
    r'Intel XPU',
]

shared_xpu_capable_tests = {
  "tests/benchmarks/test_cachegen.py",
  "tests/test_serde.py",
  "tests/v1/multiprocess/test_engine_driven_transfer.py",
  "tests/v1/test_python_ops_fallback.py",
}
blacklist_raw = os.environ.get(
    "XPU_TEST_BLACKLIST",
    "tests/v1/test_musa_support.py,tests/v1/test_*sglang*.py",
)
blacklist = {item.strip() for item in blacklist_raw.split(",") if item.strip()}
selected: set[str] = set()


def is_blacklisted(rel: str) -> bool:
  return any(fnmatch(rel, pattern) for pattern in blacklist)


def is_xpu_only_test(rel: str, text: str) -> bool:
  if name_pattern.search(rel):
    return True
  if xpu_only_skipif_pattern.search(text):
    return True
  if xpu_only_reason_pattern.search(text):
    return True
  return any(re.search(pattern, text) for pattern in xpu_only_signal_patterns)


def is_shared_xpu_capable_test(rel: str) -> bool:
  return rel in shared_xpu_capable_tests

for path in root.rglob("test_*.py"):
    rel = path.as_posix()
    if is_blacklisted(rel):
      continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    if is_shared_xpu_capable_test(rel) or is_xpu_only_test(rel, text):
      selected.add(rel)

for rel in sorted(selected):
    print(rel)
PY
}

mapfile -t XPU_TEST_FILES < <(discover_xpu_tests)
if [ "${#XPU_TEST_FILES[@]}" -eq 0 ]; then
  fail "no XPU-related test files found under tests"
fi

log "discovered ${#XPU_TEST_FILES[@]} XPU-related test files"
printf '  %s\n' "${XPU_TEST_FILES[@]}"

PYTEST_ARGS=(-q --maxfail=1)
if [ -n "${TEST_SELECTOR:-}" ]; then
  PYTEST_ARGS+=(-k "${TEST_SELECTOR}")
fi

log "running XPU-related tests"
pytest "${PYTEST_ARGS[@]}" "${XPU_TEST_FILES[@]}"
log "xpu smoke test finished successfully"