#!/usr/bin/env bash
# Per-job environment setup: installs vLLM nightly + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

echo "--- :python: Installing vLLM nightly"
# Resolve the latest nightly wheel URL directly from the nightly index.
# PEP 440 ranks stable releases (0.17.0) above pre-release nightlies
# (0.17.0rc1.devN), so pip/uv always picks the stable version when both
# indexes are available. We work around this by parsing the nightly index
# page and installing the wheel by URL.
ARCH=$(uname -m)  # x86_64 or aarch64
VLLM_NIGHTLY_URL=$(
    curl -sfL https://wheels.vllm.ai/nightly/vllm/ \
    | grep -oP 'href="\K[^"]+'"${ARCH}"'\.whl' \
    | head -1
)
if [[ -z "$VLLM_NIGHTLY_URL" ]]; then
    echo "ERROR: Could not find vLLM nightly wheel for ${ARCH}" >&2
    exit 1
fi
# href is relative (../../<commit>/vllm-....whl), resolve to absolute URL
VLLM_WHEEL_URL="https://wheels.vllm.ai/nightly/vllm/${VLLM_NIGHTLY_URL}"
echo "Resolved nightly wheel: $VLLM_WHEEL_URL"
uv pip install --prerelease=allow \
    "${VLLM_WHEEL_URL}[runai,tensorizer,flashinfer]" \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match

echo "--- :python: Installing LMCache from source"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
