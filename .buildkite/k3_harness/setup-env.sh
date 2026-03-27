#!/usr/bin/env bash
# Per-job environment setup: installs vLLM nightly + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

# Print the failing command and line number on any error.
trap 'echo "ERROR: setup-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

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
VLLM_NIGHTLY_INDEX="https://wheels.vllm.ai/nightly/vllm/"
INDEX_HTML=$(curl -sfL "$VLLM_NIGHTLY_INDEX" 2>&1) || true
VLLM_NIGHTLY_URL=$(echo "$INDEX_HTML" \
    | grep -oP 'href="\K[^"]+'"${ARCH}"'\.whl' \
    | head -1) || true
if [[ -z "$VLLM_NIGHTLY_URL" ]]; then
    echo "WARNING: Could not find vLLM nightly wheel for ${ARCH} — falling back to latest stable" >&2
    uv pip install "vllm[runai,tensorizer,flashinfer]"
else
    # href is relative (../../<commit>/vllm-....whl), resolve to absolute URL
    VLLM_WHEEL_URL="https://wheels.vllm.ai/nightly/vllm/${VLLM_NIGHTLY_URL}"
    echo "Resolved nightly wheel: $VLLM_WHEEL_URL"
    uv pip install --prerelease=allow \
        "${VLLM_WHEEL_URL}[runai,tensorizer,flashinfer]" \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match
fi

echo "--- :python: Installing LMCache from source"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
