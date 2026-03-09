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
uv pip install --prerelease=allow \
    'vllm[runai,tensorizer,flashinfer]' \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match

echo "--- :python: Installing LMCache from source"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
