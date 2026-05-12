#!/usr/bin/env bash
# Per-job environment setup for the SGLang + LMCache MP integration tests.
# Installs:
#   - LMCache from the CI checkout (editable, no build isolation).
#   - SGLang from the pre-merge fork branch that carries the MP integration.
# Patterned on setup-lmcache-only-env.sh. Drop the fork install once
# https://github.com/sgl-project/sglang/pull/24089 lands and SGLang's PyPI
# release contains the LMCache MP connector.
set -euo pipefail

trap 'echo "ERROR: setup-sglang-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

# ── LMCache (CI checkout) ───────────────────────────────────
echo "--- :python: Installing LMCache from source"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
uv pip install -e . --no-build-isolation

# ── SGLang (pre-merge fork) ─────────────────────────────────
echo "--- :python: Installing SGLang from pre-merge fork"
SGLANG_FORK_URL="git+https://github.com/Shaoting-Feng/sglang.git@shaoting/sglang-lmcache-mp-nonlayerwise#subdirectory=python"
uv pip install "${SGLANG_FORK_URL}"

echo "--- :white_check_mark: Environment ready"
python -c "import lmcache, sglang; print(f'sglang={sglang.__version__}; lmcache installed from source')"
