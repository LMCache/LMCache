#!/usr/bin/env bash
# Coordinator integration tests, run on the host agent, CPU-only:
#   1. this LMCache's MP servers against its coordinator, end to end;
#   2. the coordinator-controllers suite against this LMCache, its end-to-end
#      tests included.
# Both run; the step fails if either does.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Private repo coordinates from Buildkite secrets (or exported env).
if command -v buildkite-agent >/dev/null 2>&1; then
    CC_TOKEN="${CC_TOKEN:-$(buildkite-agent secret get CC_TM_PAT 2>/dev/null || true)}"
    CC_REPO="${CC_REPO:-$(buildkite-agent secret get CC_REPO 2>/dev/null || true)}"
fi
: "${CC_REPO:?CC_REPO not set (owner/name of the coordinator-controllers repo)}"
: "${CC_TOKEN:?CC_TM_PAT not set (PAT with contents read on ${CC_REPO})}"

# A venv and a clone per build; both removed on exit.
VENV=".venv-cc-${BUILDKITE_BUILD_ID:-local}"
CC_DIR="/tmp/coordinator-controllers-${BUILDKITE_BUILD_ID:-local}"
trap 'rm -rf "${VENV}" "${CC_DIR}"' EXIT

export PATH="$HOME/.local/bin:$PATH"   # uv, installed for the agent user
echo "--- :python: LMCache from this checkout, CPU-only"
uv venv --python 3.12 "${VENV}"
source "${VENV}/bin/activate"
uv pip install --upgrade pip setuptools wheel
# The coordinator suite touches no GPU; the CPU torch wheel is a fraction of the size.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements/common.txt
NO_GPU_EXT=1 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0+ci \
    uv pip install -e . --no-build-isolation
python -c "import lmcache; print('lmcache', lmcache.__version__)"
uv pip install pytest pytest-asyncio

status=0
echo "+++ :satellite: MP servers against the coordinator, end to end"
RUN_MP_E2E=1 LMCACHE_TRACK_USAGE=false python -m pytest -q -rs -o log_cli=false \
    tests/v1/mp_coordinator/test_mp_server_e2e.py || status=$?

# Clone the controllers at the newest release tag (or CC_REF), using only our
# token: the agent's own git credential is scoped to LMCache and 403s here.
url="https://x-access-token:${CC_TOKEN}@github.com/${CC_REPO}.git"
git_() { env GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null git -c credential.helper= "$@"; }
if [ -z "${CC_REF:-}" ]; then
    CC_REF="$(git_ ls-remote --tags --refs "${url}" | awk -F/ '{print $NF}' \
        | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1 || true)"
    CC_REF="${CC_REF:-main}"
fi
echo "--- :arrow_down: ${CC_REPO}@${CC_REF}"
git_ clone --depth 1 --branch "${CC_REF}" "${url}" "${CC_DIR}"

echo "--- :python: Installing coordinator-controllers"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_COORDINATOR_CONTROLLERS=0.0.0+ci \
    uv pip install -e "${CC_DIR}" --no-deps --no-build-isolation

echo "+++ :electric_plug: coordinator-controllers suite against this LMCache"
# RUN_E2E=1 includes its end-to-end tests, real MP servers with its
# controllers loaded; a ref without them simply has none to run.
(cd "${CC_DIR}" && RUN_E2E=1 python -m pytest -q -rs) || status=$?
exit "${status}"
