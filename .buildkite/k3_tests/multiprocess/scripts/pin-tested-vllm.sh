#!/usr/bin/env bash
# Pin the currently-installed vLLM nightly to the
# `buildkite_latest_tested_vllm` branch.
#
# Runs ONLY when the canary vllm_bench step has succeeded and
# VERIFY_AND_PIN_VLLM=true. Records the just-verified vllm wheel version so
# downstream builds can install the same version deterministically instead
# of resolving "latest nightly" again.
#
# Two files are maintained on the dedicated branch:
#   tested_vllm_versions.csv  -- JSON Lines, append-only history. Each line
#                                is one self-contained record.
#   latest_tested_vllm.txt    -- Plain text, single line: the most recent
#                                verified version. Overwritten every run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

export GIT_TERMINAL_PROMPT=0

# ── Resolve the version that's actually installed in this pod ───────────
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
if [[ -z "${VLLM_VERSION}" ]]; then
    echo "[ERROR] could not read vllm.__version__ from the live env" >&2
    exit 1
fi
echo "Verified vLLM version: ${VLLM_VERSION}"

CI_REPO="LMCache/LMCache"
CI_BRANCH="buildkite_latest_tested_vllm"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    CI_REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${CI_REPO}.git"
else
    echo "[WARN] GITHUB_TOKEN not set — push will likely fail" >&2
    CI_REPO_URL="https://github.com/${CI_REPO}.git"
fi

WORK_DIR="/tmp/pin_vllm_$$"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "--- Preparing ${CI_BRANCH} branch from ${CI_REPO}"
if ! git clone --depth=1 --branch "${CI_BRANCH}" "${CI_REPO_URL}" \
        "${WORK_DIR}" 2>/dev/null; then
    # Branch does not exist yet -- create an orphan with no parent history.
    rm -rf "${WORK_DIR}"
    mkdir -p "${WORK_DIR}"
    git -C "${WORK_DIR}" init -q
    git -C "${WORK_DIR}" remote add origin "${CI_REPO_URL}"
    git -C "${WORK_DIR}" checkout --orphan "${CI_BRANCH}"
    # Drop anything that the orphan checkout might have staged from HEAD.
    git -C "${WORK_DIR}" rm -rf --cached . >/dev/null 2>&1 || true
    find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name ".git" \
        -exec rm -rf {} +
fi

# ── Update files ────────────────────────────────────────────────────────
HISTORY_FILE="${WORK_DIR}/tested_vllm_versions.csv"
LATEST_FILE="${WORK_DIR}/latest_tested_vllm.txt"

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BUILD_URL="${BUILDKITE_BUILD_URL:-}"
BUILD_NUMBER="${BUILDKITE_BUILD_NUMBER:-}"
COMMIT_SHA="${BUILDKITE_COMMIT:-}"

# Append-only history (JSON Lines). Built via python so quoting is safe.
python - "$HISTORY_FILE" <<PY
import json, os, sys
path = sys.argv[1]
record = {
    "timestamp": "${TIMESTAMP}",
    "vllm_version": "${VLLM_VERSION}",
    "build_number": "${BUILD_NUMBER}",
    "build_url": "${BUILD_URL}",
    "commit": "${COMMIT_SHA}",
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PY

# Latest pointer (overwrite).
printf '%s\n' "${VLLM_VERSION}" > "${LATEST_FILE}"

# ── Commit + push ───────────────────────────────────────────────────────
cd "${WORK_DIR}"
git add tested_vllm_versions.csv latest_tested_vllm.txt

if git diff --cached --quiet 2>/dev/null; then
    echo "No changes to commit (version unchanged?)."
    exit 0
fi

git -c user.email="ci@lmcache.ai" -c user.name="LMCache CI" \
    commit -m "Pin verified vLLM nightly: ${VLLM_VERSION}" || true

echo "--- Pushing to ${CI_REPO} ${CI_BRANCH}"
if ! git push origin "HEAD:${CI_BRANCH}" 2>/dev/null; then
    echo "[WARN] Normal push failed, force-pushing..." >&2
    git push origin "+HEAD:${CI_BRANCH}" 2>/dev/null || {
        echo "[ERROR] Failed to push to ${CI_REPO} ${CI_BRANCH}" >&2
        exit 1
    }
fi

echo "--- Pinned vLLM ${VLLM_VERSION} successfully"
git log --oneline -1
