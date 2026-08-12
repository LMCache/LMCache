#!/usr/bin/env bash
# Pin the currently-installed vLLM nightly to a dedicated tracking branch.
#
# Supports two CI platforms via the CI_PLATFORM env var:
#   buildkite       -- resolves metadata from BUILDKITE_* env vars
#   github_actions  -- resolves metadata from GITHUB_* env vars
#
# Lives in .github/scripts/ rather than under .buildkite/ because it serves
# both platforms, same as run-cpu-e2e-validation.sh next to it; the
# buildkite pipeline invokes it by path.
#
# Target branch controlled by PIN_VLLM_BRANCH (defaults below per-platform).
# Callers that run the script only on success should set
#   PIN_VLLM_STATUS=tested.
# Callers that record a failure MUST also set PIN_VLLM_REASON and
#   PIN_VLLM_STATUS=failed so the CSV row carries a human-readable
#   reason field. A failure row does not need a version: when the caller
#   runs on a machine without vllm installed, VLLM_VERSION degrades to
#   "unknown" instead of aborting the run.
#
# Set PIN_VLLM_DRY_RUN=1 to resolve everything and print the record that
# would be written without cloning, committing or pushing anything. Use
# it to exercise this script from pull-request CI.
#
# Two files are maintained on the dedicated branch:
#   tested_vllm_versions.csv  -- JSON Lines, append-only history. Each line
#                                is one self-contained record. Now includes
#                                "status" and "reason" fields.
#   latest_tested_vllm.txt    -- Plain text, single line: the most recent
#                                *verified* version. Only overwritten when
#                                PIN_VLLM_STATUS=tested. Consumers that
#                                just `head -n1` keep working; trailing
#                                key=value lines let new consumers skip
#                                the live GitHub API lookup entirely.
#                                When OS_PLATFORM is set the name becomes
#                                latest_tested_vllm_<os>.txt, because each
#                                OS resolves its own vllm nightly and a
#                                shared file would let one OS clobber the
#                                other. Buildkite leaves OS_PLATFORM empty
#                                and keeps the original filename.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export GIT_TERMINAL_PROMPT=0

# ── Resolve CI platform & per-platform defaults ─────────────────────────
CI_PLATFORM="${CI_PLATFORM:-buildkite}"

case "${CI_PLATFORM}" in
    buildkite)
        PIN_VLLM_BRANCH="${PIN_VLLM_BRANCH:-buildkite_latest_tested_vllm}"
        ;;
    github_actions)
        PIN_VLLM_BRANCH="${PIN_VLLM_BRANCH:-github_nightly_tested_vllm}"
        ;;
    *)
        echo "[ERROR] unknown CI_PLATFORM '${CI_PLATFORM}';" \
             "expected buildkite or github_actions" >&2
        exit 1
        ;;
esac

# ── Pin status & reason ─────────────────────────────────────────────────
# Default to "tested" so existing buildkite callers (which only invoke
# this script on success) stay unchanged. Resolved before the version
# lookup below because it decides whether a missing version is fatal.
PIN_VLLM_STATUS="${PIN_VLLM_STATUS:-tested}"
PIN_VLLM_REASON="${PIN_VLLM_REASON:-}"

# Per-platform OS identifier (e.g. ubuntu-22.04, macos-latest, or
# empty for buildkite which runs on a single OS).
OS_PLATFORM="${OS_PLATFORM:-}"

# ── Resolve the version that's actually installed ────────────────────
# `|| true` keeps `set -e` from killing the script when vllm is absent:
# a failure row is recorded from an aggregator runner that never installs
# vllm, and losing that row would also lose the failure notification.
if [[ -z "${VLLM_VERSION:-}" ]]; then
    VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)' \
        2>/dev/null || true)"
fi
if [[ -z "${VLLM_VERSION}" ]]; then
    if [[ "${PIN_VLLM_STATUS}" == "tested" ]]; then
        echo "[ERROR] could not read vllm.__version__ from the live env" >&2
        exit 1
    fi
    echo "[WARN] no vllm version available; recording it as 'unknown'" >&2
    VLLM_VERSION="unknown"
fi
echo "Verified vLLM version: ${VLLM_VERSION}"

# ── Resolve commit SHAs so consumers don't need to call any external API ─
# The PEP 440 local version after `+g` is the short commit SHA, e.g.
# 0.23.1rc1.dev508+gc6dd32a81 -> c6dd32a81. Expand it to the full 40-char
# SHA via the public GitHub commits API; we already have GITHUB_TOKEN in
# the env (5000 req/h). The full SHA gives us the permanent
# wheels.vllm.ai/<full-sha>/<cuda>/ archive URL, which keeps working even
# after the rolling nightly index has dropped the wheel.
VLLM_SHORT_SHA="${VLLM_VERSION##*+g}"
if [[ "${VLLM_SHORT_SHA}" == "${VLLM_VERSION}" \
        || ! "${VLLM_SHORT_SHA}" =~ ^[0-9a-f]+$ ]]; then
    VLLM_SHORT_SHA=""
fi

VLLM_FULL_SHA=""
if [[ -n "${VLLM_SHORT_SHA}" ]]; then
    gh_auth_args=()
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        gh_auth_args=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi
    for attempt in 1 2 3; do
        VLLM_FULL_SHA="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            -H "Accept: application/vnd.github+json" \
            "${gh_auth_args[@]+"${gh_auth_args[@]}"}" \
            "https://api.github.com/repos/vllm-project/vllm/commits/${VLLM_SHORT_SHA}" \
            2>/dev/null \
            | jq -r '.sha // empty')" || true
        if [[ "${VLLM_FULL_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
            break
        fi
        VLLM_FULL_SHA=""
        echo "[INFO] GitHub commit lookup attempt ${attempt} for" \
             "${VLLM_SHORT_SHA} returned no SHA; retrying..." >&2
        sleep 2
    done
fi

if [[ -n "${VLLM_FULL_SHA}" ]]; then
    VLLM_ARCHIVE_INDEX="https://wheels.vllm.ai/${VLLM_FULL_SHA}/cu130"
    echo "Resolved full SHA: ${VLLM_FULL_SHA}"
    echo "Archive index:     ${VLLM_ARCHIVE_INDEX}"
else
    VLLM_ARCHIVE_INDEX=""
    echo "[WARN] could not resolve full SHA for short SHA" \
         "'${VLLM_SHORT_SHA:-<none>}'; archive_index_url will be empty" \
         "and consumers will fall back to live API lookup" >&2
fi

CI_REPO="LMCache/LMCache"
CI_BRANCH="${PIN_VLLM_BRANCH}"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    CI_REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${CI_REPO}.git"
else
    echo "[WARN] GITHUB_TOKEN not set — push will likely fail" >&2
    CI_REPO_URL="https://github.com/${CI_REPO}.git"
fi

WORK_DIR="/tmp/pin_vllm_$$"
trap 'rm -rf "${WORK_DIR}"' EXIT

PIN_VLLM_DRY_RUN="${PIN_VLLM_DRY_RUN:-0}"

if [[ "${PIN_VLLM_DRY_RUN}" == "1" ]]; then
    echo "--- [DRY-RUN] skipping clone of ${CI_REPO} ${CI_BRANCH}"
    mkdir -p "${WORK_DIR}"
elif ! git clone --depth=1 --branch "${CI_BRANCH}" "${CI_REPO_URL}" \
        "${WORK_DIR}" 2>/dev/null; then
    echo "--- Preparing ${CI_BRANCH} branch from ${CI_REPO}"
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
# Each OS resolves its own nightly, so give each one its own pointer file
# instead of letting the last writer win. Buildkite keeps the plain name.
if [[ -n "${OS_PLATFORM}" ]]; then
    LATEST_FILE="${WORK_DIR}/latest_tested_vllm_${OS_PLATFORM}.txt"
else
    LATEST_FILE="${WORK_DIR}/latest_tested_vllm.txt"
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Resolve CI metadata (build url / number / commit) per platform ──────
case "${CI_PLATFORM}" in
    buildkite)
        BUILD_URL="${BUILD_URL:-${BUILDKITE_BUILD_URL:-}}"
        BUILD_NUMBER="${BUILD_NUMBER:-${BUILDKITE_BUILD_NUMBER:-}}"
        COMMIT_SHA="${COMMIT_SHA:-${BUILDKITE_COMMIT:-}}"
        ;;
    github_actions)
        BUILD_NUMBER="${BUILD_NUMBER:-${GITHUB_RUN_ID:-}}"
        if [[ -n "${GITHUB_RUN_ID:-}" ]]; then
            BUILD_URL="${BUILD_URL:-${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID}}"
        else
            BUILD_URL="${BUILD_URL:-}"
        fi
        COMMIT_SHA="${COMMIT_SHA:-${GITHUB_SHA:-}}"
        ;;
esac

# Final safety net: guarantee these are always defined regardless of the
# CI_PLATFORM branch taken above, so the `set -u` heredoc below never trips
# on an unbound variable.
BUILD_URL="${BUILD_URL:-}"
BUILD_NUMBER="${BUILD_NUMBER:-}"
COMMIT_SHA="${COMMIT_SHA:-}"

# Append-only history (JSON Lines). Built via python so quoting is safe.
# Use a quoted heredoc (<<'PY') to disable Bash expansion inside the script,
# and pass variables via the environment to avoid syntax errors from special
# characters in values like BUILD_URL.
TIMESTAMP="${TIMESTAMP}" \
VLLM_VERSION="${VLLM_VERSION}" \
VLLM_SHORT_SHA="${VLLM_SHORT_SHA}" \
VLLM_FULL_SHA="${VLLM_FULL_SHA}" \
VLLM_ARCHIVE_INDEX="${VLLM_ARCHIVE_INDEX}" \
BUILD_NUMBER="${BUILD_NUMBER}" \
BUILD_URL="${BUILD_URL}" \
COMMIT_SHA="${COMMIT_SHA}" \
PIN_VLLM_STATUS="${PIN_VLLM_STATUS}" \
PIN_VLLM_REASON="${PIN_VLLM_REASON}" \
OS_PLATFORM="${OS_PLATFORM}" \
CI_PLATFORM="${CI_PLATFORM}" \
python - "$HISTORY_FILE" <<'PY'
import json, os, sys
path = sys.argv[1]
record = {
    "timestamp": os.environ.get("TIMESTAMP", ""),
    "vllm_version": os.environ.get("VLLM_VERSION", ""),
    "vllm_short_sha": os.environ.get("VLLM_SHORT_SHA", ""),
    "vllm_full_sha": os.environ.get("VLLM_FULL_SHA", ""),
    "archive_index_url": os.environ.get("VLLM_ARCHIVE_INDEX", ""),
    "build_number": os.environ.get("BUILD_NUMBER", ""),
    "build_url": os.environ.get("BUILD_URL", ""),
    "commit": os.environ.get("COMMIT_SHA", ""),
    "status": os.environ.get("PIN_VLLM_STATUS", "tested"),
    "reason": os.environ.get("PIN_VLLM_REASON", ""),
    "os_platform": os.environ.get("OS_PLATFORM", ""),
    "ci_platform": os.environ.get("CI_PLATFORM", "buildkite"),
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PY

# Latest pointer — only overwritten when the verification actually passed.
# Consumers that just `head -n1` keep working; trailing key=value lines
# let new consumers skip the live GitHub API lookup entirely.
if [[ "${PIN_VLLM_STATUS}" == "tested" ]]; then
    {
        printf '%s\n' "${VLLM_VERSION}"
        printf 'short_sha=%s\n' "${VLLM_SHORT_SHA}"
        printf 'full_sha=%s\n' "${VLLM_FULL_SHA}"
        printf 'archive_index_url=%s\n' "${VLLM_ARCHIVE_INDEX}"
    } > "${LATEST_FILE}"
fi

# ── Commit + push ───────────────────────────────────────────────────────
cd "${WORK_DIR}"

if [[ "${PIN_VLLM_DRY_RUN}" == "1" ]]; then
    echo "--- [DRY-RUN] would append to $(basename "${HISTORY_FILE}"):"
    tail -n1 "${HISTORY_FILE}"
    if [[ -f "${LATEST_FILE}" ]]; then
        echo "--- [DRY-RUN] would write $(basename "${LATEST_FILE}"):"
        cat "${LATEST_FILE}"
    fi
    echo "--- [DRY-RUN] nothing pushed to ${CI_REPO} ${CI_BRANCH}"
    exit 0
fi

git add -A

if git diff --cached --quiet 2>/dev/null; then
    echo "No changes to commit (version unchanged?)."
    exit 0
fi

COMMIT_MSG="Pin vLLM nightly (${PIN_VLLM_STATUS})"
if [[ "${PIN_VLLM_STATUS}" != "tested" ]]; then
    COMMIT_MSG="${COMMIT_MSG}: ${VLLM_VERSION} [${OS_PLATFORM:-unknown}]"
else
    COMMIT_MSG="${COMMIT_MSG}: ${VLLM_VERSION}"
fi

git -c user.email="ci@lmcache.ai" -c user.name="LMCache CI" \
    commit -m "${COMMIT_MSG}"

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
