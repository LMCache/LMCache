#!/usr/bin/env bash
# Publish a successfully verified SGLang nightly to its tracking branch.
#
# Required inputs:
#   SGLANG_VERSION       Nightly version discovered before the test.
#   SGLANG_SHORT_SHA     Source revision encoded by that nightly.
#
# Optional inputs:
#   SGLANG_FULL_SHA      Avoids the GitHub API lookup when already known.
#   SGLANG_INDEX_URL     Official nightly index containing the wheel.
#   PIN_SGLANG_BRANCH    Tracking branch to update.
#   PIN_SGLANG_REPO_URL  Repository or mirror containing the tracking branch.
#   PIN_SGLANG_DRY_RUN=1 Print the files without cloning or pushing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SGLANG_VERSION="${SGLANG_VERSION:?SGLANG_VERSION is required}"
SGLANG_SHORT_SHA="${SGLANG_SHORT_SHA:?SGLANG_SHORT_SHA is required}"
SGLANG_FULL_SHA="${SGLANG_FULL_SHA:-}"
SGLANG_INDEX_URL="${SGLANG_INDEX_URL:-https://sgl-project.github.io/whl/cu130}"
PIN_SGLANG_BRANCH="${PIN_SGLANG_BRANCH:-buildkite_latest_tested_sglang}"
PIN_SGLANG_REPO_URL="${PIN_SGLANG_REPO_URL:-}"
PIN_SGLANG_DRY_RUN="${PIN_SGLANG_DRY_RUN:-0}"

if [[ ! "${SGLANG_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+\+g[0-9a-f]{7,40}$ ]]; then
    echo "[ERROR] Invalid SGLang nightly version: ${SGLANG_VERSION}" >&2
    exit 1
fi

if [[ ! "${SGLANG_SHORT_SHA}" =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "[ERROR] Invalid SGLang short SHA: ${SGLANG_SHORT_SHA}" >&2
    exit 1
fi

if [[ "${SGLANG_VERSION##*+g}" != "${SGLANG_SHORT_SHA}" ]]; then
    echo "[ERROR] SGLang version and short SHA refer to different commits" >&2
    exit 1
fi

if [[ -z "${SGLANG_FULL_SHA}" ]]; then
    github_headers=(-H "Accept: application/vnd.github+json")
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        github_headers+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi

    for attempt in 1 2 3; do
        SGLANG_FULL_SHA="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            "${github_headers[@]}" \
            "https://api.github.com/repos/sgl-project/sglang/commits/${SGLANG_SHORT_SHA}" \
            2>/dev/null | jq -r '.sha // empty')" || true
        if [[ "${SGLANG_FULL_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
            break
        fi
        SGLANG_FULL_SHA=""
        echo "[WARN] Could not expand SGLang SHA on attempt ${attempt}/3" >&2
        sleep 2
    done
fi

if [[ ! "${SGLANG_FULL_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[ERROR] Could not resolve the full SGLang commit SHA" >&2
    exit 1
fi

if [[ "${SGLANG_FULL_SHA}" != "${SGLANG_SHORT_SHA}"* ]]; then
    echo "[ERROR] Full SHA does not match short SHA ${SGLANG_SHORT_SHA}" >&2
    exit 1
fi

if [[ -n "${PIN_SGLANG_REPO_URL}" ]]; then
    repo_url="${PIN_SGLANG_REPO_URL}"
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/LMCache/LMCache.git"
else
    repo_url="https://github.com/LMCache/LMCache.git"
    if [[ "${PIN_SGLANG_DRY_RUN}" != "1" ]]; then
        echo "[WARN] GITHUB_TOKEN is unset; pushing will probably fail" >&2
    fi
fi

work_dir="$(mktemp -d /tmp/pin-sglang.XXXXXX)"
trap 'rm -rf "${work_dir}"' EXIT

if [[ "${PIN_SGLANG_DRY_RUN}" == "1" ]]; then
    echo "--- [DRY-RUN] Skipping clone of ${PIN_SGLANG_BRANCH}"
else
    git clone --depth=1 --branch "${PIN_SGLANG_BRANCH}" \
        "${repo_url}" "${work_dir}"
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
build_number="${BUILDKITE_BUILD_NUMBER:-}"
build_url="${BUILDKITE_BUILD_URL:-}"
lmcache_commit="${BUILDKITE_COMMIT:-}"
history_file="${work_dir}/tested_runtimes.jsonl"
latest_file="${work_dir}/latest_tested_sglang.txt"

TIMESTAMP="${timestamp}" \
SGLANG_VERSION="${SGLANG_VERSION}" \
SGLANG_SHORT_SHA="${SGLANG_SHORT_SHA}" \
SGLANG_FULL_SHA="${SGLANG_FULL_SHA}" \
SGLANG_INDEX_URL="${SGLANG_INDEX_URL}" \
BUILD_NUMBER="${build_number}" \
BUILD_URL="${build_url}" \
LMCACHE_COMMIT="${lmcache_commit}" \
python3 - "${history_file}" <<'PY'
import json
import os
import sys

record = {
    "backend": "cuda",
    "runtime_id": "linux-cuda-source",
    "installation_form": "source",
    "status": "tested",
    "validated_at": os.environ["TIMESTAMP"],
    "validator": "buildkite",
    "build_number": os.environ["BUILD_NUMBER"],
    "build_url": os.environ["BUILD_URL"],
    "commit": os.environ["LMCACHE_COMMIT"],
    "sglang_version": os.environ["SGLANG_VERSION"],
    "sglang_short_sha": os.environ["SGLANG_SHORT_SHA"],
    "sglang_full_sha": os.environ["SGLANG_FULL_SHA"],
    "nightly_index_url": os.environ["SGLANG_INDEX_URL"],
}
with open(sys.argv[1], "a", encoding="utf-8") as history:
    history.write(json.dumps(record, sort_keys=True) + "\n")
PY

{
    printf '%s\n' "${SGLANG_VERSION}"
    printf 'short_sha=%s\n' "${SGLANG_SHORT_SHA}"
    printf 'full_sha=%s\n' "${SGLANG_FULL_SHA}"
    printf 'index_url=%s\n' "${SGLANG_INDEX_URL}"
} > "${latest_file}"

if [[ "${PIN_SGLANG_DRY_RUN}" == "1" ]]; then
    echo "--- [DRY-RUN] Would append to tested_runtimes.jsonl:"
    tail -n 1 "${history_file}"
    echo "--- [DRY-RUN] Would write latest_tested_sglang.txt:"
    cat "${latest_file}"
    exit 0
fi

record="$(tail -n 1 "${history_file}")"
latest_content="$(cat "${latest_file}")"
commit_message="Pin SGLang nightly: ${SGLANG_VERSION}"

commit_update() {
    git -C "${work_dir}" add latest_tested_sglang.txt tested_runtimes.jsonl
    git -C "${work_dir}" \
        -c user.email="ci@lmcache.ai" \
        -c user.name="LMCache CI" \
        commit --signoff -m "${commit_message}"
}

commit_update

for attempt in 1 2 3; do
    if git -C "${work_dir}" push origin "HEAD:${PIN_SGLANG_BRANCH}"; then
        echo "--- Pinned SGLang ${SGLANG_VERSION} successfully"
        git -C "${work_dir}" log --oneline -1
        exit 0
    fi

    if [[ "${attempt}" -eq 3 ]]; then
        break
    fi
    echo "[WARN] Pin branch changed; replaying update (${attempt}/3)" >&2
    if ! git -C "${work_dir}" fetch origin \
            "+refs/heads/${PIN_SGLANG_BRANCH}:refs/remotes/origin/${PIN_SGLANG_BRANCH}"; then
        echo "[WARN] Could not refresh the pin branch; retrying push" >&2
        sleep 2
        continue
    fi
    git -C "${work_dir}" reset --hard "origin/${PIN_SGLANG_BRANCH}"
    printf '%s\n' "${record}" >> "${history_file}"
    printf '%s\n' "${latest_content}" > "${latest_file}"
    commit_update
done

echo "[ERROR] Failed to update ${PIN_SGLANG_BRANCH} after 3 attempts" >&2
exit 1
