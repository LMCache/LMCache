#!/usr/bin/env bash
# Resolve the vLLM nightly version that this build should install.
#
# Resolution order (first non-empty wins):
#   1. PINNED_VLLM_VERSION env var  -- explicit per-build override.
#   2. latest_tested_vllm.txt fetched from
#      https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/latest_tested_vllm.txt
#      -- the most recent vLLM nightly that the canary build verified.
#   3. Empty string -- caller falls back to "latest nightly".
#
# Toggles:
#   USE_PINNED_VLLM=false  -- skip step 2 (always probe the latest nightly).
#                             Used by the canary build itself, since pinning
#                             to its own previous result would defeat the
#                             purpose of a freshness check.
#
# Usage:
#   source .buildkite/k3_harness/resolve-pinned-vllm.sh
#   echo "Resolved: ${PINNED_VLLM_VERSION:-<unpinned, using nightly>}"
#
# After sourcing, PINNED_VLLM_VERSION is set (possibly empty) and exported.
# The script never fails the build: a missing/unreachable pin file just
# falls through to the unpinned path, mirroring the previous behaviour.

# Allow re-sourcing without "unbound variable" complaints under set -u.
PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-}"
USE_PINNED_VLLM="${USE_PINNED_VLLM:-true}"

# Override URL if you mirror the pin file elsewhere (e.g. an internal
# raw-file proxy for offline CI).
LMCACHE_VLLM_PIN_URL="${LMCACHE_VLLM_PIN_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/latest_tested_vllm.txt}"

if [[ -z "${PINNED_VLLM_VERSION}" && "${USE_PINNED_VLLM}" == "true" ]]; then
    if command -v curl >/dev/null 2>&1; then
        # 5s connect, 10s total -- pin lookup must never dominate setup time.
        fetched="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            "${LMCACHE_VLLM_PIN_URL}" 2>/dev/null || true)"
        # Strip whitespace/comments; keep the first non-empty, non-comment
        # line as the version string.
        PINNED_VLLM_VERSION="$(printf '%s\n' "${fetched}" \
            | sed -e 's/[[:space:]]\+$//' \
                  -e '/^[[:space:]]*$/d' \
                  -e '/^[[:space:]]*#/d' \
            | head -n1 || true)"
    fi
fi

export PINNED_VLLM_VERSION

if [[ -n "${PINNED_VLLM_VERSION}" ]]; then
    echo "[resolve-pinned-vllm] Pinned vLLM version: ${PINNED_VLLM_VERSION}" >&2
else
    echo "[resolve-pinned-vllm] No pinned vLLM; will install latest nightly" >&2
fi
