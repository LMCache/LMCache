#!/usr/bin/env bash
# Resolve the pinned vLLM wheel version and XPU runtime image for this build.
#
# Resolution order (first non-empty wins):
#   1. PINNED_VLLM_VERSION env var  -- explicit per-build override.
#   2. latest_tested_vllm.txt fetched from
#      https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/latest_tested_vllm.txt
#      -- the most recent vLLM nightly that the canary build verified.
#   3. Empty string -- caller falls back to "latest nightly".
#
# The pin file's first non-blank, non-comment line is the bare version
# (kept this way so older `head -n1` consumers still work). Subsequent
# key=value lines carry pre-resolved metadata so consumers can install
# the wheel without making any extra API calls:
#   short_sha=<short-commit-sha>
#   full_sha=<40-char-commit-sha>
#   archive_index_url=https://wheels.vllm.ai/<full-sha>/cu130
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
#   echo "Archive : ${PINNED_VLLM_ARCHIVE_INDEX_URL:-<none>}"
#
# After sourcing the following are set (possibly empty) and exported:
#   PINNED_VLLM_VERSION         -- e.g. 0.23.1rc1.dev508+gc6dd32a81
#   PINNED_VLLM_FULL_SHA        -- 40-char commit SHA, if recorded in pin
#   PINNED_VLLM_ARCHIVE_INDEX_URL
#                               -- permanent PEP 503 simple index, if recorded
#
# The script never fails the build: a missing/unreachable pin file just
# falls through to the unpinned path, mirroring the previous behaviour.
#
# XPU image resolution follows the same pattern. It selects the most recent
# successful Buildkite XPU image record from tested_runtimes.jsonl and exports
# PINNED_XPU_IMAGE. Without a pin, it falls back to the upstream nightly tag so
# Buildkite can validate and pin the first candidate.

# Allow re-sourcing without "unbound variable" complaints under set -u.
PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-}"
PINNED_VLLM_ARCHIVE_INDEX_URL="${PINNED_VLLM_ARCHIVE_INDEX_URL:-}"
PINNED_VLLM_FULL_SHA="${PINNED_VLLM_FULL_SHA:-}"
USE_PINNED_VLLM="${USE_PINNED_VLLM:-true}"

# Override URL if you mirror the pin file elsewhere (e.g. an internal
# raw-file proxy for offline CI).
LMCACHE_VLLM_PIN_URL="${LMCACHE_VLLM_PIN_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/latest_tested_vllm.txt}"

if [[ -z "${PINNED_VLLM_VERSION}" && "${USE_PINNED_VLLM}" == "true" ]]; then
    if command -v curl >/dev/null 2>&1; then
        # 5s connect, 10s total -- pin lookup must never dominate setup time.
        fetched="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            "${LMCACHE_VLLM_PIN_URL}" 2>/dev/null || true)"
        # The pin file's first non-blank, non-comment line is the bare
        # version (back-compat for older readers); subsequent key=value
        # lines carry resolved metadata so consumers can skip the live
        # GitHub API lookup. A single awk pass extracts everything.
        # Empty / missing keys are tolerated -- we'll just fall back to
        # resolving them on demand later in the pipeline.
        eval "$(printf '%s\n' "${fetched}" | awk '
            BEGIN { ver=""; idx=""; sha="" }
            /^[[:space:]]*(#|$)/ { next }
            ver == "" {
                line=$0
                sub(/[[:space:]]+$/, "", line)
                ver=line
                next
            }
            /^archive_index_url=/ {
                v=$0; sub(/^archive_index_url=/, "", v); idx=v; next
            }
            /^full_sha=/ {
                v=$0; sub(/^full_sha=/, "", v); sha=v; next
            }
            END {
                printf("PINNED_VLLM_VERSION=%s\n", ver)
                printf("PINNED_VLLM_ARCHIVE_INDEX_URL=%s\n", idx)
                printf("PINNED_VLLM_FULL_SHA=%s\n", sha)
            }
        ')"
    fi
fi

export PINNED_VLLM_VERSION
export PINNED_VLLM_ARCHIVE_INDEX_URL
export PINNED_VLLM_FULL_SHA

if [[ -n "${PINNED_VLLM_VERSION}" ]]; then
    echo "[resolve-pinned-vllm] Pinned vLLM version: ${PINNED_VLLM_VERSION}" >&2
    if [[ -n "${PINNED_VLLM_ARCHIVE_INDEX_URL}" ]]; then
        echo "[resolve-pinned-vllm] Archive index:" \
             "${PINNED_VLLM_ARCHIVE_INDEX_URL}" >&2
    fi
else
    echo "[resolve-pinned-vllm] No pinned vLLM; will install latest nightly" >&2
fi

# XPU runtime image pin. Kept in this resolver so wheel and image runtime
# coordinates share one loader and one append-only history source.
PINNED_XPU_IMAGE="${PINNED_XPU_IMAGE:-}"
USE_PINNED_XPU_RUNTIME="${USE_PINNED_XPU_RUNTIME:-true}"
LMCACHE_XPU_RUNTIME_HISTORY_URL="${LMCACHE_XPU_RUNTIME_HISTORY_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/tested_runtimes.jsonl}"
XPU_NIGHTLY_IMAGE="${XPU_NIGHTLY_IMAGE:-vllm/vllm-openai-xpu:nightly}"

if [[ -z "${PINNED_XPU_IMAGE}" && "${USE_PINNED_XPU_RUNTIME}" == "true" ]]; then
    history="$(curl -fsSL --connect-timeout 5 --max-time 15 "${LMCACHE_XPU_RUNTIME_HISTORY_URL}" 2>/dev/null || true)"
    if [[ -n "${history}" ]]; then
        PINNED_XPU_IMAGE="$(printf '%s\n' "${history}" | python3 -c '
import json
import sys

latest = ""
for line in sys.stdin:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (
        record.get("backend") == "xpu"
        and record.get("runtime_id") == "linux-intel-xpu"
        and record.get("installation_form") == "image"
        and record.get("status") == "tested"
        and record.get("validator") == "buildkite"
        and isinstance(record.get("image_ref"), str)
        and "@sha256:" in record["image_ref"]
    ):
        latest = record["image_ref"]
print(latest)
')"
    fi
fi

if [[ -z "${PINNED_XPU_IMAGE}" ]]; then
    PINNED_XPU_IMAGE="${XPU_NIGHTLY_IMAGE}"
    echo "[resolve-pinned-vllm] No Buildkite-validated XPU image; using ${PINNED_XPU_IMAGE}" >&2
else
    echo "[resolve-pinned-vllm] Using pinned XPU image: ${PINNED_XPU_IMAGE}" >&2
fi
export PINNED_XPU_IMAGE
