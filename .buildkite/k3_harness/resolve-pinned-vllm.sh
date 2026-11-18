#!/usr/bin/env bash
# Resolve the pinned vLLM wheel version and the XPU and RBLN runtime images
# for this build.
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
#   PINNED_XPU_IMAGE            -- XPU runtime image (never empty)
#   PINNED_RBLN_IMAGE           -- RBLN rbln-serve image (never empty)
#
# The script never fails the build: a missing/unreachable pin file just
# falls through to the unpinned path, mirroring the previous behaviour.
#
# Image resolution selects each runtime's latest Buildkite-verified digest
# from tested_runtimes.jsonl. Without one, it uses that runtime's configured
# fallback image.

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

# Image runtime pins. Kept in this resolver so wheel and image runtime
# coordinates share one loader and one append-only history source.
PINNED_XPU_IMAGE="${PINNED_XPU_IMAGE:-}"
USE_PINNED_XPU_RUNTIME="${USE_PINNED_XPU_RUNTIME:-true}"
XPU_RELEASE_IMAGE="${XPU_RELEASE_IMAGE:-vllm/vllm-openai-xpu:v0.26.0}"
PINNED_RBLN_IMAGE="${PINNED_RBLN_IMAGE:-}"
USE_PINNED_RBLN_RUNTIME="${USE_PINNED_RBLN_RUNTIME:-true}"
# Used until the RBLN nightly lane pins its first verified image.
RBLN_FALLBACK_IMAGE="${RBLN_FALLBACK_IMAGE:-harbor.k8s.rebellions.in/rebellions-sw/rbln-serve:0.12.0rc0-ubuntu24.04-py3.12}"
LMCACHE_RUNTIME_HISTORY_URL="${LMCACHE_RUNTIME_HISTORY_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/tested_runtimes.jsonl}"

# Print the image_ref of the last Buildkite-tested, digest-pinned record of
# one image runtime in a tested_runtimes.jsonl history.
latest_image_from_history() {
    local history="$1" backend="$2" runtime_id="$3"

    printf '%s\n' "${history}" | awk \
        -v backend="\"backend\": \"${backend}\"" \
        -v runtime_id="\"runtime_id\": \"${runtime_id}\"" '
        index($0, backend) &&
        index($0, runtime_id) &&
        /"installation_form": "image"/ &&
        /"status": "tested"/ &&
        /"validator": "buildkite"/ &&
        /"image_ref": "[^"]*@sha256:/ {
            image_ref = $0
            sub(/^.*"image_ref": "/, "", image_ref)
            sub(/".*$/, "", image_ref)
            latest = image_ref
        }
        END { print latest }
    '
}

runtime_history=""
if [[ ( -z "${PINNED_XPU_IMAGE}" && "${USE_PINNED_XPU_RUNTIME}" == "true" ) \
    || ( -z "${PINNED_RBLN_IMAGE}" && "${USE_PINNED_RBLN_RUNTIME}" == "true" ) ]]; then
    runtime_history="$(curl -fsSL --connect-timeout 5 --max-time 15 "${LMCACHE_RUNTIME_HISTORY_URL}" 2>/dev/null || true)"
fi

if [[ -z "${PINNED_XPU_IMAGE}" && "${USE_PINNED_XPU_RUNTIME}" == "true" ]]; then
    PINNED_XPU_IMAGE="$(latest_image_from_history "${runtime_history}" xpu linux-intel-xpu)"
fi
if [[ -z "${PINNED_XPU_IMAGE}" ]]; then
    PINNED_XPU_IMAGE="${XPU_RELEASE_IMAGE}"
    echo "[resolve-pinned-vllm] No verified XPU nightly image; using XPU release ${PINNED_XPU_IMAGE}" >&2
else
    echo "[resolve-pinned-vllm] Using pinned XPU image: ${PINNED_XPU_IMAGE}" >&2
fi
export PINNED_XPU_IMAGE

if [[ -z "${PINNED_RBLN_IMAGE}" && "${USE_PINNED_RBLN_RUNTIME}" == "true" ]]; then
    PINNED_RBLN_IMAGE="$(latest_image_from_history "${runtime_history}" rbln linux-rebellions-rbln)"
fi
if [[ -z "${PINNED_RBLN_IMAGE}" ]]; then
    PINNED_RBLN_IMAGE="${RBLN_FALLBACK_IMAGE}"
    echo "[resolve-pinned-vllm] No verified RBLN nightly image; using fallback ${PINNED_RBLN_IMAGE}" >&2
else
    echo "[resolve-pinned-vllm] Using pinned RBLN image: ${PINNED_RBLN_IMAGE}" >&2
fi
export PINNED_RBLN_IMAGE
