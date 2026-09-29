#!/usr/bin/env bash
# Resolve the SGLang revision most recently verified by Buildkite.
#
# Resolution order (first non-empty value wins):
#   1. PINNED_SGLANG_VERSION/PINNED_SGLANG_SHA environment overrides.
#   2. latest_tested_sglang.txt from the buildkite_latest_tested_sglang branch.
#   3. Empty values, allowing the caller to fall back to SGLang main.
#
# The pin file starts with the nightly version and may contain metadata:
#   0.5.21.dev20260928+gcc012abd21
#   short_sha=cc012abd21
#   full_sha=<40-character-commit-sha>
#   index_url=https://sgl-project.github.io/whl/cu130
#
# Set USE_PINNED_SGLANG=false to skip the remote pin. This is intended for the
# future canary job that validates the newest nightly before updating the pin.
# The script never fails setup solely because the remote pin is unavailable.

PINNED_SGLANG_VERSION="${PINNED_SGLANG_VERSION:-}"
PINNED_SGLANG_SHA="${PINNED_SGLANG_SHA:-}"
PINNED_SGLANG_INDEX_URL="${PINNED_SGLANG_INDEX_URL:-}"
USE_PINNED_SGLANG="${USE_PINNED_SGLANG:-true}"

LMCACHE_SGLANG_PIN_URL="${LMCACHE_SGLANG_PIN_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_sglang/latest_tested_sglang.txt}"

if [[ -z "${PINNED_SGLANG_VERSION}" \
        && -z "${PINNED_SGLANG_SHA}" \
        && "${USE_PINNED_SGLANG}" == "true" ]]; then
    if command -v curl >/dev/null 2>&1; then
        fetched="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            "${LMCACHE_SGLANG_PIN_URL}" 2>/dev/null || true)"
        fetched_short_sha=""
        fetched_full_sha=""

        while IFS= read -r line; do
            line="${line%$'\r'}"
            [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue

            if [[ -z "${PINNED_SGLANG_VERSION}" ]]; then
                PINNED_SGLANG_VERSION="${line%${line##*[![:space:]]}}"
                continue
            fi

            case "${line}" in
                short_sha=*)
                    fetched_short_sha="${line#short_sha=}"
                    ;;
                full_sha=*)
                    fetched_full_sha="${line#full_sha=}"
                    ;;
                index_url=*)
                    PINNED_SGLANG_INDEX_URL="${line#index_url=}"
                    ;;
            esac
        done <<< "${fetched}"

        PINNED_SGLANG_SHA="${fetched_full_sha:-${fetched_short_sha}}"
    fi
fi

# Old or manually-authored pin files may contain only the nightly version.
# SGLang versions encode the source revision after "+g".
if [[ -z "${PINNED_SGLANG_SHA}" \
        && "${PINNED_SGLANG_VERSION}" =~ \+g([0-9a-fA-F]+)$ ]]; then
    PINNED_SGLANG_SHA="${BASH_REMATCH[1]}"
fi

export PINNED_SGLANG_VERSION
export PINNED_SGLANG_SHA
export PINNED_SGLANG_INDEX_URL

if [[ -n "${PINNED_SGLANG_VERSION}" || -n "${PINNED_SGLANG_SHA}" ]]; then
    if [[ -n "${PINNED_SGLANG_VERSION}" ]]; then
        echo "[resolve-pinned-sglang] Pinned SGLang version:" \
             "${PINNED_SGLANG_VERSION}" >&2
    fi
    echo "[resolve-pinned-sglang] Source revision:" \
         "${PINNED_SGLANG_SHA:-unknown}" >&2
else
    echo "[resolve-pinned-sglang] No pinned SGLang; using upstream main" >&2
fi
