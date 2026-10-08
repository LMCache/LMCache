#!/usr/bin/env bash
# Resolve the SGLang revision to install for this build.
#
# Resolution order (first non-empty value wins):
#   1. PINNED_SGLANG_VERSION/PINNED_SGLANG_SHA environment overrides.
#   2. With USE_PINNED_SGLANG=true, latest_tested_sglang.txt from the
#      buildkite_latest_tested_sglang branch.
#   3. With USE_PINNED_SGLANG=false, the newest official SGLang nightly.
#   4. Empty values, allowing normal CI to fall back to SGLang main when its
#      tested-pin file is temporarily unavailable.
#
# The pin file starts with the nightly version and may contain metadata:
#   0.5.21.dev20260928+gcc012abd21
#   short_sha=cc012abd21
#   full_sha=<40-character-commit-sha>
#   index_url=https://sgl-project.github.io/whl/cu130
#
# Set USE_PINNED_SGLANG=false for the canary build. Normal tested-pin lookup
# never fails setup solely because the tracking file is unavailable, whereas
# the canary fails if it cannot identify the newest nightly.

PINNED_SGLANG_VERSION="${PINNED_SGLANG_VERSION:-}"
PINNED_SGLANG_SHA="${PINNED_SGLANG_SHA:-}"
PINNED_SGLANG_INDEX_URL="${PINNED_SGLANG_INDEX_URL:-}"
USE_PINNED_SGLANG="${USE_PINNED_SGLANG:-true}"

LMCACHE_SGLANG_PIN_URL="${LMCACHE_SGLANG_PIN_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_sglang/latest_tested_sglang.txt}"
SGLANG_NIGHTLY_INDEX_URL="${SGLANG_NIGHTLY_INDEX_URL:-https://sgl-project.github.io/whl/cu130}"
SGLANG_NIGHTLY_PROJECT_URL="${SGLANG_NIGHTLY_PROJECT_URL:-${SGLANG_NIGHTLY_INDEX_URL%/}/sglang/}"

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

if [[ -z "${PINNED_SGLANG_VERSION}" \
        && -z "${PINNED_SGLANG_SHA}" \
        && "${USE_PINNED_SGLANG}" == "false" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] curl is required to resolve the latest SGLang nightly" >&2
        exit 1
    fi

    if ! index_html="$(curl -fsSL --retry 3 --retry-all-errors \
            --connect-timeout 5 --max-time 30 \
            "${SGLANG_NIGHTLY_PROJECT_URL}")"; then
        echo "[ERROR] Could not fetch the SGLang nightly index" >&2
        exit 1
    fi

    if ! metadata="$(python3 -c '
import re
import sys

html = sys.stdin.read().replace("%2B", "+").replace("%2b", "+")
pattern = re.compile(
    r"sglang-(?P<version>"
    r"(?P<major>[0-9]+)\."
    r"(?P<minor>[0-9]+)\."
    r"(?P<patch>[0-9]+)\.dev"
    r"(?P<date>[0-9]+)\+g(?P<sha>[0-9a-fA-F]+)"
    r")-cp[0-9]+-cp[0-9]+-[^\"<>\s]+\.whl"
)

candidates = {}
for match in pattern.finditer(html):
    version = match.group("version")
    candidates[version] = (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        int(match.group("date")),
        match.group("sha").lower(),
    )

if not candidates:
    raise SystemExit("No SGLang nightly wheel found in the official index")

version, fields = max(candidates.items(), key=lambda item: item[1][:4])
print(f"{version}\t{fields[4]}")
' <<< "${index_html}")"; then
        echo "[ERROR] Could not parse the SGLang nightly index" >&2
        exit 1
    fi

    IFS=$'\t' read -r PINNED_SGLANG_VERSION PINNED_SGLANG_SHA \
        <<< "${metadata}"
    if [[ -z "${PINNED_SGLANG_VERSION}" \
            || ! "${PINNED_SGLANG_SHA}" =~ ^[0-9a-f]{7,40}$ ]]; then
        echo "[ERROR] Invalid SGLang nightly metadata: ${metadata}" >&2
        exit 1
    fi
    PINNED_SGLANG_INDEX_URL="${SGLANG_NIGHTLY_INDEX_URL%/}"
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
export USE_PINNED_SGLANG

if [[ -n "${PINNED_SGLANG_VERSION}" || -n "${PINNED_SGLANG_SHA}" ]]; then
    if [[ -n "${PINNED_SGLANG_VERSION}" ]]; then
        echo "[resolve-pinned-sglang] Resolved SGLang version:" \
             "${PINNED_SGLANG_VERSION}" >&2
    fi
    echo "[resolve-pinned-sglang] Source revision:" \
         "${PINNED_SGLANG_SHA:-unknown}" >&2
else
    echo "[resolve-pinned-sglang] No pinned SGLang; using upstream main" >&2
fi
