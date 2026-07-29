#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Aggregate per-platform nightly vLLM CPU verification results and
# either pin the version (all passed) or create/update a tracking
# issue (any platform failed).
#
# Expects artifacts under verify-results/*/verify_result.json (one
# subdirectory per platform, from download-artifact).
#
# Required env:
#   GITHUB_TOKEN         -- for gh CLI and pin-tested-vllm.sh push
#   GITHUB_REPOSITORY    -- "owner/repo"
#   GITHUB_SERVER_URL    -- e.g. https://github.com
#   GITHUB_RUN_ID        -- run identifier
#
# Reads from env (set by caller):
#   PIN_SCRIPT           -- path to pin-tested-vllm.sh

set -euo pipefail

# ── Parse per-platform artifacts ────────────────────────────────────
declare -A OS_STATUS OS_VERSION OS_REASON
for d in verify-results/*/; do
    f="${d}verify_result.json"
    if [ ! -f "$f" ]; then
        continue
    fi
    os="$(python3 -c "import json; print(json.load(open('$f'))['os'])")"
    st="$(python3 -c "import json; print(json.load(open('$f'))['status'])")"
    ver="$(python3 -c "import json; print(json.load(open('$f')).get('vllm_version',''))")"
    rsn="$(python3 -c "import json; print(json.load(open('$f')).get('reason',''))")"
    OS_STATUS["$os"]="$st"
    OS_VERSION["$os"]="$ver"
    OS_REASON["$os"]="$rsn"
    echo "  $os  status=$st  version=$ver"
done

total="${#OS_STATUS[@]}"
passed=0
failed_oss=""
success_oss=""
for os in "${!OS_STATUS[@]}"; do
    if [ "${OS_STATUS[$os]}" = "ok" ]; then
        passed=$((passed + 1))
        success_oss="$success_oss $os"
    else
        failed_oss="$failed_oss $os"
    fi
done

echo "=== Summary: $passed / $total platforms passed ==="
NOW="$(date -u +'%Y-%m-%d')"

if [ "$total" -eq 0 ]; then
    echo "::warning::No verify results found — both platforms may have"
    echo "failed before producing artifacts. Skipping pin/report."
    exit 0
fi

# ── All platforms passed: pin as tested ─────────────────────────────
if [ "$passed" -eq "$total" ]; then
    echo "All platforms passed — marking as tested."

    VLLM_VER=""
    for os in $success_oss; do
        VLLM_VER="${OS_VERSION[$os]}"
        [ -n "$VLLM_VER" ] && break
    done

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '::notice::[DRY-RUN] Would pin vLLM %s as tested\n' \
            "${VLLM_VER}"
        exit 0
    fi

    export CI_PLATFORM=github_actions
    export PIN_VLLM_STATUS=tested
    export VLLM_VERSION="${VLLM_VER}"
    bash "${PIN_SCRIPT}"

    printf '::notice::vLLM %s verified on all CPU platforms\n' \
        "${VLLM_VER}"

    EXISTING="$(gh issue list --repo "$GITHUB_REPOSITORY" \
        --search "Nightly CPU vLLM verify FAILED" --state open \
        --limit 1 --json number -q '.[0].number // ""')"
    if [ -n "$EXISTING" ]; then
        gh issue comment "$EXISTING" \
            --body "Resolved: all platforms passed on $NOW (vLLM ${VLLM_VER})."
        gh issue close "$EXISTING" --reason completed
    fi
    exit 0
fi

# ── Some platforms failed: record + issue ───────────────────────────
echo "Some platforms FAILED — recording failure rows."

BODY="One or more CPU platforms failed the nightly vLLM verification"
BODY="$BODY on **$NOW**.\n\n"
BODY="$BODY**Run:** $GITHUB_SERVER_URL/$GITHUB_REPOSITORY"
BODY="$BODY/actions/runs/$GITHUB_RUN_ID\n\n"
BODY="$BODY| Platform | Status | vLLM Version |\n"
BODY="$BODY| --- | --- | --- |\n"
for os in $success_oss $failed_oss; do
    st="${OS_STATUS[$os]}"
    ver="${OS_VERSION[$os]}"
    emoji=":white_check_mark:"
    [ "$st" != "ok" ] && emoji=":x:"
    BODY="$BODY| $os | $emoji $st | $ver |\n"
done
BODY="$BODY\n**Failure details:**\n"
for os in $failed_oss; do
    BODY="$BODY- **$os**: ${OS_REASON[$os]}\n"
done
BODY="$BODY\n---\n"
BODY="$BODY*This issue was auto-created by the nightly vLLM CPU"
BODY="$BODY verification workflow.*"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "::notice::[DRY-RUN] Would record failures and create issue:"
    printf '%b\n' "$BODY"
    exit 0
fi

for os in $failed_oss; do
    export CI_PLATFORM=github_actions
    export PIN_VLLM_STATUS=failed
    export OS_PLATFORM="$os"
    export PIN_VLLM_REASON="${OS_REASON[$os]}"
    bash "${PIN_SCRIPT}"
done

ISSUE_TITLE="Nightly CPU vLLM verify FAILED — $NOW"
EXISTING="$(gh issue list --repo "$GITHUB_REPOSITORY" \
    --search "Nightly CPU vLLM verify FAILED" --state open \
    --limit 1 --json number -q '.[0].number // ""' 2>/dev/null || true)"
if [ -n "$EXISTING" ]; then
    gh issue comment "$EXISTING" --body "$(printf '%b' "$BODY")"
    printf '::warning::Commented on existing issue #%s\n' "$EXISTING"
else
    gh issue create --title "$ISSUE_TITLE" \
        --body "$(printf '%b' "$BODY")" \
        --label "CI" --label "bug"
    echo "::warning::Created new vLLM verify failure issue"
fi
