#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Aggregate per-platform nightly vLLM CPU verification results, pin every
# platform that passed and open/update a tracking issue when any of them
# failed.
#
# Each OS resolves its own vllm nightly (the pytorch CPU index does not
# carry the same torch build for every platform), so results are recorded
# per OS rather than requiring one shared version across all of them.
# pin-tested-vllm.sh writes latest_tested_vllm_<os>.txt accordingly.
#
# Results come from cpu_device.yml, whose matrix is (os x model), so one
# OS contributes several legs. An OS is only pinned as tested when every
# one of its legs passed the full CPU device suite (server bench + e2e);
# "importable" is not enough.
#
# Expects artifacts under verify-results/*/verify_result.json (one
# subdirectory per matrix leg, from download-artifact).
#
# Required env:
#   GITHUB_TOKEN         -- for gh CLI and pin-tested-vllm.sh push
#   GITHUB_REPOSITORY    -- "owner/repo"
#   GITHUB_SERVER_URL    -- e.g. https://github.com
#   GITHUB_RUN_ID        -- run identifier
#
# Reads from env (set by caller):
#   PIN_SCRIPT           -- path to pin-tested-vllm.sh
#   EXPECTED_OSS         -- space separated OS list the matrix should have
#                           produced; a missing artifact is reported as a
#                           failure instead of being silently dropped
#   DRY_RUN              -- 1 to run the pin script in dry-run mode and
#                           skip all issue mutations

set -euo pipefail

ISSUE_SEARCH='Nightly CPU vLLM verify FAILED in:title'

# Find the open tracking issue, if any. Restricted to titles so we never
# touch an issue that merely mentions the phrase in its body.
find_tracking_issue() {
    gh issue list --repo "$GITHUB_REPOSITORY" \
        --search "$ISSUE_SEARCH" --state open \
        --limit 1 --json number -q '.[0].number // ""' 2>/dev/null || true
}

# ── Parse per-leg artifacts and fold them per OS ────────────────────
# OS_LEGS keeps the individual "model=status" verdicts for the report;
# OS_STATUS is the folded result: failed as soon as any leg failed.
declare -A OS_STATUS OS_VERSION OS_REASON OS_LEGS
for d in verify-results/*/; do
    f="${d}verify_result.json"
    if [ ! -f "$f" ]; then
        continue
    fi
    # Read all fields in a single python3 invocation (tab-separated) to
    # avoid spawning one interpreter per field.
    IFS=$'\t' read -r os model st ver rsn < <(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print("\t".join([
    d["os"],
    d.get("model", "-"),
    d["status"],
    d.get("vllm_version", ""),
    d.get("reason", ""),
]))
' "$f")
    OS_LEGS["$os"]="${OS_LEGS[$os]:-} ${model}=${st}"
    # Record a version from any leg that reported one, even a failing
    # one, so the report can name the build that broke.
    [ -n "$ver" ] && OS_VERSION["$os"]="$ver"

    # First failure decides the OS verdict and keeps its reason; a later
    # passing leg must not overwrite it.
    if [ "${OS_STATUS[$os]:-}" = "failed" ]; then
        continue
    fi
    OS_STATUS["$os"]="$st"
    OS_REASON["$os"]="$rsn"
done

# A cancelled job or a failed artifact upload leaves no result file at
# all. Treat that as a failure of the expected platform, otherwise the
# run would look like "everything that reported is green".
for os in ${EXPECTED_OSS:-}; do
    if [ -z "${OS_STATUS[$os]:-}" ]; then
        OS_STATUS["$os"]="failed"
        OS_VERSION["$os"]=""
        OS_REASON["$os"]="no verify result artifact produced on $os"
        OS_LEGS["$os"]=" none"
    fi
done

# Sort so log order, CSV append order and issue table order are stable
# across runs; bash iterates associative arrays in hash order.
ALL_OSS="$(printf '%s\n' "${!OS_STATUS[@]}" | sort)"

total=0
passed=0
failed_oss=""
success_oss=""
for os in $ALL_OSS; do
    total=$((total + 1))
    echo "  $os  status=${OS_STATUS[$os]}  version=${OS_VERSION[$os]:-}" \
        " legs:${OS_LEGS[$os]:-}"
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
    echo "::warning::No verify results found and EXPECTED_OSS is empty —" \
        "nothing to pin or report."
    exit 0
fi

# ── Record one row per platform ─────────────────────────────────────
# Every platform is pinned independently: a green ubuntu leg still
# publishes its version even when macOS is broken, and vice versa.
for os in $ALL_OSS; do
    # Record the observed version on failure rows too: the CSV is the
    # append-only history used to answer "which builds broke before", and
    # dropping it there left a useless "unknown". Only the latest_*.txt
    # pointer is gated on status, and the pin script still falls back to
    # "unknown" when no leg ever reported a version.
    pin_version="${OS_VERSION[$os]:-}"
    if [ "${OS_STATUS[$os]}" = "ok" ]; then
        pin_status=tested
        pin_reason=""
    else
        pin_status=failed
        pin_reason="${OS_REASON[$os]}"
    fi

    # A single bad platform must not abort the loop, or the issue below
    # would never be created.
    if ! CI_PLATFORM=github_actions \
        PIN_VLLM_STATUS="$pin_status" \
        PIN_VLLM_REASON="$pin_reason" \
        PIN_VLLM_DRY_RUN="${DRY_RUN:-0}" \
        OS_PLATFORM="$os" \
        VLLM_VERSION="$pin_version" \
        bash "${PIN_SCRIPT}"; then
        printf '::warning::Failed to record %s result for %s\n' \
            "$pin_status" "$os"
    fi
done

# ── All platforms passed: close any open tracking issue ─────────────
if [ "$passed" -eq "$total" ]; then
    printf '::notice::vLLM verified on all %s CPU platforms\n' "$total"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "::notice::[DRY-RUN] would close the tracking issue if open"
        exit 0
    fi

    EXISTING="$(find_tracking_issue)"
    if [ -n "$EXISTING" ]; then
        # Best-effort: the issue may already be closed/edited by a human,
        # which must not fail the (successful) pin run.
        gh issue comment "$EXISTING" \
            --body "Resolved: all platforms passed on $NOW." \
            2>/dev/null || true
        gh issue close "$EXISTING" --reason completed 2>/dev/null || true
    fi
    exit 0
fi

# ── Some platforms failed: build the report ─────────────────────────
BODY="One or more CPU platforms failed the nightly vLLM verification"
BODY="$BODY on **$NOW**.\n\n"
BODY="$BODY**Run:** $GITHUB_SERVER_URL/$GITHUB_REPOSITORY"
BODY="$BODY/actions/runs/$GITHUB_RUN_ID\n\n"
BODY="$BODY Each platform resolves its own vLLM nightly, so the versions"
BODY="$BODY below are not expected to match.\n\n"
BODY="$BODY| Platform | Status | vLLM Version | Legs |\n"
BODY="$BODY| --- | --- | --- | --- |\n"
for os in $ALL_OSS; do
    st="${OS_STATUS[$os]}"
    ver="${OS_VERSION[$os]:-}"
    emoji=":white_check_mark:"
    [ "$st" != "ok" ] && emoji=":x:"
    BODY="$BODY| $os | $emoji $st | ${ver:-n/a} |${OS_LEGS[$os]:-} |\n"
done
BODY="$BODY\n**Failure details:**\n"
for os in $failed_oss; do
    BODY="$BODY- **$os**: ${OS_REASON[$os]}\n"
done
BODY="$BODY\n---\n"
BODY="$BODY*This issue was auto-created by the nightly vLLM CPU"
BODY="$BODY verification workflow.*"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "::notice::[DRY-RUN] would create or update issue with body:"
    printf '%b\n' "$BODY"
    exit 0
fi

EXISTING="$(find_tracking_issue)"
if [ -n "$EXISTING" ]; then
    gh issue comment "$EXISTING" --body "$(printf '%b' "$BODY")"
    printf '::warning::Commented on existing issue #%s\n' "$EXISTING"
else
    gh issue create --title "Nightly CPU vLLM verify FAILED — $NOW" \
        --body "$(printf '%b' "$BODY")" \
        --label "ci/cd" --label "bug"
    echo "::warning::Created new vLLM verify failure issue"
fi
