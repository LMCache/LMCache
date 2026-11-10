#!/usr/bin/env bash
# Usage: bash wait-for-required-checks.sh OWNER/REPO PR_NUMBER EXPECTED_HEAD_SHA
# Exit 0 only when the fast checks admit this exact, still-current PR head.
# Callers decide whether a non-PR build needs a gate and supply trusted code.
set -euo pipefail

repository="${1:?GitHub repository is required}"
pr_number="${2:?PR number is required}"
expected_sha="${3:?Expected PR head SHA is required}"
timeout_seconds="${REQUIRED_CHECKS_TIMEOUT_SECONDS:-1800}"
poll_seconds="${REQUIRED_CHECKS_POLL_SECONDS:-10}"
api_url="${GITHUB_API_URL:-https://api.github.com}"

if [[ ! "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ||
      ! "$pr_number" =~ ^[1-9][0-9]*$ ||
      ! "$expected_sha" =~ ^[0-9a-f]{40}$ ||
      ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ||
      ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "required-checks: invalid repository, PR, SHA, or timeout" >&2
    exit 1
fi
for command in curl jq; do
    command -v "$command" >/dev/null || exit 1
done

deadline=$((SECONDS + timeout_seconds))
headers=(-H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28")
token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [[ -n "$token" ]]; then
    headers+=(-H "Authorization: Bearer ${token}")
else
    echo "required-checks: no GitHub token; anonymous API rate limits apply" >&2
fi

# Keep HTTP retries inside the overall deadline. Authentication and permission
# errors need operator action; retry only transport errors, 429, and 5xx.
github_api() {
    local path="$1" attempt remaining request_timeout response http_status
    for attempt in 1 2 3; do
        remaining=$((deadline - SECONDS))
        if (( remaining <= 0 )); then
            echo "required-checks: API deadline exceeded" >&2
            return 1
        fi
        request_timeout=$remaining
        if (( request_timeout > 30 )); then request_timeout=30; fi
        if response="$(curl --silent --show-error --location \
            --connect-timeout "$request_timeout" --max-time "$request_timeout" \
            "${headers[@]}" --write-out $'\n%{http_code}' "${api_url}${path}")"; then
            http_status="${response##*$'\n'}"
            case "$http_status" in
                200) printf '%s\n' "${response%$'\n'*}"; return 0 ;;
                429|5??) ;;
                *) echo "required-checks: GitHub returned HTTP ${http_status} for ${path}" >&2; return 1 ;;
            esac
        else
            http_status="transport error"
        fi
        echo "required-checks: ${http_status}, API attempt ${attempt}/3" >&2
        if (( attempt < 3 && SECONDS < deadline )); then sleep 1; fi
    done
    return 1
}

# Both check runs and legacy commit statuses can span multiple API pages.
github_list() {
    local path="$1" selector="$2" page=1 response items all_items='[]' separator='?'
    if [[ "$path" == *\?* ]]; then separator='&'; fi
    while :; do
        response="$(github_api "${path}${separator}per_page=100&page=${page}")" || return 1
        items="$(jq -ce "${selector} | if type == \"array\" then . else error(\"expected an array\") end" <<<"$response")" || return 1
        all_items="$(jq -cs '.[0] + .[1]' <<<"${all_items}"$'\n'"${items}")" || return 1
        if (( $(jq 'length' <<<"$items") < 100 )); then
            printf '%s\n' "$all_items"
            return 0
        fi
        page=$((page + 1))
    done
}

check_current_head() {
    local pr head state
    pr="$(github_api "/repos/${repository}/pulls/${pr_number}")" || return 1
    head="$(jq -er '.head.sha' <<<"$pr")" || return 1
    state="$(jq -er '.state' <<<"$pr")" || return 1
    if [[ "$head" != "$expected_sha" || "$state" != "open" ]]; then
        echo "required-checks: stale build; expected ${expected_sha}, PR head is ${head} (${state})" >&2
        return 1
    fi
}

# Return 0 for admitted, 2 for pending, and 1 for rejected/invalid results.
check_result() {
    local name="$1" record="$2" status conclusion
    if [[ -z "$record" ]]; then
        echo "required-checks: ${name}: not reported yet"
        return 2
    fi
    status="$(jq -er '.status' <<<"$record")" || return 1
    case "$status" in
        queued|in_progress|waiting|pending|requested)
            echo "required-checks: ${name}: ${status}"
            return 2
            ;;
        completed) ;;
        *) echo "required-checks: ${name}: invalid status ${status}" >&2; return 1 ;;
    esac
    conclusion="$(jq -er '.conclusion' <<<"$record")" || return 1
    echo "required-checks: ${name}: ${conclusion}"
    case "$conclusion" in
        success) return 0 ;;
        neutral|skipped)
            if [[ "$name" == "Check code quality" ]]; then return 0; fi
            ;;
    esac
    echo "required-checks: ${name} did not pass; refusing admission" >&2
    return 1
}

echo "required-checks: waiting on ${repository} PR #${pr_number} at ${expected_sha}"
while (( SECONDS < deadline )); do
    check_current_head
    # Select the latest IDs ourselves: GitHub's default filter uses completion
    # times, which must not hide a pending rerun or another suite's path check.
    checks="$(github_list "/repos/${repository}/commits/${expected_sha}/check-runs?filter=all" '.check_runs')"
    pending=0
    for check_name in "Check code quality" "DCO"; do
        record="$(jq -cr --arg name "$check_name" \
            '[.[] | select(.name == $name)] | sort_by(.id) | last // empty' <<<"$checks")"
        # A skipped quality job is intentional only if its own path-selection
        # job succeeded. A failed dependency also produces a skipped check.
        if [[ "$check_name" == "Check code quality" && -n "$record" ]] &&
            [[ "$(jq -r '.conclusion' <<<"$record")" == "skipped" ]]; then
            suite_id="$(jq -er '.check_suite.id' <<<"$record")"
            record="$(jq -cr --argjson suite "$suite_id" '
                [.[] | select(.name == "Detect changes" and .check_suite.id == $suite)]
                | sort_by(.id) | last // empty' <<<"$checks")"
            check_name="Code Quality path selection"
        fi
        # A running check has no conclusion. It must not be replaced by an
        # older success from the legacy status API.
        if [[ -z "$record" && "$check_name" == "DCO" ]]; then
            statuses="$(github_list "/repos/${repository}/commits/${expected_sha}/statuses" '.')"
            record="$(jq -cr '
                [.[] | select(.context == "DCO")] | sort_by(.id) | last
                | if . == null then empty else
                    {status: (if .state == "pending" then "pending" else "completed" end),
                     conclusion: .state}
                  end' <<<"$statuses")"
        fi
        if check_result "$check_name" "$record"; then
            :
        else
            result=$?
            if (( result != 2 )); then exit 1; fi
            pending=1
        fi
    done
    if (( pending == 0 )); then
        check_current_head
        echo "required-checks: fast checks passed for ${expected_sha}"
        exit 0
    fi
    remaining=$((deadline - SECONDS))
    if (( remaining <= 0 )); then break; fi
    delay=$poll_seconds
    if (( delay > remaining )); then delay=$remaining; fi
    sleep "$delay"
    poll_seconds=$((poll_seconds * 2))
    if (( poll_seconds > 60 )); then poll_seconds=60; fi
done

echo "required-checks: timed out after ${timeout_seconds}s waiting for fast checks" >&2
exit 1
