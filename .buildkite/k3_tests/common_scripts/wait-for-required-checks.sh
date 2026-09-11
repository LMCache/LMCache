#!/usr/bin/env bash
# Wait for the fast GitHub checks that protect the PR before uploading the
# expensive Buildkite pipeline steps.
#
# Buildkite receives the pull-request webhook at the same time as GitHub
# Actions.  The initial Buildkite step is intentionally tiny, but without
# this gate it immediately uploads GPU/integration jobs while Code Quality and
# DCO are still running.  Keeping the wait in the common upload path applies
# the same ordering to every K3 lane.

set -euo pipefail

if [[ -z "${BUILDKITE_PULL_REQUEST:-}" ||
    "${BUILDKITE_PULL_REQUEST:-}" == "false" ]]; then
    echo "required-checks: not a pull-request build; no GitHub gate needed"
    exit 0
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
    echo "required-checks: curl and jq are required to inspect GitHub checks" >&2
    exit 1
fi

remote_url="$(git remote get-url origin 2>/dev/null || true)"
repo="${remote_url#https://github.com/}"
repo="${repo#http://github.com/}"
repo="${repo#git@github.com:}"
repo="${repo#ssh://git@github.com/}"
repo="${repo%.git}"

if [[ ! "$repo" =~ ^[^/]+/[^/]+$ ]]; then
    echo "required-checks: could not derive a GitHub repository from origin: ${remote_url}" >&2
    exit 1
fi

api_args=(
    --fail
    --silent
    --show-error
    --location
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: 2022-11-28"
)
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    api_args+=( -H "Authorization: Bearer ${GITHUB_TOKEN}" )
fi

github_api() {
    curl "${api_args[@]}" "https://api.github.com${1}"
}

pr_number="${BUILDKITE_PULL_REQUEST}"
pr_json="$(github_api "/repos/${repo}/pulls/${pr_number}")"
head_sha="$(jq -r '.head.sha // empty' <<<"$pr_json")"
if [[ -z "$head_sha" ]]; then
    echo "required-checks: GitHub did not return a head SHA for PR #${pr_number}" >&2
    exit 1
fi

timeout_seconds="${BUILDKITE_REQUIRED_CHECKS_TIMEOUT_SECONDS:-1800}"
poll_seconds="${BUILDKITE_REQUIRED_CHECKS_POLL_SECONDS:-10}"
max_poll_seconds=60
deadline=$((SECONDS + timeout_seconds))

echo "required-checks: waiting for Code Quality and DCO on ${repo}@${head_sha}"

while (( SECONDS < deadline )); do
    if ! checks_json="$(github_api "/repos/${repo}/commits/${head_sha}/check-runs?per_page=100")"; then
        echo "required-checks: failed to query GitHub check runs" >&2
        exit 1
    fi
    statuses_json=""
    pending=0

    for check_name in "Check code quality" "DCO"; do
        check_json="$(jq -c --arg name "$check_name" '
            [.check_runs[] | select(.name == $name)]
            | sort_by(.id) | last // {}' <<<"$checks_json")"
        status="$(jq -r '.status // empty' <<<"$check_json")"
        conclusion="$(jq -r '.conclusion // empty' <<<"$check_json")"

        # DCO has historically been reported as either a check run or a
        # commit status.  Fall back to the latter so the gate remains
        # compatible with both forms.
        if [[ -z "$status" && "$check_name" == "DCO" ]]; then
            if [[ -z "$statuses_json" ]]; then
                if ! statuses_json="$(github_api "/repos/${repo}/commits/${head_sha}/status")"; then
                    echo "required-checks: failed to query GitHub commit statuses" >&2
                    exit 1
                fi
            fi
            status="$(jq -r '[.statuses[] | select(.context == "DCO")]
                | sort_by(.id) | last.state // empty' <<<"$statuses_json")"
            conclusion="$(jq -r 'if . == "success" then "success"
                elif . == "pending" then "" else . end' <<<"$status")"
        fi

        case "$conclusion" in
            success|neutral)
                echo "required-checks: ${check_name}: ${conclusion}"
                ;;
            skipped)
                if [[ "$check_name" == "Check code quality" ]]; then
                    echo "required-checks: ${check_name}: skipped (path-filtered)"
                else
                    echo "required-checks: ${check_name}: skipped; refusing to run downstream tests" >&2
                    exit 1
                fi
                ;;
            failure|cancelled|timed_out|action_required|stale)
                echo "required-checks: ${check_name}: ${conclusion}; refusing to run downstream tests" >&2
                exit 1
                ;;
            *)
                pending=1
                echo "required-checks: ${check_name}: ${status:-not reported yet}"
                ;;
        esac
    done

    if (( pending == 0 )); then
        echo "required-checks: Code Quality and DCO passed; uploading downstream pipeline"
        exit 0
    fi

    sleep "$poll_seconds"
    if (( poll_seconds < max_poll_seconds )); then
        poll_seconds=$((poll_seconds * 2))
        if (( poll_seconds > max_poll_seconds )); then
            poll_seconds="$max_poll_seconds"
        fi
    fi
done

echo "required-checks: timed out after ${timeout_seconds}s waiting for Code Quality and DCO" >&2
exit 1
