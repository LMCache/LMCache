#!/usr/bin/env bash
# Open or update a sticky GitHub issue tracking nightly vLLM x LMCache
# gate failures. Called from .github/workflows/nightly_build.yml when the
# upstream `nightly-mp-gate` job fails.
#
# Behaviour:
#   * The "sticky" issue is identified by the label nightly-gate-failure.
#   * If no open issue with that label exists, a new one is created.
#   * If one exists, today's failure is appended as a comment.
#
# All operations go through `gh` (REST is intentionally avoided per repo
# convention -- `gh` is pre-installed on GitHub-hosted runners).
set -euo pipefail

LABEL="nightly-gate-failure"
TODAY="$(date -u +'%Y-%m-%d')"
TITLE="Nightly vLLM x LMCache MP gate failure tracker (since ${TODAY})"

# Inputs (all required, populated by the caller workflow):
#   GH_REPO                    e.g. LMCache/LMCache
#   GATE_BUILD_URL             buildkite build web url that failed
#   GATE_BUILD_STATE           e.g. failed | timed_out
GH_REPO="${GH_REPO:?GH_REPO must be set}"
GATE_BUILD_URL="${GATE_BUILD_URL:?GATE_BUILD_URL must be set}"
GATE_BUILD_STATE="${GATE_BUILD_STATE:?GATE_BUILD_STATE must be set}"

ensure_label() {
    if ! gh label list --repo "${GH_REPO}" --limit 200 \
            | awk '{print $1}' | grep -qx "${LABEL}"; then
        echo "Creating label: ${LABEL}"
        gh label create "${LABEL}" \
            --repo "${GH_REPO}" \
            --color "B60205" \
            --description "Auto-opened by nightly gate when vllm-nightly breaks LMCache" \
            || true
    fi
}

find_open_issue() {
    gh issue list --repo "${GH_REPO}" --state open --label "${LABEL}" \
        --json number --jq '.[0].number // empty'
}

create_issue_body() {
    cat <<EOF
This sticky issue tracks failures of the **nightly vLLM x LMCache MP integration gate**.

The gate runs once per day (right before the nightly build) against the freshly
published \`vllm-nightly\` wheel and the latest LMCache \`dev\` branch. When it
fails, today's nightly artifacts are NOT published, and a new comment is added
below with the failing build link.

Close this issue once the upstream regression is resolved and the gate is green
again. If a new failure appears later, a fresh issue will be opened automatically.

---

**First failure:** ${TODAY}
**Buildkite build:** ${GATE_BUILD_URL}
**Build state:** \`${GATE_BUILD_STATE}\`
EOF
}

create_comment_body() {
    cat <<EOF
Gate failed again on **${TODAY}**.

* Build: ${GATE_BUILD_URL}
* State: \`${GATE_BUILD_STATE}\`
EOF
}

main() {
    ensure_label

    local issue_number
    issue_number="$(find_open_issue || true)"

    if [[ -z "${issue_number}" ]]; then
        echo "No open sticky issue found, creating one..."
        local body
        body="$(create_issue_body)"
        gh issue create \
            --repo "${GH_REPO}" \
            --title "${TITLE}" \
            --label "${LABEL}" \
            --body "${body}"
    else
        echo "Updating existing sticky issue #${issue_number}..."
        local body
        body="$(create_comment_body)"
        gh issue comment "${issue_number}" \
            --repo "${GH_REPO}" \
            --body "${body}"
    fi
}

main "$@"
