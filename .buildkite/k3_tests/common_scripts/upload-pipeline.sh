#!/usr/bin/env bash
# Wraps `buildkite-agent pipeline upload` with a path-based skip check.
#
# Usage (called from each test's buildkite-pipeline.yml upload step):
#   command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh \
#       .buildkite/k3_tests/<test-name>/pipeline.yml
#
# If every changed file in this build is trivial (markdown, LICENSE, .github,
# etc.) and none touch .buildkite/, this script:
#   - Annotates the build with a "skipped" note
#   - Exits 0 without uploading any further steps → the build is green
# Otherwise it execs `buildkite-agent pipeline upload <pipeline.yml>`, adding
# the real test steps to the build.
#
# Add a "force-ci" label to the PR on GitHub to bypass the check.

set -euo pipefail

PIPELINE_FILE="${1:?Usage: upload-pipeline.sh <path/to/pipeline.yml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck source=.buildkite/k3_tests/common_scripts/helpers.sh
source "${SCRIPT_DIR}/helpers.sh"
# shellcheck source=.buildkite/k3_tests/common_scripts/path-filter.sh
source "${SCRIPT_DIR}/path-filter.sh"

case "${PIPELINE_FILE}" in
    *k3_tests/integration/pipeline.yml)
        good_first_issue_pipeline="integration"
        ;;
    *k3_tests/multiprocess/pipeline.yml)
        good_first_issue_pipeline="multiprocess"
        ;;
    *k3_tests/sglang/pipeline.yml)
        good_first_issue_pipeline="sglang"
        ;;
    *)
        good_first_issue_pipeline=""
        ;;
esac

if [[ -n "${good_first_issue_pipeline}" ]] && \
    should_skip_k3_pipeline_for_good_first_issue "${good_first_issue_pipeline}"; then
    echo "+++ :fast_forward: Skipping ${good_first_issue_pipeline} CI for good first issue PR"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent annotate \
            --style success \
            --context "good-first-issue-skip" \
            "Skipped: PR has the \`good first issue\` label. Add a \`force-ci\` label to run the full ${good_first_issue_pipeline} suite." \
            || true
    fi
    exit 0
fi

merge_pr_base_branch

if should_skip_ci "${PIPELINE_FILE}"; then
    echo "+++ :fast_forward: Skipping CI — no relevant files changed for ${PIPELINE_FILE}"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent annotate \
            --style success \
            --context "path-filter-skip" \
            "Skipped: no relevant files changed for ${PIPELINE_FILE}. Add a \`force-ci\` label to the PR to force a full run." \
            || true
    fi
    exit 0
fi

if [[ "${PIPELINE_FILE}" == */xpu/*/pipeline.yml ]]; then
    # The XPU template interpolates this into its Kubernetes pod image.
    # shellcheck source=.buildkite/k3_harness/resolve-pinned-vllm.sh
    source "${SCRIPT_DIR}/../../k3_harness/resolve-pinned-vllm.sh"
fi

echo "--- :pipeline: Uploading ${PIPELINE_FILE}"
exec buildkite-agent pipeline upload "${PIPELINE_FILE}"
