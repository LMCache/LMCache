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
        expensive_pipeline_name="integration"
        ;;
    *k3_tests/multiprocess/pipeline.yml)
        expensive_pipeline_name="multiprocess"
        ;;
    *)
        expensive_pipeline_name=""
        ;;
esac

if [[ -n "${expensive_pipeline_name}" ]] && \
    should_skip_expensive_k3_pipeline "${expensive_pipeline_name}"; then
    echo "+++ :fast_forward: Skipping ${expensive_pipeline_name} CI"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent annotate \
            --style success \
            --context "skip expensive ci" \
            "Skipped: PR has the \`skip expensive ci\` label. Add a \`force-ci\` label to run the full ${expensive_pipeline_name} suite." \
            || true
    fi
    exit 0
fi

merge_pr_base_branch

if [[ "${XPU_CANDIDATE_VALIDATION:-0}" != "1" ]] && should_skip_ci "${PIPELINE_FILE}"; then
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
    if [[ "${PIPELINE_FILE}" == */xpu/multiprocess/pipeline.yml ]]; then
        export XPU_RELEASE_IMAGE="${XPU_RELEASE_IMAGE:-public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:ee0da84ab9e04ac7610e28580af62c365e898389-xpu}"
    fi
    if [[ "${XPU_CANDIDATE_VALIDATION:-0}" == "1" ]]; then
        XPU_CI_IMAGE="${XPU_CANDIDATE_IMAGE:?Candidate validation requires XPU_CANDIDATE_IMAGE}"
    else
        # shellcheck source=.buildkite/k3_harness/resolve-pinned-vllm.sh
        source "${SCRIPT_DIR}/../../k3_harness/resolve-pinned-vllm.sh"
        XPU_CI_IMAGE="${PINNED_XPU_IMAGE}"
    fi
    export XPU_CI_IMAGE
fi

echo "--- :pipeline: Uploading ${PIPELINE_FILE}"
exec buildkite-agent pipeline upload "${PIPELINE_FILE}"
