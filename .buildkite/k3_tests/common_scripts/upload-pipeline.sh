#!/usr/bin/env bash
# Wraps `buildkite-agent pipeline upload` with a path-based skip check.
#
# Usage (called from each test's buildkite-pipeline.yml upload step):
#   command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh \
#       .buildkite/k3_tests/<test-name>/pipeline.yml
#
# If no changed files are relevant to this pipeline (including tests-only
# changes for integration and multiprocess), this script:
#   - Annotates the build with a "skipped" note
#   - Exits 0 without uploading test steps, so Buildkite can report success
#     for its required GitHub status. The build itself must still be triggered.
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

merge_pr_base_branch

if should_skip_ci "${PIPELINE_FILE}"; then
    echo "+++ :fast_forward: Skipping test steps — no relevant files changed for ${PIPELINE_FILE}"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent annotate \
            --style success \
            --context "path-filter-skip" \
            "Test steps skipped: no relevant files changed for ${PIPELINE_FILE}. Buildkite can report success for this build. Add a \`force-ci\` label to the PR to force a full run." \
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
