#!/usr/bin/env bash
# Wraps `buildkite-agent pipeline upload` with a path-based skip check.
#
# Usage (from a buildkite-pipeline.yml step):
#   command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh \
#       .buildkite/k3_tests/<test-name>/pipeline.yml
#
# If every changed file in this build is trivial (markdown, LICENSE, etc.) and
# none touch .github/ or .buildkite/, this script:
#   - Annotates the build with a "skipped" note
#   - Exits 0 without uploading any further steps → the build is green
# Otherwise it execs the normal pipeline upload.
#
# Set K3_PATH_FILTER_DISABLE=1 in the build environment to bypass the check.

set -euo pipefail

PIPELINE_FILE="${1:?Usage: upload-pipeline.sh <path/to/pipeline.yml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck source=path-filter.sh
source "${SCRIPT_DIR}/path-filter.sh"

if should_skip_ci; then
    echo "+++ :fast_forward: Skipping CI — only trivial files changed"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent annotate \
            --style success \
            --context "path-filter-skip" \
            "Skipped: only trivial files (docs, license, etc.) changed. Set \`K3_PATH_FILTER_DISABLE=1\` to force a full run." \
            || true
    fi
    exit 0
fi

echo "--- :pipeline: Uploading ${PIPELINE_FILE}"
exec buildkite-agent pipeline upload "${PIPELINE_FILE}"
