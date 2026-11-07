#!/usr/bin/env bash
# Uploads the RBLN smoke pinned to the newest nightly-verified rbln-serve
# image. A failed lookup fails the build rather than falling back to the PR
# lane's pin, so the nightly lane never silently retests the pinned image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

RBLN_SERVE_IMAGE="$(python3 "${SCRIPT_DIR}/resolve_nightly_image.py")"
echo "Using nightly-verified rbln-serve image: ${RBLN_SERVE_IMAGE}"
export RBLN_SERVE_IMAGE

# Skips the common path filter: this lane watches rbln-serve, not LMCache
# changes, so a docs-only head on dev must not skip a manual rerun.
echo "--- :pipeline: Uploading .buildkite/k3_tests/rbln/pipeline.yml"
exec buildkite-agent pipeline upload .buildkite/k3_tests/rbln/pipeline.yml
