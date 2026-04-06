#!/usr/bin/env bash
set -euo pipefail

# Detect whether a PR only touches operator/ files.
# Exit code semantics:
#   0 -> only operator files changed (caller should skip)
#   1 -> non-operator files changed (caller should run tests)
#
# Usage in a pipeline step:
#   .buildkite/scripts/is-operator-only.sh && exit 0

# Non-PR builds always run tests.
if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
  exit 1
fi

BASE_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-dev}"
BASE_REF="origin/${BASE_BRANCH}"

if ! git fetch origin "${BASE_BRANCH}" >/dev/null 2>&1; then
  echo "Warning: failed to fetch origin/${BASE_BRANCH}" >&2
fi

CHANGED_FILES=$(git diff --name-only "${BASE_REF}...HEAD" 2>/dev/null || echo "")

# If we cannot determine changes, be conservative and run tests.
if [[ -z "${CHANGED_FILES}" ]]; then
  exit 1
fi

while IFS= read -r f; do
  case "$f" in
    operator/*)
      # Operator file — safe to skip Python/CUDA tests.
      ;;
    *)
      # Non-operator file touched — must run tests.
      exit 1
      ;;
  esac
done <<< "${CHANGED_FILES}"

echo "Only operator/ files changed; skipping Python tests."
exit 0
