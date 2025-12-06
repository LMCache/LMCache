#!/usr/bin/env bash
set -euo pipefail

# Decide whether comprehensive tests should run for this build.
# Exit code semantics:
#   0 -> run comprehensive tests
#   1 -> skip comprehensive tests (safe paths only)

# If this isn't a PR build, always run comprehensive tests.
if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
  exit 0
fi

# Determine base ref to diff against.
BASE_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-dev}"
BASE_REF="origin/${BASE_BRANCH}"

# Ensure the base ref exists; ignore fetch failures (repo might already have it).
git fetch origin "${BASE_BRANCH}" >/dev/null 2>&1 || true

# Compute changed files between the base and current HEAD.
if CHANGED_FILES=$(git diff --name-only "${BASE_REF}...HEAD" 2>/dev/null); then
  :
else
  # Fallback: diff against the previous commit if base ref is unavailable.
  CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null || echo "")
fi

# If we cannot determine changes, be conservative and run tests.
if [[ -z "${CHANGED_FILES}" ]]; then
  exit 0
fi

# If any changed file is NOT in a safe path, we must run tests.
for f in ${CHANGED_FILES}; do
  case "${f}" in
    docs/*|docs/**) ;;                      # docs
    *.md|*.rst) ;;                          # markdown / rst anywhere
    tests/*|tests/**) ;;                    # tests
    benchmarks/*|benchmarks/**) ;;          # benchmarks
    tools/*|tools/**) ;;                    # top-level tools
    lmcache/tools/*|lmcache/tools/**) ;;    # package tools
    examples/*|examples/**) ;;              # examples
    asset/*|asset/**) ;;                    # assets
    *)
      # Non-safe file touched -> run comprehensive tests.
      exit 0
      ;;
  esac
done

echo "Docs/tests/benchmarks/tools/examples/assets-only change detected; skipping comprehensive tests."
exit 1

