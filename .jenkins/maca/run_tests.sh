#!/usr/bin/env bash
# Test entry point called by Jenkinsfile.test (MetaX MACA variant).
# Writes result.json in the result contract format when done.
#
# Exit code:
#   0  all tests passed
#   1  one or more tests failed

set -euo pipefail

START=$(date +%s)
COMMIT=$(git rev-parse HEAD)
CAPABILITY="${MACA_CAPABILITY:-maca-demo}"

echo "============================================"
echo " LMCache MACA test run"
echo " Commit   : ${COMMIT}"
echo " Capability: ${CAPABILITY}"
echo " Started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"

TEST_STATUS="passed"
PYTEST_OUTPUT=$(mktemp)

if python3 -m pytest tests/ \
       -x -q \
       --timeout=120 \
       -m "not gpu" \
       --tb=short 2>&1 | tee "${PYTEST_OUTPUT}"; then
    TEST_STATUS="passed"
else
    TEST_STATUS="failed"
fi

END=$(date +%s)
DURATION=$(( END - START ))

# Collect failed test names and first error summary (max 5 lines)
FAILED_TESTS=$(grep -E "^FAILED " "${PYTEST_OUTPUT}" | head -5 | sed 's/FAILED //' || true)
ERROR_SUMMARY=$(grep -A3 "^FAILED\|^ERROR\|short test summary" "${PYTEST_OUTPUT}" | head -10 | tr '\n' ' ' | sed 's/"/\\"/g' || true)
rm -f "${PYTEST_OUTPUT}"

python3 - << PYEOF
import json
data = {
    "capability":       "${CAPABILITY}",
    "workload":         "smoke",
    "commit":           "${COMMIT}",
    "status":           "${TEST_STATUS}",
    "duration_seconds": ${DURATION},
    "failed_tests":     """${FAILED_TESTS}""".strip().splitlines(),
    "error_summary":    """${ERROR_SUMMARY}""".strip(),
}
with open("result.json", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

echo ""
echo "Result:"
cat result.json

[[ "${TEST_STATUS}" == "passed" ]] && exit 0 || exit 1
