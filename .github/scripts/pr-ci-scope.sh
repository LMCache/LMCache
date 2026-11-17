#!/usr/bin/env bash
# Resolve the CI scope a PR author picked in the "CI scope" checklist of
# .github/PULL_REQUEST_TEMPLATE.md.
#
# Input (env):
#   PR_BODY  PR description. Untrusted: only matched as data, never evaluated.
#
# Outputs (appended to $GITHUB_OUTPUT), JSON arrays:
#   python_versions  Python versions for test.yml
#   cpu_os           runner OSes for cpu_device.yml
#
# A group with no box ticked selects every entry of that group.
set -euo pipefail

ALL_PYTHON='["3.11","3.12","3.13","3.14"]'
ALL_CPU_OS='["ubuntu-22.04","macos-latest"]'

checked=$(printf '%s\n' "${PR_BODY}" | tr -d '\r' \
  | sed -nE 's/^[[:space:]]*[-*] \[[xX]\] `(py3\.1[1-4]|ubuntu-22\.04|macos-latest)`.*/\1/p' \
  | sort -u)

python_versions=$(printf '%s\n' "${checked}" | sed -n 's/^py//p' | jq -Rnc '[inputs]')
cpu_os=$(printf '%s\n' "${checked}" | sed '/^py/d' | jq -Rnc '[inputs | select(length > 0)]')

[[ "${python_versions}" == "[]" ]] && python_versions=${ALL_PYTHON}
[[ "${cpu_os}" == "[]" ]] && cpu_os=${ALL_CPU_OS}

{
  echo "python_versions=${python_versions}"
  echo "cpu_os=${cpu_os}"
} | tee -a "${GITHUB_OUTPUT}"
