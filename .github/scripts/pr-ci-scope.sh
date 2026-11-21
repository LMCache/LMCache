#!/usr/bin/env bash
# Resolve the CI scope a PR author picked in the "CI scope" checklist of
# .github/PULL_REQUEST_TEMPLATE.md.
#
# Input (env):
#   PR_BODY  PR description. Untrusted: only matched as data, never evaluated.
#
# Outputs (appended to $GITHUB_OUTPUT):
#   python_versions  JSON array of Python versions for test.yml
#   cpu_os           JSON array of runner OSes for cpu_device.yml
#   rust             true to run test.yml's Rust raw-block jobs even when no
#                    raw-block path changed
#   gpu_devices      JSON array of Layer 3 GPU devices ticked in the PR body
#                    (empty array = auto-detect from changed files)
#   gpu_full         true when `full` is ticked in the GPU scope checklist
#
# A Python or OS group with no box ticked selects every entry of that group.
# Unticked `rust` keeps the path filter's decision.
# Unticked GPU devices let Buildkite auto-detect from changed files.
set -euo pipefail

ALL_PYTHON='["3.11","3.12","3.13","3.14"]'
ALL_CPU_OS='["ubuntu-22.04","macos-latest"]'

checked=$(printf '%s\n' "${PR_BODY:-}" | tr -d '\r' \
  | sed -nE 's/^[[:space:]]*[-*] \[[xX]\] `(py3\.1[1-4]|ubuntu-22\.04|macos-latest|rust|cuda|rocm|xpu|musa|rbln|full)`.*/\1/p' \
  | sort -u)

python_versions=$(printf '%s\n' "${checked}" | sed -n 's/^py//p' | jq -Rnc '[inputs]')
cpu_os=$(printf '%s\n' "${checked}" | sed -nE '/^(ubuntu|macos)/p' | jq -Rnc '[inputs]')
rust=false
grep -qx rust <<<"${checked}" && rust=true
gpu_devices=$(printf '%s\n' "${checked}" | sed -nE '/^(cuda|rocm|xpu|musa|rbln)$/p' | jq -Rnc '[inputs]')
gpu_full=false
grep -qx full <<<"${checked}" && gpu_full=true

[[ "${python_versions}" == "[]" ]] && python_versions=${ALL_PYTHON}
[[ "${cpu_os}" == "[]" ]] && cpu_os=${ALL_CPU_OS}

{
  echo "python_versions=${python_versions}"
  echo "cpu_os=${cpu_os}"
  echo "rust=${rust}"
  echo "gpu_devices=${gpu_devices}"
  echo "gpu_full=${gpu_full}"
} | tee -a "${GITHUB_OUTPUT}"
