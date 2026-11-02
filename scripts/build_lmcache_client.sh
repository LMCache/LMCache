#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build lmcache-client from an isolated staging tree without changing files in
# the source checkout.

set -euo pipefail

usage() {
  echo "Usage: $0 [--cibuildwheel] [--output-dir DIR]"
}

build_mode="build"
output_dir="dist"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cibuildwheel)
      build_mode="cibuildwheel"
      shift
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        usage >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="$(command -v "${PYTHON_BIN:-python}")"
python_bin="$(cd "$(dirname "${python_bin}")" && pwd)/$(basename "${python_bin}")"

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/lmcache-client-build.XXXXXX")"
cleanup() {
  rm -rf "${stage_dir}"
}
trap cleanup EXIT

# Include tracked files and non-ignored worktree additions so the command is
# useful before a commit too. Build outputs and virtual environments remain
# excluded by the repository's ignore rules.
git -C "${repo_root}" ls-files -z --cached --others --exclude-standard \
  | tar -C "${repo_root}" --null -T - -cf - \
  | tar -C "${stage_dir}" -xf -

cp "${stage_dir}/pyproject_client.toml" "${stage_dir}/pyproject.toml"

if [[ -z "${SETUPTOOLS_SCM_PRETEND_VERSION:-}" ]]; then
  SETUPTOOLS_SCM_PRETEND_VERSION="$(${python_bin} -m setuptools_scm \
    -r "${repo_root}" -c "${repo_root}/pyproject_client.toml" -f plain)"
  export SETUPTOOLS_SCM_PRETEND_VERSION
fi
export LMCACHE_CLIENT_BUILD=1

echo "Building lmcache-client ${SETUPTOOLS_SCM_PRETEND_VERSION} from staging tree"
if [[ "${build_mode}" == "cibuildwheel" ]]; then
  "${python_bin}" -m cibuildwheel "${stage_dir}" --output-dir "${output_dir}"
else
  (
    cd "${stage_dir}"
    "${python_bin}" -m build --wheel --no-isolation
  )
  cp "${stage_dir}"/dist/lmcache_client-*.whl "${output_dir}/"
fi

find "${output_dir}" -maxdepth 1 -type f -name 'lmcache_client-*.whl' -print
