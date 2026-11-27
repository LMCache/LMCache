#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the lmcache-client wheel (#6315) from pyproject_client.toml.
#
# Exports HEAD into a temporary directory, uses pyproject_client.toml as its
# pyproject.toml and builds there, so the checkout is never modified.
#
# Usage: build_client_wheel.sh [--cibuildwheel] OUT_DIR [options]
#   options go to `python -m build`, or with --cibuildwheel to cibuildwheel,
#   which builds the manylinux release wheels ([tool.cibuildwheel] in
#   pyproject_client.toml).
# Env:
#   CLIENT_VERSION  wheel version (default: setuptools_scm on the checkout,
#                   the version the full lmcache wheel gets)
#   PYTHON          interpreter (default: python)
set -euo pipefail

if [[ "${1:-}" == --cibuildwheel ]]; then
    builder=(-m cibuildwheel --output-dir)
    shift
else
    builder=(-m build --wheel --outdir)
    # Keeps a caller's BUILD_WITH_<device> or NO_GPU_EXT=0 from selecting a
    # device profile: the local build is the CPU build.
    export NO_GPU_EXT=1
fi
out_dir="$(mkdir -p "$1" && cd "$1" && pwd)"
shift
repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
python="${PYTHON:-python}"
version="${CLIENT_VERSION:-$(cd "$repo" && "$python" -m setuptools_scm | tail -1)}"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
git -C "$repo" archive --format=tar HEAD | tar -x -C "$stage"
cp "$stage/pyproject_client.toml" "$stage/pyproject.toml"

echo "Building lmcache-client ${version} from $(git -C "$repo" rev-parse --short HEAD)"
cd "$stage"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_CLIENT="$version" LMCACHE_CLIENT_BUILD=1 \
    "$python" "${builder[@]}" "$out_dir" "$@"
