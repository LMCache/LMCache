#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the lmcache-client wheel (#6315) from pyproject_client.toml.
#
# Exports HEAD into a temporary directory, uses pyproject_client.toml as its
# pyproject.toml and builds there, so the checkout is never modified.
#
# Usage: build_client_wheel.sh OUT_DIR [python -m build options]
# Env:
#   CLIENT_VERSION  wheel version (default: setuptools_scm on the checkout,
#                   the version the full lmcache wheel gets)
#   PYTHON          interpreter (default: python)
set -euo pipefail

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
    "$python" -m build --wheel --outdir "$out_dir" "$@"
