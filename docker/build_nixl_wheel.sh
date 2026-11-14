#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build the lmcache-nixl wheel (the nixl_native L2 adapter's C++ connector,
# csrc/storage_backends/nixl) for the active venv's Python, so source-built
# images load it the same way release images load the published wheel.
#
#   docker/build_nixl_wheel.sh <out-dir>     (run from the LMCache checkout)
#
# Compiles against the headers of the NIXL release matching the installed
# nixl package and the libnixl from its nixl-cu13 wheel. The NIXL source is
# cloned into a temporary directory and removed afterwards.

set -euo pipefail

out=${1:?usage: build_nixl_wheel.sh <out-dir>}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

version=$(python3 -c 'import importlib.metadata as m; print(m.version("nixl"))')
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
git clone --quiet --depth 1 --branch "v${version}" \
    https://github.com/ai-dynamo/nixl.git "$tmp/nixl"

NIXL_INCLUDE_DIR="$tmp/nixl/src/api/cpp" \
NIXL_LIBRARY_DIR="$site/.nixl_cu13.mesonpy.libs" \
    uv build --wheel --python "$(command -v python3)" --out-dir "$out" \
    csrc/storage_backends/nixl
rm -rf csrc/storage_backends/nixl/build csrc/storage_backends/nixl/*.egg-info
