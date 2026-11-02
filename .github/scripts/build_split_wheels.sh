#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the client (lmcache) and full-runtime (lmcache-server) wheels from this
# checkout. Build-profile variables (NO_GPU_EXT, BUILD_WITH_HIP, ...) select the
# server's extensions; the client always builds only lmcache_native.
#
# Usage: build_split_wheels.sh [extra `python -m build --wheel` args]

set -euo pipefail

PYTHON="${PYTHON:-python}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

saved="$(mktemp)"
cp pyproject.toml "$saved"
trap 'cp "$saved" pyproject.toml; rm -f "$saved"' EXIT

"$PYTHON" -m build --wheel "$@"
cp pyproject_server.toml pyproject.toml
"$PYTHON" -m build --wheel "$@"
