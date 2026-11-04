#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Editable install of the client (lmcache) and the full runtime (lmcache-server)
# from this checkout. lmcache-server requires lmcache at its own version, so the
# client goes first.
#
# Usage:
#   install_lmcache_editable.sh [extra pip install args]
#   PIP_BIN="uv pip" install_lmcache_editable.sh

set -euo pipefail

PIP_BIN="${PIP_BIN:-python -m pip}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .github/scripts/split_pretend_version.sh

saved="$(mktemp)"
cp pyproject.toml "$saved"
trap 'cp "$saved" pyproject.toml; rm -f "$saved"' EXIT

${PIP_BIN} install -e ".[grpc]" --no-build-isolation "$@"
cp pyproject_server.toml pyproject.toml
${PIP_BIN} install -e . --no-build-isolation "$@"
