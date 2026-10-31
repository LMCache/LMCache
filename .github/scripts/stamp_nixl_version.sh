#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stamp an LMCache release version (PEP 440, e.g. 0.5.7 or 0.5.7rc1) into the
# lmcache-nixl package (csrc/storage_backends/nixl), so its wheel is published
# under the same version as lmcache and images can pin the two together.
#
#   stamp_nixl_version.sh <version>

set -euo pipefail

version=${1:?usage: stamp_nixl_version.sh <version>}
pyproject="$(dirname "$0")/../../csrc/storage_backends/nixl/pyproject.toml"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
    echo "unsupported release version '$version' (want X.Y.Z, X.Y.ZaN, X.Y.ZbN or X.Y.ZrcN)" >&2
    exit 1
fi

sed -i "s/^version = \".*\"/version = \"${version}\"/" "$pyproject"
grep '^version = ' "$pyproject"
