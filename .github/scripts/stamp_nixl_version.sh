#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stamp an LMCache release version (PEP 440, e.g. 0.5.7, 0.5.7rc1 or
# 0.5.7.post1) into the lmcache-nixl package (csrc/storage_backends/nixl), so
# its wheel is published under the same version as lmcache and images can pin
# the two together.
#
#   stamp_nixl_version.sh <version>

set -euo pipefail

version=${1:?usage: stamp_nixl_version.sh <version>}
pyproject="$(dirname "$0")/../../csrc/storage_backends/nixl/pyproject.toml"

# The release-version pattern is also in setup.py.
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.post[0-9]+)?$ ]]; then
    echo "unsupported release version '$version' (want X.Y.Z, X.Y.Z{a,b,rc}N or X.Y.Z.postN)" >&2
    exit 1
fi

sed -i "s/^version = \".*\"/version = \"${version}\"/" "$pyproject"
grep '^version = ' "$pyproject"
