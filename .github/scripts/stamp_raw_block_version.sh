#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stamp an LMCache release version (PEP 440, e.g. 0.5.7 or 0.5.7rc1) into the
# raw-block Rust crate, so its wheel (lmcache_rust_raw_block_io) is published
# under the same version as lmcache and images can pin the two together.
#
#   stamp_raw_block_version.sh <version>
#
# Cargo needs semver (0.5.7-rc.1); maturin maps it back to PEP 440 (0.5.7rc1)
# when it names the wheel.

set -euo pipefail

version=${1:?usage: stamp_raw_block_version.sh <version>}
crate_dir="$(dirname "$0")/../../rust/raw_block"

if [[ "$version" =~ ^([0-9]+\.[0-9]+\.[0-9]+)((a|b|rc)([0-9]+))?$ ]]; then
    semver=${BASH_REMATCH[1]}
    case "${BASH_REMATCH[3]}" in
        a) semver+="-alpha.${BASH_REMATCH[4]}" ;;
        b) semver+="-beta.${BASH_REMATCH[4]}" ;;
        rc) semver+="-rc.${BASH_REMATCH[4]}" ;;
    esac
else
    echo "unsupported release version '$version' (want X.Y.Z, X.Y.ZaN, X.Y.ZbN or X.Y.ZrcN)" >&2
    exit 1
fi

# The first `version = ` line is the [package] version. Cargo.lock records
# the crate's own version too; update it so `--locked` builds still pass.
sed -i "0,/^version = \".*\"/s//version = \"${semver}\"/" "$crate_dir/Cargo.toml"
sed -i "/^name = \"lmcache_rust_raw_block_io\"$/{n;s/^version = \".*\"/version = \"${semver}\"/}" \
    "$crate_dir/Cargo.lock"
grep -m1 '^version = ' "$crate_dir/Cargo.toml"
grep -A1 '^name = "lmcache_rust_raw_block_io"$' "$crate_dir/Cargo.lock" | tail -1
