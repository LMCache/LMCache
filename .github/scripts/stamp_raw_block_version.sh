#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stamp an LMCache release version (PEP 440, e.g. 0.5.7, 0.5.7rc1 or 0.5.7.post1)
# as the static [project] version of the raw-block crate's pyproject, so its
# wheel (lmcache_rust_raw_block_io) is published under the same version as
# lmcache and images can pin the two together.
#
#   stamp_raw_block_version.sh <version>
#
# maturin reads [project].version verbatim and it takes precedence over
# Cargo.toml, so this accepts any PEP 440 release version -- including
# post-releases. Cargo's SemVer has no post-release concept and there is no
# SemVer spelling maturin maps back to a PEP 440 .postN, so the version is
# carried on the Python side (pyproject) rather than round-tripped through
# Cargo.toml. The crate's own Cargo.toml/Cargo.lock version is internal (never
# published to crates.io) and is left untouched, so `cargo --locked` builds
# stay consistent.

set -euo pipefail

version=${1:?usage: stamp_raw_block_version.sh <version>}
pyproject="$(dirname "$0")/../../rust/raw_block/pyproject.toml"

# Accept the PEP 440 release forms we tag: X.Y.Z, pre-release (aN/bN/rcN) and
# post-release (.postN), optionally combined (e.g. 0.5.7rc1.post1).
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.post[0-9]+)?$ ]]; then
    echo "unsupported release version '$version' (want X.Y.Z, X.Y.Z{a,b,rc}N or X.Y.Z.postN)" >&2
    exit 1
fi

# Switch the crate's pyproject from a Cargo-derived dynamic version to a static
# [project] version carrying the exact PEP 440 string; maturin then names the
# wheel exactly this version. Idempotent: a re-run updates the static line.
if grep -qxF 'dynamic = ["version"]' "$pyproject"; then
    sed -i 's/^dynamic = \["version"\]$/version = "'"${version}"'"/' "$pyproject"
elif grep -qE '^version = "' "$pyproject"; then
    sed -i 's/^version = ".*"/version = "'"${version}"'"/' "$pyproject"
else
    echo "no 'dynamic = [\"version\"]' or 'version = ...' line found in $pyproject" >&2
    exit 1
fi

grep -nE '^(version|dynamic) = ' "$pyproject"
