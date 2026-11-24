#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build the wheel of the raw_block L2 adapter's Rust module
# (lmcache_rust_raw_block_io, rust/raw_block) for the active venv's Python.
# The adapter imports it lazily, so an image without it starts fine and then
# fails on the first L2 store.
#
#   docker/build_raw_block_wheel.sh <out-dir>     (run from the LMCache checkout)
#
# The Rust toolchain and maturin are installed into a temporary directory and
# removed afterwards, so neither ends up in the image.

set -euo pipefail

out=${1:?usage: build_raw_block_wheel.sh <out-dir>}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

export RUSTUP_HOME="$tmp/rustup" CARGO_HOME="$tmp/cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --quiet --profile minimal --no-modify-path
uv venv --quiet "$tmp/maturin"
uv pip install --quiet --python "$tmp/maturin/bin/python" maturin

PATH="$CARGO_HOME/bin:$PATH" "$tmp/maturin/bin/maturin" build --release --locked \
    --manifest-path rust/raw_block/Cargo.toml \
    --interpreter "$(command -v python3)" \
    --out "$out"
rm -rf rust/raw_block/target
