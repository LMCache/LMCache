#!/bin/bash
# Stage the NIXL SDK cibuildwheel needs to compile lmcache.lmcache_nixl.
#
# The nixl wheel ships libnixl and its plugins but no C++ headers, and the
# NIXL source ships headers but no binaries, so the two halves come from
# different places: headers from the pinned source release, libnixl unpacked
# from the nixl-cuXX wheel that matches this build's CUDA line. Runs inside
# the manylinux container (cibuildwheel before-all), once per container.
#
# Usage: fetch_nixl_sdk.sh <cu12|cu13> [nixl-version]
#   /opt/nixl-source/src/api/cpp/nixl.h              -> NIXL_INCLUDE_DIR
#   /opt/nixl-libs/.nixl_<variant>.mesonpy.libs/     -> NIXL_LIBRARY_DIR
set -euo pipefail

variant="${1:?usage: fetch_nixl_sdk.sh <cu12|cu13> [nixl-version]}"
version="${2:-1.3.1}"
# Any interpreter the image provides will do: this only downloads and unzips.
py="$(ls -d /opt/python/cp3*/bin/python | head -1)"

rm -rf /opt/nixl-source /opt/nixl-libs /tmp/nixl-wheel
git clone --depth 1 --branch "v${version}" \
    https://github.com/ai-dynamo/nixl.git /opt/nixl-source

# The bundled libraries are identical across the per-Python wheels; take one.
"$py" -m pip download --no-deps --only-binary=:all: \
    --platform "manylinux_2_28_$(uname -m)" --python-version 3.12 \
    "nixl-${variant}==${version}" -d /tmp/nixl-wheel
set -- /tmp/nixl-wheel/*.whl
"$py" -m zipfile -e "$1" /opt/nixl-libs

test -f /opt/nixl-source/src/api/cpp/nixl.h
test -f "/opt/nixl-libs/.nixl_${variant}.mesonpy.libs/libnixl.so"
echo "NIXL ${version} SDK staged for ${variant}: headers in /opt/nixl-source, libnixl in /opt/nixl-libs"
