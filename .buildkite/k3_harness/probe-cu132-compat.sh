#!/usr/bin/env bash
set -euo pipefail

command -v nvidia-smi curl sha256sum apt-get dpkg-query
test "$(id -u)" = 0
. /etc/os-release
test "${ID}" = ubuntu && test "${VERSION_ID}" = 24.04
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
python .buildkite/k3_harness/verify-cu132-compat.py --allow-unsupported-ptx

compat_host_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
if (( ${compat_host_driver%%.*} < 595 )); then
    compat_package_path="/tmp/lmcache-cuda-compat-13-2.deb"
    curl --fail --location --retry 3 \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-2_595.112.07-1ubuntu1_amd64.deb \
        --output "${compat_package_path}"
    printf '%s  %s\n' acc3ba5b576b41ab304ad1c6215a64e3e57bd7658afd97f5c82a882e40db4300 \
        "${compat_package_path}" | sha256sum --check
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "${compat_package_path}"
    dpkg-query -W -f='${Package} ${Version}\n' cuda-compat-13-2
    test -f /usr/local/cuda-13.2/compat/libcuda.so.1
    export LD_LIBRARY_PATH="/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "Host driver ${compat_host_driver} already supports the CUDA 13.2 PTX toolchain"
fi
python .buildkite/k3_harness/verify-cu132-compat.py
