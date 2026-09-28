#!/usr/bin/env bash
# Per-job env setup for the SGLang + LMCache MP integration tests.
# CI callers may override SGLANG_INSTALL_SPEC with a branch, wheel, or
# commit-pinned git URL. Otherwise the latest Buildkite-verified nightly is
# resolved to its source commit.
set -euo pipefail
trap 'echo "ERROR: setup-sglang-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80
merge_pr_base_branch

# The official SGLang nightly wheel is currently built for CPython 3.10,
# while the LMCache CI base image uses Python 3.12. Resolve the verified
# nightly's source revision and install that exact commit instead.
source "${REPO_ROOT}/.buildkite/k3_harness/resolve-pinned-sglang.sh"

echo "--- :wrench: System tools (rustup, protoc, libnuma1)"
# rustup: sglang-grpc needs Rust 1.85+ (apt's rustc is too old).
# protoc: sglang-grpc's prost-build shells out to it.
# libnuma1: sgl_kernel's sm100 .so dynamically links to libnuma.so.1.
if ! command -v rustc >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
        -y --default-toolchain stable --profile minimal --no-modify-path
fi
# shellcheck disable=SC1091
. "${HOME}/.cargo/env"
rustc --version

APT_NEEDED=()
command -v protoc >/dev/null 2>&1 || APT_NEEDED+=("protobuf-compiler")
[[ -e /usr/lib/x86_64-linux-gnu/libnuma.so.1 || -e /lib/x86_64-linux-gnu/libnuma.so.1 ]] || APT_NEEDED+=("libnuma1")
if [[ ${#APT_NEEDED[@]} -gt 0 ]]; then
    apt-get update
    apt-get install -y --no-install-recommends "${APT_NEEDED[@]}"
fi
protoc --version

echo "--- :package: SGLang + LMCache install"
SGLANG_INDEX_ARGS=()
if [[ -z "${SGLANG_INSTALL_SPEC:-}" ]]; then
    if [[ -n "${PINNED_SGLANG_SHA}" ]]; then
        SGLANG_INSTALL_SPEC="git+https://github.com/sgl-project/sglang.git@${PINNED_SGLANG_SHA}#subdirectory=python"
    elif [[ -n "${PINNED_SGLANG_VERSION}" ]]; then
        SGLANG_INSTALL_SPEC="sglang==${PINNED_SGLANG_VERSION}"
        if [[ -n "${PINNED_SGLANG_INDEX_URL}" ]]; then
            SGLANG_INDEX_ARGS+=(--extra-index-url "${PINNED_SGLANG_INDEX_URL}")
        fi
    else
        SGLANG_INSTALL_SPEC="git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"
    fi
fi
echo "Installing SGLang from ${SGLANG_INSTALL_SPEC}"
uv pip install \
    "${SGLANG_INDEX_ARGS[@]+"${SGLANG_INDEX_ARGS[@]}"}" \
    "${SGLANG_INSTALL_SPEC}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"

uv pip uninstall cupy-cuda12x 2>/dev/null || true
uv pip install -e . --no-build-isolation

uv pip install -r requirements/proto.txt
echo "--- :gear: Generating LMCache gRPC bindings"
python "${REPO_ROOT}/lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py"

python -c "import lmcache, sglang; print(f'sglang={sglang.__version__}; lmcache OK')"
python -c "import cupy; print(f'cupy={cupy.__version__}')"
