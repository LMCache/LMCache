#!/usr/bin/env bash
# Per-job environment setup for the SGLang + LMCache MP integration tests.
# Installs:
#   - LMCache from the CI checkout (editable, no build isolation).
#   - SGLang from the pre-merge fork branch that carries the MP integration.
# Patterned on setup-lmcache-only-env.sh. Drop the fork install once
# https://github.com/sgl-project/sglang/pull/24089 lands and SGLang's PyPI
# release contains the LMCache MP connector.
set -euo pipefail

trap 'echo "ERROR: setup-sglang-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

# ── Rust toolchain (for sglang's grpc Rust extension) ───────
# SGLang's python/pyproject.toml declares a setuptools-rust extension at
# sglang.srt.grpc._core which uses Cargo edition 2024, requiring Rust 1.85+.
# Ubuntu apt's rustc is too old; install via rustup.
echo "--- :rust: Installing Rust toolchain via rustup"
if ! command -v rustc >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
        -y --default-toolchain stable --profile minimal --no-modify-path
fi
# shellcheck disable=SC1091
. "${HOME}/.cargo/env"
rustc --version

# ── apt packages (system libs sglang needs at runtime) ─────
# - protobuf-compiler: sglang-grpc's build.rs invokes prost-build which
#   shells out to protoc to compile .proto files into Rust.
# - libnuma1: sgl_kernel's sm100/common_ops.abi3.so dynamically links to
#   libnuma.so.1 (NUMA-aware allocation). The nvidia/cuda base image
#   doesn't ship it; without it, every sglang launch dies with
#   `ImportError: libnuma.so.1: cannot open shared object file`.
# Pod runs as root.
echo "--- :package: Installing apt deps (protoc + libnuma1)"
NEEDED=()
command -v protoc >/dev/null 2>&1 || NEEDED+=("protobuf-compiler")
[[ -e /usr/lib/x86_64-linux-gnu/libnuma.so.1 || -e /lib/x86_64-linux-gnu/libnuma.so.1 ]] || NEEDED+=("libnuma1")
if [[ ${#NEEDED[@]} -gt 0 ]]; then
    apt-get update
    apt-get install -y --no-install-recommends "${NEEDED[@]}"
fi
protoc --version

# ── SGLang (pre-merge fork) FIRST so it pins torch ─────────
# SGLang's pyproject.toml pins torch==2.9.1+cu130 via [tool.uv.sources].
# Install it BEFORE LMCache so the c_ops extension compiled by LMCache's
# editable install in the next step links against the final torch ABI.
# Reversing this order (the prior bug) leaves LMCache's c_ops.so built
# against the base image's pre-installed torch, then sglang upgrades torch,
# and at runtime c_ops fails with `undefined symbol: _ZN3c104cuda29c10_cuda
# _check_implementation...`. The same ABI mismatch also breaks sglang's
# sgl_kernel architecture-specific common_ops load on Blackwell SM 12.0.
echo "--- :python: Installing SGLang from pre-merge fork"
SGLANG_FORK_URL="git+https://github.com/Shaoting-Feng/sglang.git@shaoting/sglang-lmcache-mp-nonlayerwise#subdirectory=python"
uv pip install "${SGLANG_FORK_URL}"

# ── LMCache (CI checkout) AFTER torch is pinned ────────────
echo "--- :python: Installing LMCache from source"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready"
python -c "import lmcache, sglang; print(f'sglang={sglang.__version__}; lmcache installed from source')"
