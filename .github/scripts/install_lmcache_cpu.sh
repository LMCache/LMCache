#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install lmcache in editable, CPU-only mode (NO_GPU_EXT=1), or build and
# install the lmcache-client wheel when LMCACHE_INSTALL_CLIENT_WHEEL=1.
# Assumes vLLM (and therefore torch) was already installed by
# install_vllm_cpu.sh, so we use --no-deps to keep that pinned torch
# in place.
#
# Usage:
#   install_lmcache_cpu.sh           # plain pip
#   PIP_BIN="uv pip" install_lmcache_cpu.sh

set -euo pipefail

PIP_BIN="${PIP_BIN:-pip}"

export NO_GPU_EXT=1
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"

${PIP_BIN} install --upgrade pip
${PIP_BIN} install -r requirements/build.txt
${PIP_BIN} install -r requirements/common.txt
${PIP_BIN} install -r requirements/cli.txt
if [[ "${LMCACHE_INSTALL_CLIENT_WHEEL:-0}" == "1" ]]; then
  ${PIP_BIN} install build
  client_wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/lmcache-client-wheel.XXXXXX")"
  trap 'rm -rf "${client_wheel_dir}"' EXIT
  scripts/build_lmcache_client.sh --output-dir "${client_wheel_dir}"
  client_wheels=("${client_wheel_dir}"/lmcache_client-*.whl)
  if [[ ${#client_wheels[@]} -ne 1 || ! -f "${client_wheels[0]}" ]]; then
    echo "Expected exactly one lmcache-client wheel" >&2
    exit 1
  fi
  ${PIP_BIN} install "${client_wheels[0]}" --no-deps --force-reinstall
  repo_root="${PWD}"
  (cd /tmp && python "${repo_root}/.github/scripts/assert_client_wheel.py")
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "LMCACHE_CLI_MODE=module" >> "${GITHUB_ENV}"
  fi
  rm -rf "${client_wheel_dir}"
  trap - EXIT
else
  ${PIP_BIN} install -e . --no-deps --no-build-isolation
fi

echo "Generating LMCache gRPC bindings"
python lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py

python -c "import lmcache, vllm; \
print('lmcache:', lmcache.__version__, 'vllm:', vllm.__version__)"
