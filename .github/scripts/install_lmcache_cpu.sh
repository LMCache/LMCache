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
  cp pyproject.toml /tmp/lmcache-pyproject.toml
  trap 'cp /tmp/lmcache-pyproject.toml pyproject.toml' EXIT
  cp pyproject_client.toml pyproject.toml
  rm -rf dist/
  LMCACHE_CLIENT_BUILD=1 python -m build --wheel --no-isolation
  ${PIP_BIN} install dist/lmcache_client-*.whl --no-deps --force-reinstall
  repo_root="${PWD}"
  (cd /tmp && python "${repo_root}/.github/scripts/assert_client_wheel.py")
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "LMCACHE_CLI_MODE=module" >> "${GITHUB_ENV}"
  fi
  cp /tmp/lmcache-pyproject.toml pyproject.toml
  trap - EXIT
else
  ${PIP_BIN} install -e . --no-deps --no-build-isolation
fi

echo "Generating LMCache gRPC bindings"
python lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py

python -c "import lmcache, vllm; \
print('lmcache:', lmcache.__version__, 'vllm:', vllm.__version__)"
