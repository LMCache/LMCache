#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install lmcache in editable, CPU-only mode (NO_GPU_EXT=1).
# Assumes vLLM (and therefore torch) was already installed by
# install_vllm_cpu.sh, so we use --no-deps to keep that pinned torch
# in place.
#
# With LMCACHE_SERVER_VENV set, install the lmcache-client wheel instead, and
# the full lmcache wheel into a new venv there for the server, both built from
# this checkout.
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
if [[ -n "${LMCACHE_SERVER_VENV:-}" ]]; then
  ${PIP_BIN} install build -r requirements/client.txt
  # Against vLLM's torch, as the editable install below builds.
  wheels="$(mktemp -d)"
  CLIENT_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" \
    .github/scripts/build_client_wheel.sh "${wheels}" --no-isolation --skip-dependency-check
  python -m build --wheel --no-isolation --skip-dependency-check --outdir "${wheels}"
  ls "${wheels}"
  ${PIP_BIN} install --no-deps "${wheels}"/lmcache_client-*.whl
  if ${PIP_BIN} show lmcache >/dev/null 2>&1; then
    echo "The full lmcache distribution must not be in vLLM's environment" >&2
    exit 1
  fi

  # The server venv gets the torch both wheels were just built against.
  python -m venv "${LMCACHE_SERVER_VENV}"
  "${LMCACHE_SERVER_VENV}/bin/pip" install --index-url https://download.pytorch.org/whl/cpu \
    "torch==$(python -c 'import torch; print(torch.__version__)')"
  "${LMCACHE_SERVER_VENV}/bin/pip" install "${wheels}"/lmcache-*.whl -r requirements/cli.txt
  "${LMCACHE_SERVER_VENV}/bin/pip" list | grep -i '^lmcache'
else
  ${PIP_BIN} install -r requirements/common.txt
  ${PIP_BIN} install -r requirements/cli.txt
  ${PIP_BIN} install -e . --no-deps --no-build-isolation

  echo "Generating LMCache gRPC bindings"
  python lmcache/v1/transport/grpc/_proto_gen/_generate.py
fi

# Outside the checkout, so lmcache is the installed one.
(cd /tmp && python -I -c "import lmcache, vllm; \
print('lmcache:', lmcache.__version__, lmcache.__file__, 'vllm:', vllm.__version__)")
