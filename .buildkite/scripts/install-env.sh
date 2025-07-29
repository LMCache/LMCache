#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 <BUILDKITE_BUILD_ID>" >&2
  exit 1
fi

BUILDKITE_BUILD_ID="$1"

VENV_DIR=".venv-$BUILDKITE_BUILD_ID"
PYTHON_BIN="/usr/bin/python3.10"
if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python 3.10 at: $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

# CUDA version
CUDA_VERSION="12.1"

uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements/common.txt
uv pip install -r requirements/test.txt

# Export CUDA variables
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"
