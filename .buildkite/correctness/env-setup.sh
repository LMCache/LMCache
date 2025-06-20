#!/bin/bash
set -e

# ASSUMPTION: CUDA is installed in /usr/local/cuda-{version}
CUDA_VERSION="12.1"
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Ensure that the `uv` binary is available. Install it on-the-fly if missing
# (common on fresh Buildkite agents).
if ! command -v uv >/dev/null 2>&1; then
  echo "[env-setup] 'uv' not found – installing via standalone script..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops an activation helper into ~/.local/bin/env which adds
  # ~/.local/bin to PATH. Source it so that `uv` is immediately available in
  # the current shell without requiring a new login shell.
  if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env"
  fi
fi

# If a cached virtual-environment already exists (restored by Buildkite cache),
# simply activate it and return early to make this script idempotent.
if [ -d ".venv" ]; then
  echo "[env-setup] Reusing previously cached Python virtual environment (.venv)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exit 0
fi

uv venv --python 3.12
source .venv/bin/activate

# Extra dependencies needed for the MMLU scripts
uv pip install requests pandas numpy tqdm matplotlib fastapi transformers
# Install lmcache from source
# the lmcache wheel also gives us access to:
# lmcache_server entrypoint
# lmcache_controller entrypoint
cd ../../
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[env-setup] 'nvcc' not found – installing LMCache without CUDA extensions"
  export NO_CUDA_EXT=1
fi

uv pip install -e .

uv pip install vllm

# come back to the correctness directory
cd $SCRIPT_DIR
# Download the MMLU dataset
wget -q --show-progress https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar