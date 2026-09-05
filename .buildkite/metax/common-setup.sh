#!/usr/bin/env bash
# Shared environment setup for both the PR smoke-test and the post-merge full
# unit-test run on the "metax-maca" bare-metal queue. Sourced, not executed
# directly -- both run-smoke-tests.sh and run-unit-tests.sh `source` this
# before their respective pytest invocations, so a change to how the venv,
# MACA SDK env, or install steps work only needs to happen in one place.
set -euo pipefail

echo "$PWD" # for debugging

# This queue's host has limited disk space, not the effectively-unlimited
# storage an ephemeral K8s pod gets, so this reuses one fixed venv path
# (recreated from scratch each run) instead of a new one per build, plus a
# disk-space guard so a creeping leak fails loudly instead of degrading into
# mysterious ENOSPC errors weeks later.
MIN_FREE_GB=10
free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
if (( free_gb < MIN_FREE_GB )); then
    echo "ERROR: only ${free_gb}GiB free on this host (need >= ${MIN_FREE_GB}GiB)." >&2
    echo "Investigate disk usage (du -sh .venv ~/.cache/uv coverage-test) before retrying." >&2
    exit 1
fi

# This host's proxy (set via the agent's own environment, for GitHub/PyPI
# reachability) is otherwise honored by urllib/requests for any destination,
# including 127.0.0.1 -- which breaks tests that spin up a real local
# HTTPServer and immediately query it, since the proxy has no idea what to
# do with a random localhost port. Exempt localhost unconditionally.
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

# MACA SDK env (cu-bridge nvcc-compatible compiler + runtime libs).
export MACA_PATH=/opt/maca
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export PATH="${CUCC_PATH}/bin:${CUCC_PATH}/tools:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"

# MetaX GPUs require this whenever multiple processes access the same GPU
# concurrently, which several tests in this suite do.
export MACA_MPS_MODE=1

# Stale artifacts from previous runs (this queue's single agent reuses the
# same checkout dir; Buildkite's default checkout does not git-clean
# untracked files between builds).
rm -rf .venv coverage-test durations .pytest_cache
find . -name "__pycache__" -type d -prune -exec rm -rf {} +

uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel

# MACA-enabled torch is published on MetaX's own pip index, not PyPI.
uv pip install torch \
    --index-url https://repos.metax-tech.com/r/maca-pypi/simple

uv pip install -r requirements/common.txt
uv pip install -r requirements/test.txt
# maca_core.txt lists mcpy (MetaX's cupy equivalent, MP mode only); also on
# MetaX's index, not PyPI.
uv pip install -r requirements/maca_core.txt \
    --index-url https://repos.metax-tech.com/r/maca-pypi/simple

BUILD_WITH_MACA=1 uv pip install -e . --no-build-isolation
uv pip freeze

# Keep uv's package cache from growing unbounded across builds (pruned here,
# before tests run, so a test failure doesn't skip it).
uv cache prune

# HuggingFace model/tokenizer cache is left alone here; clear it manually
# via the existing "Kill all" pipeline (.buildkite/pipelines/clean.yml) if
# disk pressure shows up.
