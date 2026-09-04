#!/usr/bin/env bash
# Shared environment setup for both the PR smoke-test and the post-merge full
# unit-test run on the "metax-maca" bare-metal queue. Sourced, not executed
# directly -- both run-smoke-tests.sh and run-unit-tests.sh `source` this
# before their respective pytest invocations, so a change to how the venv,
# MACA SDK env, or install steps work only needs to happen in one place.
set -euo pipefail

echo "$PWD" # for debugging

# This queue's host has only tens of GB of disk, all of it shared with the
# repo checkout, the venv, and uv's package cache -- unlike the k3_tests/
# vendor pipelines, which run in ephemeral K8s pods that discard their entire
# filesystem after each build. A per-build-id venv (the k3_tests/ convention)
# would accumulate forever here, so this reuses one fixed venv path,
# recreated from scratch each run, plus a disk-space guard so a creeping leak
# fails loudly instead of degrading into mysterious ENOSPC errors weeks later.
MIN_FREE_GB=10
free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
if (( free_gb < MIN_FREE_GB )); then
    echo "ERROR: only ${free_gb}GiB free on this host (need >= ${MIN_FREE_GB}GiB)." >&2
    echo "Investigate disk usage (du -sh .venv ~/.cache/uv coverage-test) before retrying." >&2
    exit 1
fi

# MACA SDK env (cu-bridge nvcc-compatible compiler + runtime libs).
export MACA_PATH=/opt/maca
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export PATH="${CUCC_PATH}/bin:${CUCC_PATH}/tools:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"

# Confirmed 2026-09-04: without this, cross-process CUDA event import/wait
# (tests/v1/platform/test_event_ipc_ordering.py's producer/consumer handoff)
# times out waiting on a handle that never arrives; with it, the same test
# passes in ~16s. Does NOT fix every cross-process issue on this queue --
# tests/v1/multiprocess/test_mq.py's MP-server registration timeouts are
# unaffected by this flag and have a separate, still-unidentified cause.
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

# uv's package cache (~/.cache/uv by default) is what makes recreating .venv
# from scratch each run cheap (installs hardlink from it instead of
# re-downloading/rebuilding); it persists across builds by design. Bound its
# growth by dropping anything not referenced by the venv just built, rather
# than letting every historical package version pile up indefinitely.
# Pruned here (before tests run) rather than at the end so a test failure
# (which stops the script under set -e) doesn't skip it.
uv cache prune

# HuggingFace model/tokenizer cache (~/.cache/huggingface), if tests/v1/
# populates one, is NOT cleared -- unlike .venv, redownloading it every build
# would trade a disk problem for a network/time one. If disk pressure shows
# up again, check `du -sh ~/.cache/huggingface` first; clearing it is a
# manual step via the existing "Kill all" pipeline (.buildkite/pipelines/clean.yml),
# not automatic here.
