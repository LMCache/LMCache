#!/usr/bin/env bash
# MetaX MACA bare-metal PR smoke tests. Runs on every PR build so contributors
# get quick MACA feedback without waiting on the full suite (several thousand
# tests across the whole tests/ tree; see run-unit-tests.sh, which runs
# instead on push to dev). Same environment
# setup as the full suite (common-setup.sh); only the test scope differs.
#
# Scope is a first-pass split, not a scientifically-tuned one: it keeps the
# smaller, single-process directories (core compute kernels, platform
# detection/dispatch, native extension bindings, connector/API glue) and
# excludes the largest, subprocess-/multi-node-heavy directories --
# distributed/ (66 test files), multiprocess/ (51), mp_observability/ (33),
# mp_coordinator/ (31), storage_backend/ (35, also I/O-heavy) -- which stay
# in the full post-merge run. Revisit this split using real per-test timing
# (durations/test.html from a full run) once available; this is sized by
# file count and known subprocess-heaviness, not measured smoke-run duration.
set -euo pipefail

# shellcheck source=./common-setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-setup.sh"

LMCACHE_TRACK_USAGE="false" \
pytest --maxfail=1 \
    tests/v1/compute \
    tests/v1/platform \
    tests/v1/lmcache_native \
    tests/v1/gpu_connector \
    tests/v1/cache_controller \
    tests/v1/internal_api_server \
    tests/v1/shm_allocator \
    tests/v1/lookup_client \
    tests/v1/plugin \
    tests/v1/cli \
    --ignore=tests/v1/platform/test_cuda_ipc_wrapper.py \
    --ignore=tests/v1/platform/test_timeline_semaphore_event_ipc.py

# Whole files ignored (not deselected test-by-test) -- root cause CONFIRMED
# 2026-09-03 and the same for every failure/hang found in both:
# lmcache/v1/platform/cuda/utils.py's _import_cuda_bindings() lazily imports
# NVIDIA's `cuda.bindings` (cuda-python) package for raw driver-level IPC
# calls, which is not installed on MACA and structurally isn't expected to
# work there even if it were (it binds to NVIDIA's own driver ABI). Both
# files' own pytestmark already exclude ROCm from this exact area; MACA just
# isn't covered by that check since it's ROCm-specific.
#
# Symptom is either a clean FAILED (`ModuleNotFoundError: No module named
# 'cuda'`, confirmed via isolated -v --tb=long runs) or a genuine HANG when
# the same crash happens inside a spawned child process instead of the
# test's own process (the child dies before writing its pipe message, and
# the parent's recv_bytes() has no timeout). Reproduced this way at least 3
# times across both files -- see run-unit-tests.sh for the full evidence
# per test.
#
# Excluding both whole files (rather than continuing to chase individual
# hangs one at a time, which is how the first 3 of these were found) is a
# deliberate, broad, temporary choice. Tests in this same area that only
# reference the isolated_ipc *config switch* without touching
# RawCudaIPCWrapper/cuda.bindings (test_event_ipc.py, test_isolated_ipc.py,
# and outside tests/v1/platform: test_vllm_mp_adapter.py,
# test_trtllm_integration.py, tests/v1/multiprocess/test_config.py) were
# NOT excluded. The real fix is upstream: generalize both files' ROCm-only
# skipif to also cover MACA.

# Matches the main CUDA pipeline.yml's own end-of-step cleanup on its
# bare-metal queue: delete the whole workspace so the next build starts from
# a fresh checkout instead of accumulating anything (build artifacts, stray
# temp files, .git growth) beyond what common-setup.sh's targeted cleanup
# already knows to remove. Requires passwordless sudo for this agent user.
export TARGET="$PWD"
echo "Deleting current workspace $TARGET"
cd /
sudo rm -rf "$TARGET"
