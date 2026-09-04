#!/usr/bin/env bash
# MetaX MACA bare-metal PR smoke tests. Runs on every PR build so contributors
# get quick MACA feedback without waiting on the full suite (several thousand
# tests across the whole tests/ tree; see run-unit-tests.sh, which runs
# instead on push to dev). Same environment
# setup as the full suite (common-setup.sh); only the test scope differs.
#
# Scope keeps the smaller, single-process directories (core compute
# kernels, platform detection/dispatch, native extension bindings,
# connector/API glue) and excludes the largest, subprocess-/multi-node-heavy
# directories -- distributed/, multiprocess/, mp_observability/,
# mp_coordinator/, storage_backend/ (also I/O-heavy) -- which stay in the
# full post-merge run. Can be refined further with real per-test timing
# (durations/test.html from a full run) as it becomes available.
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

# Whole files ignored -- both depend on NVIDIA's `cuda.bindings`
# (cuda-python) package for raw driver-level IPC calls
# (lmcache/v1/platform/cuda/utils.py's _import_cuda_bindings()), which isn't
# installed on MACA and isn't expected to work there even if it were -- it
# binds to NVIDIA's own driver ABI. Both files already skip this area for
# ROCm for the same reason; MACA just isn't covered by that check since it's
# ROCm-specific. Affected tests either fail cleanly or hang (a spawned child
# process crashes before writing to a pipe the parent blocks on with no
# timeout) -- see run-unit-tests.sh for the full exclusion list and evidence.
# The real fix is upstream: generalize the ROCm-only skip to cover any
# environment without `cuda.bindings`.

# Matches the main CUDA pipeline.yml's own end-of-step cleanup on its
# bare-metal queue: delete the whole workspace so the next build starts from
# a fresh checkout instead of accumulating anything (build artifacts, stray
# temp files, .git growth) beyond what common-setup.sh's targeted cleanup
# already knows to remove. Requires passwordless sudo for this agent user.
export TARGET="$PWD"
echo "Deleting current workspace $TARGET"
cd /
sudo rm -rf "$TARGET"
