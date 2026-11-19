#!/usr/bin/env bash
# MetaX MACA bare-metal PR smoke tests. Runs on every PR build so contributors
# get quick MACA feedback without waiting on the manually triggered full suite
# (several thousand tests across the whole tests/ tree; see run-unit-tests.sh).
# Same environment setup as the full suite (common-setup.sh);
# only the test scope differs.
#
# Scope keeps platform detection/dispatch, native extension bindings, and
# connector/API glue, while excluding the largest subprocess and multi-node
# directories. Can be refined further with real per-test timing
# (durations/test.html from a full run) as it becomes available.
set -euo pipefail

# shellcheck source=./common-setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-setup.sh"

LMCACHE_TRACK_USAGE="false" \
pytest --maxfail=1 \
    tests/v1/platform \
    tests/v1/lmcache_native \
    tests/v1/gpu_connector \
    tests/v1/plugin \
    tests/v1/cli \
    --ignore=tests/v1/platform/devices/cuda/test_cuda_ipc_wrapper.py \
    --ignore=tests/v1/platform/devices/cuda/test_timeline_semaphore_event_ipc.py \
    --ignore=tests/v1/platform/devices/cuda/test_vmm_ipc_wrapper.py

# Whole files ignored -- all three depend on NVIDIA's `cuda.bindings`
# (cuda-python) package for raw driver-level IPC calls
# (lmcache/v1/platform/devices/cuda/utils.py's _import_cuda_bindings()), which isn't
# installed on MACA and isn't expected to work there even if it were -- it
# binds to NVIDIA's own driver ABI. All three already skip this area for
# ROCm for the same reason; MACA just isn't covered by that check since it's
# ROCm-specific. Affected tests either fail cleanly or hang (a spawned child
# process crashes before writing to a pipe the parent blocks on with no
# timeout) -- see run-unit-tests.sh for the full exclusion list and evidence.
# The real fix is upstream: generalize the ROCm-only skip to cover any
# environment without `cuda.bindings`.
#
# Matches the main CUDA pipeline.yml's own end-of-step cleanup on its
# bare-metal queue: delete the whole workspace so the next build starts from
# a fresh checkout instead of accumulating anything (build artifacts, stray
# temp files, .git growth) beyond what common-setup.sh's targeted cleanup
# already knows to remove. Requires passwordless sudo for this agent user.
TARGET="${BUILDKITE_BUILD_CHECKOUT_PATH:-$PWD}"
echo "Deleting current workspace ${TARGET}"
if [[ -z "${TARGET}" || "${TARGET}" == "/" ]]; then
    echo "Refusing to delete unsafe workspace path: '${TARGET}'" >&2
    exit 1
fi
cd /
sudo rm -rf -- "${TARGET}"
