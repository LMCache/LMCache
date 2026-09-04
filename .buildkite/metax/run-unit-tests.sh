#!/usr/bin/env bash
# MetaX MACA bare-metal FULL unit test suite. Runs on push to dev (post-merge),
# not on PRs -- see pipeline.yml for the trigger split and run-smoke-tests.sh
# for the fast PR-facing subset. Runs directly on a dedicated single-GPU host
# registered to the "metax-maca" queue -- no K8s/GPU-Operator involved, unlike
# the multi-GPU vendor pipelines under k3_tests/. There is no pick-free-gpu
# step because this queue's agent owns its GPU exclusively.
set -euo pipefail

# shellcheck source=./common-setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-setup.sh"

LMCACHE_TRACK_USAGE="false" \
pytest --maxfail=1 --cov=lmcache \
    --cov-report term --cov-report=html:coverage-test \
    --cov-report=xml:coverage-test.xml --html=durations/test.html \
    --ignore=tests/disagg --ignore=tests/v1/test_pos_kernels.py \
    --ignore=tests/v1/test_nixl_batched_contains.py \
    --ignore=tests/v1/test_device_id_race.py \
    --ignore=tests/v1/test_nixl_multipath.py \
    --ignore=tests/skipped \
    --ignore=tests/v1/storage_backend/test_eic.py \
    --deselect="tests/v1/distributed/serde/test_turboquant.py::test_turboquant_direct_roundtrip_cuda[turboquant_k8v4-2.6-0.95]" \
    --ignore=tests/v1/mp_coordinator/test_instances_usage_e2e.py \
    --ignore=tests/v1/platform/test_cuda_ipc_wrapper.py \
    --ignore=tests/v1/platform/test_timeline_semaphore_event_ipc.py \
    --ignore=tests/v1/multiprocess/test_mq.py \
    --ignore=tests/v1/multiprocess/test_cb_plan_executor_gpu.py \
    --ignore=tests/v1/multiprocess/test_custom_types.py \
    --ignore=tests/v1/multiprocess/test_engine_driven_transfer.py \
    --ignore=tests/v1/multiprocess/test_free_locks.py \
    --ignore=tests/v1/multiprocess/test_query_lookup_hits.py \
    --deselect="tests/cli/commands/bench/test_server_bench.py::TestUnregisterKVCache::test_data_mode_sends_engine_driven_unregister" \
    --deselect="tests/v1/mp_coordinator/test_key_directory.py::test_token_ids_outside_uint32_leave_the_binding_unfilled" \
    --deselect="tests/v1/test_torch_ops.py::TestScenarios::test_1_scenario[cuda_ops-load_and_reshape_flash-scenario_load_and_reshape_flash]" \
    --deselect="tests/v1/test_torch_ops.py::TestScenarios::test_2_compare[multi_layer_block_kv_transfer]"

# Not deselected/ignored above for a MACA capability reason, documented
# separately so a future re-run on different hardware knows what to revisit:
#
# - test_turboquant_direct_roundtrip_cuda[turboquant_k8v4-...]: MACA's Triton
#   backend does not support the FP8 type conversion this one preset needs
#   (tl.float8e4b15 / tl.float8e4nv) -- confirmed via
#   triton.compiler.errors.CompilationError, reproducible in isolation. The
#   other 3 parametrizations of the same test function pass; only this one
#   preset is affected.
# - test_instances_usage_e2e.py (all cases): not a MACA issue -- all 7 cases
#   pass individually. They fail only when run immediately after the
#   GPU-heavy tests earlier in this suite, on this single-GPU CI host: a
#   background uvicorn thread doesn't bind its port within the test's
#   hardcoded 5s timeout under that load. Revisit if this queue ever moves
#   to less contended hardware.
# - test_cuda_ipc_wrapper.py, test_timeline_semaphore_event_ipc.py: whole
#   files ignored. Both depend on NVIDIA's `cuda.bindings` (cuda-python)
#   package for raw driver-level IPC calls (lmcache/v1/platform/cuda/utils.py's
#   _import_cuda_bindings()), which isn't installed on MACA and isn't
#   expected to work there even if it were -- it binds to NVIDIA's own
#   driver ABI, not a CUDA-API surface MACA's cu-bridge shims. Both files
#   already skip this area for ROCm for the same reason; MACA just isn't
#   covered by that check since it's ROCm-specific. Affected tests either
#   fail cleanly or hang (a spawned child process crashes before writing to
#   a pipe the parent blocks on with no timeout). The real fix is upstream:
#   generalize the ROCm-only skip to cover any environment without
#   `cuda.bindings`.

cat << EOF | buildkite-agent annotate --style "info"
  Read the <a href="artifact://coverage-test/index.html">uploaded coverage report</a>
EOF

# Matches the main CUDA pipeline.yml's own end-of-step cleanup on its
# bare-metal queue: delete the whole workspace so the next build starts from
# a fresh checkout instead of accumulating anything (build artifacts, stray
# temp files, .git growth) beyond what common-setup.sh's targeted cleanup
# already knows to remove. Requires passwordless sudo for this agent user.
export TARGET="$PWD"
echo "Deleting current workspace $TARGET"
cd /
sudo rm -rf "$TARGET"
