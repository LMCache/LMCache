#!/usr/bin/env bash
# MUSA unit-test entrypoint. Bootstrap is shared; discovery stays local
# like the XPU unit lane.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ci-common.sh"

bootstrap_musa_ci 0
pytest_base_args

discover_unit_tests() {
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

allowlist = (
    "tests/test_*.py",
    "tests/cli/**/test_*.py",
    "tests/v1/**/test_*.py",
)
excluded = {
    # These modules import optional CUDA/Triton/vLLM components during
    # collection and are covered by their platform-specific jobs.
    "tests/v1/compute/attention/test_triton_kernels.py",
    "tests/v1/test_pos_kernels.py",
    # These lanes require CUDA/NIXL-specific host services.
    "tests/v1/test_device_id_race.py",
    "tests/v1/test_nixl_batched_contains.py",
    "tests/v1/test_nixl_multipath.py",
    "tests/v1/storage_backend/test_eic.py",
}

selected: set[str] = set()
for pattern in allowlist:
    selected.update(
        path.as_posix()
        for path in Path(".").glob(pattern)
        if path.is_file()
    )

for path in sorted(selected - excluded):
    print(path)
PY
}

mapfile -t UNIT_TEST_FILES < <(discover_unit_tests)
if [ "${#UNIT_TEST_FILES[@]}" -eq 0 ]; then
    fail "no MUSA unit-test files found under tests"
fi

log "Running ${#UNIT_TEST_FILES[@]} MUSA-compatible unit-test files"
printf '  %s\n' "${UNIT_TEST_FILES[@]}"
UNIT_REPORT_ARGS=(
    --cov=lmcache
    --cov-report=term
    "--cov-report=xml:${ARTIFACT_DIR}/coverage.xml"
    "--cov-report=html:${ARTIFACT_DIR}/coverage-html"
    "--junitxml=${ARTIFACT_DIR}/junit.xml"
    "--html=${ARTIFACT_DIR}/pytest.html"
    --self-contained-html
)
run_pytest "${PYTEST_ARGS[@]}" \
    "${UNIT_REPORT_ARGS[@]}" \
    --deselect=tests/v1/distributed/serde/test_turboquant.py::test_turboquant_direct_roundtrip_cuda \
    --deselect=tests/v1/platform/musa/test_musa_ipc_wrapper_integration.py::test_musa_ipc_wrapper_two_process_round_trip \
    --deselect=tests/v1/platform/musa/test_musa_mp_block_transfer.py::test_musa_cache_context_sglang_mha_operand_round_trip \
    --deselect=tests/v1/gpu_connector/test_blocks_first_cs_kv_format.py::test_mp_gather_scatter_roundtrip \
    --deselect=tests/v1/mp_coordinator/test_key_directory.py::test_token_ids_outside_uint32_leave_the_binding_unfilled \
    "--deselect=tests/v1/multiprocess/test_engine_driven_transfer.py::test_compute_kv_layout_and_gather_scatter_roundtrip[fused_hnd]" \
    "--deselect=tests/v1/multiprocess/test_engine_driven_transfer.py::test_compute_kv_layout_and_gather_scatter_roundtrip[fused_nhd]" \
    --deselect=tests/v1/multiprocess/test_engine_driven_transfer.py::test_gather_scatter_roundtrip_hnd_layout \
    "--deselect=tests/v1/multiprocess/test_engine_driven_transfer.py::test_scatter_rounds_down_partial_block_skip_first_n_tokens[hnd]" \
    -m "not cuda and not xpu and not sglang" \
    "${UNIT_TEST_FILES[@]}" \
    2>&1 | tee "${ARTIFACT_DIR}/pytest.log"
log "MUSA unit tests finished successfully"
