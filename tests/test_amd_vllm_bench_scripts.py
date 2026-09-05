# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the AMD vLLM benchmark entry point."""

# Standard
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_amd_container_exports_shared_multiprocess_device_contract() -> None:
    """The direct AMD entry point must configure the shared launcher."""
    script = (
        ROOT / ".buildkite" / "scripts" / "amd-vllm-bench-container.sh"
    ).read_text()

    assert 'export TORCH_DEVICE_TYPE="cuda"' in script
    assert 'export VLLM_TARGET_DEVICE="cuda"' in script
    assert 'export DEVICE_AFFINITY_VAR="HIP_VISIBLE_DEVICES"' in script
    assert (
        "exec .buildkite/k3_tests/multiprocess/scripts/run-single-test.sh "
        "vllm_bench"
    ) in script
