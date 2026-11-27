# SPDX-License-Identifier: Apache-2.0
"""Public LMCache contracts used by the separately distributed in-process engine."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher
from lmcache.v1.utils.bloom_filter import BloomFilter


def test_lookup_statistics_count_a_repeated_prefix_once() -> None:
    """A prefix repeated inside one prehashed batch is recorded as one new entry."""
    bloom = BloomFilter(expected_elements=128, false_positive_rate=0.01)
    assert bloom.add_batch_with_hashes_and_check([[1, 2], [1, 2], [2, 3]]) == 2
    assert bloom.get_statistics()["item_count"] == 2


@pytest.mark.parametrize(
    ("role", "worker_id", "expected"),
    [
        ("WORKER", 1, {"all_common", "worker_shared", "worker_1_probe"}),
        ("SCHEDULER", 0, {"all_common", "scheduler_0_probe"}),
    ],
)
def test_plugin_launch_respects_role_and_worker_id(
    tmp_path: Path, role: str, worker_id: int, expected: set[str]
) -> None:
    """Only plugins matching the role and optional worker ID receive the launch."""
    # A role prefix without a worker ID targets every worker of that role.
    plugins = (
        "all_common",
        "worker_shared",
        "worker_0_probe",
        "worker_1_probe",
        "scheduler_0_probe",
    )
    for name in plugins:
        (tmp_path / f"{name}.py").write_text(
            f"#!{sys.executable}\n"
            "from pathlib import Path\n"
            "Path(__file__).with_suffix('.ran').touch()\n"
        )
    config = SimpleNamespace(
        runtime_plugin_locations=[str(tmp_path)], to_json=lambda: "{}"
    )
    launcher = RuntimePluginLauncher(config, role, worker_count=2, worker_id=worker_id)
    try:
        launcher.launch_plugins()
        for proc in launcher.plugin_processes:
            assert proc.wait(timeout=10) == 0
        assert {path.stem for path in tmp_path.glob("*.ran")} == expected
    finally:
        launcher.stop_plugins()
