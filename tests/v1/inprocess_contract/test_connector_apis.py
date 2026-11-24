# SPDX-License-Identifier: Apache-2.0
"""Public LMCache contracts used by the separately distributed in-process engine."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.utils import (
    cdiv,
    convert_tokens_to_list,
    parse_mixed_slot_mapping,
)
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher
from lmcache.v1.utils.bloom_filter import BloomFilter


@pytest.mark.parametrize(("tokens", "chunks"), [(0, 0), (16, 1), (17, 2)])
def test_chunk_count_rounds_up(tokens: int, chunks: int) -> None:
    """The connector counts a partially filled chunk exactly once."""
    assert cdiv(tokens, 16) == chunks


@pytest.mark.parametrize("tokens", [[8, 9, 10, 11], torch.tensor([8, 9, 10, 11]), None])
def test_event_token_slice(tokens: list[int] | torch.Tensor | None) -> None:
    """Event token ranges include the end index and permit missing token IDs."""
    assert convert_tokens_to_list(tokens, 1, 2) == ([] if tokens is None else [9, 10])


def test_slot_mapping_expands_inclusive_ranges_in_order() -> None:
    """The cache API accepts a mixture of individual slots and inclusive ranges."""
    assert parse_mixed_slot_mapping("5, [1,3], 8") == ([5, 1, 2, 3, 8], None)


@pytest.mark.parametrize("invalid", ["", ",,,", "[3,1]", "[[1,2]]", "[2,3", "four"])
def test_slot_mapping_rejects_invalid_input(invalid: str) -> None:
    """Invalid input returns an error without a partially parsed slot list."""
    slots, error = parse_mixed_slot_mapping(invalid)
    assert slots is None
    assert error is not None and "error" in error


def test_lookup_statistics_count_new_bloom_entries_and_reset() -> None:
    """A repeated prehashed prefix contributes once to the lookup statistics."""
    bloom = BloomFilter(expected_elements=128, false_positive_rate=0.01)
    assert bloom.add_batch_with_hashes_and_check([[1, 2], [1, 2], [2, 3]]) == 2
    assert bloom.add_batch_with_hashes_and_check([[1, 2]]) == 0
    assert bloom.get_statistics()["item_count"] == 2
    assert bloom.get_fill_rate() > 0
    bloom.add(42)
    assert bloom.contains(42)
    bloom.clear()
    assert not bloom.contains(42)
    assert bloom.get_statistics()["item_count"] == 0


@pytest.mark.parametrize(
    ("role", "worker_id", "expected"),
    [
        ("WORKER", 1, {"all_common", "worker_1_probe"}),
        ("SCHEDULER", 0, {"all_common", "scheduler_0_probe"}),
    ],
)
def test_plugin_launch_respects_role_and_worker_id(
    tmp_path: Path, role: str, worker_id: int, expected: set[str]
) -> None:
    """Only plugins matching the role and optional worker ID receive the launch."""
    for name in ("all_common", "worker_0_probe", "worker_1_probe", "scheduler_0_probe"):
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
