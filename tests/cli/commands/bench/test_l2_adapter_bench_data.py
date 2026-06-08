# SPDX-License-Identifier: Apache-2.0
"""Tests for L2 adapter benchmark memory object construction."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.command import run_l2_adapter_bench
from lmcache.cli.commands.bench.l2_adapter_bench.data import (
    create_l1_memory_desc,
    make_aligned_l1_buffer,
    make_memory_objects,
)
from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult
from lmcache.v1.memory_management import MemoryFormat


def test_make_memory_objects_allocates_views_from_l1_buffer() -> None:
    align_bytes = 64
    data_size = 100
    slot_size = 128
    start_offset = 64
    l1_buffer = make_aligned_l1_buffer(
        start_offset + 3 * slot_size,
        align_bytes=align_bytes,
    )
    l1_desc = create_l1_memory_desc(l1_buffer, align_bytes=align_bytes)

    objects = make_memory_objects(
        3,
        data_size,
        l1_buffer,
        start_offset,
        align_bytes=align_bytes,
    )

    assert len(objects) == 3
    for i, obj in enumerate(objects):
        offset = start_offset + i * slot_size
        assert obj.metadata.address == offset
        assert obj.metadata.phy_size == slot_size
        assert obj.metadata.shape == torch.Size([data_size])
        assert obj.metadata.dtype is torch.uint8
        assert obj.metadata.fmt is MemoryFormat.KV_2LTD
        assert obj.raw_data.data_ptr() == l1_desc.ptr + offset
        assert l1_desc.ptr <= obj.raw_data.data_ptr() < l1_desc.ptr + l1_desc.size
        assert obj.raw_data.numel() == slot_size
        assert torch.all(obj.raw_data[:data_size] == (i & 0xFF))
        assert torch.all(obj.raw_data[data_size:] == 0)


def test_make_memory_objects_rejects_out_of_range_allocation() -> None:
    l1_buffer = make_aligned_l1_buffer(128, align_bytes=64)

    with pytest.raises(ValueError, match="exceed l1_buffer"):
        make_memory_objects(
            3,
            64,
            l1_buffer,
            0,
            align_bytes=64,
        )


def test_run_l2_adapter_bench_passes_l1_backed_objects_to_runners() -> None:
    fake_adapter = MagicMock()
    captured_desc = None
    captured_batches = []

    def fake_create_l2_adapter(_adapter_cfg, l1_memory_desc=None):
        nonlocal captured_desc
        captured_desc = l1_memory_desc
        return fake_adapter

    def fake_bench_store(
        _adapter,
        in_flight,
        num_keys,
        data_size,
        rounds,
        keys_for_round,
        objs_for_round,
        log,
    ):
        del keys_for_round, log
        captured_batches.extend(objs_for_round(0))
        return BenchResult(
            operation="Store",
            in_flight=in_flight,
            num_keys=num_keys,
            data_size_bytes=data_size,
            round_durations=[0.1] * rounds,
            success_counts=[in_flight * num_keys] * rounds,
        )

    def fake_bench_lookup(
        _adapter,
        in_flight,
        num_keys,
        rounds,
        keys_for_round,
        log,
        expected_max_hit_rate=0.0,
        expected_hit_count=0,
    ):
        del keys_for_round, log
        return BenchResult(
            operation="Lookup",
            in_flight=in_flight,
            num_keys=num_keys,
            data_size_bytes=0,
            round_durations=[0.1] * rounds,
            success_counts=[0] * rounds,
            expected_max_hit_rate=expected_max_hit_rate,
            expected_hit_count=expected_hit_count,
        )

    def fake_bench_load(
        _adapter,
        in_flight,
        num_keys,
        data_size,
        rounds,
        keys_for_round,
        objs_for_round,
        log,
    ):
        del keys_for_round, log
        captured_batches.extend(objs_for_round(0))
        return BenchResult(
            operation="Load",
            in_flight=in_flight,
            num_keys=num_keys,
            data_size_bytes=data_size,
            round_durations=[0.1] * rounds,
            success_counts=[in_flight * num_keys] * rounds,
        )

    command = MagicMock()
    metrics = MagicMock()
    metrics.add_section.return_value = MagicMock()
    command.create_metrics.return_value = metrics
    args = argparse.Namespace(
        l2_adapter=['{"type":"mock","max_size_gb":1,"mock_bandwidth_gb":10}'],
        num_keys=2,
        in_flight=2,
        data_size_kb=1,
        rounds=1,
        warmup_rounds=0,
        lookup_max_hit_rate=0.0,
        skip_verify=True,
        only=None,
        quiet=True,
        format=None,
        output=None,
    )

    with (
        patch(
            "lmcache.v1.distributed.l2_adapters.config."
            "parse_args_to_l2_adapters_config",
            return_value=SimpleNamespace(adapters=[object()]),
        ),
        patch(
            "lmcache.v1.distributed.l2_adapters.create_l2_adapter",
            side_effect=fake_create_l2_adapter,
        ),
        patch(
            "lmcache.cli.commands.bench.l2_adapter_bench.runner.bench_store",
            side_effect=fake_bench_store,
        ),
        patch(
            "lmcache.cli.commands.bench.l2_adapter_bench.runner.bench_lookup",
            side_effect=fake_bench_lookup,
        ),
        patch(
            "lmcache.cli.commands.bench.l2_adapter_bench.runner.bench_load",
            side_effect=fake_bench_load,
        ),
    ):
        run_l2_adapter_bench(command, args)

    assert captured_desc is not None
    assert len(captured_batches) == 4
    for batch in captured_batches:
        for obj in batch:
            assert obj.raw_data.data_ptr() == captured_desc.ptr + obj.metadata.address
            assert obj.metadata.phy_size == 4096
            assert 0 <= obj.metadata.address < captured_desc.size
            assert obj.metadata.address % captured_desc.align_bytes == 0
            assert obj.raw_data.data_ptr() < captured_desc.ptr + captured_desc.size
    fake_adapter.close.assert_called_once_with()
