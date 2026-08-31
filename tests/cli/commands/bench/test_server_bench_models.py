# SPDX-License-Identifier: Apache-2.0
"""Contract tests for server-bench data models."""

# Standard
from dataclasses import FrozenInstanceError
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench.config import (
    WorkerSpec,
    parse_args_to_config,
)
from lmcache.cli.commands.bench.server_bench.runtime import (
    LookupResult,
    TransferResult,
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "rpc_url": "tcp://localhost:5555",
        "url": "http://localhost:8080",
        "mode": "gpu",
        "transfer_mode": "auto",
        "tp_size": 1,
        "use_mla": False,
        "num_tokens": 512,
        "kvcache_shape_spec": "(2,1024,16,8,128):float16:32",
        "num_blocks": 1024,
        "block_size": 16,
        "start": 0,
        "end": 3,
        "quiet": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("mode", "transfer_mode", "uses_handle"),
    [
        ("gpu", "auto", True),
        ("cpu", "auto", False),
        ("cpu", "lmcache_driven", True),
        ("gpu", "engine_driven", False),
    ],
)
def test_bench_config_preserves_cli_routing(
    mode: str,
    transfer_mode: str,
    uses_handle: bool,
) -> None:
    config = parse_args_to_config(
        _args(mode=mode, transfer_mode=transfer_mode, tp_size=0)
    )

    assert config.mode == mode
    assert config.transfer_mode == transfer_mode
    assert config.tp_size == 1
    assert config.uses_handle_transfer is uses_handle


def test_worker_spec_is_immutable_and_validated() -> None:
    spec = WorkerSpec(0, 1000, 0, 1, True)
    with pytest.raises(FrozenInstanceError):
        spec.rank = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="kv_world_size must be positive"):
        WorkerSpec(0, 1000, 0, 0, True)


@pytest.mark.parametrize(
    ("hit_chunks", "expected"),
    [(0, "is_full_miss"), (1, "is_partial_hit"), (2, "is_full_hit")],
)
def test_lookup_result_classifies_hit_ranges(
    hit_chunks: int,
    expected: str,
) -> None:
    result = LookupResult(hit_chunks, total_chunks=2, latency_ms=1.5)

    assert result.succeeded
    assert getattr(result, expected)


def test_lookup_result_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="between zero and total_chunks"):
        LookupResult(hit_chunks=3, total_chunks=2, latency_ms=1.0)


def test_transfer_result_reports_partial_failure() -> None:
    result = TransferResult(
        operation="retrieve",
        token_count=256,
        latency_ms=3.0,
        attempted_worker_ranks=(0, 1),
        successful_worker_ranks=(0,),
        failed_worker_ranks=(1,),
    )

    assert not result.succeeded


def test_transfer_result_requires_exact_worker_partition() -> None:
    with pytest.raises(ValueError, match="exactly partition"):
        TransferResult("store", 256, 2.5, (0, 1), (0,), ())
