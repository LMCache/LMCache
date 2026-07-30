# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for LMCacheEngine MemoryObj pin cleanup.

cleanup_memory_objs must only unpin objects that are currently pinned.
During layerwise retrieval, LocalCPUBackend returns lookup-owned pinned
objects, while other backends may return retrieve-owned staging objects.
retrieve_layer must preserve the former and release the latter.
"""

# Standard
from collections.abc import Generator
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus
from lmcache.v1.pin_monitor import PinMonitor

# Local
from .utils import create_test_memory_obj


@pytest.fixture
def pin_monitor() -> Generator[None, None, None]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, lmcache_instance_id="test_cleanup"
    )
    PinMonitor.GetOrCreate(config)
    yield
    PinMonitor.DestroyInstance()


def test_cleanup_memory_objs_handles_mixed_pin_state(
    caplog: pytest.LogCaptureFixture, pin_monitor: Any
) -> None:
    """
    Cleanup must unpin chunks that were pinned during prefetch and
    skip unpin for chunks that were not, leaving every chunk with a
    non-negative pin_count and no "Double unpin" warning.
    """
    pinned_obj = create_test_memory_obj()
    pinned_obj.pin()
    pinned_obj.ref_count_up()
    assert pinned_obj.metadata.pin_count == 1

    nonpinned_obj = create_test_memory_obj()
    assert nonpinned_obj.metadata.pin_count == 0

    future = MagicMock()
    future.result.return_value = [
        [(None, pinned_obj)],
        [(None, nonpinned_obj)],
    ]

    engine = SimpleNamespace(event_manager=MagicMock())
    engine.event_manager.get_event_status.return_value = EventStatus.DONE
    engine.event_manager.pop_event.return_value = future

    caplog.set_level(logging.WARNING, logger="lmcache")

    LMCacheEngine.cleanup_memory_objs(engine, "test_lookup")  # type: ignore[arg-type]

    assert pinned_obj.metadata.pin_count == 0
    assert nonpinned_obj.metadata.pin_count == 0
    assert "Double unpin" not in caplog.text
    assert "is negative" not in caplog.text

    pinned_obj.ref_count_down()


@pytest.mark.parametrize(
    ("location", "expected_pinned"),
    [
        ("LocalCPUBackend", True),
        ("LocalDiskBackend", False),
    ],
)
def test_retrieve_layer_respects_backend_pin_ownership(
    location: str, expected_pinned: bool, pin_monitor: Any
) -> None:
    """Layerwise retrieval must preserve or release pins based on ownership."""
    num_layers = 2
    chunk_size = 16
    key = CacheEngineKey("test-model", 1, 0, 1, torch.bfloat16)

    memory_objs = [create_test_memory_obj() for _ in range(num_layers)]
    for memory_obj in memory_objs:
        # For LocalCPU this simulates a lookup-owned pin on the cached object.
        # For LocalDisk it simulates a retrieve-owned pin on a staging object.
        memory_obj.pin()
        memory_obj.ref_count_up()

    def layerwise_batched_get(*args: Any, **kwargs: Any) -> Generator:
        for memory_obj in memory_objs:
            task: Future = Future()
            task.set_result([memory_obj])
            yield task

    def batched_to_gpu(*args: Any, **kwargs: Any) -> Generator:
        while True:
            yield

    storage_manager = MagicMock()
    storage_manager.contains.return_value = location
    storage_manager.layerwise_batched_get.side_effect = layerwise_batched_get

    gpu_connector = MagicMock()
    gpu_connector.batched_to_gpu.side_effect = batched_to_gpu

    token_database = MagicMock()
    token_database.process_tokens.return_value = [(0, chunk_size, key)]

    engine = SimpleNamespace(
        _get_req_id=lambda kwargs: "test-request",
        _is_passive=lambda: False,
        gpu_connector=gpu_connector,
        is_healthy=lambda: True,
        num_layers=num_layers,
        retrieve_locations=[location],
        stats_monitor=MagicMock(),
        storage_manager=storage_manager,
        token_database=token_database,
    )

    with patch("lmcache.v1.cache_engine.assert_layerwise_gpu_connector"):
        retriever = LMCacheEngine.retrieve_layer(  # type: ignore[arg-type]
            engine,
            torch.arange(chunk_size),
        )
        for _ in range(num_layers + 2):
            next(retriever)

    try:
        assert all(
            memory_obj.is_pinned is expected_pinned for memory_obj in memory_objs
        )
    finally:
        for memory_obj in memory_objs:
            if memory_obj.is_pinned:
                memory_obj.unpin()
            memory_obj.ref_count_down()
