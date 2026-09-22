# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for LMCacheEngine.cleanup_memory_objs.

Some storage backends (LocalCPU, LocalDisk, P2P, Maru) pin the MemoryObj
they return via async prefetch, while others (Remote, Nixl, plugin tiers)
do not. cleanup_memory_objs must skip unpin for the latter to avoid
driving pin_count below zero.
"""

# Standard
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import TokenDatabase

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


@pytest.mark.no_shared_allocator
def test_retrieve_cleanup_ref_count_and_unpin(pin_monitor: Any) -> None:
    """A real engine releases both pinned and unpinned retrieval references."""
    objects = [create_test_memory_obj(), create_test_memory_obj()]
    objects[0].pin()
    keys = [CacheEngineKey("test", 1, 0, index, torch.bfloat16) for index in range(2)]
    database = MagicMock(spec=TokenDatabase)
    database.process_tokens.return_value = [(0, 8, keys[0]), (8, 16, keys[1])]
    storage = MagicMock(spec=StorageManager)
    storage.get_block_mapping.return_value = {
        "LocalCPUBackend": [(keys[0], 0, 8), (keys[1], 8, 16)]
    }
    storage.batched_get.return_value = objects
    shape = (16, 2, 8, 1, 128)
    engine = LMCacheEngine(
        LMCacheEngineConfig.from_defaults(chunk_size=8, py_enable_gc=True),
        LMCacheMetadata("test", 1, 1, 0, 0, torch.bfloat16, shape),
        database,
        MockGPUConnector(shape),
        lambda tensor, src: None,
        lambda obj, src: obj,
    )
    engine.storage_manager = storage
    try:
        assert engine.retrieve(list(range(16))).all()
        assert all(obj.get_ref_count() == 0 for obj in objects)
        assert all(obj.metadata.pin_count == 0 for obj in objects)
    finally:
        engine.close()
