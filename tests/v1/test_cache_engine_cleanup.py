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


def test_retrieve_cleanup_preserves_resident_pins() -> None:
    """
    Regression test for #5090: retrieve()'s cleanup must not unpin objects
    resident in the CPU hot cache — their pin belongs to lookup(pin=True)
    and is released via lookup_unpin(). Staging buffers are still unpinned.
    """
    resident_pinned = MagicMock()
    resident_pinned.is_pinned = True

    staging_pinned = MagicMock()
    staging_pinned.is_pinned = True

    engine = MagicMock(spec=LMCacheEngine)
    engine.is_healthy.return_value = True
    engine.remove_after_retrieve = False
    engine._is_passive.return_value = False
    engine.save_only_first_rank = False
    engine._get_req_id.return_value = "req_123"
    engine.async_loading = False
    engine._is_sync_pd_backend.return_value = False

    k_resident = CacheEngineKey("test", 1, 0, 0, torch.bfloat16)
    k_staging = CacheEngineKey("test", 1, 0, 1, torch.bfloat16)
    engine._process_tokens_internal.return_value = (
        [(k_resident, resident_pinned, 0, 10), (k_staging, staging_pinned, 10, 20)],
        1024,
    )
    engine.storage_manager = MagicMock()
    engine.storage_manager.is_hot_cache_object.side_effect = (
        lambda key, obj: obj is resident_pinned
    )

    engine.stats_monitor = MagicMock()
    mock_stats = MagicMock()
    mock_stats.time_to_retrieve.return_value = 1.0
    engine.stats_monitor.on_retrieve_request.return_value = mock_stats
    engine.gpu_connector = MagicMock()

    LMCacheEngine.retrieve(
        engine,
        tokens=torch.zeros(20, dtype=torch.long),
        kvcaches=torch.zeros(20),
        slot_mapping=torch.zeros(20, dtype=torch.long),
    )

    resident_pinned.unpin.assert_not_called()
    resident_pinned.ref_count_down.assert_called_once()
    staging_pinned.unpin.assert_called_once()
    staging_pinned.ref_count_down.assert_called_once()
