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
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.pin_monitor import PinMonitor

# Local
from .utils import create_test_memory_obj


def create_cleanup_engine() -> LMCacheEngine:
    """Create a minimal engine exercising the public cleanup lifecycle."""
    engine = object.__new__(LMCacheEngine)
    engine.event_manager = EventManager()
    engine._async_lookup_cleanup_lock = threading.Lock()
    engine._active_async_lookups = set()
    engine._pending_async_lookup_cleanups = set()
    return engine


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

    engine = SimpleNamespace(
        event_manager=MagicMock(),
        _async_lookup_cleanup_lock=threading.Lock(),
        _active_async_lookups={"test_lookup"},
        _pending_async_lookup_cleanups=set(),
    )
    engine.event_manager.get_event_status.return_value = EventStatus.DONE
    engine.event_manager.pop_event.return_value = future

    caplog.set_level(logging.WARNING, logger="lmcache")

    LMCacheEngine.cleanup_memory_objs(engine, "test_lookup")  # type: ignore[arg-type]

    assert pinned_obj.metadata.pin_count == 0
    assert nonpinned_obj.metadata.pin_count == 0
    assert "Double unpin" not in caplog.text
    assert "is negative" not in caplog.text

    pinned_obj.ref_count_down()


def test_cleanup_memory_objs_deferred_until_lookup_finishes() -> None:
    """An early or duplicate cleanup request must release the result once."""
    lookup_id = "deferred_lookup"
    engine = create_cleanup_engine()
    engine._active_async_lookups.add(lookup_id)

    memory_obj = MagicMock()
    memory_obj.is_pinned = False
    memory_obj.get_ref_count.return_value = 1
    future = MagicMock()
    future.result.return_value = [[(None, memory_obj)]]

    # Cleanup arrives before event registration, then is repeated while ongoing.
    engine.cleanup_memory_objs(lookup_id)
    engine.event_manager.add_event(EventType.LOADING, lookup_id, future)
    engine.cleanup_memory_objs(lookup_id)
    memory_obj.ref_count_down.assert_not_called()

    engine.event_manager.update_event_status(
        EventType.LOADING, lookup_id, EventStatus.DONE
    )
    engine.finish_async_lookup(lookup_id)

    memory_obj.ref_count_down.assert_called_once_with()
    assert (
        engine.event_manager.get_event_status(EventType.LOADING, lookup_id)
        == EventStatus.NOT_FOUND
    )

    # A late duplicate cleanup is ignored after the active event was consumed.
    engine.cleanup_memory_objs(lookup_id)
    memory_obj.ref_count_down.assert_called_once_with()


def test_cleanup_memory_objs_after_lookup_finishes() -> None:
    """Cleanup arriving after completion must release immediately."""
    lookup_id = "completed_lookup"
    engine = create_cleanup_engine()
    engine._active_async_lookups.add(lookup_id)

    memory_obj = MagicMock()
    memory_obj.is_pinned = False
    memory_obj.get_ref_count.return_value = 1
    future = MagicMock()
    future.result.return_value = [[(None, memory_obj)]]
    engine.event_manager.add_event(EventType.LOADING, lookup_id, future)
    engine.event_manager.update_event_status(
        EventType.LOADING, lookup_id, EventStatus.DONE
    )

    engine.cleanup_memory_objs(lookup_id)

    memory_obj.ref_count_down.assert_called_once_with()
    assert (
        engine.event_manager.get_event_status(EventType.LOADING, lookup_id)
        == EventStatus.NOT_FOUND
    )


def test_finish_async_lookup_without_event_discards_cleanup_request() -> None:
    """A no-hit lookup must discard an early cleanup tombstone."""
    lookup_id = "no_hit_lookup"
    engine = create_cleanup_engine()
    engine._active_async_lookups.add(lookup_id)

    engine.cleanup_memory_objs(lookup_id)
    engine.finish_async_lookup(lookup_id, event_registered=False)

    assert lookup_id not in engine._active_async_lookups
    assert lookup_id not in engine._pending_async_lookup_cleanups


def test_cleanup_memory_objs_consumes_failed_lookup_event() -> None:
    """A failed completed lookup must not retain its future after abort."""
    lookup_id = "failed_lookup"
    engine = create_cleanup_engine()
    engine._active_async_lookups.add(lookup_id)

    future = MagicMock()
    future.result.side_effect = RuntimeError("prefetch failed")
    engine.event_manager.add_event(EventType.LOADING, lookup_id, future)
    engine.cleanup_memory_objs(lookup_id)
    engine.event_manager.update_event_status(
        EventType.LOADING, lookup_id, EventStatus.DONE
    )

    engine.finish_async_lookup(lookup_id)

    assert (
        engine.event_manager.get_event_status(EventType.LOADING, lookup_id)
        == EventStatus.NOT_FOUND
    )
