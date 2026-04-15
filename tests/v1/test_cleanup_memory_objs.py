# SPDX-License-Identifier: Apache-2.0
"""
Regression test for GitHub issue #3027.

cleanup_memory_objs() must propagate unpin to storage backends via
storage_manager.batched_unpin(), not just call memory_obj.unpin() locally.
"""
# Standard
from unittest.mock import MagicMock
import asyncio

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.event_manager import EventManager, EventStatus, EventType


class FakeMemoryObj:
    """Tracks unpin() and ref_count_down() calls."""

    def __init__(self):
        self.unpin_called = False
        self.ref_count_down_called = False

    def unpin(self):
        self.unpin_called = True

    def ref_count_down(self):
        self.ref_count_down_called = True


def _make_engine_stub():
    """Build a minimal object with only the attrs cleanup_memory_objs needs."""
    engine = object.__new__(LMCacheEngine)  # skip __init__
    engine.event_manager = EventManager()
    engine.storage_manager = MagicMock()  # spy on batched_unpin
    return engine


def _seed_done_event(engine, lookup_id, memory_objs_nested):
    """Register a completed LOADING event containing memory_objs_nested.

    The EventManager requires events to be added as ONGOING first, then
    moved to DONE.  We simulate a finished async prefetch by adding the
    future and immediately marking it DONE.
    """
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    future.set_result(memory_objs_nested)

    engine.event_manager.add_event(EventType.LOADING, lookup_id, future)
    engine.event_manager.update_event_status(
        EventType.LOADING, lookup_id, EventStatus.DONE
    )
    loop.close()


class TestCleanupMemoryObjsUnpinPropagation:
    """Issue #3027: backend unpin must be called, not just local unpin."""

    def test_batched_unpin_called_with_all_keys(self):
        """After fix, cleanup_memory_objs must call
        storage_manager.batched_unpin(keys) so backends with server-side
        pin state are properly notified."""
        engine = _make_engine_stub()
        key_a = MagicMock(name="key_a")
        key_b = MagicMock(name="key_b")
        mem_a, mem_b = FakeMemoryObj(), FakeMemoryObj()

        _seed_done_event(engine, "lu-1", [[(key_a, mem_a), (key_b, mem_b)]])

        engine.cleanup_memory_objs("lu-1")

        engine.storage_manager.batched_unpin.assert_called_once_with(
            [key_a, key_b]
        )

    def test_ref_count_down_still_called(self):
        """ref_count_down must still happen for every memory object."""
        engine = _make_engine_stub()
        mem_a, mem_b = FakeMemoryObj(), FakeMemoryObj()
        _seed_done_event(
            engine,
            "lu-2",
            [[(MagicMock(), mem_a)], [(MagicMock(), mem_b)]],
        )

        engine.cleanup_memory_objs("lu-2")

        assert mem_a.ref_count_down_called
        assert mem_b.ref_count_down_called

    def test_local_unpin_not_called_directly(self):
        """memory_obj.unpin() should NOT be called directly — unpin is
        delegated to storage_manager.batched_unpin()."""
        engine = _make_engine_stub()
        mem = FakeMemoryObj()
        _seed_done_event(engine, "lu-3", [[(MagicMock(), mem)]])

        engine.cleanup_memory_objs("lu-3")

        assert not mem.unpin_called

    def test_multi_tier_keys_collected(self):
        """Keys from multiple tiers are all collected into one batch."""
        engine = _make_engine_stub()
        keys = [MagicMock(name=f"k{i}") for i in range(4)]
        mems = [FakeMemoryObj() for _ in range(4)]
        _seed_done_event(
            engine,
            "lu-4",
            [
                [(keys[0], mems[0]), (keys[1], mems[1])],
                [(keys[2], mems[2]), (keys[3], mems[3])],
            ],
        )

        engine.cleanup_memory_objs("lu-4")

        engine.storage_manager.batched_unpin.assert_called_once_with(keys)
        for m in mems:
            assert m.ref_count_down_called

    def test_ref_count_down_before_batched_unpin(self):
        """ref_count_down must complete for ALL objects before
        batched_unpin to prevent premature eviction."""
        call_log = []
        engine = _make_engine_stub()
        mem = FakeMemoryObj()
        mem.ref_count_down = lambda: call_log.append("ref_count_down")
        engine.storage_manager.batched_unpin.side_effect = (
            lambda keys: call_log.append("batched_unpin")
        )
        _seed_done_event(engine, "lu-order", [[(MagicMock(), mem)]])

        engine.cleanup_memory_objs("lu-order")

        assert call_log == ["ref_count_down", "batched_unpin"]

    def test_no_completed_event_is_noop(self):
        """If no DONE event exists, cleanup is a no-op."""
        engine = _make_engine_stub()

        engine.cleanup_memory_objs("nonexistent")  # should not raise

        engine.storage_manager.batched_unpin.assert_not_called()
