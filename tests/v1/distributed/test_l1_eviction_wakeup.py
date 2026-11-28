# SPDX-License-Identifier: Apache-2.0

"""Tests that L1 growth wakes the L1 eviction loop instead of waiting a period."""

# Standard
from collections.abc import Iterator
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus

L1_BYTES = 64 * 1024 * 1024
# 2 MiB per object: 20 objects fill ~62% of L1.
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024, 1024])], dtypes=[torch.float16])
# 64 KiB per object, for write storms that stay under the watermark.
SMALL_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([32, 1024])], dtypes=[torch.float16])
# Well under the loop's 1 s idle period, so passing requires a wakeup.
EVICTION_DEADLINE_S = 0.5
# Long enough for an already-woken cycle and the 50 ms gap after it to pass,
# short enough that the deadline above still ends before the next idle cycle.
SETTLE_S = 0.2


def _keys(n: int, start: int = 0) -> list[ObjectKey]:
    """Build ``n`` distinct object keys, numbered from ``start``."""
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="m",
            kv_rank=0,
            object_group_id=0,
        )
        for i in range(start, start + n)
    ]


def _usage(l1_manager: DramL1Manager) -> float:
    """Return the L1 memory usage ratio."""
    used, total = l1_manager.get_memory_usage()
    return used / total


def _wait_for_usage_below(
    l1_manager: DramL1Manager, ratio: float, deadline_s: float
) -> bool:
    """Poll until L1 usage drops below ``ratio`` or ``deadline_s`` passes."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if _usage(l1_manager) < ratio:
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def l1_manager() -> Iterator[DramL1Manager]:
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=L1_BYTES,
            use_lazy=False,
            init_size_in_bytes=L1_BYTES,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = DramL1Manager(config)
    yield mgr
    mgr.close()


@pytest.fixture
def loop_ticks() -> Iterator[list[Event]]:
    """Enable the global event bus and collect L1 eviction-loop ticks."""
    bus = init_event_bus(EventBusConfig(enabled=True))
    ticks: list[Event] = []
    bus.subscribe(EventType.L1_EVICTION_LOOP_TICK, ticks.append)
    bus.start()
    try:
        yield ticks
    finally:
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


@pytest.fixture
def controller(l1_manager: DramL1Manager) -> Iterator[L1EvictionController]:
    ctrl = L1EvictionController(
        l1_manager=l1_manager,
        eviction_config=EvictionConfig(
            eviction_policy="LRU", trigger_watermark=0.5, eviction_ratio=0.5
        ),
    )
    ctrl.start()
    # Let the loop enter its idle wait so the writes below land mid-period.
    time.sleep(0.1)
    yield ctrl
    ctrl.stop()


class TestL1EvictionWakeup:
    def test_reservation_over_watermark_evicts_older_keys(
        self, l1_manager: DramL1Manager, controller: L1EvictionController
    ) -> None:
        """A write reservation that pushes usage over the watermark wakes the
        loop, which evicts older unlocked keys while the new keys are still
        being written."""
        keys = _keys(20)
        resident, incoming = keys[:12], keys[12:]
        reserved = l1_manager.reserve_write(resident, [False] * len(resident), LAYOUT)
        assert all(reserved[key][0] == L1Error.SUCCESS for key in resident)
        l1_manager.finish_write(resident)
        # 12 objects of 2 MiB stay under the watermark; let the cycle these
        # writes woke run, so only the next reservation can wake the loop.
        time.sleep(SETTLE_S)

        reserved = l1_manager.reserve_write(incoming, [False] * len(incoming), LAYOUT)
        assert all(reserved[key][0] == L1Error.SUCCESS for key in incoming)

        assert _wait_for_usage_below(l1_manager, 0.5, EVICTION_DEADLINE_S)

    def test_finished_writes_over_watermark_are_evicted_promptly(
        self, l1_manager: DramL1Manager, controller: L1EvictionController
    ) -> None:
        """Keys that were write-locked while usage crossed the watermark are
        evicted promptly once their writes finish."""
        keys = _keys(20)
        l1_manager.reserve_write(keys, [False] * len(keys), LAYOUT)
        # Let the cycle the reservation woke run while every key is still
        # write-locked, so only finish_write can wake the next one.
        time.sleep(SETTLE_S)
        assert _usage(l1_manager) >= 0.5

        l1_manager.finish_write(keys)

        assert _wait_for_usage_below(l1_manager, 0.5, EVICTION_DEADLINE_S)

    def test_finished_reads_over_watermark_are_evicted_promptly(
        self, l1_manager: DramL1Manager, controller: L1EvictionController
    ) -> None:
        """Keys that were read-locked while usage crossed the watermark are
        evicted promptly once their reads finish."""
        keys = _keys(20)
        l1_manager.reserve_write(keys, [False] * len(keys), LAYOUT)
        l1_manager.finish_write_and_reserve_read(keys)
        # Let the woken cycle run while every key is still read-locked.
        time.sleep(SETTLE_S)
        assert _usage(l1_manager) >= 0.5

        l1_manager.finish_read(keys)

        assert _wait_for_usage_below(l1_manager, 0.5, EVICTION_DEADLINE_S)

    def test_usage_below_watermark_is_left_alone(
        self, l1_manager: DramL1Manager, controller: L1EvictionController
    ) -> None:
        """Growth that stays under the watermark evicts nothing."""
        keys = _keys(5)
        l1_manager.reserve_write(keys, [False] * len(keys), LAYOUT)
        l1_manager.finish_write(keys)
        before = _usage(l1_manager)

        time.sleep(EVICTION_DEADLINE_S)

        assert _usage(l1_manager) == pytest.approx(before)

    def test_loop_rate_is_bounded_under_a_write_storm(
        self, l1_manager: DramL1Manager, loop_ticks: list[Event]
    ) -> None:
        """A steady stream of small writes below the watermark runs the loop
        at most once per 50 ms instead of once per write."""
        ctrl = L1EvictionController(
            l1_manager=l1_manager,
            eviction_config=EvictionConfig(
                eviction_policy="LRU", trigger_watermark=0.5, eviction_ratio=0.5
            ),
        )
        ctrl.start()
        try:
            time.sleep(0.1)
            before = len(loop_ticks)
            storm_s, writes = 0.5, 0
            end = time.monotonic() + storm_s
            while time.monotonic() < end:
                key = _keys(1, start=1000 + writes)
                l1_manager.reserve_write(key, [True], SMALL_LAYOUT)
                l1_manager.finish_write(key)
                writes += 1
                time.sleep(0.001)
        finally:
            ctrl.stop()
        # Let the bus drain the last ticks.
        time.sleep(0.1)

        assert writes > 100
        # One cycle per 50 ms over the storm, plus slack for scheduling.
        assert len(loop_ticks) - before <= storm_s / 0.05 + 4

    def test_reads_that_leave_keys_locked_do_not_wake_the_loop(
        self, l1_manager: DramL1Manager, loop_ticks: list[Event]
    ) -> None:
        """Above the watermark with every key read-locked, a stream of extra
        reads on those keys leaves nothing evictable, so the loop keeps its
        once-per-second idle period instead of rescanning every 50 ms."""
        ctrl = L1EvictionController(
            l1_manager=l1_manager,
            eviction_config=EvictionConfig(
                eviction_policy="LRU", trigger_watermark=0.5, eviction_ratio=0.5
            ),
        )
        keys = _keys(20)
        l1_manager.reserve_write(keys, [False] * len(keys), LAYOUT)
        l1_manager.finish_write_and_reserve_read(keys)
        ctrl.start()
        try:
            time.sleep(SETTLE_S)
            before = len(loop_ticks)
            storm_s = 2.0
            end = time.monotonic() + storm_s
            while time.monotonic() < end:
                l1_manager.reserve_read(keys[:1])
                l1_manager.finish_read(keys[:1])
                time.sleep(0.005)
            assert _usage(l1_manager) >= 0.5
        finally:
            ctrl.stop()
        time.sleep(0.1)
        l1_manager.finish_read(keys)

        # The idle period allows about 2 cycles in 2 s; waking on each of the
        # hundreds of finished reads would allow up to 40.
        assert len(loop_ticks) - before <= 4
