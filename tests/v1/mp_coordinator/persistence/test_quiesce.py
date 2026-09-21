# SPDX-License-Identifier: Apache-2.0
"""Tests for the quiesce lock: what it holds still, and what it refuses."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    PersistenceType,
)
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from tests.v1.mp_coordinator.persistence.conftest import capture_consistently


class TestQuiesceExcludesBatches:
    def test_a_batch_arriving_during_a_quiesce_waits_for_it(self):
        """The whole point: while a capture holds the barrier, no state
        may change underneath it."""
        barrier = QuiesceLock()
        applied = threading.Event()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with barrier.quiesced():
                future = pool.submit(_apply, barrier, applied)
                assert not applied.wait(timeout=0.2), (
                    "a batch was applied while ingest was quiesced"
                )
            assert applied.wait(timeout=2.0), "the batch never resumed"
            future.result(timeout=2.0)

    def test_a_quiesce_waits_for_the_batch_already_running(self):
        """A quiesce must not cut a batch in half, so it waits for the
        one in flight rather than interrupting it."""
        barrier = QuiesceLock()
        release = threading.Event()
        entered = threading.Event()

        def slow_batch() -> None:
            with barrier.applying():
                entered.set()
                release.wait(timeout=5.0)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(slow_batch)
            assert entered.wait(timeout=2.0)
            quiesced_at = []

            def capture() -> None:
                with barrier.quiesced(timeout=5.0):
                    quiesced_at.append(time.monotonic())

            with ThreadPoolExecutor(max_workers=1) as capture_pool:
                capture_future = capture_pool.submit(capture)
                time.sleep(0.2)
                assert not quiesced_at, "the quiesce began mid-batch"
                released_at = time.monotonic()
                release.set()
                capture_future.result(timeout=5.0)

            assert quiesced_at[0] >= released_at
            future.result(timeout=5.0)

    def test_a_stuck_batch_times_out_instead_of_stalling_ingest(self):
        """A capture is best effort. Blocking forever on one wedged batch
        would take the event stream down with it, and the timeout has to
        reach observability rather than being a bare builtin.
        """
        barrier = QuiesceLock()
        release = threading.Event()
        entered = threading.Event()

        def wedged_batch() -> None:
            with barrier.applying():
                entered.set()
                release.wait(timeout=5.0)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(wedged_batch)
            assert entered.wait(timeout=2.0)

            with pytest.raises(LMCacheTimeoutError, match="in flight"):
                with barrier.quiesced(timeout=0.2):
                    pass  # pragma: no cover - the quiesce never starts

            # The failed quiesce must not leave ingest parked.
            applied = threading.Event()
            release.set()
            future.result(timeout=5.0)
            _apply(barrier, applied)
            assert applied.is_set()


class _SlowComponent:
    """A durable component whose capture takes measurable time."""

    def __init__(self, delay: float, started: threading.Event) -> None:
        self._delay = delay
        self._started = started

    @property
    def name(self) -> str:
        return "slow"

    @property
    def persistence_type(self) -> PersistenceType:
        return PersistenceType.CHECKPOINT

    def capture(self) -> dict[str, object]:
        self._started.set()
        time.sleep(self._delay)
        return {}

    def restore(self, state: object) -> None:
        raise NotImplementedError


class TestQuiesceLockOrdering:
    def test_an_ingest_arriving_mid_capture_does_not_deadlock(self):
        """The quiesce must be the outermost lock on the mutating path.

        Acquired *inside* the gate's lock, an ingest arriving during a
        capture takes that lock and then parks waiting for the quiesce --
        while the capture, holding the quiesce, waits for the same lock to
        read the cursors. Opposite orders, permanent deadlock: the entry
        timeout is long spent by then.
        """
        quiesce = QuiesceLock()
        gate = EventGate(CacheEventBroadcaster(), quiesce)
        capture_started = threading.Event()
        slow = _SlowComponent(delay=0.4, started=capture_started)
        batch = CacheEventBatch(
            instance_id="node-a",
            incarnation=1,
            seq=1,
            event_type=CacheEventType.STORE,
            tier=Tier.L2,
            backend="fs",
            entries=[],
        )

        # Daemon threads, not a pool: under the bug both threads wedge
        # forever, and a pool would hang the whole suite on shutdown
        # instead of failing this one test.
        captured: dict[str, object] = {}
        finished = threading.Event()

        def capture() -> None:
            captured.update(capture_consistently(quiesce, [slow, gate]))
            finished.set()

        threading.Thread(target=capture, daemon=True).start()
        assert capture_started.wait(timeout=2.0)
        threading.Thread(target=gate.ingest, args=(batch,), daemon=True).start()

        assert finished.wait(timeout=5.0), (
            "capture never returned: an ingest holding the gate lock is "
            "waiting for the quiesce this capture holds"
        )
        assert sorted(captured) == ["slow", "stream_cursors"]


def _apply(barrier: QuiesceLock, applied: threading.Event) -> None:
    """Apply one no-op batch through the barrier."""
    with barrier.applying():
        applied.set()
