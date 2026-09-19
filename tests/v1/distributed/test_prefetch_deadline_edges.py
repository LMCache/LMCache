# SPDX-License-Identifier: Apache-2.0
"""Edge-case regressions for the optional L2 prefetch/load deadline.

Deliberately a separate module from ``test_prefetch_deadline.py``: that file
installs an autouse fixture shrinking ``PREFETCH_LOOP_POLL_TIMEOUT_MS`` to 20 ms,
which would hide exactly the bug :class:`TestPollBound` pins. Everything here
runs with the production 500 ms poll interval.

Each test in this module corresponds to a concrete defect found by review of
the original change:

* ``TestDisabledNeverReadsTheClock`` -- the loop read the clock on every
  iteration even with the feature off.
* ``TestPollBound`` -- the poll bound only inspected the pending queue's *head*,
  so an unarmed WARM entry there hid an armed request behind it and the loop
  slept the full default poll past that request's deadline.
* ``TestAdmissionAfterDeadline`` -- a request could be pulled off the pending
  queue and started on L2 after its deadline had already passed.
* ``TestCompletedTaskBuffers`` -- a *failed* key belonging to an adapter task
  that had already returned stayed write-reserved until the slowest adapter in
  the same request finished.
* ``TestQueuedFallbackTouchesRetained`` -- the queued-timeout fallback did not
  refresh LRU recency for the keys it published, unlike normal completion.
"""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import PrefetchMode, PrefetchRequestSpec, TrimPolicy
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_controllers import prefetch_controller
from tests.v1.distributed.test_prefetch_deadline import (
    FakeClock,
    in_flight_count,
    l1_manager,
    make_controller,
    make_controller_multi,
    make_gated_adapter,
    make_layout,
    make_object_key,
    result_ready,
    store_keys_in_l2,
    submit,
    wait_until,
)

# ``l1_manager`` is a pytest fixture defined in the sibling module and reused
# here; pytest resolves it by name, so the import is not "unused".
__all__ = ["l1_manager"]

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


class ExplodingClock:
    """A clock that records every read and refuses to be read.

    Used to prove the disabled path never touches the clock: a call count of
    zero is the assertion, the exception only makes an accidental call loud.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.calls += 1
        raise AssertionError("the prefetch loop read the clock while disabled")


def submit_mode(ctrl, keys, layout, mode, policy=TrimPolicy.PREFIX) -> int:
    spec = PrefetchRequestSpec(
        keys=keys,
        group_layout_descs={0: layout},
        num_kv_readers=1,
        policy=policy,
        mode=mode,
    )
    return ctrl.submit_prefetch_request(spec)


class TestDisabledNeverReadsTheClock:
    def test_disabled_loop_does_not_call_the_clock(self, l1_manager):  # noqa: F811
        """With ``l2_load_timeout=None`` the loop must not evaluate the clock on
        any iteration -- not even as the argument of a call that immediately
        returns. The clock here raises, so a single read both trips the count
        assertion and surfaces in the loop's exception log."""
        adapter = make_gated_adapter()
        adapter.release_loads()
        clock = ExplodingClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=None)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, req))
            assert ctrl.query_prefetch_result(req) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            # Let the loop spin through several more idle iterations.
            time.sleep(1.2)
            assert clock.calls == 0, (
                f"the disabled prefetch loop read the clock {clock.calls} times"
            )
        finally:
            ctrl.stop()
            adapter.close()


class TestPollBound:
    def test_warm_head_does_not_hide_a_queued_lookup_deadline(self, l1_manager):  # noqa: F811
        """An unarmed WARM entry at the head of the pending queue must not hide
        the deadline of an armed LOOKUP request queued behind it.

        Runs at the production 500 ms poll with the real monotonic clock and no
        other source of wakeups, so the measured fallback latency is exactly the
        poll bound under test: ~125 ms when the scan finds the armed entry,
        ~500 ms (the next default tick) when it only looks at the head.
        """
        assert prefetch_controller.PREFETCH_LOOP_POLL_TIMEOUT_MS == 500, (
            "this regression only means anything at the production poll interval"
        )
        budget = 0.125
        adapter = make_gated_adapter()  # loads never complete: no load wakeups
        ctrl = make_controller(
            l1_manager, adapter, time.monotonic, timeout=budget, max_in_flight=1
        )
        ctrl.start()
        try:
            layout = make_layout()
            occupant = [make_object_key(800 + i) for i in range(2)]
            warm = [make_object_key(810 + i) for i in range(2)]
            armed = [make_object_key(820 + i) for i in range(2)]
            for ks in (occupant, warm, armed):
                store_keys_in_l2(adapter, ks, layout)

            # WARM takes the single slot and is never armed, so _armed stays
            # empty and only the pending queue can bound the poll.
            submit_mode(ctrl, occupant, layout, PrefetchMode.WARM)
            assert adapter.load_entered.wait(10.0)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            # A second WARM request parks unarmed at the head of the queue.
            submit_mode(ctrl, warm, layout, PrefetchMode.WARM)
            assert wait_until(lambda: ctrl.report_status()["pending_queue_size"] == 1)
            assert ctrl._armed == {}  # noqa: SLF001 (white-box precondition)

            started = time.monotonic()
            req = submit(ctrl, armed, layout)  # LOOKUP: armed, queued behind WARM
            assert wait_until(lambda: result_ready(ctrl, req), timeout=3.0)
            elapsed = time.monotonic() - started

            assert ctrl.query_prefetch_result(req) is not None
            assert ctrl.report_status()["deadline_timeout_count"] == 1
            assert elapsed < 0.35, (
                f"queued LOOKUP fell back after {elapsed * 1000:.0f} ms; expected "
                f"~{budget * 1000:.0f} ms, not the next "
                f"{prefetch_controller.PREFETCH_LOOP_POLL_TIMEOUT_MS} ms poll tick"
            )
        finally:
            ctrl.stop()
            adapter.close()


class TestAdmissionAfterDeadline:
    def test_admission_reroutes_a_request_already_past_its_deadline(
        self,
        l1_manager,  # noqa: F811
    ):
        """Time passes between the expiry sweep and admission, so a request can
        be popped off the queue after its deadline. It must take the queued
        fallback path, not start fresh L2 work whose result can never be
        published.

        Driven directly on the loop thread's own helper with the controller
        stopped, so the window is exercised deterministically rather than raced.
        """
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        try:
            layout = make_layout()
            keys = [make_object_key(830 + i) for i in range(2)]
            spec = PrefetchRequestSpec(
                keys=keys, group_layout_descs={0: layout}, num_kv_readers=1
            )
            started: list[int] = []
            ctrl._start_lookup_phase = (  # noqa: SLF001 (test shim)
                lambda rid, sp, dl: started.append(rid)
            )
            # Queued with a deadline that has already passed when admission runs.
            deadline_at = clock() - 0.001
            ctrl._pending_queue.append((7, spec, deadline_at))  # noqa: SLF001
            ctrl._status_pending_count += 1  # noqa: SLF001

            ctrl._start_pending_requests()  # noqa: SLF001

            assert started == [], "expired request was started on L2 anyway"
            assert ctrl._pending_queue == []  # noqa: SLF001
            status = ctrl.report_status()
            assert status["deadline_timeout_count"] == 1
            assert status["pending_queue_size"] == 0
            assert ctrl.query_prefetch_result(7) is not None  # fallback published
        finally:
            adapter.close()


class TestCompletedTaskBuffers:
    def test_failed_keys_of_a_completed_task_are_released_at_the_deadline(
        self,
        l1_manager,  # noqa: F811
    ):
        """A key that an adapter reported as *not loaded* must have its buffer
        released as soon as that adapter's task returns -- it cannot still be
        written. Holding it until the slowest adapter in the request finishes
        pins an L1 write reservation for the whole drain.

        ``fast`` returns immediately reporting zero keys loaded; ``slow`` sits on
        its gate. At the deadline the fast adapter's keys must be gone from L1
        (deleted), while the slow adapter's keys stay write-reserved.
        """
        slow = make_gated_adapter()  # owns keys 0, 2; never released here
        fast = make_gated_adapter()
        fast.fail_loads()  # its task completes reporting zero keys loaded
        fast.release_loads()
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(840 + i) for i in range(4)]
            store_keys_in_l2(slow, [keys[0], keys[2]], layout)
            store_keys_in_l2(fast, [keys[1], keys[3]], layout)
            req = submit(ctrl, keys, layout, policy=TrimPolicy.SPARSE)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            assert slow.load_entered.wait(10.0)
            assert fast.load_result_consumed.wait(10.0)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            retained = ctrl.query_prefetch_result(req)
            assert retained is not None
            assert set(retained.get_indices_list()) == set()  # nothing loaded

            probe = l1_manager.unsafe_read(keys)
            # Fast adapter has returned: its failed buffers are released now.
            assert probe[keys[1]][0] == L1Error.KEY_NOT_EXIST
            assert probe[keys[3]][0] == L1Error.KEY_NOT_EXIST
            # Slow adapter may still be writing: its reservations are untouched.
            assert probe[keys[0]][0] == L1Error.KEY_NOT_READABLE
            assert probe[keys[2]][0] == L1Error.KEY_NOT_READABLE

            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            slow.close()
            fast.close()


class TestQueuedFallbackTouchesRetained:
    def test_queued_fallback_touches_the_keys_it_publishes(self, l1_manager):  # noqa: F811
        """LRU parity with normal completion: the keys a queued-timeout fallback
        publishes are the ones it serves, and locking never refreshes recency,
        so they must be touched.

        A first, ordinary prefetch loads the keys into L1 and read-locks them for
        its caller, so the queued fallback below has a non-empty L1 hit to
        publish and touch. The read lock is kept for the duration: under the
        default prefetch policy these objects are temporary, and releasing the
        last read lock deletes them.
        """
        adapter = make_gated_adapter()
        adapter.release_loads()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(850 + i) for i in range(2)]
            store_keys_in_l2(adapter, keys, layout)
            warm = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, warm))
            first = ctrl.query_prefetch_result(warm)
            assert first is not None and first.popcount() == len(keys)
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            probe = l1_manager.unsafe_read(keys)
            assert all(v[0] == L1Error.SUCCESS for v in probe.values())
        finally:
            # Join the loop thread so the helper below runs without racing it.
            ctrl.stop()

        try:
            touched: list[list] = []
            ctrl._l1_manager = _TouchRecorder(l1_manager, touched)  # noqa: SLF001
            spec = PrefetchRequestSpec(
                keys=keys, group_layout_descs={0: layout}, num_kv_readers=1
            )
            ctrl._expire_queued_request(9, spec, clock() - 0.001)  # noqa: SLF001
            published = ctrl.query_prefetch_result(9)
            assert published is not None
            assert published.popcount() == len(keys)  # L1 hit is not discarded
            assert touched == [keys], (
                "queued fallback published keys without refreshing LRU recency"
            )
            # Release both read locks: the first prefetch's and the fallback's.
            l1_manager.finish_read(keys)
            l1_manager.finish_read(keys)
        finally:
            adapter.close()


class _TouchRecorder:
    """Delegating proxy that records ``touch_keys`` calls."""

    def __init__(self, inner, sink: list) -> None:
        self._inner = inner
        self._sink = sink

    def touch_keys(self, keys):
        self._sink.append(list(keys))
        return self._inner.touch_keys(keys)

    def __getattr__(self, name):
        return getattr(self._inner, name)
