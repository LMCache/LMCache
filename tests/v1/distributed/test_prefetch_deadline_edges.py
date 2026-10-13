# SPDX-License-Identifier: Apache-2.0
"""Deadline regressions at the production poll interval (no fast-loop fixture)."""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import PrefetchLockMode
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
from tests.v1.distributed.utils import single_row_spec

__all__ = ["l1_manager"]

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


class ExplodingClock:
    """Record and reject clock reads on the disabled path."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.calls += 1
        raise AssertionError("the prefetch loop read the clock while disabled")


def submit_mode(ctrl, keys, layout, mode, policy="prefix") -> int:
    spec = single_row_spec(
        keys, layout, num_kv_readers=1, fetching_policy=policy, lock_mode=mode
    )
    return ctrl.submit_prefetch_request(spec)


class TestDisabledNeverReadsTheClock:
    def test_disabled_loop_does_not_call_the_clock(self, l1_manager):  # noqa: F811
        """Disabled prefetches never consult the deadline clock."""
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
            time.sleep(1.2)
            assert clock.calls == 0, (
                f"the disabled prefetch loop read the clock {clock.calls} times"
            )
        finally:
            ctrl.stop()
            adapter.close()


class TestPollBound:
    def test_warm_head_does_not_hide_a_queued_lookup_deadline(self, l1_manager):  # noqa: F811
        """Queued NO_LOCK work must not hide an armed request's deadline."""
        assert prefetch_controller.PREFETCH_LOOP_POLL_TIMEOUT_MS == 500, (
            "this regression only means anything at the production poll interval"
        )
        budget = 0.125
        adapter = make_gated_adapter()
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

            submit_mode(ctrl, occupant, layout, PrefetchLockMode.NO_LOCK)
            assert adapter.load_entered.wait(10.0)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            submit_mode(ctrl, warm, layout, PrefetchLockMode.NO_LOCK)
            assert wait_until(lambda: ctrl.report_status()["pending_queue_size"] == 1)
            assert ctrl._armed == {}  # noqa: SLF001 (white-box precondition)

            started = time.monotonic()
            req = submit(ctrl, armed, layout)
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
        """Expired requests must not start L2 I/O at admission."""
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        try:
            layout = make_layout()
            keys = [make_object_key(830 + i) for i in range(2)]
            spec = single_row_spec(keys, layout, num_kv_readers=1)
            started: list[int] = []
            ctrl._start_lookup_phase = (  # noqa: SLF001 (test shim)
                lambda rid, sp, dl: started.append(rid)
            )
            deadline_at = clock() - 0.001
            ctrl._pending_queue.append((7, spec, deadline_at))  # noqa: SLF001

            ctrl._start_pending_requests()  # noqa: SLF001

            assert started == [], "expired request was started on L2 anyway"
            assert not ctrl._pending_queue  # noqa: SLF001
            status = ctrl.report_status()
            assert status["deadline_timeout_count"] == 1
            assert status["pending_queue_size"] == 0
            assert ctrl.query_prefetch_result(7) is not None
        finally:
            adapter.close()


class TestCompletedTaskBuffers:
    def test_failed_keys_of_a_completed_task_are_released_at_the_deadline(
        self,
        l1_manager,  # noqa: F811
    ):
        """Completed failed buffers are freed while pending buffers stay reserved."""
        slow = make_gated_adapter()
        fast = make_gated_adapter()
        fast.fail_loads()
        fast.release_loads()
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(840 + i) for i in range(4)]
            store_keys_in_l2(slow, [keys[0], keys[2]], layout)
            store_keys_in_l2(fast, [keys[1], keys[3]], layout)
            req = submit(ctrl, keys, layout, policy="full")
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            assert slow.load_entered.wait(10.0)
            assert fast.load_result_consumed.wait(10.0)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            retained = ctrl.query_prefetch_result(req)
            assert retained is not None
            assert set(retained.hit_cells[0].get_indices_list()) == set()

            probe = l1_manager.unsafe_read(keys)
            assert probe[keys[1]][0] == L1Error.KEY_NOT_EXIST
            assert probe[keys[3]][0] == L1Error.KEY_NOT_EXIST
            assert probe[keys[0]][0] == L1Error.KEY_NOT_EXIST
            assert probe[keys[2]][0] == L1Error.KEY_NOT_EXIST
            status = l1_manager.report_status()
            assert status["staging_object_count"] == 2
            assert status["write_locked_count"] == 2

            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert l1_manager.report_status()["staging_object_count"] == 0
        finally:
            ctrl.stop()
            slow.close()
            fast.close()
