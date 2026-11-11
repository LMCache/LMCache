# SPDX-License-Identifier: Apache-2.0
"""Tests for the fill tracker of pipelined L2 loads."""

# Standard
from collections.abc import Iterator
from typing import cast
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed import pending_fill
from lmcache.v1.distributed.pending_fill import FillFailedError, FillTracker
from lmcache.v1.memory_management import MemoryObj


class _Obj:
    """Just enough of a MemoryObj for the registry: a data pointer."""

    def __init__(self, data_ptr: int) -> None:
        self.data_ptr = data_ptr


def _objs(*ptrs: int) -> list[MemoryObj]:
    return [cast(MemoryObj, _Obj(ptr)) for ptr in ptrs]


@pytest.fixture(autouse=True)
def empty_registry() -> Iterator[None]:
    yield
    leaked = dict(pending_fill._fills)
    pending_fill._fills.clear()
    assert not leaked, "a test left a fill registered"


def test_find_serves_registered_objects_until_release() -> None:
    a, b, other = _objs(1, 2, 3)
    fill = FillTracker([a, b])
    assert pending_fill.find(a) is fill
    assert pending_fill.find(b) is fill
    assert pending_fill.find(other) is None
    fill.release()
    assert pending_fill.find(a) is None


def test_wait_for_a_prefix_returns_once_it_is_filled() -> None:
    (obj,) = _objs(1)
    fill = FillTracker([obj])
    fill.advance(100)
    fill.wait(100)
    with pytest.raises(FillFailedError):
        fill.wait(101, timeout=0.01)

    waiter = threading.Thread(target=fill.wait, args=(200,))
    waiter.start()
    fill.advance(200)
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    fill.settle(True)


def test_success_releases_before_callbacks_run() -> None:
    (obj,) = _objs(1)
    fill = FillTracker([obj])
    seen: list[tuple[bool, object]] = []
    fill.on_settled(lambda ok: seen.append((ok, pending_fill.find(obj))))
    fill.settle(True)
    assert seen == [(True, None)]
    fill.wait()
    assert fill.settled and not fill.failed
    # Late callbacks run at once; settling again is a no-op.
    fill.on_settled(lambda ok: seen.append((ok, None)))
    fill.settle(False)
    assert seen == [(True, None), (True, None)]


def test_failure_stays_registered_and_fails_readers() -> None:
    (obj,) = _objs(1)
    fill = FillTracker([obj])
    fill.advance(64)
    outcomes: list[bool] = []
    fill.on_settled(outcomes.append)
    fill.settle(False)

    assert outcomes == [False]
    assert fill.failed
    assert pending_fill.find(obj) is fill
    # Bytes written before the failure stay valid.
    fill.wait(64)
    with pytest.raises(FillFailedError):
        fill.wait(65)
    with pytest.raises(FillFailedError):
        fill.wait()
    fill.release()


def test_wait_filled_reports_per_object_validity() -> None:
    ok_obj, failed_obj, plain_obj = _objs(1, 2, 3)
    ok_fill = FillTracker([ok_obj])
    failed_fill = FillTracker([failed_obj])
    failed_fill.settle(False)

    waiter_result: list[list[bool]] = []
    waiter = threading.Thread(
        target=lambda: waiter_result.append(
            pending_fill.wait_filled([ok_obj, failed_obj, plain_obj])
        )
    )
    waiter.start()
    ok_fill.settle(True)
    waiter.join(timeout=5)
    assert waiter_result == [[True, False, True]]
    failed_fill.release()


def test_wait_filled_without_fills_is_all_valid() -> None:
    assert pending_fill.wait_filled(_objs(1, 2)) == [True, True]
