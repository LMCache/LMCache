# SPDX-License-Identifier: Apache-2.0
"""Tests for StoreFailureLog, the rate limit on failed-store WARNINGs."""

# Standard
from collections.abc import Iterator
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers import store_controller
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreFailureLog,
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    """Records from the store controller's logger, which does not propagate
    to pytest's caplog. The logger level is lowered so DEBUG is visible."""
    lg = store_controller.logger
    handler = _Capture()
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _keys(n: int) -> list[ObjectKey]:
    return [ObjectKey(ObjectKey.IntHash2Bytes(i), "m", 0) for i in range(n)]


def _warnings(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == logging.WARNING]


def test_one_warning_per_interval_while_failures_continue(records) -> None:
    clock = _Clock()
    log = StoreFailureLog(interval_s=10.0, clock=clock)

    # 100 rejected tasks within one interval: one WARNING, the rest DEBUG.
    for task_id in range(100):
        log.record(task_id, 0, _keys(2))
        clock.now += 0.05
    warnings = _warnings(records)
    assert len(warnings) == 1
    assert "Store task 0 to adapter 0 failed for keys" in warnings[0]
    debug = [r for r in records if r.levelno == logging.DEBUG]
    assert len(debug) == 99

    # After the interval the next failure warns again and reports the folded
    # count.
    clock.now += 10.0
    log.record(100, 0, _keys(3))
    warnings = _warnings(records)
    assert len(warnings) == 2
    assert "99 more failed store tasks, 198 keys" in warnings[1]


def test_isolated_failures_each_warn_with_keys(records) -> None:
    clock = _Clock()
    log = StoreFailureLog(interval_s=10.0, clock=clock)

    log.record(1, 0, _keys(1))
    clock.now += 30.0
    log.record(2, 1, _keys(1))

    warnings = _warnings(records)
    assert len(warnings) == 2
    assert all("more failed store tasks" not in w for w in warnings)
    assert "Store task 2 to adapter 1" in warnings[1]
