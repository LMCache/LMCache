# SPDX-License-Identifier: Apache-2.0

# Standard
import threading

# First Party
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.periodic_thread import ThreadRunSummary, create_periodic_thread


def test_thread_can_restart_after_irrecoverable_exception() -> None:
    """A terminated thread can be restarted through its public API."""
    attempts = 0
    restarted = threading.Event()

    def execute() -> ThreadRunSummary:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IrrecoverableException("stop first run")
        restarted.set()
        return ThreadRunSummary()

    periodic_thread = create_periodic_thread(
        name="restartable-thread",
        interval=10.0,
        execute_fn=execute,
        auto_register=False,
    )

    first_worker = periodic_thread.start()
    assert first_worker is not None
    first_worker.join(timeout=1.0)
    assert not first_worker.is_alive()
    assert not periodic_thread.is_running

    second_worker = periodic_thread.start()
    try:
        assert second_worker is not None
        assert restarted.wait(timeout=1.0)
    finally:
        periodic_thread.stop()
