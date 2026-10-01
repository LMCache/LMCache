# SPDX-License-Identifier: Apache-2.0
"""PinPacer: pinning waits for in-flight transfer submissions, bounded."""

# Standard
import threading
import time

# First Party
from lmcache.v1.memory_allocators.pin_pacer import PinPacer


def test_wait_idle_returns_immediately_when_nothing_is_submitting():
    pacer = PinPacer()
    start = time.perf_counter()
    assert pacer.wait_idle(max_wait_s=1.0) is True
    assert time.perf_counter() - start < 0.1


def test_wait_idle_blocks_until_submission_ends():
    pacer = PinPacer()
    entered = threading.Event()
    release = threading.Event()

    def submitter():
        with pacer.submitting():
            entered.set()
            release.wait()

    t = threading.Thread(target=submitter)
    t.start()
    entered.wait()

    assert pacer.wait_idle(max_wait_s=0.05) is False
    release.set()
    assert pacer.wait_idle(max_wait_s=1.0) is True
    t.join()


def test_nested_submissions_release_only_when_all_end():
    pacer = PinPacer()
    with pacer.submitting():
        with pacer.submitting():
            assert pacer.wait_idle(max_wait_s=0.01) is False
        assert pacer.wait_idle(max_wait_s=0.01) is False
    assert pacer.wait_idle(max_wait_s=0.01) is True


def test_submitting_releases_on_exception():
    pacer = PinPacer()
    try:
        with pacer.submitting():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert pacer.wait_idle(max_wait_s=0.01) is True
