# SPDX-License-Identifier: Apache-2.0
"""Tests for the IsolatedLRU-on-L1 startup guard and eviction loop hardening.

``--eviction-policy`` feeds only the L1 tier, but ``IsolatedLRU`` is an
isolation-only policy: its ``get_eviction_actions`` requires a
``cache_salt``, which only the L2 eviction controller (via
``QuotaManager``) provides. Previously, selecting it made the L1
eviction thread die with ``ValueError`` on the first over-watermark
cycle, permanently freezing L1 once full.

Covered here:

1. Startup guard: ``L1EvictionController`` rejects isolation-only
   policies at construction.
2. Loop hardening: an unexpected exception in one eviction cycle no
   longer kills the eviction thread — it logs and retries next cycle.
3. Regression: the global policies (``LRU``, ``noop``) still construct
   and ``LRU`` still evicts.
"""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)

# How long to wait for the eviction loop (it ticks once per second).
_LOOP_TIMEOUT_S = 10.0


class _FakeL1Manager:
    """Duck-typed stand-in for ``L1Manager``.

    Reports above-watermark memory usage so the eviction loop's very
    first cycle triggers eviction, and records deletions.
    """

    def __init__(self):
        self.listener = None
        self.deleted_keys: list[ObjectKey] = []

    def register_listener(self, listener) -> None:
        self.listener = listener

    def get_memory_usage(self) -> tuple[int, int]:
        return (90, 100)  # 90% > default 0.8 watermark

    def is_key_evictable(self, key: ObjectKey) -> bool:
        return True

    def delete(self, keys: list[ObjectKey]) -> None:
        self.deleted_keys.extend(keys)


class _ThreadExceptionCapture:
    """Capture uncaught exceptions from background threads."""

    def __init__(self):
        self.exceptions: list[threading.ExceptHookArgs] = []
        self._old_hook = None

    def __enter__(self):
        self._old_hook = threading.excepthook
        threading.excepthook = lambda args: self.exceptions.append(args)
        return self

    def __exit__(self, *exc_info):
        threading.excepthook = self._old_hook
        return False


def _make_key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=f"hash-{i}".encode(),
        model_name="test-model",
        kv_rank=0,
        cache_salt="tenant-a",
    )


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class TestStartupGuard:
    def test_isolated_lru_on_l1_rejected_at_construction(self):
        with pytest.raises(ValueError, match="IsolatedLRU.*not supported.*L1"):
            L1EvictionController(
                l1_manager=_FakeL1Manager(),
                eviction_config=EvictionConfig(eviction_policy="IsolatedLRU"),
            )

    @pytest.mark.parametrize("policy", ["LRU", "noop"])
    def test_global_policies_still_construct(self, policy):
        controller = L1EvictionController(
            l1_manager=_FakeL1Manager(),
            eviction_config=EvictionConfig(eviction_policy=policy),
        )
        assert controller.report_status()["eviction_policy"] == policy


class TestLoopResilience:
    def test_eviction_thread_survives_policy_exception(self):
        """A cycle that raises must not kill the thread; the next cycle
        proceeds and evicts."""
        fake_l1 = _FakeL1Manager()
        controller = L1EvictionController(
            l1_manager=fake_l1,
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
        assert fake_l1.listener is not None
        fake_l1.listener.on_l1_keys_write_finished([_make_key(i) for i in range(10)])

        # Make the policy raise exactly once, then behave normally.
        policy = controller._eviction_policy
        real_get_actions = policy.get_eviction_actions
        fail_once = {"pending": True}

        def flaky_get_actions(*args, **kwargs):
            if fail_once["pending"]:
                fail_once["pending"] = False
                raise RuntimeError("injected transient failure")
            return real_get_actions(*args, **kwargs)

        policy.get_eviction_actions = flaky_get_actions

        with _ThreadExceptionCapture() as capture:
            controller.start()
            evicted = _wait_for(lambda: len(fake_l1.deleted_keys) > 0, _LOOP_TIMEOUT_S)
            assert evicted, "eviction should recover after a transient failure"
            assert controller._thread.is_alive()
            assert controller.report_status()["is_healthy"] is True
        controller.stop()

        # The injected failure was contained — no uncaught thread exception.
        assert capture.exceptions == []
        assert fail_once["pending"] is False  # the failure actually fired


class TestLRURegression:
    def test_lru_still_evicts(self):
        fake_l1 = _FakeL1Manager()
        controller = L1EvictionController(
            l1_manager=fake_l1,
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
        assert fake_l1.listener is not None
        fake_l1.listener.on_l1_keys_write_finished([_make_key(i) for i in range(10)])
        controller.start()
        evicted = _wait_for(lambda: len(fake_l1.deleted_keys) > 0, _LOOP_TIMEOUT_S)
        controller.stop()
        assert evicted
