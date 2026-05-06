# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``LMCacheMPWorkerAdapter`` health/recovery wiring.

Two pieces are exercised:

* ``HeartbeatThread`` — the ``register_recover_callback`` hook that
  fires on the unhealthy->healthy edge before the health event is set.
* ``LMCacheMPWorkerAdapter._reregister_kv_caches`` — the worker-side
  callback that re-issues ``REGISTER_KV_CACHE`` on recovery, plus the
  thin trivial ``is_healthy`` property and the unchanged public
  ``register_kv_caches``.

Both classes' ``__init__`` perform real work (chunk-size MQ query for the
adapter, ``threading.Thread`` setup for the heartbeat). We bypass them
with ``__new__`` and inject only the attributes each method touches.
"""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    HeartbeatThread,
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.multiprocess.protocol import RequestType

# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> LMCacheMPWorkerAdapter:
    """Build a worker adapter without running its real ``__init__``."""
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter.kv_caches = {}

    adapter.mq_client = MagicMock(name="mq_client")
    adapter.instance_id = 4242
    adapter.model_name = "test-model"
    adapter._mq_timeout = 5.0

    parallel_strategy = MagicMock()
    parallel_strategy.kv_world_size = 1
    adapter.parallel_strategy = parallel_strategy

    return adapter


@pytest.fixture
def patch_send(monkeypatch):
    """Replace ``send_lmcache_request`` and friends with mocks."""
    future = MagicMock(name="future")
    future.result.return_value = None  # default: registration succeeds

    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: "fake-layout",
        raising=False,
    )
    monkeypatch.setattr(adapter_mod, "wrap_kv_caches", lambda kv: list(kv.values()))

    return send_mock, future


# ---------------------------------------------------------------------------
# is_healthy: trivial property reflecting the heartbeat's event
# ---------------------------------------------------------------------------


def test_is_healthy_reflects_event_set():
    adapter = _make_adapter()
    assert adapter.is_healthy is True


def test_is_healthy_reflects_event_cleared():
    adapter = _make_adapter()
    adapter._health_event.clear()
    assert adapter.is_healthy is False


# ---------------------------------------------------------------------------
# Public register_kv_caches: behavior preserved across the refactor
# ---------------------------------------------------------------------------


def test_register_kv_caches_public_unchanged(patch_send):
    """Public register_kv_caches still updates self.kv_caches and submits once."""
    send_mock, _ = patch_send
    adapter = _make_adapter()
    new_caches = {"layer.0": object(), "layer.1": object()}

    adapter.register_kv_caches(new_caches)

    assert adapter.kv_caches is new_caches
    assert send_mock.call_count == 1
    args, _ = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE


def test_register_kv_caches_public_raises_on_timeout(patch_send):
    """Public register_kv_caches surfaces ConnectionError on MQ timeout."""
    _, future = patch_send
    adapter = _make_adapter()
    future.result.side_effect = TimeoutError("server down")

    with pytest.raises(ConnectionError, match="did not respond"):
        adapter.register_kv_caches({"layer.0": object()})


# ---------------------------------------------------------------------------
# _reregister_kv_caches: the worker's recovery callback
# ---------------------------------------------------------------------------


def test_reregister_callback_no_prior_registration(patch_send):
    """Empty kv_caches: nothing to do, returns True without submitting."""
    send_mock, _ = patch_send
    adapter = _make_adapter()
    assert adapter.kv_caches == {}

    assert adapter._reregister_kv_caches() is True
    send_mock.assert_not_called()


def test_reregister_callback_success_submits(patch_send):
    """Populated kv_caches: submits REGISTER_KV_CACHE and returns True."""
    send_mock, _ = patch_send
    adapter = _make_adapter()
    adapter.kv_caches = {"layer.0": object()}

    assert adapter._reregister_kv_caches() is True
    assert send_mock.call_count == 1
    args, _ = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE


def test_reregister_callback_returns_false_on_timeout(patch_send):
    """Server times out -> ConnectionError -> callback returns False."""
    send_mock, future = patch_send
    adapter = _make_adapter()
    adapter.kv_caches = {"layer.0": object()}
    future.result.side_effect = TimeoutError("server gone")

    assert adapter._reregister_kv_caches() is False
    assert send_mock.call_count == 1


# ---------------------------------------------------------------------------
# HeartbeatThread: recover callback + _execute integration
# ---------------------------------------------------------------------------


def _make_heartbeat() -> HeartbeatThread:
    """Build a HeartbeatThread without invoking PeriodicThread.__init__."""
    hb = HeartbeatThread.__new__(HeartbeatThread)
    hb._mq_client = MagicMock(name="mq_client")
    hb._health_event = threading.Event()
    hb._health_event.set()
    hb._interval = 1.0

    def noop() -> bool:
        return True

    hb._recover_callback = noop
    return hb


def test_heartbeat_steady_healthy_no_callback(monkeypatch):
    """Healthy ping while already healthy: no callback invoked, event stays set."""
    hb = _make_heartbeat()
    callback = MagicMock(return_value=True)
    hb.register_recover_callback(callback)

    monkeypatch.setattr(adapter_mod, "send_ping", lambda *a, **kw: True)
    summary = hb._execute()

    assert hb._health_event.is_set()
    callback.assert_not_called()
    assert summary.message == "healthy"


def test_heartbeat_recovery_invokes_callback_then_sets(monkeypatch):
    """Unhealthy -> healthy: callback runs; on True, event is set after."""
    hb = _make_heartbeat()
    hb._health_event.clear()  # was unhealthy

    call_order: list[str] = []

    def callback() -> bool:
        # Event must still be cleared when callback runs.
        call_order.append("set" if hb._health_event.is_set() else "clear")
        return True

    hb.register_recover_callback(callback)
    monkeypatch.setattr(adapter_mod, "send_ping", lambda *a, **kw: True)

    summary = hb._execute()

    assert call_order == ["clear"]
    assert hb._health_event.is_set()
    assert summary.message == "healthy"


def test_heartbeat_recovery_callback_failure_keeps_event_clear(monkeypatch):
    """Callback returns False on recovery: event stays cleared, retry next tick."""
    hb = _make_heartbeat()
    hb._health_event.clear()
    callback = MagicMock(side_effect=[False, True])
    hb.register_recover_callback(callback)
    monkeypatch.setattr(adapter_mod, "send_ping", lambda *a, **kw: True)

    summary1 = hb._execute()
    assert callback.call_count == 1
    assert not hb._health_event.is_set()
    assert summary1.message == "unhealthy"

    # Next heartbeat: still in the recovery state (was_healthy=False),
    # ping succeeds, callback now returns True.
    summary2 = hb._execute()
    assert callback.call_count == 2
    assert hb._health_event.is_set()
    assert summary2.message == "healthy"


def test_heartbeat_callback_skipped_when_ping_fails(monkeypatch):
    """During a sustained outage, the callback must NOT fire on each tick.

    Regression test for the recovery gate: only ping=True AND was_healthy=False
    should trigger the callback.
    """
    hb = _make_heartbeat()
    hb._health_event.clear()  # already in degraded mode
    callback = MagicMock(return_value=True)
    hb.register_recover_callback(callback)

    monkeypatch.setattr(adapter_mod, "send_ping", lambda *a, **kw: False)

    for _ in range(3):
        summary = hb._execute()
        assert summary.message == "unhealthy"
        assert not hb._health_event.is_set()

    callback.assert_not_called()


def test_heartbeat_unhealthy_does_not_invoke_callback(monkeypatch):
    """Failed ping: callback never runs (no recovery edge)."""
    hb = _make_heartbeat()
    callback = MagicMock(return_value=True)
    hb.register_recover_callback(callback)

    monkeypatch.setattr(adapter_mod, "send_ping", lambda *a, **kw: False)
    summary = hb._execute()

    callback.assert_not_called()
    assert not hb._health_event.is_set()
    assert summary.message == "unhealthy"


def test_heartbeat_recovery_without_callback(monkeypatch):
    """No callback registered: recovery still sets the event."""
    hb = _make_heartbeat()
    hb._health_event.clear()
    monkeypatch.setattr(adapter_mod, "send_ping", lambda *a, **kw: True)

    summary = hb._execute()

    assert hb._health_event.is_set()
    assert summary.message == "healthy"
