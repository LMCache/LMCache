# SPDX-License-Identifier: Apache-2.0
"""Tests for adapter-side heartbeat: tri-state `send_ping`,
cold-start grace, terminal-on-False, eager-start, and worker shutdown.

We avoid spinning up the full server; instead we mock `mq_client` and
directly drive the helpers.
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    COLD_START_FAILURE_THRESHOLD,
    DEFAULT_HEARTBEAT_INTERVAL,
    PING_SENTINEL_INSTANCE_ID,
    HeartbeatThread,
    LMCacheMPWorkerAdapter,
    send_ping,
)

# =============================================================================
# send_ping tri-state
# =============================================================================


def test_send_ping_returns_true_when_server_returns_true():
    mq = MagicMock()
    fut = MagicMock()
    fut.result.return_value = True
    mq.submit_request.return_value = fut

    result = send_ping(mq, timeout=1.0, instance_id=12345)
    assert result is True
    # PING payload should include the instance_id.
    args, _ = mq.submit_request.call_args
    assert args[1] == [12345]


def test_send_ping_returns_false_when_server_returns_false():
    mq = MagicMock()
    fut = MagicMock()
    fut.result.return_value = False
    mq.submit_request.return_value = fut

    result = send_ping(mq, timeout=1.0, instance_id=99999)
    assert result is False  # terminal — server doesn't know us


def test_send_ping_returns_none_on_timeout():
    mq = MagicMock()
    fut = MagicMock()
    fut.result.side_effect = TimeoutError()
    mq.submit_request.return_value = fut

    result = send_ping(mq, timeout=1.0, instance_id=12345)
    assert result is None


def test_send_ping_returns_none_on_other_exception():
    mq = MagicMock()
    fut = MagicMock()
    fut.result.side_effect = RuntimeError("zmq blip")
    mq.submit_request.return_value = fut

    result = send_ping(mq, timeout=1.0, instance_id=12345)
    assert result is None


def test_send_ping_distinguishes_transient_from_terminal():
    """The whole point of `Optional[bool]`: a transient zmq exception
    must NOT collapse into the terminal `False` signal. The previous
    bool-only signature conflated them."""
    mq = MagicMock()
    fut = MagicMock()

    # Transient → None
    fut.result.side_effect = TimeoutError()
    mq.submit_request.return_value = fut
    transient = send_ping(mq, timeout=1.0, instance_id=1)
    assert transient is None

    # Terminal → False (server returned False)
    fut.result.side_effect = None
    fut.result.return_value = False
    terminal = send_ping(mq, timeout=1.0, instance_id=1)
    assert terminal is False

    assert transient != terminal  # nominal sanity


# =============================================================================
# HeartbeatThread state machine
# =============================================================================


def _make_heartbeat(
    instance_id: int = 42,
) -> tuple[HeartbeatThread, threading.Event, MagicMock]:
    """Build a HeartbeatThread without starting it. Returns (hb, event, mq)."""
    health = threading.Event()
    health.set()
    mq = MagicMock()
    hb = HeartbeatThread(
        mq_client=mq,
        health_event=health,
        instance_id=instance_id,
        interval=DEFAULT_HEARTBEAT_INTERVAL,
    )
    return hb, health, mq


def _patch_send_ping(monkeypatch, *return_values):
    """Patch `send_ping` in the adapter module to return `return_values`
    in sequence (one per call)."""
    queue = list(return_values)

    def fake_send_ping(*_args, **_kwargs):
        assert queue, "send_ping called more times than scripted"
        return queue.pop(0)

    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_multi_process_adapter.send_ping",
        fake_send_ping,
    )


def test_heartbeat_healthy_sets_event_and_latches(monkeypatch):
    hb, health, _ = _make_heartbeat()
    health.clear()  # start unhealthy to verify recovery
    _patch_send_ping(monkeypatch, True)

    hb._execute()
    assert health.is_set()
    assert hb._first_success_seen is True
    assert hb._consecutive_failures == 0


def test_heartbeat_cold_start_grace_absorbs_one_failure(monkeypatch):
    """Adapter that has never seen a healthy ping must NOT clear health
    after a single transient failure — wait until we hit the threshold."""
    hb, health, _ = _make_heartbeat()
    assert COLD_START_FAILURE_THRESHOLD == 2  # guard for the assertion below

    # First failure during cold-start: stays healthy.
    _patch_send_ping(monkeypatch, None)
    hb._execute()
    assert health.is_set()
    assert hb._consecutive_failures == 1
    assert hb._first_success_seen is False

    # Second failure: hits the threshold, clears.
    _patch_send_ping(monkeypatch, None)
    hb._execute()
    assert not health.is_set()


def test_heartbeat_steady_state_clears_on_first_failure(monkeypatch):
    """Once `_first_success_seen` latches, a single transient failure
    flips immediately. The cold-start grace is one-shot."""
    hb, health, _ = _make_heartbeat()

    # Latch _first_success_seen via a healthy ping.
    _patch_send_ping(monkeypatch, True)
    hb._execute()
    assert hb._first_success_seen is True
    assert health.is_set()

    # Transient failure flips us immediately.
    _patch_send_ping(monkeypatch, None)
    hb._execute()
    assert not health.is_set()


def test_heartbeat_recovery_does_not_re_extend_grace(monkeypatch):
    """After a flap (success → failure → success), we are still in
    steady-state; grace doesn't reset."""
    hb, health, _ = _make_heartbeat()

    # Latch.
    _patch_send_ping(monkeypatch, True)
    hb._execute()
    assert hb._first_success_seen is True

    # Flip unhealthy on a single failure.
    _patch_send_ping(monkeypatch, None)
    hb._execute()
    assert not health.is_set()

    # Recover.
    _patch_send_ping(monkeypatch, True)
    hb._execute()
    assert health.is_set()
    assert hb._first_success_seen is True

    # New single failure: still steady-state, flips immediately.
    _patch_send_ping(monkeypatch, None)
    hb._execute()
    assert not health.is_set()


def test_heartbeat_terminal_false_clears_and_stops_thread(monkeypatch):
    """Server returned False → we've been reaped. Thread must stop and
    health must clear, regardless of cold-start state."""
    hb, health, _ = _make_heartbeat()
    _patch_send_ping(monkeypatch, False)

    hb._execute()
    assert not health.is_set()
    # `stop(timeout=0.0)` was called — running flag flips.
    assert hb._running is False


def test_heartbeat_terminal_false_during_cold_start(monkeypatch):
    """Even before any healthy ping, an explicit False from the server
    is terminal — the cold-start grace does not apply to False, only
    to None (transient failures)."""
    hb, health, _ = _make_heartbeat()
    _patch_send_ping(monkeypatch, False)

    hb._execute()
    assert not health.is_set()
    assert hb._running is False
    # _first_success_seen never latched (we never saw True).
    assert hb._first_success_seen is False


# =============================================================================
# Worker eager-start + shutdown
# =============================================================================


def _make_parallel_strategy():
    """Minimal `ParallelStrategy` for tests that need one."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ParallelStrategy,
    )

    return ParallelStrategy(
        use_mla=False,
        kv_world_size=1,
        kv_worker_id=0,
        actual_world_size=1,
        actual_worker_id=0,
        tp_size=1,
        pp_size=1,
    )


def _bare_worker_adapter() -> LMCacheMPWorkerAdapter:
    """Build a worker adapter without going through __init__ (which
    requires a real MQ + chunk-size handshake). Wire only the fields
    the methods we test reach."""
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.mq_client = MagicMock()
    adapter._mq_timeout = 5.0
    adapter._heartbeat = None
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = DEFAULT_HEARTBEAT_INTERVAL
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter.instance_id = 0xCAFEBABE
    return adapter


def test_worker_register_kv_caches_starts_heartbeat_after_ack():
    """Eager start: `register_kv_caches` must call
    `_ensure_heartbeat_started()` only AFTER the REGISTER ack succeeds.
    On REGISTER timeout, no heartbeat thread is created."""
    adapter = _bare_worker_adapter()
    adapter.model_name = "m"
    adapter.parallel_strategy = _make_parallel_strategy()
    adapter.kv_caches = {}

    # Mock the REGISTER future to succeed.
    fut = MagicMock()
    fut.result.return_value = True
    adapter.mq_client.submit_request.return_value = fut

    # Patch `_ensure_heartbeat_started` so we can detect the call.
    with (
        patch.object(adapter, "_ensure_heartbeat_started") as ehs,
        patch(
            "lmcache.integration.vllm.vllm_multi_process_adapter.wrap_kv_caches",
            return_value=[],
        ),
        patch(
            "lmcache.integration.vllm.utils.vllm_layout_hints",
            return_value=None,
            create=True,
        ),
    ):
        adapter.register_kv_caches({})
    ehs.assert_called_once()


def test_worker_register_kv_caches_does_not_start_heartbeat_on_timeout():
    """REGISTER timeout → ConnectionError raised, heartbeat NOT started."""
    adapter = _bare_worker_adapter()
    adapter.model_name = "m"
    adapter.parallel_strategy = _make_parallel_strategy()
    adapter.kv_caches = {}

    fut = MagicMock()
    fut.result.side_effect = TimeoutError()
    adapter.mq_client.submit_request.return_value = fut

    with (
        patch.object(adapter, "_ensure_heartbeat_started") as ehs,
        patch(
            "lmcache.integration.vllm.vllm_multi_process_adapter.wrap_kv_caches",
            return_value=[],
        ),
        patch(
            "lmcache.integration.vllm.utils.vllm_layout_hints",
            return_value=None,
            create=True,
        ),
    ):
        with pytest.raises(ConnectionError):
            adapter.register_kv_caches({})
    ehs.assert_not_called()


def test_worker_shutdown_stops_heartbeat_first():
    """`shutdown` must stop the heartbeat before sending UNREGISTER so a
    stray PING doesn't race the closing mq_client."""
    adapter = _bare_worker_adapter()
    adapter.kv_caches = {}
    adapter.request_telemetry = MagicMock()

    # Hook up a mock heartbeat so we can verify stop() ordering.
    hb_mock = MagicMock()
    adapter._heartbeat = hb_mock

    fut = MagicMock()
    fut.result.return_value = None
    adapter.mq_client.submit_request.return_value = fut

    call_order: list[str] = []
    hb_mock.stop.side_effect = lambda *a, **k: call_order.append("hb_stop")

    real_submit = adapter.mq_client.submit_request

    def tracked_submit(*args, **kwargs):
        call_order.append("submit_request")
        return real_submit(*args, **kwargs)

    adapter.mq_client.submit_request = tracked_submit

    adapter.shutdown()

    assert call_order[0] == "hb_stop"
    assert "submit_request" in call_order
    assert call_order.index("hb_stop") < call_order.index("submit_request")


# =============================================================================
# Worker random instance_id
# =============================================================================


def test_worker_instance_id_is_not_pid_and_is_63_bit_positive(monkeypatch):
    """Verify `instance_id` is a freshly-generated 63-bit positive int.

    This guards against a future regression where someone "fixes" a test
    by hard-coding `os.getpid()` again.
    """
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ParallelStrategy,
    )

    captured: dict[str, object] = {}

    def fake_getrandbits(n):
        assert n == 63
        captured["called"] = True
        captured["n"] = n
        return 0x1234567890ABCDEF

    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_multi_process_adapter.random.getrandbits",
        fake_getrandbits,
    )

    # The constructor needs an MQ that responds to GET_CHUNK_SIZE.
    fut = MagicMock()
    fut.result.return_value = 256  # chunk size
    mq = MagicMock()
    mq.submit_request.return_value = fut

    with patch(
        "lmcache.integration.vllm.vllm_multi_process_adapter.MessageQueueClient",
        return_value=mq,
    ):
        ps = ParallelStrategy(
            use_mla=False,
            kv_world_size=1,
            kv_worker_id=0,
            actual_world_size=1,
            actual_worker_id=0,
            tp_size=1,
            pp_size=1,
        )
        adapter = LMCacheMPWorkerAdapter(
            server_url="tcp://x:1",
            context=MagicMock(),
            model_name="m",
            vllm_block_size=16,
            parallel_strategy=ps,
        )

    assert captured.get("called") is True
    assert adapter.instance_id == 0x1234567890ABCDEF
    # 63 bits → must be in [0, 2**63 - 1].
    assert 0 <= adapter.instance_id < (1 << 63)
    # And it must not collide with the sentinel.
    assert adapter.instance_id != PING_SENTINEL_INSTANCE_ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
