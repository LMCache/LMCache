# SPDX-License-Identifier: Apache-2.0
"""Unit tests for reclaiming workers whose client connection closed.

The transport side (reporting which connection closed) is covered in
test_mq.py; these tests drive the liveness side through public interfaces,
simulating the transport with ``bind_request_peer``. No GPU is required.
"""

# Standard
from multiprocessing.synchronize import Event as EventClass
from typing import Callable
from unittest.mock import MagicMock
import multiprocessing as mp
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as gpu_mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.request_handler import bind_request_peer
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.periodic_thread import PeriodicThreadRegistry

# Test helpers
from tests.v1.multiprocess.transport_test_utils import request_server_config

PEER_A = b"\x00peer-a"
PEER_B = b"\x00peer-b"
# Far enough in the future that a countdown never expires during a test.
LONG = 1e6


@pytest.fixture(autouse=True)
def _reset_periodic_registry():
    """Keep reaper threads out of the global registry across tests."""
    PeriodicThreadRegistry.reset()
    yield
    PeriodicThreadRegistry.reset()


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> LMCacheDrivenTransferModule:
    """A transfer module whose registration path needs no GPU."""
    monkeypatch.setattr(
        gpu_mod,
        "create_cache_context",
        lambda *a, **kw: MagicMock(
            num_layers=2, **{"kv_layer_groups_manager.num_object_groups": 1}
        ),
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        gpu_mod,
        "get_event_ipc_backend",
        MagicMock(return_value=MagicMock(name="event_backend")),
    )
    # Bypass __init__, which starts a device host-func dispatcher.
    mod = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    mod._ctx = MagicMock(name="ctx")
    mod._cache_contexts = {}
    mod._lock = threading.Lock()
    return mod


def _register(
    module: LMCacheDrivenTransferModule,
    instance_id: int,
    peer: bytes | None,
    *,
    ping: bool = True,
) -> None:
    """Register ``instance_id`` over ``peer``, optionally ping-proving it."""
    with bind_request_peer(peer):
        module.register_kv_cache(
            instance_id, MagicMock(), "model", 1, MagicMock(), MagicMock(), []
        )
        if ping:
            module.touch_instance(instance_id)


def _reap(module: LMCacheDrivenTransferModule) -> list[int]:
    """Run one reaper scan with the default silence budgets."""
    return module.reap_stale_instances(120.0, 3600.0)


def test_closed_connection_reaps_after_countdown(module) -> None:
    """A ping-proven worker is reaped once its disconnect countdown expires,
    long before the silence budget, and not before the countdown."""
    _register(module, 1, PEER_A)
    _register(module, 2, PEER_B)

    module.mark_peer_disconnected(PEER_A, LONG, LONG)
    module.mark_peer_disconnected(PEER_B, 0.0, LONG)
    status = module.report_status()["cache_context_meta"]

    assert status["1"]["connection_closed"] is True
    assert _reap(module) == [2]
    assert module.tracked_instance_count() == 1
    assert 2 not in module.context_entries_snapshot()


def test_never_pinged_worker_gets_the_unproven_countdown(module) -> None:
    """A warming worker cannot prove liveness, so it gets the longer
    unproven countdown rather than the ping-proven one."""
    _register(module, 1, PEER_A, ping=True)
    _register(module, 2, PEER_A, ping=False)

    module.mark_peer_disconnected(PEER_A, 0.0, LONG)

    assert _reap(module) == [1]
    module.mark_peer_disconnected(PEER_A, 0.0, 0.0)  # already counting down
    assert _reap(module) == []


def test_only_instances_on_the_closed_connection_are_marked(module) -> None:
    """Other connections' instances are untouched."""
    _register(module, 1, PEER_A)
    _register(module, 2, PEER_B)

    module.mark_peer_disconnected(PEER_A, 0.0, 0.0)

    assert _reap(module) == [1]
    status = module.report_status()["cache_context_meta"]
    assert status["2"]["connection_closed"] is False


@pytest.mark.parametrize("touch", ["ping", "transfer", "register"])
def test_request_over_a_new_connection_cancels_countdown(module, touch) -> None:
    """Any request for the instance over another connection means it
    reconnected: the countdown is cancelled and the entry follows it."""
    _register(module, 1, PEER_A)
    module.mark_peer_disconnected(PEER_A, 0.0, 0.0)

    with bind_request_peer(PEER_B):
        if touch == "ping":
            module.touch_instance(1)
        elif touch == "transfer":
            module.get_and_touch_context_entry(1)
        else:
            module.register_kv_cache(
                1, MagicMock(), "model", 1, MagicMock(), MagicMock(), []
            )

    assert _reap(module) == []
    module.mark_peer_disconnected(PEER_A, 0.0, 0.0)  # the old connection
    assert _reap(module) == []
    module.mark_peer_disconnected(PEER_B, 0.0, 0.0)  # the new one
    assert _reap(module) == [1]


def test_requests_from_the_closed_connection_do_not_cancel(module) -> None:
    """Requests still queued from the dead connection, and calls with no
    known connection (e.g. HTTP APIs), do not keep the instance alive."""
    _register(module, 1, PEER_A)
    module.mark_peer_disconnected(PEER_A, 0.0, 0.0)

    with bind_request_peer(PEER_A):
        module.get_and_touch_context_entry(1)
        module.touch_instance(1)
    module.get_and_touch_context_entry(1)

    assert _reap(module) == [1]


def test_untracked_registration_is_never_marked(module) -> None:
    """Without a known connection (e.g. gRPC), only the timeouts apply."""
    _register(module, 1, None)

    module.mark_peer_disconnected(PEER_A, 0.0, 0.0)

    assert _reap(module) == []
    assert module.tracked_instance_count() == 1


class _RecordingTarget(InstanceLivenessTarget):
    """Liveness target recording disconnect notifications."""

    def __init__(self) -> None:
        self.marked: list[tuple[bytes, float, float]] = []

    def mark_peer_disconnected(
        self, peer: bytes, proven_grace_s: float, unproven_grace_s: float
    ) -> None:
        self.marked.append((peer, proven_grace_s, unproven_grace_s))


def test_management_forwards_disconnects_with_both_countdowns() -> None:
    """A closed connection reaches every target with the disconnect grace
    for ping-proven workers and the reap timeout for never-pinged ones."""
    targets = [_RecordingTarget(), _RecordingTarget()]
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=targets,
        worker_reap_timeout_seconds=120.0,
        worker_registration_grace_seconds=3600.0,
        worker_disconnect_grace_seconds=30.0,
    )
    try:
        assert mgmt.reclaims_on_disconnect is True
        mgmt.on_peer_disconnected(PEER_A)

        assert [t.marked for t in targets] == [[(PEER_A, 30.0, 120.0)]] * 2
        assert mgmt._reaper is not None
        assert mgmt._reaper.interval == pytest.approx(10.0)
        status = mgmt.report_status()["worker_liveness"]
        assert status["disconnect_grace_seconds"] == 30.0
    finally:
        mgmt.close()


@pytest.mark.parametrize(
    ("reap_timeout", "disconnect_grace"), [(0.0, 30.0), (120.0, 0.0)]
)
def test_management_ignores_disconnects_when_disabled(
    reap_timeout: float, disconnect_grace: float
) -> None:
    """Disabled reaping or a zero grace turns connection-loss reclaim off
    and leaves the reaper interval at reap_timeout / 4."""
    target = _RecordingTarget()
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[target],
        worker_reap_timeout_seconds=reap_timeout,
        worker_registration_grace_seconds=3600.0,
        worker_disconnect_grace_seconds=disconnect_grace,
    )
    try:
        assert mgmt.reclaims_on_disconnect is False
        mgmt.on_peer_disconnected(PEER_A)

        assert target.marked == []
        if mgmt._reaper is not None:
            assert mgmt._reaper.interval == pytest.approx(reap_timeout / 4)
        status = mgmt.report_status()["worker_liveness"]
        assert status["disconnect_grace_seconds"] == 0.0
    finally:
        mgmt.close()


def _register_ping_then_hang(
    server_url: str, instance_id: int, ready: EventClass
) -> None:
    """Worker stand-in process: register, ping once, then idle until killed."""
    client = RequestClientFactory.create(server_url)
    client.register_kv_cache(
        instance_id, [], "model", 1, EngineType.VLLM, {}, []
    ).result(timeout=30)
    client.ping(instance_id).result(timeout=30)
    ready.set()
    time.sleep(120)


def _wait_for(predicate: Callable[[], bool], timeout: float) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def test_killed_worker_is_reaped_through_the_zmq_server(module) -> None:
    """End to end over ZMQ: of two registered, ping-proven workers, the one
    that is SIGKILLed is reaped after the disconnect countdown -- far below
    the reap timeout -- and the live one is kept."""
    server_url = "tcp://127.0.0.1:16044"
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[module],
        worker_reap_timeout_seconds=60.0,
        worker_registration_grace_seconds=3600.0,
        worker_disconnect_grace_seconds=0.3,
    )
    server = create_request_server(
        [mgmt, module],
        request_server_config("zmq", server_url),
        on_peer_disconnected=mgmt.on_peer_disconnected,
    )
    server.start()
    spawn = mp.get_context("spawn")
    workers = []
    try:
        for instance_id in (1, 2):
            ready = spawn.Event()
            proc = spawn.Process(
                target=_register_ping_then_hang,
                args=(server_url, instance_id, ready),
                daemon=True,
            )
            proc.start()
            workers.append((proc, ready))
        for _, ready in workers:
            assert ready.wait(timeout=60)
        assert module.tracked_instance_count() == 2

        killed_at = time.monotonic()
        workers[0][0].kill()

        assert _wait_for(lambda: module.tracked_instance_count() == 1, timeout=10)
        assert time.monotonic() - killed_at < 5.0
        assert list(module.context_entries_snapshot()) == [2]
    finally:
        for proc, _ in workers:
            if proc.is_alive():
                proc.kill()
        server.close()
        mgmt.close()


def test_grpc_transport_builds_without_disconnect_reporting() -> None:
    """gRPC cannot report closed connections; the callback is ignored and
    the server still builds (workers fall back to the heartbeat timeout)."""
    mgmt = ManagementModule(MagicMock())
    server = create_request_server(
        [mgmt],
        request_server_config("grpc", "grpc://127.0.0.1:16045"),
        on_peer_disconnected=mgmt.on_peer_disconnected,
    )
    server.close()


@pytest.mark.parametrize("grace", [0.0, 30.0, 60.0, 120.0])
def test_config_accepts_disconnect_grace(grace: float) -> None:
    """0 disables; otherwise any value from 30s up to the reap timeout."""
    config = MPServerConfig(
        worker_reap_timeout_seconds=120.0, worker_disconnect_grace_seconds=grace
    )
    assert config.worker_disconnect_grace_seconds == grace


@pytest.mark.parametrize("grace", [10.0, 121.0, -1.0, float("nan"), float("inf")])
def test_config_rejects_bad_disconnect_grace(grace: float) -> None:
    """Sub-floor, above-timeout and non-finite values are rejected."""
    with pytest.raises(ValueError, match="disconnect grace"):
        MPServerConfig(
            worker_reap_timeout_seconds=120.0,
            worker_disconnect_grace_seconds=grace,
        )


def test_config_disconnect_grace_defaults() -> None:
    """The default is on at 30s and stays valid when reaping is disabled."""
    assert MPServerConfig().worker_disconnect_grace_seconds == 30.0
    MPServerConfig(
        worker_reap_timeout_seconds=0.0, worker_registration_grace_seconds=0.0
    )
