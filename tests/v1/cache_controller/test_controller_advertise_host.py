# SPDX-License-Identifier: Apache-2.0
"""Tests for the controller's advertised host.

At registration, workers receive a heartbeat URL built from the controller's
bind address. Without an advertise host a bind-all address is replaced by the
controller's own IP; with one (e.g. a Kubernetes Service name) workers are
given that host instead, so they can still reach a controller that restarted
with a new IP.
"""

# Standard
from collections.abc import Iterator
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.config import ControllerConfig
from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import RegisterMsg, RegisterRetMsg
from lmcache.v1.rpc_utils import get_ip

ADVERTISE_HOST = "lmcache-controller.default.svc.cluster.local"


def _free_port() -> int:
    """Return a TCP port that is currently free on all interfaces."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _make_manager(heartbeat_port: int, advertise_host: str) -> LMCacheControllerManager:
    return LMCacheControllerManager(
        controller_urls={
            "pull": f"127.0.0.1:{_free_port()}",
            "reply": f"127.0.0.1:{_free_port()}",
            "heartbeat": f"0.0.0.0:{heartbeat_port}",
        },
        health_check_interval=-1,
        lmcache_worker_timeout=30,
        advertise_host=advertise_host,
    )


@pytest.fixture
def heartbeat_port() -> int:
    return _free_port()


@pytest.fixture
def default_manager(heartbeat_port: int) -> Iterator[LMCacheControllerManager]:
    manager = _make_manager(heartbeat_port, advertise_host="")
    yield manager
    manager.close()


@pytest.fixture
def advertising_manager(heartbeat_port: int) -> Iterator[LMCacheControllerManager]:
    manager = _make_manager(heartbeat_port, advertise_host=ADVERTISE_HOST)
    yield manager
    manager.close()


async def _register_heartbeat_url(
    manager: LMCacheControllerManager, worker_ip: str
) -> str:
    """Register one worker and return the heartbeat URL it is given."""
    ret = await manager.handle_worker_req_message(
        RegisterMsg(
            instance_id="test_instance",
            worker_id=0,
            ip=worker_ip,
            port=1,
            peer_init_url=None,
        )
    )
    assert isinstance(ret, RegisterRetMsg)
    return ret.extra_config["heartbeat_url"]


@pytest.mark.asyncio
async def test_heartbeat_url_defaults_to_controller_ip(
    default_manager: LMCacheControllerManager, heartbeat_port: int
) -> None:
    # A worker on another host is given the controller's IP.
    url = await _register_heartbeat_url(default_manager, worker_ip="192.0.2.10")
    assert url == f"{get_ip()}:{heartbeat_port}"


@pytest.mark.asyncio
async def test_heartbeat_url_uses_advertise_host(
    advertising_manager: LMCacheControllerManager, heartbeat_port: int
) -> None:
    url = await _register_heartbeat_url(advertising_manager, worker_ip="192.0.2.10")
    assert url == f"{ADVERTISE_HOST}:{heartbeat_port}"


@pytest.mark.asyncio
async def test_advertise_host_applies_to_same_host_workers(
    advertising_manager: LMCacheControllerManager, heartbeat_port: int
) -> None:
    # Without an advertise host a same-host worker would get 127.0.0.1; the
    # advertise host must win so every worker keeps a restart-proof address.
    url = await _register_heartbeat_url(advertising_manager, worker_ip=get_ip())
    assert url == f"{ADVERTISE_HOST}:{heartbeat_port}"


def test_config_advertise_host_default_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LMCACHE_CONTROLLER_CONTROLLER_ADVERTISE_HOST", raising=False)
    assert ControllerConfig.from_env().controller_advertise_host == ""


def test_config_advertise_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMCACHE_CONTROLLER_CONTROLLER_ADVERTISE_HOST", ADVERTISE_HOST)
    assert ControllerConfig.from_env().controller_advertise_host == ADVERTISE_HOST
