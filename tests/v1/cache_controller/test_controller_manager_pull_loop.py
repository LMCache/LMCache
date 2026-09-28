# SPDX-License-Identifier: Apache-2.0
"""Tests for the controller manager's PULL receive loop.

The PULL loop must keep receiving worker messages after a frame that cannot
be decoded or a message whose handler raises: ending the loop silently stops
the controller from tracking KV admissions/evictions for the rest of its life.
"""

# Standard
from collections.abc import Iterator
from unittest.mock import AsyncMock
import asyncio
import socket

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import DeRegisterMsg


def _free_port() -> int:
    """Return a TCP port that is currently free on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakePullSocket:
    """Minimal stand-in for a ZMQ PULL socket.

    Returns the queued multipart batches in order, then raises
    ``asyncio.CancelledError`` to end the receive loop.
    """

    def __init__(self, batches: list[list[bytes]]) -> None:
        self._batches = list(batches)

    async def recv_multipart(self) -> list[bytes]:
        """Return the next queued batch, or cancel once all are consumed."""
        if not self._batches:
            raise asyncio.CancelledError
        return self._batches.pop(0)


def _deregister(worker_id: int) -> DeRegisterMsg:
    return DeRegisterMsg(
        instance_id="test_instance", worker_id=worker_id, ip="127.0.0.1", port=1
    )


@pytest.fixture
def manager() -> Iterator[LMCacheControllerManager]:
    controller_manager = LMCacheControllerManager(
        controller_urls={
            "pull": f"127.0.0.1:{_free_port()}",
            "reply": f"127.0.0.1:{_free_port()}",
        },
        health_check_interval=-1,
        lmcache_worker_timeout=30,
    )
    yield controller_manager
    controller_manager.close()


@pytest.mark.asyncio
async def test_pull_loop_skips_undecodable_frames(
    manager: LMCacheControllerManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = AsyncMock()
    monkeypatch.setattr(manager, "handle_worker_message", handler)
    first, second = _deregister(0), _deregister(1)
    fake_socket = _FakePullSocket(
        [
            [
                # An HTTP request on the ZMQ port: "G" decodes as a msgpack int.
                b"GET /metrics HTTP/1.1\r\nHost: controller\r\n\r\n",
                b"{not json",
                b"\xff\xfe\xfd",
                msgspec.msgpack.encode(first),
            ],
            [msgspec.msgpack.encode(second)],
        ]
    )

    with pytest.raises(asyncio.CancelledError):
        await manager.handle_batched_push_request(fake_socket)

    assert [call.args[0] for call in handler.await_args_list] == [first, second]


@pytest.mark.asyncio
async def test_pull_loop_survives_handler_exception(
    manager: LMCacheControllerManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = AsyncMock(side_effect=[RuntimeError("handler failed"), None])
    monkeypatch.setattr(manager, "handle_worker_message", handler)
    first, second = _deregister(0), _deregister(1)
    fake_socket = _FakePullSocket(
        [[msgspec.msgpack.encode(first)], [msgspec.msgpack.encode(second)]]
    )

    with pytest.raises(asyncio.CancelledError):
        await manager.handle_batched_push_request(fake_socket)

    assert [call.args[0] for call in handler.await_args_list] == [first, second]
