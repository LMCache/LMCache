# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CoordinatorClient failure-mode handling."""

# Standard
from unittest.mock import Mock
import socket as _socket

# Third Party
import zmq

# First Party
from lmcache.v1.mp_coordinator.client import CoordinatorClient
from lmcache.v1.rpc_utils import close_zmq_socket


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _client() -> CoordinatorClient:
    return CoordinatorClient(
        instance_id="i",
        reply_url=f"127.0.0.1:{_free_port()}",
        heartbeat_url=f"127.0.0.1:{_free_port()}",
        pull_url=f"127.0.0.1:{_free_port()}",
        control_port=_free_port(),
        advertise_ip="127.0.0.1",
        heartbeat_interval=0.1,
    )


def test_heartbeat_once_rebuilds_socket_on_failure():
    client = _client()
    bad = Mock()
    bad.send.side_effect = zmq.ZMQError("boom")

    new_socket = client._heartbeat_once(bad, b"hb")

    # The failed socket was closed and a fresh one returned, so the loop
    # recovers instead of staying wedged in the REQ state machine.
    bad.close.assert_called_once()
    assert new_socket is not bad
    close_zmq_socket(new_socket)


def test_heartbeat_once_keeps_socket_on_success():
    client = _client()
    ok = Mock()
    ok.recv.return_value = b""

    same_socket = client._heartbeat_once(ok, b"hb")

    assert same_socket is ok
    ok.send.assert_called_once_with(b"hb")
    ok.close.assert_not_called()
