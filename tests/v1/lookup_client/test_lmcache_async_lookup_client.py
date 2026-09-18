# SPDX-License-Identifier: Apache-2.0
"""Tests for asynchronous lookup cancellation lifecycle handling."""

# Standard
import threading

# Third Party
import msgspec

# First Party
from lmcache.v1.lookup_client.async_lookup_message import LookupCleanupMsg
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)


class RecordingSocket:
    """Record ZMQ-compatible send calls without opening a real socket."""

    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def send(self, message: bytes, copy: bool = False) -> None:
        """Record one serialized message.

        Args:
            message: Serialized lookup message.
            copy: Compatibility argument matching ``zmq.Socket.send``.
        """
        self.messages.append(message)


def test_cancel_lookup_sends_cleanup_immediately_once() -> None:
    """Cancellation must not wait for another lookup before notifying workers."""
    socket = RecordingSocket()
    client = object.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.RLock()
    client.world_size = 1
    client.push_sockets = [socket]
    client.reqs_status = {"aborted": None}
    client.res_for_each_worker = {}
    client.first_lookup_time = {"aborted": 1.0}
    client.aborted_lookups = set()

    client.cancel_lookup("aborted")
    client.cancel_lookup("aborted")

    assert len(socket.messages) == 1
    message = msgspec.msgpack.decode(socket.messages[0], type=LookupCleanupMsg)
    assert message.lookup_id == "aborted"
