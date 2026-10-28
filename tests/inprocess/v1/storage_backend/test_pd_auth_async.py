# SPDX-License-Identifier: Apache-2.0

"""Tests for PD allocation-plane authentication on the wire and in the
receiver handler."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import asyncio

# Third Party
import msgspec
import pytest

# First Party
from lmcache.inprocess.v1.storage_backend.pd_backend_async import (
    AllocRequest,
    AllocResponse,
    CancelNotif,
    PDBackendAsync,
    PDMsg,
)


class TestMessageRoundTrip:
    def test_alloc_request_carries_token(self):
        req = AllocRequest(
            keys=["k0"],
            fmt=1,
            shape=[2, 4],
            dtype="bfloat16",
            last_chunk_toks=16,
            req_id="r0",
            total_chunks=1,
            auth_token=b"secret",
        )
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=PDMsg)
        assert isinstance(decoded, AllocRequest)
        assert decoded.auth_token == b"secret"

    def test_alloc_request_without_token_decodes_empty(self):
        """Peers that do not know about the field keep decoding cleanly."""
        legacy_payload = {
            "keys": ["k0"],
            "fmt": 1,
            "shape": [2, 4],
            "dtype": "bfloat16",
            "last_chunk_toks": 16,
            "req_id": "r0",
            "is_last_batch": True,
            "total_chunks": 1,
        }
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(legacy_payload), type=AllocRequest
        )
        assert decoded.auth_token == b""

    def test_cancel_notif_carries_token(self):
        msg = CancelNotif(req_id="r0", keys=["k0"], auth_token=b"secret")
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(msg), type=PDMsg)
        assert isinstance(decoded, CancelNotif)
        assert decoded.auth_token == b"secret"


def _make_receiver(token):
    """Build a receiver with only the state the handler touches."""
    backend = object.__new__(PDBackendAsync)
    backend._pd_auth_token = token
    backend._router_send_lock = asyncio.Lock()
    backend._async_allocate_and_put = AsyncMock(
        return_value=AllocResponse(remote_indexes=[0])
    )
    backend.remove = MagicMock()
    backend._req_allocated_keys = {}
    backend._recv_reservation_mgr = MagicMock()
    backend._recv_reservation_mgr.async_release_reservation = AsyncMock()
    return backend


def _make_socket():
    socket = MagicMock()
    socket.send_multipart = AsyncMock()
    return socket


def _last_response(socket):
    frames = socket.send_multipart.await_args.args[0]
    return msgspec.msgpack.decode(frames[-1], type=AllocResponse)


def _alloc_request(**overrides):
    args = {
        "keys": ["k0", "k1"],
        "fmt": 1,
        "shape": [2, 4],
        "dtype": "bfloat16",
        "last_chunk_toks": 16,
        "req_id": "r0",
        "total_chunks": 2,
    }
    args.update(overrides)
    return AllocRequest(**args)


class TestHandleAllocRequestAuth:
    def test_unauthenticated_alloc_request_is_rejected(self):
        backend = _make_receiver(b"secret")
        socket = _make_socket()
        payload = msgspec.msgpack.encode(_alloc_request())

        asyncio.run(backend._handle_alloc_request(socket, b"peer", payload))

        backend._async_allocate_and_put.assert_not_awaited()
        assert _last_response(socket).remote_indexes == [-1, -1]

    def test_authenticated_alloc_request_is_served(self):
        backend = _make_receiver(b"secret")
        socket = _make_socket()
        payload = msgspec.msgpack.encode(_alloc_request(auth_token=b"secret"))

        asyncio.run(backend._handle_alloc_request(socket, b"peer", payload))

        backend._async_allocate_and_put.assert_awaited_once()
        assert _last_response(socket).remote_indexes == [0]

    def test_forged_cancel_notif_does_not_release_keys(self):
        backend = _make_receiver(b"secret")
        socket = _make_socket()
        payload = msgspec.msgpack.encode(CancelNotif(req_id="r0", keys=["k0"]))

        asyncio.run(backend._handle_alloc_request(socket, b"peer", payload))

        backend.remove.assert_not_called()
        backend._recv_reservation_mgr.async_release_reservation.assert_not_awaited()
        assert _last_response(socket).remote_indexes == [-1]

    def test_authentication_disabled_keeps_legacy_behaviour(self):
        backend = _make_receiver(None)
        socket = _make_socket()
        payload = msgspec.msgpack.encode(_alloc_request())

        asyncio.run(backend._handle_alloc_request(socket, b"peer", payload))

        backend._async_allocate_and_put.assert_awaited_once()
        assert _last_response(socket).remote_indexes == [0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
