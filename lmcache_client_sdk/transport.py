# SPDX-License-Identifier: Apache-2.0
"""Synchronous ZMQ transport for the standalone client.

The client speaks the LMCache multiprocess wire protocol directly: a
MessagePack-encoded frame of ``[request_uid, operation, *payloads]`` per
request and ``[request_uid, operation, *response]`` per reply. A DEALER
socket is used instead of REQ because REQ inserts an extra empty delimiter
frame that would shift the ROUTER-side frame layout; DEALER preserves the
exact wire contract the server expects. Calls stay synchronous: one request
at a time, with a receive timeout.
"""

# Standard
from collections.abc import Sequence
from typing import Any
import itertools

# Third Party
import msgspec
import zmq

# First Party
from lmcache_client_sdk.key import IPCCacheServerKey
from lmcache_client_sdk.logging import init_logger
from lmcache_client_sdk.protocol import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)

logger = init_logger(__name__)

_REQUEST_TIMEOUT_MS = 60_000


def _encode_payload(obj: Any, cls: Any) -> bytes:
    """Encode one payload to its wire bytes.

    Args:
        obj: The payload object to encode.
        cls: The declared wire type (used for ``bool``/``int`` coercion).

    Returns:
        The MessagePack bytes for the payload.
    """
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def _decode_payload(data: bytes, cls: Any) -> Any:
    """Decode one response frame.

    Args:
        data: MessagePack bytes from the server.
        cls: The declared wire type.

    Returns:
        The decoded response object.
    """
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(data))
    return msgspec.msgpack.decode(data, type=cls)


class ZmqSyncClient:
    """Synchronous ZMQ request client for the LMCache server.

    One client owns one DEALER socket. Callers must invoke a request and
    consume its response before issuing the next one.

    Args:
        server_url: The server endpoint (``tcp://`` for ZMQ).
        timeout: Per-request timeout in seconds. Defaults to 60.
    """

    def __init__(self, server_url: str, timeout: float = 60.0) -> None:
        self._timeout_ms = int(timeout * 1000)
        self._uid_counter = itertools.count()
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(server_url)

    def _send_request(
        self,
        operation: str,
        payloads: Sequence[Any],
        payload_types: Sequence[Any],
        response_type: Any,
    ) -> Any:
        """Send one RPC request and return the decoded response.

        Args:
            operation: Stable snake-case RPC name.
            payloads: Ordered request payload objects.
            payload_types: Declared wire types, one per payload.
            response_type: Declared wire type of the response.

        Returns:
            The decoded response object.

        Raises:
            TimeoutError: If the server does not reply within the timeout.
            RuntimeError: If the reply is malformed.
        """
        request_uid = next(self._uid_counter)
        b_uid = msgspec.msgpack.encode(request_uid)
        b_operation = msgspec.msgpack.encode(operation)
        b_payloads = [
            _encode_payload(payload, cls)
            for payload, cls in zip(payloads, payload_types, strict=True)
        ]
        try:
            self._socket.send_multipart([b_uid, b_operation, *b_payloads])
            reply = self._socket.recv_multipart()
        except zmq.Again as exc:
            raise TimeoutError(
                f"LMCache server did not reply to {operation!r} within "
                f"{self._timeout_ms} ms"
            ) from exc
        if len(reply) < 2:
            raise RuntimeError(
                f"Malformed LMCache reply for {operation!r}: "
                f"expected at least [request_uid, operation, *response]"
            )
        b_reply_uid, b_reply_operation, *b_response = reply
        reply_uid = msgspec.msgpack.decode(b_reply_uid, type=int)
        if reply_uid != request_uid:
            raise RuntimeError(
                f"LMCache reply uid mismatch for {operation!r}: "
                f"expected {request_uid}, got {reply_uid}"
            )
        if msgspec.msgpack.decode(b_reply_operation, type=str) != operation:
            raise RuntimeError(f"LMCache reply operation mismatch for {operation!r}")
        if not b_response:
            raise RuntimeError(f"LMCache reply carries no response for {operation!r}")
        return _decode_payload(b_response[0], response_type)

    def register_kv_cache_engine_driven_context(
        self, payload: RegisterEngineDrivenContextPayload
    ) -> RegisterEngineDrivenContextResponse:
        """Register the client's engine-driven transfer context."""
        return self._send_request(
            "register_kv_cache_engine_driven_context",
            [payload],
            [RegisterEngineDrivenContextPayload],
            RegisterEngineDrivenContextResponse,
        )

    def unregister_kv_cache_engine_driven_context(self, instance_id: int) -> None:
        """Unregister the client's engine-driven transfer context."""
        self._send_request(
            "unregister_kv_cache_engine_driven_context",
            [instance_id],
            [int],
            type(None),
        )

    def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        """Submit a LOOKUP request for the given key."""
        self._send_request(
            "lookup", [key, tp_size], [IPCCacheServerKey, int], type(None)
        )

    def query_prefetch_status(self, request_id: str) -> int | None:
        """Return the prefetch token count for a LOOKUP, if finished."""
        return self._send_request(
            "query_prefetch_status", [request_id], [str], int | None
        )

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        skipped_chunks: tuple[tuple[int, ...], ...] = (),
    ) -> PrepareStoreResponse:
        """Prepare an engine-driven store for the given key."""
        return self._send_request(
            "prepare_store",
            [key, instance_id, skipped_chunks],
            [IPCCacheServerKey, int, tuple[tuple[int, ...], ...]],
            PrepareStoreResponse,
        )

    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, data: bytes
    ) -> bool:
        """Commit a serialized tensor envelope as a store."""
        return self._send_request(
            "commit_store",
            [key, instance_id, data],
            [IPCCacheServerKey, int, bytes],
            bool,
        )

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> PrepareRetrieveResponse:
        """Prepare an engine-driven retrieve for the given key."""
        return self._send_request(
            "prepare_retrieve",
            [key, instance_id],
            [IPCCacheServerKey, int],
            PrepareRetrieveResponse,
        )

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit an engine-driven retrieve (no-op for the inline path)."""
        return self._send_request(
            "commit_retrieve",
            [key, instance_id],
            [IPCCacheServerKey, int],
            bool,
        )

    def end_session(self, request_id: str) -> None:
        """End a session and clean up server-side session state."""
        self._send_request("end_session", [request_id], [str], type(None))

    def get_chunk_size(self) -> int:
        """Return the server's chunk size in tokens."""
        return self._send_request("get_chunk_size", [], [], int)

    def ping(self, instance_id: int | None) -> bool:
        """Ping the server; returns whether it is responsive."""
        return self._send_request(
            "ping", [instance_id], [int | None], bool
        )

    def close(self) -> None:
        """Close the REQ socket and its ZMQ context."""
        self._socket.close(linger=0)
        self._ctx.term()
