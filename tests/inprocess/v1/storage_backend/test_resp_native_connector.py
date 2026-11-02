# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for bounded native RESP authentication replies."""

# Standard
import queue
import socket
import threading

# Third Party
import pytest
import torch  # noqa: F401

_MAX_RESP_LINE_LENGTH = 1024 * 1024


def _native_client_available() -> bool:
    """Return whether the native Redis connector can be imported."""
    try:
        # First Party
        from lmcache.lmcache_redis import LMCacheRedisClient  # noqa: F401

        return True
    except ImportError:
        return False


requires_native = pytest.mark.skipif(
    not _native_client_available(),
    reason="C++ Redis extension (lmcache_redis) not available",
)


def _serve_oversized_auth_reply(
    listener: socket.socket,
    peer_closed: queue.Queue[bool],
) -> None:
    """Send an unterminated oversized AUTH reply and observe peer shutdown."""
    try:
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(5)
            conn.recv(4096)
            try:
                conn.sendall(b"X" * (_MAX_RESP_LINE_LENGTH + 1))
                closed = conn.recv(1) == b""
            except (BrokenPipeError, ConnectionResetError):
                closed = True
            except socket.timeout:
                closed = False
            peer_closed.put(closed)
    finally:
        listener.close()


@requires_native
def test_oversized_auth_reply_closes_connection(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Reject an AUTH response that exceeds the bounded RESP line length."""
    # First Party
    from lmcache.lmcache_redis import LMCacheRedisClient

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    peer_closed: queue.Queue[bool] = queue.Queue()
    server = threading.Thread(
        target=_serve_oversized_auth_reply,
        args=(listener, peer_closed),
    )
    server.start()

    client = LMCacheRedisClient("127.0.0.1", port, 1, "", "password")
    try:
        assert peer_closed.get(timeout=10)
    finally:
        client.close()
        server.join(timeout=10)

    assert not server.is_alive()
    assert "RESP line exceeds maximum length" in capfd.readouterr().err
