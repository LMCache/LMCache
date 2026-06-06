# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for LMCServerConnector reconnect-on-EPIPE behavior.

Regression for https://github.com/LMCache/LMCache/issues/3565: the connector
opened its TCP socket once in ``__init__`` and never reconnected, so after the
remote lmcache-server restarted every PUT failed silently with
``[Errno 32] Broken pipe``. These tests drive the write/read paths against a
fake socket that fails once (simulating the dead FD) and assert the connector
transparently reconnects and retries.
"""

# Standard
from unittest.mock import MagicMock, patch
import asyncio
import errno

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector import lm_connector
from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector


class _FakeSocket:
    """A socket whose send path fails with a chosen errno for the first
    ``fail_times`` calls, then succeeds; tracks whether it was closed.

    ``connect_fail`` makes ``connect()`` raise once (to simulate the server
    pod not yet being back up during a reconnect). ``recv_data`` lets a test
    drive the read path (``b""`` simulates a peer that closed the connection).
    """

    def __init__(
        self,
        fail_times: int = 0,
        fail_errno: int = errno.EPIPE,
        connect_fail: bool = False,
        recv_data: bytes = b"",
    ):
        self._fail_times = fail_times
        self._fail_errno = fail_errno
        self._connect_fail = connect_fail
        self._recv_data = recv_data
        self.closed = False
        self.sent: list[bytes] = []

    def connect(self, addr: tuple) -> None:
        if self._connect_fail:
            self._connect_fail = False
            raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

    def sendall(self, data: bytes) -> None:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise OSError(self._fail_errno, "Broken pipe")
        self.sent.append(bytes(data))

    def recv(self, n: int) -> bytes:
        chunk, self._recv_data = self._recv_data[:n], self._recv_data[n:]
        return chunk

    def close(self) -> None:
        self.closed = True


def _make_connector(sockets: list) -> tuple:
    """Build an LMCServerConnector whose socket factory yields ``sockets`` in
    order (first for __init__, the rest for reconnects)."""
    it = iter(sockets)

    def fake_socket(*args, **kwargs):
        return next(it)

    backend = MagicMock()
    backend.config = MagicMock()
    backend.metadata = MagicMock()

    loop = asyncio.new_event_loop()

    # ``loop.sock_sendall`` is what put() uses; delegate to the fake socket's
    # sendall so a dead FD raises the same OSError the kernel would.
    async def sock_sendall(sock, data):
        sock.sendall(data)

    with patch.object(lm_connector.socket, "socket", side_effect=fake_socket):
        with patch.object(lm_connector.RemoteConnector, "__init__", return_value=None):
            conn = LMCServerConnector.__new__(LMCServerConnector)
            conn.host = "127.0.0.1"
            conn.port = 12345
            conn._closed = False
            conn.client_socket = conn._open_socket()
            conn.loop = loop
            conn.local_cpu_backend = backend
            conn.async_socket_lock = asyncio.Lock()
    loop.sock_sendall = sock_sendall  # type: ignore[method-assign]
    return conn, loop


async def _instant_sleep(_seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Make the retry backoff instant so tests stay fast."""
    with patch.object(lm_connector.asyncio, "sleep", side_effect=_instant_sleep):
        yield


def test_put_reconnects_after_broken_pipe():
    """A PUT that hits EPIPE on a stale socket must reconnect and succeed."""
    dead = _FakeSocket(fail_times=1)  # first sendall raises EPIPE
    fresh = _FakeSocket(fail_times=0)
    conn, loop = _make_connector([dead, fresh])

    obj = MagicMock()
    obj.byte_array = b"payload"
    obj.get_shape.return_value = (0, 0, 0, 0)
    obj.get_dtype.return_value = "float16"
    obj.get_memory_format.return_value = 1

    key = MagicMock()
    key.to_string.return_value = "k"
    with patch(
        "lmcache.v1.storage_backend.connector.lm_connector.ClientMetaMessage"
    ) as mm:
        mm.return_value.serialize.return_value = b"meta"
        loop.run_until_complete(conn.put(key, obj))

    assert dead.closed, "stale socket should be closed on reconnect"
    assert fresh.sent, "fresh socket should have received the retried PUT"
    loop.close()


def test_put_gives_up_after_max_attempts():
    """If every reconnect also fails, the error eventually propagates rather
    than looping forever."""
    socks = [_FakeSocket(fail_times=1) for _ in range(5)]
    conn, loop = _make_connector(socks)

    obj = MagicMock()
    obj.byte_array = b"payload"
    obj.get_shape.return_value = (0, 0, 0, 0)
    obj.get_dtype.return_value = "float16"
    obj.get_memory_format.return_value = 1
    key = MagicMock()

    with patch(
        "lmcache.v1.storage_backend.connector.lm_connector.ClientMetaMessage"
    ) as mm:
        mm.return_value.serialize.return_value = b"meta"
        with pytest.raises(OSError):
            loop.run_until_complete(conn.put(key, obj))
    loop.close()


def test_non_retryable_error_propagates_without_reconnect():
    """A non-connection OSError (e.g. EACCES) must not trigger a reconnect."""
    sock = _FakeSocket(fail_times=1, fail_errno=errno.EACCES)
    conn, loop = _make_connector([sock])  # only one socket: no reconnect allowed

    obj = MagicMock()
    obj.byte_array = b"payload"
    obj.get_shape.return_value = (0, 0, 0, 0)
    obj.get_dtype.return_value = "float16"
    obj.get_memory_format.return_value = 1
    key = MagicMock()

    with patch(
        "lmcache.v1.storage_backend.connector.lm_connector.ClientMetaMessage"
    ) as mm:
        mm.return_value.serialize.return_value = b"meta"
        with pytest.raises(OSError) as exc_info:
            loop.run_until_complete(conn.put(key, obj))
    assert exc_info.value.errno == errno.EACCES
    assert not sock.closed, "non-retryable error must not reconnect"
    loop.close()


def test_close_prevents_reconnect():
    """After close(), a reconnect attempt is a no-op (don't resurrect)."""
    sock = _FakeSocket()
    conn, loop = _make_connector([sock])
    loop.run_until_complete(conn.close())
    assert conn._closed
    conn._reconnect()  # must not open a new socket
    assert conn.client_socket is sock
    loop.close()


def test_put_retries_when_reconnect_initially_refused():
    """The K8s window: the stale FD breaks, then the first reconnect's
    connect() is refused because the server pod is still coming up, and a
    later attempt finally succeeds. The connect() failure must be retried
    within the budget, not abort the whole RPC."""
    dead = _FakeSocket(fail_times=1)  # send raises EPIPE
    refused = _FakeSocket(connect_fail=True)  # reconnect connect() refused once
    fresh = _FakeSocket()  # final reconnect succeeds
    conn, loop = _make_connector([dead, refused, fresh])

    obj = MagicMock()
    obj.byte_array = b"payload"
    obj.get_shape.return_value = (0, 0, 0, 0)
    obj.get_dtype.return_value = "float16"
    obj.get_memory_format.return_value = 1
    key = MagicMock()

    with patch(
        "lmcache.v1.storage_backend.connector.lm_connector.ClientMetaMessage"
    ) as mm:
        mm.return_value.serialize.return_value = b"meta"
        loop.run_until_complete(conn.put(key, obj))

    assert fresh.sent, "PUT should succeed after a refused reconnect then a good one"
    loop.close()


def test_recv_exact_raises_on_peer_close():
    """`_recv_exact` must raise a (retryable) ConnectionResetError when the
    peer closes the connection, so a restarted server doesn't feed a truncated
    header into deserialize()."""
    sock = _FakeSocket(recv_data=b"")  # peer closed: recv returns b""
    conn, loop = _make_connector([sock])
    with pytest.raises(ConnectionResetError):
        conn._recv_exact(8)
    # And the error is classified retryable, so the RPC layer would reconnect.
    assert conn._is_retryable(ConnectionResetError())
    loop.close()
