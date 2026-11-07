# SPDX-License-Identifier: Apache-2.0
"""Out-of-band delivery of POSIX file descriptors to the LMCache server.

Some IPC handles are process-local fds (CUDA VMM memory exported as
``POSIX_FILE_DESCRIPTOR``) and cannot ride the serialized request: the
kernel must duplicate them into the receiver with ``SCM_RIGHTS``. A worker
connects to the server's channel, sends each fd keyed by the wrapper's
export id, and keeps the connection open while the server registers the
wrappers; the server's IPC wrappers claim the fds through the
:func:`~lmcache.v1.platform.base.ipc_wrapper.set_fd_resolver` hook. Fds a
connection delivered but the server never claimed are closed when that
connection closes, so a failed or no-op registration leaks nothing.

The channel is an ``AF_UNIX`` ``SOCK_SEQPACKET`` socket at
:func:`fd_channel_path`. ``/dev/shm`` is shared by every deployment that
shares device memory between the processes (``--ipc host``), so worker
containers reach the server's socket there. Each message carries one
16-byte export id plus one fd; the server acknowledges it once stored.
Design: ``docs/design/v1/platform/devices/cuda/vmm_cuda_ipc.md``.
"""

# Future
from __future__ import annotations

# Standard
from types import TracebackType
from urllib.parse import urlsplit
import errno
import os
import socket
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_FD_CHANNEL_DIR = "/dev/shm"
_EXPORT_ID_BYTES = 16
_ACK = b"\x01"


def _remove_stale_socket(path: str) -> None:
    """Unlink ``path`` if it is a socket nobody listens on.

    A server killed without cleanup leaves its socket file behind, which
    makes ``bind`` fail; a live server's socket is left alone.

    Raises:
        OSError: ``EADDRINUSE`` if another process serves ``path``.
    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        probe.connect(path)
    except FileNotFoundError:
        return
    except ConnectionRefusedError:
        os.unlink(path)
        return
    finally:
        probe.close()
    raise OSError(errno.EADDRINUSE, f"another process serves the fd channel {path}")


class FdChannelServer:
    """Server side of the fd channel: receives fds and hands them out once.

    Binds the socket on construction and accepts connections on a daemon
    thread, one daemon thread per connection. Thread-safe.
    """

    def __init__(self, path: str) -> None:
        """Bind ``path`` and start accepting worker connections.

        Args:
            path: Socket path, normally :func:`fd_channel_path`.

        Raises:
            OSError: If the socket cannot be bound, e.g. another live server
                serves ``path``.
        """
        self._path = path
        # Guards _fds, _connections and _closed.
        self._lock = threading.Lock()
        # Delivered and not yet claimed: export id -> (fd, owning connection).
        self._fds: dict[bytes, tuple[int, socket.socket]] = {}
        self._connections: set[socket.socket] = set()
        self._closed = False

        _remove_stale_socket(path)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self._listener.bind(path)
            # Workers may run as other users (e.g. in other MP containers).
            # Connecting only lets a peer deliver fds, which the server hands
            # out solely by the export ids its registrations carry.
            os.chmod(path, 0o666)
            self._listener.listen()
        except OSError:
            self._listener.close()
            raise
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="lmcache-fd-channel", daemon=True
        )
        self._accept_thread.start()
        logger.info("LMCache fd channel listening on %s", path)

    @property
    def path(self) -> str:
        """The socket path this server listens on."""
        return self._path

    def take(self, export_id: bytes) -> int:
        """Claim the fd delivered for ``export_id``.

        Usable as the resolver of
        :func:`~lmcache.v1.platform.base.ipc_wrapper.set_fd_resolver`.

        Args:
            export_id: The export id the worker delivered the fd under.

        Returns:
            The fd; the caller owns it and must close it.

        Raises:
            RuntimeError: If no fd is pending for ``export_id``: never
                delivered, already claimed, or dropped because its
                connection closed.
        """
        with self._lock:
            entry = self._fds.pop(export_id, None)
        if entry is None:
            raise RuntimeError(
                f"No fd was delivered for export id {export_id.hex()} on the "
                f"fd channel {self._path}: the worker must deliver it, and keep "
                "its connection open, until the server has registered the "
                "wrapper."
            )
        return entry[0]

    def close(self) -> None:
        """Stop accepting, drop every connection and close unclaimed fds.

        Idempotent.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = list(self._connections)
            self._connections.clear()
            pending = [fd for fd, _ in self._fds.values()]
            self._fds.clear()
        # shutdown() wakes the threads blocked in accept()/recvmsg().
        for sock in (self._listener, *connections):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                logger.debug("fd channel socket already shut down", exc_info=True)
        self._accept_thread.join()
        self._listener.close()
        for fd in pending:
            os.close(fd)
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                with self._lock:
                    if self._closed:
                        return
                # Registrations needing an fd then fail in take() with a
                # pointed error instead of this thread spinning on accept.
                logger.exception("fd channel on %s stopped accepting", self._path)
                return
            with self._lock:
                if self._closed:
                    conn.close()
                    return
                self._connections.add(conn)
            threading.Thread(
                target=self._serve_connection,
                args=(conn,),
                name="lmcache-fd-channel-conn",
                daemon=True,
            ).start()

    def _serve_connection(self, conn: socket.socket) -> None:
        """Store each delivered fd until the connection closes, then close
        the ones never claimed."""
        try:
            while self._receive_one(conn):
                pass
        except OSError:
            logger.warning("fd channel connection failed", exc_info=True)
        finally:
            with self._lock:
                self._connections.discard(conn)
                dropped = [
                    export_id
                    for export_id, (_, owner) in self._fds.items()
                    if owner is conn
                ]
                fds = [self._fds.pop(export_id)[0] for export_id in dropped]
            for fd in fds:
                os.close(fd)
            if fds:
                logger.debug("Closed %d unclaimed fds of a closed connection", len(fds))
            conn.close()

    def _receive_one(self, conn: socket.socket) -> bool:
        """Receive and acknowledge one fd; False at end of stream or on a
        malformed message (which drops the connection)."""
        export_id, fds, flags, _ = socket.recv_fds(conn, _EXPORT_ID_BYTES + 1, 1)
        if not export_id and not fds:
            return False
        if (
            len(export_id) != _EXPORT_ID_BYTES
            or len(fds) != 1
            or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC)
        ):
            for fd in fds:
                os.close(fd)
            logger.warning(
                "Dropping fd channel connection after a malformed message "
                "(%d payload bytes, %d fds, flags 0x%x)",
                len(export_id),
                len(fds),
                flags,
            )
            return False
        with self._lock:
            replaced = self._fds.get(export_id)
            self._fds[export_id] = (fds[0], conn)
        if replaced is not None:
            os.close(replaced[0])
        conn.sendall(_ACK)
        return True


class FdChannelClient:
    """Worker side of the fd channel: one connection to the server.

    Fds sent over it stay claimable by the server until :meth:`close`.
    Not thread-safe; use one client per registration.
    """

    def __init__(self, path: str, timeout: float) -> None:
        """Connect to the server's fd channel.

        Args:
            path: The server's socket path (:func:`fd_channel_path`).
            timeout: Seconds to wait for the connection and each
                acknowledgement.

        Raises:
            ConnectionError: If the server's channel is unreachable at
                ``path``.
        """
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._sock.settimeout(timeout)
        try:
            self._sock.connect(path)
        except OSError as exc:
            self._sock.close()
            raise ConnectionError(
                f"Cannot reach the LMCache server's fd channel at {path}: "
                f"{exc}. The server must run on this host, and both processes "
                "must share /dev/shm (--ipc host)."
            ) from exc

    def send(self, export_id: bytes, fd: int) -> None:
        """Deliver ``fd`` under ``export_id`` and wait until it is stored.

        The server receives its own duplicate; the caller keeps ``fd``.

        Args:
            export_id: The 16-byte export id the server will claim it by.
            fd: An open file descriptor.

        Raises:
            ValueError: If ``export_id`` is not 16 bytes.
            ConnectionError: If the server does not acknowledge it.
        """
        if len(export_id) != _EXPORT_ID_BYTES:
            raise ValueError(f"export id must be {_EXPORT_ID_BYTES} bytes")
        try:
            socket.send_fds(self._sock, [export_id], [fd])
            ack = self._sock.recv(len(_ACK))
        except OSError as exc:
            raise ConnectionError(f"fd channel delivery failed: {exc}") from exc
        if ack != _ACK:
            raise ConnectionError("fd channel closed before acknowledging an fd")

    def close(self) -> None:
        """Close the connection; the server drops fds it has not claimed."""
        self._sock.close()

    def __enter__(self) -> FdChannelClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def fd_channel_path(port: int) -> str:
    """Return the fd channel socket path of the LMCache server on ``port``.

    Args:
        port: The server's request port (``MPServerConfig.port``).

    Returns:
        The ``AF_UNIX`` socket path, ``/dev/shm/lmcache_fd_<port>.sock``.
    """
    return os.path.join(_FD_CHANNEL_DIR, f"lmcache_fd_{port}.sock")


def fd_channel_path_for_url(server_url: str) -> str:
    """Return the fd channel socket path of the server at ``server_url``.

    Args:
        server_url: The worker's server URL (``tcp://host:port``,
            ``grpc://host:port`` or a bare ``host:port``).

    Returns:
        :func:`fd_channel_path` of the URL's port, or ``""`` when the URL
        names no port (``ipc://``, ``inproc://``, ``grpc+unix://``), for
        which no channel path is defined.
    """
    url = server_url.strip()
    if "://" not in url:
        url = f"tcp://{url}"
    port = urlsplit(url).port
    return "" if port is None else fd_channel_path(port)
