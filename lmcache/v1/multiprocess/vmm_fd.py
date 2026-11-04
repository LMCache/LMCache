# SPDX-License-Identifier: Apache-2.0
"""Same-node channel delivering POSIX fds from engine workers to the MP server.

A VMM KV-cache allocation exported as a POSIX fd cannot ride the
``REGISTER_KV_CACHE`` descriptor: an fd is an index into the exporting
process's own file table, so the number is meaningless to the server. Only
``SCM_RIGHTS`` over an ``AF_UNIX`` socket makes the kernel install the same
file in the server's table. The MP server listens on one socket per request port
(:func:`fd_channel_path`). On the worker, ``VmmCudaIPCWrapper.wrap`` hands
each exported fd to the installed sender (:func:`send_fd`) before the
REGISTER descriptor is sent; the server's register handler then resolves each
fd by export id (:meth:`VmmFdResolver.resolve`).

The export id is a random UUID4 that the worker sends only to the server,
so a peer cannot attach an fd to an allocation it did not export.
Both ends additionally require the other side to run under the same
effective uid (``SO_PEERCRED``), and the socket is owner-only.

Linux only (``SO_PEERCRED``, ``AF_UNIX`` ``SOCK_SEQPACKET``).
"""

# Standard
from urllib.parse import urlparse
import errno
import os
import socket
import stat
import struct
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.periodic_thread import (
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

_EXPORT_ID_BYTES = 16

_ACCEPT_POLL_S = 1.0
# Matches the request transports' connect timeout (zmq/grpc CONNECT_TIMEOUT_MS).
_PEER_IO_TIMEOUT_S = 5.0


def fd_channel_path(port: int) -> str:
    """Return the fd channel socket path of the MP server on ``port``.

    Args:
        port: The MP server's request port (``--port``).

    Returns:
        The ``AF_UNIX`` socket path, under ``/dev/shm`` so that any
        deployment already sharing ``/dev/shm`` for event IPC can reach it.
    """
    return f"/dev/shm/lmcache_fd_{port}.sock"


def fd_channel_path_for_server_url(server_url: str) -> str | None:
    """Return the fd channel path of the MP server behind ``server_url``.

    Args:
        server_url: The worker's MP server URL, e.g. ``tcp://host:5555``,
            ``grpc://host:5555``, or bare ``host:5555``.

    Returns:
        :func:`fd_channel_path` for the URL's port, or ``None`` when the URL
        carries no port (``ipc://``, ``inproc://``, ``grpc+unix://``).

    Raises:
        ValueError: If the URL's port is not a valid port number.
    """
    url = server_url.strip()
    if "://" not in url:
        url = f"tcp://{url}"
    port = urlparse(url).port
    return None if port is None else fd_channel_path(port)


def _peer_uid(sock: socket.socket) -> int:
    """Return the effective uid of the process on the other end of ``sock``."""
    # struct ucred: pid_t pid; uid_t uid; gid_t gid
    creds = sock.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII")
    )
    _pid, uid, _gid = struct.unpack("iII", creds)
    return uid


def _is_stale_socket(path: str) -> bool:
    """Whether ``path`` is a socket nobody listens on."""
    if not stat.S_ISSOCK(os.lstat(path).st_mode):
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as probe:
        try:
            probe.connect(path)
        except ConnectionRefusedError:
            return True
    return False


def _bind_owner_only(listener: socket.socket, path: str) -> None:
    """Bind ``listener`` to ``path`` (mode 0600), reclaiming a stale socket."""
    try:
        listener.bind(path)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE or not _is_stale_socket(path):
            raise
        # Left behind by a server that died without close().
        os.unlink(path)
        listener.bind(path)
    # Before listen(): no peer can connect until the mode is final.
    os.chmod(path, 0o600)


def send_fd(path: str, export_id: bytes, fd: int) -> None:
    """Send one exported fd to the MP server listening on ``path``.

    Installed on the worker as the VMM fd sender
    (``functools.partial(send_fd, path)``, see
    ``set_vmm_fd_sender``). Does not take ownership of ``fd``.

    Args:
        path: The MP server's fd channel socket path.
        export_id: The exporting wrapper's export id.
        fd: The exported fd.

    Raises:
        RuntimeError: If the server's channel is unreachable or owner-only
            for another uid, or the listener is not owned by this uid.
        OSError: If sending the fd fails (e.g. times out).
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(_PEER_IO_TIMEOUT_S)
        try:
            sock.connect(path)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise RuntimeError(
                f"Cannot reach the MP server's VMM fd channel at {path}. "
                "POSIX-fd VMM registration needs an MP server on this node "
                "started with --enable-vmm-fd-channel whose channel came up "
                "(see its log), with a shared /dev/shm; use a "
                "fabric-exportable pool (IMEX channel) otherwise."
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"Permission denied on the MP server's VMM fd channel at "
                f"{path}: the socket is owner-only, so the worker and the "
                "server must run under the same uid."
            ) from exc
        # The path lives in world-writable /dev/shm: refuse to hand device
        # memory to a listener some other user planted there.
        if _peer_uid(sock) != os.geteuid():
            raise RuntimeError(
                f"The VMM fd channel at {path} is not owned by this "
                "process's uid; refusing to send KV-cache fds."
            )
        socket.send_fds(sock, [export_id], [fd])


class VmmFdResolver:
    """MP-server end of the fd channel: receives fds and hands them out.

    The constructor binds the socket and starts the accept thread and the
    periodic thread that closes fds left unresolved past their TTL. Received
    fds are kept under their export id until :meth:`resolve` takes them.

    Args:
        path: Socket path to listen on.
        resolve_timeout: Seconds :meth:`resolve` waits for an fd to arrive.
        unresolved_ttl: Seconds an fd may stay unresolved before it is closed.
        expiry_interval: Seconds between scans for expired fds.

    Raises:
        OSError: If the socket cannot be bound (e.g. a live server already
            owns ``path``, or a non-socket file is in the way).
    """

    # The worker sends every fd before REGISTER, so the fd is already queued
    # on the server socket; this only covers the accept thread not having
    # drained it yet.
    DEFAULT_RESOLVE_TIMEOUT = 5.0
    # Unresolved fds pin the exporter's device memory. They appear when a
    # re-registration is a no-op (instance already registered) or REGISTER
    # fails. At least the default worker mq_timeout (300s), since REGISTER
    # may queue that long behind its fds.
    DEFAULT_UNRESOLVED_TTL = 300.0
    DEFAULT_EXPIRY_INTERVAL = 30.0

    def __init__(
        self,
        path: str,
        resolve_timeout: float = DEFAULT_RESOLVE_TIMEOUT,
        unresolved_ttl: float = DEFAULT_UNRESOLVED_TTL,
        expiry_interval: float = DEFAULT_EXPIRY_INTERVAL,
    ) -> None:
        self._path = path
        self._resolve_timeout = resolve_timeout
        self._unresolved_ttl = unresolved_ttl
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            _bind_owner_only(listener, path)
            listener.listen()
        except BaseException:
            listener.close()
            raise
        listener.settimeout(_ACCEPT_POLL_S)
        self._listener = listener

        # export_id -> (fd, monotonic arrival time) of fds received but not yet
        # resolved or expired; guarded by _cond. Every fd here is owned by the
        # resolver and closed on resolve hand-off, expiry, or close().
        self._pending: dict[bytes, tuple[int, float]] = {}
        self._cond = threading.Condition()
        # Set once by close(); stops both threads and fails pending resolves.
        self._stop = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._serve, name="lmcache-vmm-fd-channel", daemon=True
        )
        self._accept_thread.start()
        self._expiry_thread = create_periodic_thread(
            name=f"lmcache-vmm-fd-expiry-{id(self):x}",
            interval=expiry_interval,
            execute_fn=self._expiry_cycle,
            level=ThreadLevel.MEDIUM,
        )
        self._expiry_thread.start()
        logger.info("VMM fd channel listening on %s", path)

    @property
    def path(self) -> str:
        """The socket path this resolver listens on."""
        return self._path

    def resolve(self, export_id: bytes) -> int:
        """Take the fd delivered for ``export_id``, waiting briefly for it.

        Matches the ``set_vmm_fd_resolver`` contract: ownership of the
        returned fd passes to the caller.

        Args:
            export_id: The wrapper's export id.

        Returns:
            The received fd.

        Raises:
            RuntimeError: If no fd arrives for ``export_id`` in time, or the
                resolver is closed.
        """
        deadline = time.monotonic() + self._resolve_timeout
        with self._cond:
            while export_id not in self._pending:
                remaining = deadline - time.monotonic()
                if self._stop.is_set() or remaining <= 0:
                    raise RuntimeError(
                        f"No fd arrived on the VMM fd channel {self._path} for "
                        f"export id {export_id.hex()} within "
                        f"{self._resolve_timeout}s; the worker must deliver it "
                        "before REGISTER."
                    )
                self._cond.wait(remaining)
            fd, _arrived = self._pending.pop(export_id)
        return fd

    def close(self) -> None:
        """Stop both threads, remove the socket, and close unresolved fds.

        Idempotent.
        """
        if self._stop.is_set():
            return
        self._stop.set()
        self._accept_thread.join()
        PeriodicThreadRegistry.get_instance().unregister(self._expiry_thread.name)
        self._expiry_thread.stop()
        self._listener.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        with self._cond:
            pending, self._pending = self._pending, {}
            # Wake resolve() waiters so they see the stop and fail fast.
            self._cond.notify_all()
        for fd, _arrived in pending.values():
            os.close(fd)

    def _serve(self) -> None:
        """Accept loop: handle one worker connection at a time."""
        while not self._stop.is_set():
            try:
                conn, _addr = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                logger.exception("VMM fd channel accept failed")
                # Persistent errors (e.g. EMFILE) would otherwise spin.
                self._stop.wait(_ACCEPT_POLL_S)
                continue
            try:
                with conn:
                    self._receive_from(conn)
            except Exception:
                logger.exception("VMM fd channel dropped a worker connection")

    def _receive_from(self, conn: socket.socket) -> None:
        """Store every ``(export_id, fd)`` message a worker sends."""
        conn.settimeout(_PEER_IO_TIMEOUT_S)
        uid = _peer_uid(conn)
        if uid != os.geteuid():
            logger.warning(
                "VMM fd channel rejected a peer with uid %d (expected %d)",
                uid,
                os.geteuid(),
            )
            return
        while not self._stop.is_set():
            data, fds, flags, _addr = socket.recv_fds(conn, _EXPORT_ID_BYTES, 1)
            if not data and not fds:
                return  # worker closed the connection
            truncated = flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
            if truncated or len(data) != _EXPORT_ID_BYTES or len(fds) != 1:
                for fd in fds:
                    os.close(fd)
                logger.warning(
                    "VMM fd channel dropped a malformed message "
                    "(%d bytes, %d fds, flags=%#x)",
                    len(data),
                    len(fds),
                    flags,
                )
                return
            self._store(data, fds[0])

    def _store(self, export_id: bytes, fd: int) -> None:
        """Keep ``fd`` for ``export_id``; a duplicate id's fd is closed."""
        with self._cond:
            duplicate = export_id in self._pending
            if not duplicate:
                self._pending[export_id] = (fd, time.monotonic())
                self._cond.notify_all()
        if duplicate:
            os.close(fd)
            logger.warning(
                "VMM fd channel ignored a duplicate fd for export id %s",
                export_id.hex(),
            )

    def _expiry_cycle(self) -> ThreadRunSummary:
        """Close fds unresolved for longer than the TTL."""
        cutoff = time.monotonic() - self._unresolved_ttl
        with self._cond:
            expired = [
                eid for eid, (_fd, arrived) in self._pending.items() if arrived < cutoff
            ]
            fds = [self._pending.pop(eid)[0] for eid in expired]
        for fd in fds:
            os.close(fd)
        if fds:
            logger.debug("VMM fd channel closed %d unresolved fd(s)", len(fds))
        return ThreadRunSummary(success=True, message=f"expired={len(fds)}")
