# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the out-of-band fd channel to the MP server."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import os
import select
import socket
import stat
import sys
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.platform.fd_channel import (
    FdChannelClient,
    FdChannelServer,
    fd_channel_path,
    fd_channel_path_for_url,
)

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="AF_UNIX SOCK_SEQPACKET"
    ),
]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[FdChannelServer]:
    channel = FdChannelServer(str(tmp_path / "fd.sock"))
    yield channel
    channel.close()


@pytest.fixture
def pipe() -> Iterator[tuple[int, int]]:
    """A pipe whose read end sees EOF once every copy of the write end,
    including those delivered to the server, is closed. Tests own (and
    close) the write end."""
    read_fd, write_fd = os.pipe()
    yield read_fd, write_fd
    os.close(read_fd)


def _all_copies_closed(read_fd: int, timeout: float) -> bool:
    ready, _, _ = select.select([read_fd], [], [], timeout)
    return bool(ready) and os.read(read_fd, 1) == b""


def _deliver_and_drop_local_copy(client: FdChannelClient, write_fd: int) -> bytes:
    export_id = uuid.uuid4().bytes
    client.send(export_id, write_fd)
    os.close(write_fd)
    return export_id


def test_delivered_fd_is_claimed_once(
    server: FdChannelServer, pipe: tuple[int, int]
) -> None:
    """take() hands out the server's own copy of the delivered fd, once."""
    read_fd, write_fd = pipe
    with FdChannelClient(server.path, timeout=10) as client:
        export_id = _deliver_and_drop_local_copy(client, write_fd)
        received = server.take(export_id)
        os.write(received, b"x")
        assert os.read(read_fd, 1) == b"x"
        with pytest.raises(RuntimeError, match="No fd was delivered"):
            server.take(export_id)
        os.close(received)
    assert _all_copies_closed(read_fd, timeout=10)


def test_unclaimed_fds_are_closed_with_their_connection(
    server: FdChannelServer, pipe: tuple[int, int]
) -> None:
    """A registration that never claims its fds leaves none open once the
    worker closes the connection."""
    read_fd, write_fd = pipe
    client = FdChannelClient(server.path, timeout=10)
    export_id = _deliver_and_drop_local_copy(client, write_fd)
    assert not _all_copies_closed(read_fd, timeout=0.1)
    client.close()
    assert _all_copies_closed(read_fd, timeout=10)
    with pytest.raises(RuntimeError, match="No fd was delivered"):
        server.take(export_id)


def test_connections_scope_only_their_own_fds(server: FdChannelServer) -> None:
    """Closing one worker's connection keeps the fds another one delivered."""
    kept_read, kept_write = os.pipe()
    dropped_read, dropped_write = os.pipe()
    try:
        with FdChannelClient(server.path, timeout=10) as staying:
            kept = _deliver_and_drop_local_copy(staying, kept_write)
            with FdChannelClient(server.path, timeout=10) as leaving:
                _deliver_and_drop_local_copy(leaving, dropped_write)
            assert _all_copies_closed(dropped_read, timeout=10)
            assert not _all_copies_closed(kept_read, timeout=0.1)
            os.close(server.take(kept))
    finally:
        os.close(kept_read)
        os.close(dropped_read)


def test_close_drops_pending_fds_and_unlinks_the_socket(
    tmp_path: Path, pipe: tuple[int, int]
) -> None:
    read_fd, write_fd = pipe
    path = str(tmp_path / "fd.sock")
    channel = FdChannelServer(path)
    with FdChannelClient(path, timeout=10) as client:
        _deliver_and_drop_local_copy(client, write_fd)
        channel.close()
        channel.close()
        assert _all_copies_closed(read_fd, timeout=10)
    assert not os.path.exists(path)
    with pytest.raises(ConnectionError, match="fd channel"):
        FdChannelClient(path, timeout=1)


def test_stale_socket_file_is_replaced(tmp_path: Path) -> None:
    """A socket file left behind by a killed server does not block a restart."""
    path = str(tmp_path / "fd.sock")
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    stale.bind(path)
    stale.close()
    FdChannelServer(path).close()


def test_workers_of_other_users_can_connect(server: FdChannelServer) -> None:
    """MP containers may run the engine and the server as different users;
    connecting to a Unix socket needs write permission on it."""
    mode = os.stat(server.path).st_mode
    assert mode & stat.S_IWOTH and mode & stat.S_IWGRP


def test_live_server_socket_is_not_taken_over(server: FdChannelServer) -> None:
    with pytest.raises(OSError, match="another process serves"):
        FdChannelServer(server.path)


def test_malformed_message_drops_the_connection(
    server: FdChannelServer, pipe: tuple[int, int]
) -> None:
    read_fd, write_fd = pipe
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.settimeout(10)
    sock.connect(server.path)
    socket.send_fds(sock, [b"short"], [write_fd])
    os.close(write_fd)
    assert sock.recv(1) == b""
    sock.close()
    assert _all_copies_closed(read_fd, timeout=10)


def test_client_rejects_bad_export_id(
    server: FdChannelServer, pipe: tuple[int, int]
) -> None:
    with FdChannelClient(server.path, timeout=10) as client:
        with pytest.raises(ValueError, match="16 bytes"):
            client.send(b"short", pipe[1])
    os.close(pipe[1])


@pytest.mark.parametrize(
    ("url", "port"),
    [
        ("tcp://localhost:5555", 5555),
        ("tcp://10.0.0.1:16555", 16555),
        ("localhost:6000", 6000),
        ("grpc://host:7000", 7000),
        ("ipc:///tmp/lmcache.sock", None),
        ("grpc+unix:///tmp/lmcache.sock", None),
    ],
)
def test_channel_path_follows_the_server_port(url: str, port: int | None) -> None:
    """Workers derive the path the server binds from the port alone."""
    expected = "" if port is None else fd_channel_path(port)
    assert fd_channel_path_for_url(url) == expected
    if port is not None:
        assert expected == f"/dev/shm/lmcache_fd_{port}.sock"
