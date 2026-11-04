# SPDX-License-Identifier: Apache-2.0
"""Tests for the worker -> MP server POSIX fd channel and its server module.

Pipes stand in for VMM-exported fds: a byte written into the pipe is
readable through the fd the server resolves only if the kernel duplicated
the same open file across the socket.
"""

# Standard
from typing import Literal
from unittest.mock import MagicMock
import os
import random
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.modules.vmm_fd_channel import VmmFdChannelModule
from lmcache.v1.multiprocess.vmm_fd import (
    VmmFdResolver,
    fd_channel_path,
    fd_channel_path_for_server_url,
    send_fd,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="fd channel needs Linux SO_PEERCRED and AF_UNIX SOCK_SEQPACKET",
)


def _deliver_pipe(path: str) -> tuple[bytes, int]:
    """Deliver a fresh pipe's read end under a random export id.

    Returns:
        ``(export_id, write_fd)``; the local read end is already closed, as
        ``VmmCudaIPCWrapper.wrap`` closes its fd after delivery.
    """
    export_id = os.urandom(16)
    read_fd, write_fd = os.pipe()
    try:
        send_fd(path, export_id, read_fd)
    finally:
        os.close(read_fd)
    return export_id, write_fd


def _assert_same_pipe(resolved_fd: int, write_fd: int) -> None:
    try:
        os.write(write_fd, b"x")
        assert os.read(resolved_fd, 1) == b"x"
    finally:
        os.close(resolved_fd)
        os.close(write_fd)


def _wait_reader_closed(write_fd: int, timeout: float = 5.0) -> None:
    """Block until every read end of the pipe behind ``write_fd`` is closed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.write(write_fd, b"x")
        except BrokenPipeError:
            return
        time.sleep(0.01)
    raise AssertionError("the pipe's read end is still open")


@pytest.fixture
def channel_path(tmp_path) -> str:
    return str(tmp_path / "fd.sock")


@pytest.fixture
def channel(channel_path):
    server = VmmFdResolver(channel_path)
    yield server
    server.close()


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------


def test_channel_path_derives_from_port() -> None:
    assert fd_channel_path(5555) == "/dev/shm/lmcache_fd_5555.sock"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("tcp://localhost:5555", "/dev/shm/lmcache_fd_5555.sock"),
        ("localhost:6000", "/dev/shm/lmcache_fd_6000.sock"),
        ("grpc://10.0.0.1:7000", "/dev/shm/lmcache_fd_7000.sock"),
        ("ipc:///tmp/lmcache.ipc", None),
        ("grpc+unix:///tmp/lmcache.sock", None),
    ],
)
def test_channel_path_for_server_url(url: str, expected: str | None) -> None:
    assert fd_channel_path_for_server_url(url) == expected


# ---------------------------------------------------------------------------
# Delivery and resolving
# ---------------------------------------------------------------------------


def test_delivered_fds_are_resolvable_by_export_id(channel) -> None:
    first_id, first_w = _deliver_pipe(channel.path)
    second_id, second_w = _deliver_pipe(channel.path)

    _assert_same_pipe(channel.resolve(second_id), second_w)
    _assert_same_pipe(channel.resolve(first_id), first_w)


def test_deliver_does_not_take_ownership(channel) -> None:
    read_fd, write_fd = os.pipe()
    try:
        send_fd(channel.path, os.urandom(16), read_fd)
        os.fstat(read_fd)  # still open in the caller
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_unreachable_channel_raises(channel_path) -> None:
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(RuntimeError, match="Cannot reach"):
            send_fd(channel_path, os.urandom(16), read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_resolve_waits_for_a_late_fd(channel) -> None:
    export_id = os.urandom(16)
    resolved: list[int] = []
    waiter = threading.Thread(
        target=lambda: resolved.append(channel.resolve(export_id))
    )
    waiter.start()

    read_fd, write_fd = os.pipe()
    try:
        send_fd(channel.path, export_id, read_fd)
    finally:
        os.close(read_fd)
    waiter.join(timeout=5)

    assert len(resolved) == 1
    _assert_same_pipe(resolved[0], write_fd)


def test_resolve_times_out_for_unknown_export_id(channel_path) -> None:
    server = VmmFdResolver(channel_path, resolve_timeout=0.1)
    try:
        with pytest.raises(RuntimeError, match="No fd arrived"):
            server.resolve(os.urandom(16))
    finally:
        server.close()


def test_fd_is_resolvable_only_once(channel_path) -> None:
    server = VmmFdResolver(channel_path, resolve_timeout=0.1)
    try:
        export_id, write_fd = _deliver_pipe(channel_path)

        _assert_same_pipe(server.resolve(export_id), write_fd)
        with pytest.raises(RuntimeError, match="No fd arrived"):
            server.resolve(export_id)
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Resolver lifecycle
# ---------------------------------------------------------------------------


def test_unresolved_fds_expire(channel_path) -> None:
    server = VmmFdResolver(
        channel_path, resolve_timeout=0.1, unresolved_ttl=0.0, expiry_interval=0.05
    )
    try:
        export_id, write_fd = _deliver_pipe(channel_path)
        try:
            # The sender closed its copy, so this observes the expiry.
            _wait_reader_closed(write_fd)
            with pytest.raises(RuntimeError, match="No fd arrived"):
                server.resolve(export_id)
        finally:
            os.close(write_fd)
    finally:
        server.close()


def test_socket_is_owner_only(channel) -> None:
    assert stat.S_IMODE(os.stat(channel.path).st_mode) == 0o600


def test_close_removes_socket_and_closes_unresolved_fds(channel_path) -> None:
    server = VmmFdResolver(channel_path)
    export_id, write_fd = _deliver_pipe(channel_path)
    try:
        server.close()
        server.close()  # idempotent

        assert not os.path.exists(channel_path)
        _wait_reader_closed(write_fd)
    finally:
        os.close(write_fd)


def test_stale_socket_is_reclaimed(channel_path) -> None:
    # A bound socket closed without unlink: what a crashed server leaves.
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as stale:
        stale.bind(channel_path)

    server = VmmFdResolver(channel_path)
    try:
        export_id, write_fd = _deliver_pipe(channel_path)
        _assert_same_pipe(server.resolve(export_id), write_fd)
    finally:
        server.close()


def test_live_channel_is_not_taken_over(channel) -> None:
    with pytest.raises(OSError):
        VmmFdResolver(channel.path)

    export_id, write_fd = _deliver_pipe(channel.path)
    _assert_same_pipe(channel.resolve(export_id), write_fd)


def test_non_socket_file_is_left_alone(channel_path) -> None:
    with open(channel_path, "w") as f:
        f.write("not a socket")

    with pytest.raises(OSError):
        VmmFdResolver(channel_path)
    with open(channel_path) as f:
        assert f.read() == "not a socket"


# ---------------------------------------------------------------------------
# Server module (built only with --enable-vmm-fd-channel)
# ---------------------------------------------------------------------------


requires_cuda_server = pytest.mark.skipif(
    torch_device_type != "cuda",
    reason="the module only opens the channel on a CUDA server",
)


@pytest.fixture
def server_port():
    """A request port whose fd channel path is free; removes the socket
    file afterwards if a test left it behind."""
    port = random.randint(40000, 60000)
    while os.path.lexists(fd_channel_path(port)):
        port = random.randint(40000, 60000)
    yield port
    path = fd_channel_path(port)
    if os.path.lexists(path):
        os.unlink(path)


def _create_module(
    port: int,
    enable_vmm_fd_channel: bool,
    supported_transfer_mode: Literal[
        "lmcache_driven", "engine_driven", "auto"
    ] = "lmcache_driven",
):
    """Run ``VmmFdChannelModule.create`` for a server on ``port``."""
    build_ctx = MagicMock(name="build_ctx")
    build_ctx.mp_config = MPServerConfig(
        port=port,
        enable_vmm_fd_channel=enable_vmm_fd_channel,
        supported_transfer_mode=supported_transfer_mode,
    )
    return VmmFdChannelModule.create(build_ctx)


def test_module_is_not_built_without_flag(server_port) -> None:
    assert _create_module(server_port, enable_vmm_fd_channel=False) is None
    assert not os.path.exists(fd_channel_path(server_port))


def test_module_is_not_built_for_engine_driven_only_server(server_port) -> None:
    module = _create_module(
        server_port,
        enable_vmm_fd_channel=True,
        supported_transfer_mode="engine_driven",
    )
    assert module is None
    assert not os.path.exists(fd_channel_path(server_port))


@pytest.mark.skipif(torch_device_type == "cuda", reason="needs a non-CUDA server")
def test_module_is_not_built_off_cuda(server_port) -> None:
    assert _create_module(server_port, enable_vmm_fd_channel=True) is None
    assert not os.path.exists(fd_channel_path(server_port))


@requires_cuda_server
def test_module_is_not_built_when_socket_is_taken(server_port) -> None:
    path = fd_channel_path(server_port)
    with open(path, "w") as f:
        f.write("not a socket")

    assert _create_module(server_port, enable_vmm_fd_channel=True) is None
    with open(path) as f:
        assert f.read() == "not a socket"


@requires_cuda_server
def test_module_serves_the_channel_until_close(server_port) -> None:
    path = fd_channel_path(server_port)
    module = _create_module(server_port, enable_vmm_fd_channel=True)
    assert module is not None
    try:
        assert stat.S_ISSOCK(os.stat(path).st_mode)
        assert module.report_status()["path"] == path
    finally:
        module.close()
    assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# Peers under a different uid (needs root to switch uids in a child)
# ---------------------------------------------------------------------------

_NOBODY = 65534

requires_root = pytest.mark.skipif(
    os.geteuid() != 0, reason="switching uids needs root"
)


def _in_child_as_nobody(body) -> str:
    """Fork, drop to uid/gid nobody, run ``body()`` and return what it
    reports: its return value, or the exception's type and message."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        try:
            os.setgid(_NOBODY)
            os.setuid(_NOBODY)
            result = str(body())
        except BaseException as exc:  # report everything to the parent
            result = f"{type(exc).__name__}: {exc}"
        os.write(write_fd, result.encode())
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as reader:
        result = reader.read().decode()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    return result


@pytest.fixture
def shared_dir():
    """A directory both root and nobody can create sockets in (pytest's
    tmp_path sits under a root-only parent)."""
    path = tempfile.mkdtemp(prefix="lmc-fd-", dir="/tmp")
    os.chmod(path, 0o777)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@requires_root
def test_worker_under_other_uid_cannot_connect(shared_dir) -> None:
    """The owner-only socket refuses a worker running as another uid."""
    path = os.path.join(shared_dir, "fd.sock")
    server = VmmFdResolver(path)
    try:

        def deliver() -> str:
            export_id, write_fd = _deliver_pipe(path)
            os.close(write_fd)
            return "delivered"

        result = _in_child_as_nobody(deliver)
    finally:
        server.close()
    assert result.startswith("RuntimeError: Permission denied"), result


@requires_root
def test_server_rejects_peer_under_other_uid(shared_dir) -> None:
    """A peer that gets past the socket mode is still rejected by
    SO_PEERCRED, and its fd never becomes resolvable."""
    path = os.path.join(shared_dir, "fd.sock")
    server = VmmFdResolver(path, resolve_timeout=0.5)
    export_id = os.urandom(16)
    try:
        os.chmod(path, 0o666)  # stand-in for a peer that can reach the socket

        def send_raw() -> str:
            read_fd, _write_fd = os.pipe()
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
                sock.connect(path)
                try:
                    socket.send_fds(sock, [export_id], [read_fd])
                except OSError:
                    pass  # the server may already have dropped us
            return "connected"

        assert _in_child_as_nobody(send_raw) == "connected"
        with pytest.raises(RuntimeError, match="No fd arrived"):
            server.resolve(export_id)
    finally:
        server.close()


@requires_root
def test_worker_refuses_listener_under_other_uid(shared_dir) -> None:
    """A listener planted by another uid does not receive KV-cache fds."""
    path = os.path.join(shared_dir, "fd.sock")
    ready_r, ready_w = os.pipe()
    stop_r, stop_w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: the planted listener
        # Plain socket calls only: a forked child must not start threads or
        # log, since the parent's threads may hold locks across the fork.
        os.close(ready_r)
        os.close(stop_w)
        try:
            os.setgid(_NOBODY)
            os.setuid(_NOBODY)
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as planted:
                planted.bind(path)
                os.chmod(path, 0o666)
                planted.listen()
                os.write(ready_w, b"1")
                os.read(stop_r, 1)
        finally:
            os._exit(0)
    os.close(ready_w)
    os.close(stop_r)
    try:
        assert os.read(ready_r, 1) == b"1"
        with pytest.raises(RuntimeError, match="not owned by this"):
            _deliver_pipe(path)
    finally:
        os.write(stop_w, b"1")
        os.close(stop_w)
        os.close(ready_r)
        os.waitpid(pid, 0)
