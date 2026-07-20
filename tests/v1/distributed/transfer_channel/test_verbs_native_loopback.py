# SPDX-License-Identifier: Apache-2.0

# Standard
import ctypes
import mmap
import os
import socket
import threading
import time

# Third Party
import pytest

_DEVICE = os.environ.get("LMCACHE_RDMA_TEST_DEVICE")
_GID_INDEX = os.environ.get("LMCACHE_RDMA_TEST_GID_INDEX")
_CONTROL_IP = os.environ.get("LMCACHE_RDMA_TEST_IP")

pytestmark = pytest.mark.skipif(
    not (_DEVICE and _GID_INDEX and _CONTROL_IP),
    reason=(
        "set LMCACHE_RDMA_TEST_DEVICE, LMCACHE_RDMA_TEST_GID_INDEX, and "
        "LMCACHE_RDMA_TEST_IP on an RDMA test host"
    ),
)


def _native():
    return pytest.importorskip("lmcache.rdma_l1_ops")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_CONTROL_IP, 0))  # type: ignore[arg-type]
        return int(sock.getsockname()[1])


def _contexts(size: int):
    native = _native()
    source_buffer = mmap.mmap(-1, size)
    target_buffer = mmap.mmap(-1, size)

    def pointer(buffer):
        return ctypes.addressof(ctypes.c_char.from_buffer(buffer))

    base_port = _free_port()
    kwargs = {
        "device_name": _DEVICE,
        "port_num": 1,
        "gid_index": int(_GID_INDEX),  # type: ignore[arg-type]
        "queue_depth": 4096,
        "handshake_timeout_ms": 2_000,
    }
    source = native.RdmaContext(
        base_address=pointer(source_buffer),
        length=size,
        listen_url=f"{_CONTROL_IP}:{base_port}",
        advertise_url=f"{_CONTROL_IP}:{base_port}",
        **kwargs,
    )
    try:
        target = native.RdmaContext(
            base_address=pointer(target_buffer),
            length=size,
            listen_url=f"{_CONTROL_IP}:{base_port + 1}",
            advertise_url=f"{_CONTROL_IP}:{base_port + 1}",
            **kwargs,
        )
    except Exception:
        source.close()
        source_buffer.close()
        target_buffer.close()
        raise
    return source, target, source_buffer, target_buffer, base_port


def test_native_rdma_loopback_checksum_and_inflight_close():
    size = 16 * 1024 * 1024
    source, target, source_buffer, target_buffer, base_port = _contexts(size)
    client = None
    try:
        for index in range(size):
            source_buffer[index] = (index * 17 + 3) & 0xFF
        client = target.connect(f"{_CONTROL_IP}:{base_port}")
        task_id = client.submit_read([0], [0], [size])
        deadline = time.monotonic() + 10
        while True:
            finished, succeeded, count = client.query_read_status(task_id)
            if finished:
                assert succeeded and count == 1
                break
            if time.monotonic() > deadline:
                raise TimeoutError("native RDMA loopback did not complete")
            time.sleep(0.001)
        assert bytes(source_buffer) == bytes(target_buffer)

        # A close must quiesce an in-flight QP before the registered L1 mapping
        # is released. This also exercises CQ flush/event acknowledgement.
        client.submit_read([0], [0], [size])
        started = time.monotonic()
        client.close()
        client = None
        assert time.monotonic() - started < 5
    finally:
        if client is not None:
            client.close()
        target.close()
        source.close()
        source_buffer.close()
        target_buffer.close()


def test_native_context_close_interrupts_handshake():
    native = _native()
    size = 4096
    buffer = mmap.mmap(-1, size)
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    port = _free_port()
    context = native.RdmaContext(
        base_address=pointer,
        length=size,
        listen_url=f"{_CONTROL_IP}:{port}",
        advertise_url=f"{_CONTROL_IP}:{port}",
        device_name=_DEVICE,
        port_num=1,
        gid_index=int(_GID_INDEX),  # type: ignore[arg-type]
        queue_depth=16,
        handshake_timeout_ms=5_000,
    )
    peer = socket.create_connection((_CONTROL_IP, port))  # type: ignore[arg-type]
    try:
        peer.sendall(b"x")
        time.sleep(0.05)
        started = time.monotonic()
        context.close()
        assert time.monotonic() - started < 5
    finally:
        peer.close()
        buffer.close()


def test_native_context_close_interrupts_outbound_handshake():
    native = _native()
    size = 4096
    buffer = mmap.mmap(-1, size)
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    context_port = _free_port()
    peer_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    peer_listener.bind((_CONTROL_IP, 0))  # type: ignore[arg-type]
    peer_listener.listen(1)
    peer_port = int(peer_listener.getsockname()[1])
    context = native.RdmaContext(
        base_address=pointer,
        length=size,
        listen_url=f"{_CONTROL_IP}:{context_port}",
        advertise_url=f"{_CONTROL_IP}:{context_port}",
        device_name=_DEVICE,
        port_num=1,
        gid_index=int(_GID_INDEX),  # type: ignore[arg-type]
        queue_depth=16,
        handshake_timeout_ms=5_000,
    )
    errors = []

    def connect() -> None:
        try:
            context.connect(f"{_CONTROL_IP}:{peer_port}")
        except Exception as error:  # noqa: BLE001 - asserted below
            errors.append(error)

    thread = threading.Thread(target=connect)
    thread.start()
    peer, _ = peer_listener.accept()
    try:
        started = time.monotonic()
        context.close()
        thread.join(timeout=2)
        assert time.monotonic() - started < 2
        assert not thread.is_alive()
        assert errors
    finally:
        peer.close()
        peer_listener.close()
        if thread.is_alive():
            thread.join(timeout=5)
        context.close()
        buffer.close()
