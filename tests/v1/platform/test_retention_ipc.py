# SPDX-License-Identifier: Apache-2.0
"""CPU checks for dev FD ownership, wire compatibility and independent leases."""

from unittest.mock import MagicMock
import os
import struct

import msgspec
import pytest

from lmcache.v1.platform import fd_channel, kv_wrap
from lmcache.v1.platform.base.ipc_wire import (
    TensorMetadata,
    VmmCudaIPCData,
    VmmRocmIPCData,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.cuda import ipc_wrapper as ipc
from lmcache.v1.platform.devices.rocm.ipc_wrapper import VmmRocmIPCWrapper


def descriptor(wire, creation_id=None):
    return wire(
        version=1,
        metadata=TensorMetadata("uint8", (8,), (1,), 0, "device"),
        export_id=b"r" * 16,
        kind="posix_fd",
        fabric_blob=None,
        alloc_size=4096,
        alloc_offset=16,
        nbytes=8,
        creation_id=creation_id,
    )


@pytest.mark.parametrize(
    "wire, cls, tag",
    [
        (VmmCudaIPCData, ipc.VmmCudaIPCWrapper, "cuda_vmm"),
        (VmmRocmIPCData, VmmRocmIPCWrapper, "rocm_vmm"),
    ],
)
def test_default_wire_compatibility(wire, cls, tag):
    w = cls.from_wire(descriptor(wire))
    data = DeviceIPCWrapper.Serialize(w)
    decoded = msgspec.msgpack.decode(data)
    assert decoded["type"] == tag and "creation_id" not in decoded
    assert type(DeviceIPCWrapper.Deserialize(data)) is cls
    assert DeviceIPCWrapper.Deserialize(data).to_wire() == w.to_wire()


@pytest.mark.parametrize("observed", [False, True])
def test_fd_close_at_registration_and_ledger(monkeypatch, observed):
    # The client double holds an independent OS fd just as the ACK'd server does.
    monkeypatch.delenv("LMCACHE_ZC_RUN_ID", raising=False)
    if observed:
        monkeypatch.setenv("LMCACHE_ZC_RUN_ID", "run")
    log = MagicMock()
    monkeypatch.setattr(ipc.retention_allocator, "record", log)
    w = ipc.VmmCudaIPCWrapper.from_wire(
        descriptor(VmmCudaIPCData, "creation" if observed else None)
    )
    reader, w._fd = os.pipe()
    dup = []
    client = MagicMock()
    client.send.side_effect = lambda eid, fd: dup.append(os.dup(fd))
    monkeypatch.setattr(kv_wrap, "FdChannelClient", lambda *args: client)
    try:
        with kv_wrap.kv_wrapper_fds_delivered([w], "channel", 1):
            assert w.fd_payload() is None
            registered = DeviceIPCWrapper.Deserialize(DeviceIPCWrapper.Serialize(w))
            assert registered.creation_id == w.creation_id
            os.write(dup[0], b"x")
            assert os.read(reader, 1) == b"x"
        events = [c.args[0] for c in log.call_args_list]
        assert events == (["fd_sent", "sender_fd_closed"] if observed else [])
        assert all(c.kwargs["creation_id"] == "creation" for c in log.call_args_list)
    finally:
        w.close()
        for fd in dup:
            os.close(fd)
        os.close(reader)


def test_last_lease_owns_driver_release(monkeypatch):
    monkeypatch.setenv("LMCACHE_ZC_RUN_ID", "run")
    log = MagicMock()
    monkeypatch.setattr(ipc.retention_allocator, "record", log)
    cls = ipc.VmmCudaIPCWrapper
    w = cls.from_wire(descriptor(VmmCudaIPCData, "creation"))
    mock_driver = MagicMock()
    mock_driver.CUresult.CUDA_SUCCESS = 0
    for method in ("cuMemUnmap", "cuMemAddressFree", "cuMemRelease"):
        getattr(mock_driver, method).return_value = (0,)
    monkeypatch.setattr(cls, "_driver", staticmethod(lambda: mock_driver))
    monkeypatch.setattr(cls, "_bind_context", staticmethod(lambda dev: None))
    monkeypatch.setattr(cls, "_get_device_index_from_uuid", staticmethod(lambda uid: 0))
    monkeypatch.setattr(
        cls, "_tensor_view", lambda self, base: base + self._alloc_offset
    )
    w._opens = 1
    entry = ipc._VmmMapping(1024, 77, 1)
    monkeypatch.setitem(ipc._MAPPED_VMM_ALLOCATIONS, w._export_id, entry)
    lease = w.acquire_mapping_lease()
    w.close()
    assert lease.tensor == 1040 and lease.creation_id == "creation"
    mock_driver.cuMemUnmap.assert_not_called()
    lease.close()
    lease.close()
    mock_driver.cuMemUnmap.assert_called_once_with(1024, 4096)
    mock_driver.cuMemRelease.assert_called_once_with(77)
    assert [c.args[0] for c in log.call_args_list].count("lease_drop") == 1


def test_peer_uid_policy_is_opt_in(monkeypatch):
    sock = MagicMock()
    monkeypatch.setattr(fd_channel.socket, "SO_PEERCRED", 17, raising=False)
    sock.getsockopt.return_value = struct.pack("3i", 100, os.geteuid() + 1, 7)
    monkeypatch.delenv("LMCACHE_ZC_RUN_ID", raising=False)
    fd_channel._check_retention_peer(sock)
    sock.getsockopt.assert_not_called()
    monkeypatch.setenv("LMCACHE_ZC_RUN_ID", "run")
    with pytest.raises(PermissionError):
        fd_channel._check_retention_peer(sock)
