# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated cross-process tests for NPU IPC events."""

# Standard
import multiprocessing as mp

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.npu import NpuDeviceSpec

pytestmark = [
    pytest.mark.npu,
    pytest.mark.no_shared_allocator,
]

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU hardware is required",
)


def _child_event_producer(conn) -> None:
    # Third Party
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        payload = torch.full((1024,), 7.0, device="npu")
        payload.mul_(2)
    event = torch.npu.Event(enable_timing=False, interprocess=True)
    event.record(stream)
    conn.send({"handle": event.ipc_handle(), "expected": 14.0})
    conn.recv()  # keep the producer alive until the parent is done


@requires_npu
def test_cross_process_event_import_and_sync() -> None:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_child_event_producer, args=(child_conn,))
    process.start()
    try:
        message = parent_conn.recv()
        imported = torch.npu.Event.from_ipc_handle(
            torch.device("npu:0"), message["handle"]
        )
        imported.synchronize()
        assert imported.query() is True
    finally:
        parent_conn.send("done")
        process.join(timeout=30)
    assert process.exitcode == 0


@requires_npu
def test_backend_roundtrip_on_device() -> None:
    backend = NpuDeviceSpec().event_ipc_backend
    device = torch.device("npu:0")
    backend.check_event_support(device)
    event = backend.create_event(device)
    stream = torch.npu.Stream()
    payload = torch.ones(8, device=device)
    payload.add_(1)
    backend.record_event(event, stream)
    backend.synchronize_event(event, device)
    assert backend.query_event(event) is True
    handle = backend.export_event(event, device)
    # Same-process ``from_ipc_handle`` fails with driver error 17 (ACL
    # 507899) on this CANN build, so handle import is verified across
    # processes in ``test_cross_process_event_import_and_sync``.
    assert isinstance(handle, bytes) and len(handle) > 0
