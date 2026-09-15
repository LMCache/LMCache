# SPDX-License-Identifier: Apache-2.0
"""Tests for the CUDA KV-cache IPC wrappers.

Cross-process round trips are parametrized over both wrappers where the
environment allows (``CudaIPCWrapper`` needs a shared ``/dev/shm``, which
same-host test processes have); raw-wrapper-only cases cover what the
torch storage path cannot do or what only the raw path must get right
(interior-pointer offsets).
"""

# Standard
from multiprocessing import get_context
from multiprocessing.connection import Connection

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import resolve_kv_wrapper_factory
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cuda import CudaDeviceSpec
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper, RawCudaIPCWrapper
from lmcache.v1.platform.isolated_ipc import is_isolated_ipc, set_isolated_ipc

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
    # ROCm reports torch.cuda.is_available() but has no cuda.bindings.
    pytest.mark.skipif(
        torch.version.hip is not None,
        reason="raw CUDA IPC wrapping is NVIDIA-only (cuda.bindings)",
    ),
]

DEVICE = "cuda:0"


@pytest.fixture
def restore_isolated_ipc():
    """Restore the process-global isolated-IPC switch after the test."""
    previous = is_isolated_ipc()
    yield
    set_isolated_ipc(previous)


def _make_wrapper(kind: str, tensor: torch.Tensor) -> DeviceIPCWrapper:
    """Construct a wrapper by parametrization key."""
    if kind == "raw":
        return RawCudaIPCWrapper(tensor)
    return CudaIPCWrapper(tensor)


# ---------------------------------------------------------------------------
# Factory selection under the isolated-IPC switch
# ---------------------------------------------------------------------------


def test_spec_selects_torch_wrapper_by_default(restore_isolated_ipc) -> None:
    set_isolated_ipc(False)
    assert CudaDeviceSpec().ipc_wrapper_cls is CudaIPCWrapper


def test_spec_selects_raw_wrapper_when_isolated(restore_isolated_ipc) -> None:
    set_isolated_ipc(True)
    assert CudaDeviceSpec().ipc_wrapper_cls is RawCudaIPCWrapper


def test_factory_dispatch_follows_switch(restore_isolated_ipc) -> None:
    """resolve_kv_wrapper_factory (the registration entry point) follows
    the switch at call time -- the wrapper class is not cached anywhere.
    """
    set_isolated_ipc(True)
    tensor = torch.arange(64, device=DEVICE, dtype=torch.float32)
    wrapper = resolve_kv_wrapper_factory("cuda")(tensor)
    assert type(wrapper) is RawCudaIPCWrapper

    set_isolated_ipc(False)
    wrapper = resolve_kv_wrapper_factory("cuda")(tensor)
    assert type(wrapper) is CudaIPCWrapper


# ---------------------------------------------------------------------------
# Wrapping semantics (same-process)
# ---------------------------------------------------------------------------


def test_raw_wrapper_records_interior_offset() -> None:
    """An interior view wraps with the same allocation handle as the
    base tensor and a strictly larger in-allocation offset.
    """
    base = torch.arange(1 << 16, device=DEVICE, dtype=torch.float32)
    interior = base[4096:8192]

    base_wrapper = RawCudaIPCWrapper(base)
    interior_wrapper = RawCudaIPCWrapper(interior)

    assert (
        interior_wrapper._ipc_handle_reserved == base_wrapper._ipc_handle_reserved  # noqa: SLF001
    )
    offset_delta = (
        interior_wrapper._alloc_offset - base_wrapper._alloc_offset  # noqa: SLF001
    )
    assert offset_delta == 4096 * base.element_size()


def test_raw_wrapper_rejects_unpermutable_noncontiguous() -> None:
    """A strided slice cannot be permuted contiguous and must be refused
    (the flat-bytes reconstruction would silently reorder elements).
    """
    base = torch.arange(1024, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous"):
        RawCudaIPCWrapper(base[::2])


# ---------------------------------------------------------------------------
# Mapping lifecycle (close / refcount / leak)
# ---------------------------------------------------------------------------


def _pool_exporter(conn: Connection, pool_mib: int, exit_after_send: bool) -> None:
    """Child: export wrappers for two interior views of one pool.

    Sends ``(wrapper_a, wrapper_b)`` pickled, then either exits
    immediately (dead-exporter case) or holds the pool until released.
    """
    torch.cuda.init()
    torch.cuda.set_device(0)
    pool = torch.zeros(pool_mib * (1 << 20) // 4, device=DEVICE, dtype=torch.float32)
    view_a = pool[1024:2048]
    view_b = pool[4096:8192]
    view_a.copy_(torch.arange(1024, device=DEVICE, dtype=torch.float32))
    view_b.copy_(torch.arange(4096, device=DEVICE, dtype=torch.float32))
    torch.cuda.synchronize()
    conn.send_bytes(DeviceIPCWrapper.Serialize(RawCudaIPCWrapper(view_a)))
    conn.send_bytes(DeviceIPCWrapper.Serialize(RawCudaIPCWrapper(view_b)))
    if not exit_after_send:
        conn.recv()  # hold the pool until the parent finishes


def test_close_refcounts_shared_allocation() -> None:
    """Closing one of two wrappers sharing an allocation must keep the
    other's tensor valid; after the last close a fresh import works
    (the mapping was genuinely released, not leaked or half-closed).
    """
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    child = ctx.Process(target=_pool_exporter, args=(child_conn, 64, False))
    child.start()
    try:
        wrapper_a = DeviceIPCWrapper.Deserialize(parent_conn.recv_bytes())
        wrapper_b = DeviceIPCWrapper.Deserialize(parent_conn.recv_bytes())

        tensor_a = wrapper_a.to_tensor()
        tensor_b = wrapper_b.to_tensor()
        assert tensor_a[100].item() == 100.0

        wrapper_a.close()
        wrapper_a.close()  # idempotent
        # The shared mapping must still serve wrapper_b's tensor.
        assert tensor_b[4095].item() == 4095.0

        wrapper_b.close()
        # Fresh import after the last close proves a clean unmap.
        reopened = wrapper_b.to_tensor()
        assert reopened[1].item() == 1.0
        wrapper_b.close()
        parent_conn.send("release")
    finally:
        child.join(timeout=60)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0


def test_close_releases_dead_exporters_memory() -> None:
    """A dead exporter's pool stays resident while this process holds the
    mapping and is returned by close() -- the Kubernetes pod-restart leak
    in miniature.
    """
    pool_mib = 2048
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    child = ctx.Process(target=_pool_exporter, args=(child_conn, pool_mib, True))
    child.start()
    try:
        wrapper_a = DeviceIPCWrapper.Deserialize(parent_conn.recv_bytes())
        wrapper_b = DeviceIPCWrapper.Deserialize(parent_conn.recv_bytes())
        # Import while the exporter is alive; handles cannot be opened
        # after it exits.
        tensor_a = wrapper_a.to_tensor()
        assert tensor_a[100].item() == 100.0
        wrapper_b.to_tensor()
    finally:
        child.join(timeout=60)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0

    torch.cuda.synchronize()
    free_pinned, _ = torch.cuda.mem_get_info(0)
    wrapper_a.close()
    wrapper_b.close()
    free_released, _ = torch.cuda.mem_get_info(0)

    released = free_released - free_pinned
    assert released >= int(0.8 * pool_mib * (1 << 20)), (
        f"close released only {released / (1 << 20):.0f} MiB of the dead "
        f"exporter's {pool_mib} MiB pool"
    )


# ---------------------------------------------------------------------------
# Cache-context integration of close
# ---------------------------------------------------------------------------


class _CloseRecordingWrapper(DeviceIPCWrapper):
    """Fake wrapper recording close calls; hands back the original tensor
    (no real IPC -- neither CUDA transport supports same-process import).
    """

    device_type = "cuda"

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self.closed = False
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())
        self.device_uuid = self._get_device_uuid(tensor.device.index)

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        self.closed = True


def _make_vllm_style_kv(num_layers: int) -> list[torch.Tensor]:
    """Small vLLM-format KV tensors: [2, num_blocks, block_size, heads, hd]."""
    return [
        torch.randn(2, 64, 16, 4, 32, device=DEVICE, dtype=torch.float16)
        for _ in range(num_layers)
    ]


def test_cache_context_close_closes_wrappers() -> None:
    """GPUCacheContext.close() must close every KV wrapper -- the hook the
    server's _release_entries teardown relies on.
    """
    # First Party
    from lmcache.v1.platform.cache_context import create_cache_context

    recorders = [_CloseRecordingWrapper(t) for t in _make_vllm_style_kv(2)]
    wrappers: list[DeviceIPCWrapper] = list(recorders)
    context = create_cache_context(wrappers, lmcache_tokens_per_chunk=256)
    assert not any(w.closed for w in recorders)
    context.close()
    assert all(w.closed for w in recorders)


def test_cache_context_init_failure_rolls_back_wrappers() -> None:
    """A registration that fails mid-construction must close the wrappers
    it already imported instead of pinning the worker's pool.
    """
    # First Party
    from lmcache.v1.platform.cuda.cache_context import GPUCacheContext

    good = [_CloseRecordingWrapper(t) for t in _make_vllm_style_kv(1)]
    bad = _CloseRecordingWrapper(
        torch.randn(7, device=DEVICE, dtype=torch.float16)  # not a KV shape
    )
    recorders = [*good, bad]
    wrappers: list[DeviceIPCWrapper] = list(recorders)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        GPUCacheContext(kv_caches=wrappers)
    assert all(w.closed for w in recorders)


# ---------------------------------------------------------------------------
# Cross-process round trip
# ---------------------------------------------------------------------------


def _consumer(conn: Connection) -> None:
    """Child: reconstruct the wrapped tensor, verify the producer's
    pattern, overwrite it with an acknowledgement pattern, and report.
    """
    torch.cuda.init()
    torch.cuda.set_device(0)
    wrapper = DeviceIPCWrapper.Deserialize(conn.recv_bytes())
    tensor = wrapper.to_tensor()

    expected = torch.arange(
        tensor.numel(), device=tensor.device, dtype=tensor.dtype
    ).reshape(tensor.shape)
    matches = bool(torch.equal(tensor, expected))

    tensor.fill_(-1.0)
    torch.cuda.synchronize()
    conn.send(matches)
    conn.recv()  # hold until the parent verified the write-back


@pytest.mark.parametrize("kind", ["torch", "raw"])
def test_cross_process_round_trip_of_interior_view(kind: str) -> None:
    """A consumer process sees the producer's exact bytes for an
    *interior* tensor view, and its writes land in the producer's
    memory (both directions prove the mapping targets the right offset).
    """
    pool = torch.zeros(1 << 16, device=DEVICE, dtype=torch.float32)
    view = pool[4096 : 4096 + 2048].reshape(64, 32)
    view.copy_(
        torch.arange(view.numel(), device=DEVICE, dtype=torch.float32).reshape(
            view.shape
        )
    )
    torch.cuda.synchronize()

    wrapper = _make_wrapper(kind, view)
    payload = DeviceIPCWrapper.Serialize(wrapper)

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    child = ctx.Process(target=_consumer, args=(child_conn,))
    child.start()
    try:
        parent_conn.send_bytes(payload)
        assert parent_conn.recv() is True  # consumer saw the exact pattern

        torch.cuda.synchronize()
        assert bool((view == -1.0).all().item())  # consumer wrote back
        # Neighbors outside the view must be untouched (offset correct).
        assert bool((pool[:4096] == 0).all().item())
        assert bool((pool[4096 + 2048 :] == 0).all().item())
        parent_conn.send("release")
    finally:
        child.join(timeout=60)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0
