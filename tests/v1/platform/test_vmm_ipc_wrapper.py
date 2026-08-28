# SPDX-License-Identifier: Apache-2.0
"""Tests for the CUDA VMM KV-cache IPC wrapper.

VMM (``cuMemCreate``/``cuMemMap``) memory has no legacy CUDA IPC handle;
:class:`VmmCudaIPCWrapper` shares it through
``cuMemExportToShareableHandle`` instead. POSIX-fd delivery is out of
band -- these tests stand in for the fd transport with a
``socket.send_fds`` pair and a test-installed resolver.
"""

# Standard
from multiprocessing import get_context
from multiprocessing.connection import Connection
import os
import pickle
import socket

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import resolve_kv_wrapper_factory
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cuda.ipc_wrapper import (
    VmmCudaIPCWrapper,
    set_vmm_fd_resolver,
)
from lmcache.v1.platform.cuda.utils import _cuda
from lmcache.v1.platform.isolated_ipc import is_isolated_ipc, set_isolated_ipc
from lmcache.v1.platform.vmm_ipc import is_use_vmm_api, set_use_vmm_api

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
    # ROCm reports torch.cuda.is_available() but has no cuda.bindings.
    pytest.mark.skipif(
        torch.version.hip is not None,
        reason="VMM IPC wrapping is NVIDIA-only (cuda.bindings)",
    ),
]

DEVICE = "cuda:0"
GRANULARITY = 2 * 1024 * 1024  # minimum cuMemCreate granularity (2 MiB)

# Fabric tests run only where IMEX channels are provisioned (the driver
# created this directory's channel nodes -- NVreg_CreateImexChannel0 or
# equivalent). No channel dir means no IMEX support on the node: skip
# rather than chase manual setup. Note the dir alone does not guarantee
# a fabric-capable GPU; that combination fails the test loudly instead
# of silently skipping.
IMEX_CHANNEL_DIR = "/dev/nvidia-caps-imex-channels"


def _imex_channel_present() -> bool:
    """Return whether an IMEX channel device exists on this node."""
    return os.path.isdir(IMEX_CHANNEL_DIR) and bool(os.listdir(IMEX_CHANNEL_DIR))


@pytest.fixture
def restore_switches():
    """Restore both process-global wrapper switches after the test."""
    previous_isolated = is_isolated_ipc()
    previous_vmm = is_use_vmm_api()
    yield
    set_isolated_ipc(previous_isolated)
    set_use_vmm_api(previous_vmm)


@pytest.fixture
def clear_fd_resolver():
    """Uninstall any test-installed fd resolver after the test."""
    yield
    set_vmm_fd_resolver(None)


class _VmmPool:
    """A VMM-backed device buffer: N chunks mapped into one VA range.

    Stands in for a vLLM cumem-allocator segment (``num_chunks=1``, the
    CUDA path) or a chunked ROCm-style allocation (``num_chunks>1``).
    """

    def __init__(
        self,
        num_chunks: int = 1,
        handle_type: object | None = None,
        device_index: int = 0,
    ) -> None:
        driver = _cuda.driver
        if handle_type is None:
            handle_types = driver.CUmemAllocationHandleType
            handle_type = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        self.size = num_chunks * GRANULARITY
        self.device_index = device_index

        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_index
        prop.requestedHandleTypes = handle_type

        with torch.cuda.device(device_index):
            err, base = driver.cuMemAddressReserve(self.size, 0, 0, 0)
            assert int(err) == 0, f"cuMemAddressReserve: {err}"
            self.base = int(base)
            self.handles = []
            for i in range(num_chunks):
                err, handle = driver.cuMemCreate(GRANULARITY, prop, 0)
                assert int(err) == 0, f"cuMemCreate: {err}"
                self.handles.append(handle)
                (err,) = driver.cuMemMap(
                    self.base + i * GRANULARITY, GRANULARITY, 0, handle, 0
                )
                assert int(err) == 0, f"cuMemMap: {err}"
            access = driver.CUmemAccessDesc()
            access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device_index
            access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            (err,) = driver.cuMemSetAccess(self.base, self.size, [access], 1)
            assert int(err) == 0, f"cuMemSetAccess: {err}"

    def tensor(self, offset_bytes: int, num_floats: int) -> torch.Tensor:
        """A float32 torch view at ``offset_bytes`` into the pool."""
        # Third Party
        import cupy

        nbytes = num_floats * 4
        with cupy.cuda.Device(self.device_index):
            mem = cupy.cuda.UnownedMemory(self.base, self.size, owner=self)
            memptr = cupy.cuda.MemoryPointer(mem, offset_bytes)
            cp_flat = cupy.ndarray(nbytes, dtype=cupy.uint8, memptr=memptr)
        return torch.from_dlpack(cp_flat).view(torch.float32)

    def close(self) -> None:
        driver = _cuda.driver
        with torch.cuda.device(self.device_index):
            driver.cuMemUnmap(self.base, self.size)
            for handle in self.handles:
                driver.cuMemRelease(handle)
            driver.cuMemAddressFree(self.base, self.size)


@pytest.fixture
def vmm_pool():
    pool = _VmmPool()
    yield pool
    pool.close()


# ---------------------------------------------------------------------------
# Factory selection under the use_vmm_api switch
# ---------------------------------------------------------------------------


def test_factory_selects_vmm_wrapper_when_enabled(restore_switches, vmm_pool) -> None:
    """use_vmm_api routes registration through VmmCudaIPCWrapper."""
    set_isolated_ipc(False)
    set_use_vmm_api(True)
    tensor = vmm_pool.tensor(0, 1024)
    wrapper = resolve_kv_wrapper_factory("cuda")(tensor)
    assert type(wrapper) is VmmCudaIPCWrapper
    wrapper.close()


def test_factory_default_unaffected(restore_switches) -> None:
    """Switch off keeps the legacy default wrapper."""
    # First Party
    from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

    set_isolated_ipc(False)
    set_use_vmm_api(False)
    tensor = torch.arange(64, device=DEVICE, dtype=torch.float32)
    wrapper = resolve_kv_wrapper_factory("cuda")(tensor)
    assert type(wrapper) is CudaIPCWrapper


def test_vmm_composes_with_isolated_per_kind(restore_switches) -> None:
    """use_vmm_api + isolated_ipc compose: the VMM wrapper is selected,
    and a POSIX-fd-only allocation is rejected at wrap time (fd passing
    needs a shared filesystem path, which zero-share rules out).
    """
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceSpec

    set_isolated_ipc(True)
    set_use_vmm_api(True)
    assert CudaDeviceSpec().ipc_wrapper_cls is VmmCudaIPCWrapper

    pool = _VmmPool()  # posix_fd handle type
    try:
        with pytest.raises(RuntimeError, match="isolated_ipc"):
            VmmCudaIPCWrapper(pool.tensor(0, 1024))
    finally:
        pool.close()


@pytest.mark.skipif(
    not _imex_channel_present(),
    reason="no IMEX channel on this node "
    f"({IMEX_CHANNEL_DIR}; see NVreg_CreateImexChannel0)",
)
def test_fabric_kind_allowed_under_isolated(restore_switches) -> None:
    """A fabric-exportable allocation wraps fine under isolated_ipc --
    the blob travels inline and needs no shared filesystem.
    """
    driver = _cuda.driver
    set_isolated_ipc(True)
    set_use_vmm_api(True)
    pool = _VmmPool(
        handle_type=driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    )
    try:
        wrapper = VmmCudaIPCWrapper(pool.tensor(0, 1024))
        assert wrapper._kind == "fabric"  # noqa: SLF001
        wrapper.close()
    finally:
        pool.close()


def test_vmm_wrap_rejects_non_vmm_memory(restore_switches) -> None:
    """The switch and the memory must agree: VmmCudaIPCWrapper on
    caching-allocator memory fails loudly at wrap.
    """
    tensor = torch.arange(64, device=DEVICE, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="VMM"):
        VmmCudaIPCWrapper(tensor)


# ---------------------------------------------------------------------------
# Wrapping semantics (same-process)
# ---------------------------------------------------------------------------


def test_wrap_records_chunk_offset(vmm_pool) -> None:
    """An interior view records its byte offset within the mapped chunk."""
    offset_bytes = 8192
    tensor = vmm_pool.tensor(offset_bytes, 2048)
    wrapper = VmmCudaIPCWrapper(tensor)
    try:
        assert wrapper._alloc_offset == offset_bytes  # noqa: SLF001
        assert wrapper._nbytes == 2048 * 4  # noqa: SLF001
        assert wrapper._kind == "posix_fd"  # noqa: SLF001
        assert wrapper.fd_payload() is not None
    finally:
        wrapper.close()


def test_wrap_rejects_multi_chunk() -> None:
    """A tensor spanning two physical chunks must be refused -- one
    exported handle maps exactly one chunk.
    """
    pool = _VmmPool(num_chunks=2)
    try:
        spanning = pool.tensor(0, pool.size // 4)  # covers both chunks
        with pytest.raises(RuntimeError, match="chunk"):
            VmmCudaIPCWrapper(spanning)
    finally:
        pool.close()


def test_wrap_rejects_unexportable_allocation() -> None:
    """requestedHandleTypes=NONE memory cannot be exported by any
    mechanism; wrap must fail loudly with the actionable hint.
    """
    driver = _cuda.driver
    pool = _VmmPool(
        handle_type=driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    )
    try:
        with pytest.raises(RuntimeError, match="requestedHandleTypes"):
            VmmCudaIPCWrapper(pool.tensor(0, 1024))
    finally:
        pool.close()


def test_wrap_rejects_unpermutable_noncontiguous(vmm_pool) -> None:
    """Same contiguity contract as RawCudaIPCWrapper."""
    base = vmm_pool.tensor(0, 1024)
    with pytest.raises(ValueError, match="contiguous"):
        VmmCudaIPCWrapper(base[::2])


def test_pickle_strips_process_local_state(vmm_pool) -> None:
    """The fd and registry references never travel with the pickle."""
    wrapper = VmmCudaIPCWrapper(vmm_pool.tensor(0, 1024))
    try:
        assert wrapper.fd_payload() is not None
        clone = pickle.loads(pickle.dumps(wrapper))
        assert clone.fd_payload() is None
        assert clone._opens == 0  # noqa: SLF001
        assert clone._export_id == wrapper._export_id  # noqa: SLF001
    finally:
        wrapper.close()


def test_exporter_close_closes_fd(vmm_pool) -> None:
    """close() on the exporting side must release the exported fd."""
    wrapper = VmmCudaIPCWrapper(vmm_pool.tensor(0, 1024))
    payload = wrapper.fd_payload()
    assert payload is not None
    _export_id, fd = payload
    os.fstat(fd)  # open before close
    wrapper.close()
    assert wrapper.fd_payload() is None
    with pytest.raises(OSError):
        os.fstat(fd)


def test_to_tensor_without_resolver_raises(vmm_pool, clear_fd_resolver) -> None:
    """An fd-kind wrapper cannot import without the out-of-band fd."""
    set_vmm_fd_resolver(None)
    wrapper = VmmCudaIPCWrapper(vmm_pool.tensor(0, 1024))
    try:
        clone = pickle.loads(pickle.dumps(wrapper))
        with pytest.raises(RuntimeError, match="resolver"):
            clone.to_tensor()
    finally:
        wrapper.close()


# ---------------------------------------------------------------------------
# Cross-process round trip (fd over SCM_RIGHTS)
# ---------------------------------------------------------------------------


def _vmm_consumer(conn: Connection, sock: socket.socket) -> None:
    """Child: receive the wrapper (pipe) and its fd (SCM_RIGHTS), import,
    verify the producer's pattern, write back, and hold until released.
    """
    torch.cuda.init()
    torch.cuda.set_device(0)

    wrapper = DeviceIPCWrapper.Deserialize(conn.recv_bytes())
    assert isinstance(wrapper, VmmCudaIPCWrapper)
    export_id, fds, _flags, _addr = socket.recv_fds(sock, 4096, 1)
    assert export_id == wrapper._export_id  # noqa: SLF001

    # Stand-in for the fd transport: each resolve hands out a dup so a
    # re-import after close also works.
    set_vmm_fd_resolver(lambda eid: os.dup(fds[0]))

    tensor = wrapper.to_tensor()
    again = wrapper.to_tensor()  # idempotent second import, same mapping
    same_mapping = tensor.data_ptr() == again.data_ptr()

    expected = torch.arange(
        tensor.numel(), device=tensor.device, dtype=tensor.dtype
    ).reshape(tensor.shape)
    matches = bool(torch.equal(tensor, expected)) and same_mapping

    tensor.fill_(-1.0)
    torch.cuda.synchronize()

    wrapper.close()
    reopened = wrapper.to_tensor()  # fresh import proves a clean unmap
    matches = matches and bool((reopened == -1.0).all().item())
    wrapper.close()
    os.close(fds[0])

    conn.send(matches)
    conn.recv()  # hold until the parent verified the write-back


def test_cross_process_round_trip_of_interior_view(clear_fd_resolver) -> None:
    """A consumer process sees the producer's exact bytes for an interior
    VMM view, and its writes land in the producer's memory.
    """
    pool = _VmmPool()
    try:
        offset_bytes = 3 * 4096
        view = pool.tensor(offset_bytes, 2048).reshape(64, 32)
        neighbors = pool.tensor(0, pool.size // 4)
        neighbors.zero_()
        view.copy_(
            torch.arange(view.numel(), device=DEVICE, dtype=torch.float32).reshape(
                view.shape
            )
        )
        torch.cuda.synchronize()

        wrapper = VmmCudaIPCWrapper(view)
        payload = DeviceIPCWrapper.Serialize(wrapper)
        fd_payload = wrapper.fd_payload()
        assert fd_payload is not None
        export_id, fd = fd_payload

        parent_sock, child_sock = socket.socketpair()
        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        child = ctx.Process(target=_vmm_consumer, args=(child_conn, child_sock))
        child.start()
        try:
            parent_conn.send_bytes(payload)
            socket.send_fds(parent_sock, [export_id], [fd])
            # poll() before recv(): a crashed consumer must fail the test,
            # not block recv() forever.
            assert parent_conn.poll(120), "consumer died before reporting"
            assert parent_conn.recv() is True

            torch.cuda.synchronize()
            assert bool((view == -1.0).all().item())  # consumer wrote back
            # Neighbors outside the view must be untouched (offset correct).
            flat = neighbors.view(-1)
            n_before = offset_bytes // 4
            n_after_start = (offset_bytes + 2048 * 4) // 4
            assert bool((flat[:n_before] == 0).all().item())
            assert bool((flat[n_after_start:] == 0).all().item())
            parent_conn.send("release")
        finally:
            child.join(timeout=60)
            if child.is_alive():
                child.kill()
                child.join()
        assert child.exitcode == 0
        wrapper.close()
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Fabric kind (requires an IMEX channel; validates the blob path)
# ---------------------------------------------------------------------------


def _fabric_consumer(conn: Connection) -> None:
    """Child: import a fabric-kind wrapper (blob travels inline)."""
    torch.cuda.init()
    torch.cuda.set_device(0)
    wrapper = DeviceIPCWrapper.Deserialize(conn.recv_bytes())
    tensor = wrapper.to_tensor()
    expected = torch.arange(
        tensor.numel(), device=tensor.device, dtype=tensor.dtype
    ).reshape(tensor.shape)
    conn.send(bool(torch.equal(tensor, expected)))
    wrapper.close()
    conn.recv()


@pytest.mark.skipif(
    not _imex_channel_present(),
    reason="no IMEX channel on this node "
    f"({IMEX_CHANNEL_DIR}; see NVreg_CreateImexChannel0)",
)
def test_fabric_round_trip() -> None:
    """Fabric-kind wrappers need no fd channel: the 64-byte blob rides
    the pickle and imports across processes on its own.
    """
    driver = _cuda.driver
    pool = _VmmPool(
        handle_type=driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    )
    try:
        view = pool.tensor(4096, 1024)
        view.copy_(torch.arange(1024, device=DEVICE, dtype=torch.float32))
        torch.cuda.synchronize()

        wrapper = VmmCudaIPCWrapper(view)
        assert wrapper._kind == "fabric"  # noqa: SLF001
        assert wrapper.fd_payload() is None

        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        child = ctx.Process(target=_fabric_consumer, args=(child_conn,))
        child.start()
        try:
            parent_conn.send_bytes(DeviceIPCWrapper.Serialize(wrapper))
            # poll() before recv(): a crashed consumer must fail the test,
            # not block recv() forever.
            assert parent_conn.poll(120), "consumer died before reporting"
            assert parent_conn.recv() is True
            parent_conn.send("release")
        finally:
            child.join(timeout=60)
            if child.is_alive():
                child.kill()
                child.join()
        assert child.exitcode == 0
        wrapper.close()
    finally:
        pool.close()
