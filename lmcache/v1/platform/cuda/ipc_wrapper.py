# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC wrapper implementations.

:class:`CudaIPCWrapper` shares tensors through PyTorch's storage IPC
(needs a shared ``/dev/shm``); :class:`RawCudaIPCWrapper` shares them
through driver-level CUDA IPC memory handles alone, which work across
fully isolated containers and also cover tensors allocated outside
PyTorch (e.g. TRT-LLM's ``cudaMalloc``'d pool);
:class:`VmmCudaIPCWrapper` shares CUDA VMM (``cuMemCreate``) memory,
which no legacy IPC handle can express (vLLM's cumem allocator, torch
``expandable_segments``).

``device_type="cuda"`` binds to one of the three via
:attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`,
driven by two mutually exclusive process-global switches:
:class:`CudaIPCWrapper` by default, :class:`RawCudaIPCWrapper` under
``isolated_ipc`` (``lmcache/v1/platform/isolated_ipc.py``), and
:class:`VmmCudaIPCWrapper` under ``use_vmm_api``
(``lmcache/v1/platform/vmm_ipc.py``). The multiprocess adapter
dispatches through
:func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`; the TRT-LLM
adapter instantiates :class:`RawCudaIPCWrapper` directly.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Callable, ClassVar
import os
import threading
import uuid

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cuda.utils import _cuda
from lmcache.v1.platform.isolated_ipc import is_isolated_ipc

logger = init_logger(__name__)

_NON_IPC_MEMORY_HINT = (
    "CUDA IPC memory handles only support cudaMalloc-style allocations. "
    "Memory created through the CUDA VMM API cannot be shared this way -- "
    "common sources are PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "and vLLM's sleep mode (CuMemAllocator). Disable those or turn off "
    "isolated IPC for this deployment."
)

_VMM_NOT_EXPORTABLE_HINT = (
    "the VMM allocation was created without an exportable handle type "
    "(requestedHandleTypes has neither POSIX_FILE_DESCRIPTOR nor FABRIC). "
    "Exportability is fixed at cuMemCreate time and cannot be added later. "
    "vLLM's cumem allocator only requests exportable memory on "
    "fabric-capable devices (falling back to a POSIX fd when IMEX is not "
    "provisioned); on devices without fabric support it needs a patch to "
    "request CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR."
)

_VMM_MULTI_CHUNK_HINT = (
    "the tensor spans more than one cuMemCreate physical chunk "
    "(cuMemGetAddressRange reports per-chunk ranges, and one exported "
    "handle maps exactly one chunk). Multi-chunk VMM allocations (e.g. "
    "vLLM's ROCm cumem path) are not supported."
)

# Refcounted registry of allocations this process has mapped via
# cudaIpcOpenMemHandle, keyed by handle bytes: handle -> [mapped_ptr, opens].
# The driver returns ONE mapping per (process, allocation) no matter how
# often it is opened, and a single cudaIpcCloseMemHandle unmaps it for
# every user -- so per-layer tensors sharing one allocation must be
# counted, and the mapping unmapped only when the last wrapper closes.
# An unclosed mapping pins the exporting process's device memory even
# after the exporter dies (a dead vLLM worker's KV pool stays resident
# until the server closes or exits). Keyed by handle bytes alone: a KV
# allocation is only ever imported on the wrapper's own device.
_MAPPED_ALLOCATIONS: dict[bytes, list[int]] = {}
_MAPPINGS_LOCK = threading.Lock()


@dataclass
class _VmmMapping:
    """One imported VMM mapping held by this process.

    Attributes:
        base_ptr: Mapped base address in this process.
        alloc_handle: The imported ``CUmemGenericAllocationHandle``
            (opaque cuda-bindings object; released on last close).
        refcount: References taken by ``to_tensor`` calls.
    """

    base_ptr: int
    alloc_handle: object
    refcount: int


# Refcounted registry of VMM allocations this process has imported via
# cuMemImportFromShareableHandle, keyed by the wrapper's export id.
# Unlike the legacy registry above there is no driver-side dedup to
# mirror -- each import creates an independent mapping -- so the
# registry only serves repeated to_tensor() calls on (copies of) the
# same wrapper, and close() releases the mapping when the last
# reference drops.
_MAPPED_VMM_ALLOCATIONS: dict[bytes, _VmmMapping] = {}

# Receiver-side hook resolving an export id to a POSIX fd delivered out
# of band (fds cannot ride the pickled wrapper). Installed by the fd
# transport (tests inject a socketpair-based resolver); unset means fd
# delivery is not configured in this process.
_VMM_FD_RESOLVER: Callable[[bytes], int] | None = None


def set_vmm_fd_resolver(resolver: Callable[[bytes], int] | None) -> None:
    """Install the process-wide export-id -> fd resolver.

    The resolver is called by :meth:`VmmCudaIPCWrapper.to_tensor` on the
    importing side to obtain the POSIX fd for an export id. Ownership of
    the returned fd transfers to the wrapper (it is closed after import).

    Args:
        resolver: Callable mapping an export id to an open fd, or ``None``
            to uninstall.
    """
    global _VMM_FD_RESOLVER
    _VMM_FD_RESOLVER = resolver


def _resolve_vmm_fd(export_id: bytes) -> int:
    """Resolve an export id to a POSIX fd via the installed resolver.

    Args:
        export_id: The wrapper's export id.

    Returns:
        An open fd owned by the caller.

    Raises:
        RuntimeError: If no resolver is installed.
    """
    if _VMM_FD_RESOLVER is None:
        raise RuntimeError(
            "No VMM fd resolver installed in this process: the wrapper's "
            "POSIX fd must be delivered out of band (SCM_RIGHTS) and "
            "registered via set_vmm_fd_resolver() before to_tensor()."
        )
    return _VMM_FD_RESOLVER(export_id)


class CudaIPCWrapper(DeviceIPCWrapper):
    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "CudaIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by PyTorch's caching allocator.

        Returns:
            A new :class:`CudaIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        # Offset is preserved by the wrapper's storage_offset field.
        tensor = attempt_permute_to_contiguous_view(tensor)

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """
        Note:
            This function may break if the accelerator is not initialized.
            We should call ``torch_dev.init()`` before using this function
            (guarded by hasattr since not all backends expose init()).
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t


class RawCudaIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper that shares CUDA tensors through driver-level IPC only.

    ``CudaIPCWrapper`` rides PyTorch's ``UntypedStorage._share_cuda_()``,
    which requires a shared ``/dev/shm`` between the processes (torch
    keeps its IPC reference counter there). This wrapper instead calls
    ``cudaIpcGetMemHandle`` on the tensor's allocation and reconstructs
    on the receiving side via ``cudaIpcOpenMemHandle`` plus a CuPy
    ``UnownedMemory`` → DLPack → ``torch`` round-trip. CUDA IPC *memory*
    handles rendezvous in the kernel driver, so this works across fully
    isolated containers -- no shared IPC namespace, no common /dev/shm.

    Two caller groups use it:

    - the MP registration path selects it via
      :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
      when the isolated-IPC switch is on
      (``lmcache/v1/platform/isolated_ipc.py``);
    - the TRT-LLM adapter instantiates it directly for its
      ``cudaMalloc``'d KV pool, which ``_share_cuda_()`` cannot wrap at
      all.

    An IPC mem handle always maps the *whole allocation* it was taken
    from, and opening it returns the allocation's *base* pointer. Torch
    caching-allocator tensors usually sit at an interior pointer, so the
    wrapper ships ``data_ptr - cuMemGetAddressRange(data_ptr).base`` and
    the consumer reads at that offset from the opened base.

    Sharing the ``DeviceIPCWrapper`` base (rather than introducing a
    parallel class with its own msgspec ext code) is load-bearing —
    msgspec does not support unions of custom ext-encoded types. With a
    common base, ``KVCache = list[DeviceIPCWrapper]`` type-checks, the
    single ext code 1 round-trips every wrapper, and pickle preserves
    the concrete subclass identity through the wire so ``to_tensor``
    dispatches correctly.
    """

    #: Same ``torch.device.type`` as ``CudaIPCWrapper``; exposed on
    #: :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
    #: under isolated IPC, instantiated directly by the TRT-LLM adapter.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "RawCudaIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by ``cudaMalloc``-style memory
                (PyTorch caching-allocator tensors included).

        Returns:
            A new :class:`RawCudaIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Same layout normalization as CudaIPCWrapper: permute
        # non-contiguous views (e.g. vLLM's NHD-over-HND) into contiguous
        # ones, metadata-only. The flat-bytes reconstruction below only
        # supports contiguous tensors, so anything still non-contiguous
        # is rejected rather than silently reordered.
        tensor = attempt_permute_to_contiguous_view(tensor)
        if not tensor.is_contiguous():
            raise ValueError(
                "RawCudaIPCWrapper requires a tensor that is contiguous "
                f"(possibly after permutation); got shape={tuple(tensor.shape)} "
                f"stride={tuple(tensor.stride())}"
            )

        data_ptr = tensor.data_ptr()
        range_result = _cuda.driver.cuMemGetAddressRange(
            _cuda.driver.CUdeviceptr(data_ptr)
        )
        if range_result[0] != 0:
            raise RuntimeError(
                f"cuMemGetAddressRange failed: {range_result[0]} "
                f"(ptr=0x{data_ptr:x}). " + _NON_IPC_MEMORY_HINT
            )
        _err, alloc_base, _alloc_size = range_result

        err, ipc_handle = _cuda.runtime.cudaIpcGetMemHandle(int(alloc_base))
        if err != _cuda.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed: {err} (ptr=0x{data_ptr:x}). "
                + _NON_IPC_MEMORY_HINT
            )

        # Store only what's needed for reconstruction. The handle maps
        # the whole allocation; the offset locates the tensor within it.
        self._ipc_handle_reserved = bytes(ipc_handle.reserved)
        self._alloc_offset = data_ptr - int(
            alloc_base
        )  # offset in bytes not the same as storage offset
        self._nbytes = tensor.numel() * tensor.element_size()

        # DeviceIPCWrapper interface fields. ``handle`` is unused —
        # ``to_tensor`` is overridden to bypass it — but kept (None) so
        # the base-class equality check has a value to compare.
        # ``storage_offset`` is 0 because ``data_ptr`` (folded into
        # ``_alloc_offset``) already points at the tensor's first element.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = 0

        # Opens this wrapper holds on the shared mapping registry;
        # released by close(). Travels through pickle as 0 (the producer
        # never opens), so a receiver always starts at 0.
        self._opens = 0

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC.

        Opens the allocation's mem handle through the process-wide
        refcounted registry (one physical mapping per allocation).
        Every call takes one reference; :meth:`close` releases all
        references this wrapper holds.
        """
        # Third Party
        import cupy

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        with _MAPPINGS_LOCK:
            entry = _MAPPED_ALLOCATIONS.get(self._ipc_handle_reserved)
            if entry is None:
                handle = _cuda.runtime.cudaIpcMemHandle_t()
                handle.reserved = self._ipc_handle_reserved
                with torch.cuda.device(device_index):
                    err, ptr = _cuda.runtime.cudaIpcOpenMemHandle(
                        handle, _cuda.runtime.cudaIpcMemLazyEnablePeerAccess
                    )
                if err != _cuda.runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")
                entry = [int(ptr), 0]
                _MAPPED_ALLOCATIONS[self._ipc_handle_reserved] = entry
            entry[1] += 1
            self._opens += 1
            base_ptr = entry[0]

        # Wrap as a flat ``uint8`` CuPy array at the allocation offset,
        # DLPack to torch, then view as the original dtype/shape.
        # ``uint8`` avoids dtype-conversion gaps (bfloat16, fp8 have no
        # direct CuPy/NumPy equivalent without ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(
                base_ptr, self._alloc_offset + self._nbytes, owner=self
            )
            memptr = cupy.cuda.MemoryPointer(mem, self._alloc_offset)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)

    def close(self) -> None:
        """Release this wrapper's references on the imported mapping.

        Unmaps the allocation (``cudaIpcCloseMemHandle``) when the last
        reference across all wrappers is released, returning the
        exporter's device memory once the exporter itself has freed it.
        Idempotent; safe on wrappers that never imported. Tensors from
        :meth:`to_tensor` must no longer be dereferenced after the last
        close -- their later garbage collection is harmless (the CuPy
        memory is unowned; no device call is issued on collection).

        Unmap failures are logged, not raised: close runs on teardown
        paths (the worker reaper) where raising would abort cleanup of
        the remaining entries.
        """
        opens = getattr(self, "_opens", 0)
        if opens <= 0:
            return
        with _MAPPINGS_LOCK:
            self._opens = 0
            entry = _MAPPED_ALLOCATIONS.get(self._ipc_handle_reserved)
            if entry is None:
                return
            entry[1] -= opens
            if entry[1] > 0:
                return
            del _MAPPED_ALLOCATIONS[self._ipc_handle_reserved]
            device_index = self._get_device_index_from_uuid(self.device_uuid)
            with torch.cuda.device(device_index):
                (err,) = _cuda.runtime.cudaIpcCloseMemHandle(entry[0])
            if err != _cuda.runtime.cudaError_t.cudaSuccess:
                logger.warning(
                    "cudaIpcCloseMemHandle failed: %s (ptr=0x%x)", err, entry[0]
                )


class VmmCudaIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for CUDA VMM (``cuMemCreate``/``cuMemMap``) memory.

    Legacy CUDA IPC fails on VMM-backed pointers (vLLM's cumem
    allocator, torch ``expandable_segments``); VMM has its own IPC:
    export the allocation as an OS shareable handle, import + map it in
    the receiver. The shareable form is fixed at ``cuMemCreate`` time
    (``requestedHandleTypes``); the wrapper reads it off the allocation
    and picks per allocation:

    - ``POSIX_FILE_DESCRIPTOR``: an fd -- meaningless outside this
      process, so it never rides the pickle; it is delivered out of
      band (``SCM_RIGHTS``) and looked up via :func:`set_vmm_fd_resolver`
      by the wrapper's export id.
    - ``FABRIC``: a 64-byte blob, travels inline; needs IMEX channel
      access in both processes.
    - neither: wrap fails loudly (nothing can export the allocation).

    Each wrapper exports/imports its own mapping (N views of one
    allocation cost N fds + N mappings; dedup only if that
    materializes). Unlike legacy IPC, VMM handles are driver-refcounted:
    an imported mapping keeps the physical memory alive after the
    exporter releases it, so :meth:`close` is mandatory on the importing
    side (cache-context teardown calls it). Full rationale:
    ``docs/design/v1/platform/cuda/vmm_cuda_ipc.md``.
    """

    #: Same ``torch.device.type`` as the other CUDA wrappers; exposed on
    #: :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
    #: when the ``use_vmm_api`` switch is on.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "VmmCudaIPCWrapper":
        """Factory mirror of the other wrappers' ``wrap``.

        Args:
            tensor: A CUDA tensor backed by VMM (``cuMemMap``'d) memory.

        Returns:
            A new :class:`VmmCudaIPCWrapper` wrapping ``tensor``.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Same layout normalization + contiguity contract as
        # RawCudaIPCWrapper: flat-bytes reconstruction supports only
        # contiguous tensors.
        tensor = attempt_permute_to_contiguous_view(tensor)
        if not tensor.is_contiguous():
            raise ValueError(
                "VmmCudaIPCWrapper requires a tensor that is contiguous "
                f"(possibly after permutation); got shape={tuple(tensor.shape)} "
                f"stride={tuple(tensor.stride())}"
            )

        data_ptr = tensor.data_ptr()
        device_index = tensor.device.index
        nbytes = tensor.numel() * tensor.element_size()

        kind: str = ""
        payload: object = None
        with torch.cuda.device(device_index):
            # Recover the allocation handle from the bare pointer (works
            # for interior pointers; adds one driver reference we must
            # release below).
            err, alloc_handle = _cuda.driver.cuMemRetainAllocationHandle(int(data_ptr))
            if err != _cuda.driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuMemRetainAllocationHandle failed: {err} "
                    f"(ptr=0x{data_ptr:x}); the tensor does not appear to "
                    "be VMM-backed."
                )
            try:
                kind, payload = self._export_allocation(alloc_handle)

                # The fabric kind is isolation-clean (inline blob, IMEX
                # channel device); a POSIX fd is not -- it must cross via
                # SCM_RIGHTS over a shared filesystem path, which the
                # zero-share isolated-IPC model rules out.
                if kind == "posix_fd" and is_isolated_ipc():
                    raise RuntimeError(
                        "VMM allocation is only POSIX-fd exportable, but "
                        "isolated_ipc is enabled: fd passing needs a shared "
                        "filesystem path between the processes. Provision an "
                        "IMEX channel (NVreg_CreateImexChannel0 / "
                        "NVIDIA_IMEX_CHANNELS) so the pool is fabric-"
                        "exportable, or disable isolated_ipc."
                    )

                # One exported handle maps exactly one physical chunk.
                # cuMemGetAddressRange on VMM reports the mapped chunk
                # containing data_ptr, so the tensor must fit inside it.
                range_result = _cuda.driver.cuMemGetAddressRange(
                    _cuda.driver.CUdeviceptr(data_ptr)
                )
                if range_result[0] != _cuda.driver.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(
                        f"cuMemGetAddressRange failed: {range_result[0]} "
                        f"(ptr=0x{data_ptr:x})"
                    )
                _err, chunk_base, chunk_size = range_result
                alloc_offset = data_ptr - int(chunk_base)
                if alloc_offset + nbytes > int(chunk_size):
                    raise RuntimeError(
                        f"tensor bytes [{alloc_offset}, {alloc_offset + nbytes}) "
                        f"exceed the {int(chunk_size)}-byte physical chunk at "
                        f"0x{int(chunk_base):x}: " + _VMM_MULTI_CHUNK_HINT
                    )
            except BaseException:
                if kind == "posix_fd" and isinstance(payload, int):
                    os.close(payload)
                raise
            finally:
                # The exported fd/blob (or, failing that, the exporter's
                # own live mapping) keeps the allocation reachable; the
                # retained reference is ours to drop.
                (release_err,) = _cuda.driver.cuMemRelease(alloc_handle)
                if release_err != _cuda.driver.CUresult.CUDA_SUCCESS:
                    logger.warning(
                        "cuMemRelease of retained handle failed: %s", release_err
                    )

        self._export_id = uuid.uuid4().bytes
        self._kind = kind
        self._fabric_blob = payload if kind == "fabric" else None
        self._fd = payload if kind == "posix_fd" else None
        self._alloc_size = int(chunk_size)
        self._alloc_offset = alloc_offset
        self._nbytes = nbytes

        # DeviceIPCWrapper interface fields. ``handle`` carries the
        # export id so the base-class equality check compares identity;
        # ``to_tensor`` is overridden and never reads it.
        # ``storage_offset`` is 0 because ``data_ptr`` (folded into
        # ``_alloc_offset``) already points at the tensor's first element.
        self.handle = self._export_id
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = 0
        self.device_uuid = self._get_device_uuid(device_index)

        # References this wrapper holds on the imported-mapping registry;
        # producer-side wrappers never import, so receivers start at 0.
        self._opens = 0

    def fd_payload(self) -> "tuple[bytes, int] | None":
        """Return the out-of-band fd payload, if this wrapper carries one.

        Returns:
            ``(export_id, fd)`` on the exporting side of a POSIX-fd
            wrapper, ``None`` otherwise (fabric wrappers, or any
            unpickled copy -- the fd never travels with the pickle).
            The fd remains owned by the wrapper until :meth:`close`.
        """
        fd = getattr(self, "_fd", None)
        if fd is None:
            return None
        return (self._export_id, fd)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via VMM IPC.

        Imports and maps the allocation through the process-wide
        refcounted registry (one mapping per export id, however often
        this wrapper -- or a pickled copy sharing its export id -- is
        materialized). Every call takes one reference; :meth:`close`
        releases all references this wrapper holds.

        Returns:
            The reconstructed tensor, viewing the imported mapping.

        Raises:
            RuntimeError: If the fd resolver is missing (POSIX-fd kind)
                or any driver import/mapping step fails.
        """
        # Third Party
        import cupy

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        with _MAPPINGS_LOCK:
            entry = _MAPPED_VMM_ALLOCATIONS.get(self._export_id)
            if entry is None:
                with torch.cuda.device(device_index):
                    entry = self._import_and_map(device_index)
                _MAPPED_VMM_ALLOCATIONS[self._export_id] = entry
            entry.refcount += 1
            self._opens += 1
            base_ptr = entry.base_ptr

        # Same reconstruction as RawCudaIPCWrapper: flat uint8 CuPy view
        # at the chunk offset, DLPack to torch, then view as the original
        # dtype/shape (uint8 avoids bf16/fp8 dtype-conversion gaps).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(
                base_ptr, self._alloc_offset + self._nbytes, owner=self
            )
            memptr = cupy.cuda.MemoryPointer(mem, self._alloc_offset)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)

    def close(self) -> None:
        """Release everything this wrapper holds, on either side.

        Exporting side: closes the exported fd (if any and not yet
        handed off). Importing side: releases this wrapper's references
        on the imported mapping; when the last reference across all
        holders drops, unmaps the range, frees the VA reservation, and
        releases the imported handle -- returning the exporter's device
        memory once the exporter itself has released the allocation
        (VMM handles are refcounted by the driver, so an unclosed import
        pins the memory even after the exporter dies).

        Idempotent. Failures are logged, not raised: close runs on
        teardown paths where raising would abort cleanup of the
        remaining entries.
        """
        fd = getattr(self, "_fd", None)
        if fd is not None:
            self._fd = None
            try:
                os.close(fd)
            except OSError:
                logger.warning("closing exported VMM fd %d failed", fd)

        opens = getattr(self, "_opens", 0)
        if opens <= 0:
            return
        with _MAPPINGS_LOCK:
            self._opens = 0
            entry = _MAPPED_VMM_ALLOCATIONS.get(self._export_id)
            if entry is None:
                return
            entry.refcount -= opens
            if entry.refcount > 0:
                return
            del _MAPPED_VMM_ALLOCATIONS[self._export_id]
            base_ptr, alloc_handle = entry.base_ptr, entry.alloc_handle
            for what, result in (
                ("cuMemUnmap", _cuda.driver.cuMemUnmap(base_ptr, self._alloc_size)),
                (
                    "cuMemAddressFree",
                    _cuda.driver.cuMemAddressFree(base_ptr, self._alloc_size),
                ),
                ("cuMemRelease", _cuda.driver.cuMemRelease(alloc_handle)),
            ):
                if result[0] != _cuda.driver.CUresult.CUDA_SUCCESS:
                    logger.warning(
                        "%s failed during VMM mapping close: %s (ptr=0x%x)",
                        what,
                        result[0],
                        base_ptr,
                    )

    def __getstate__(self) -> dict[str, object]:
        """Pickle without process-local state (fd, registry references)."""
        state = dict(self.__dict__)
        state["_fd"] = None
        state["_opens"] = 0
        return state

    def _export_allocation(self, alloc_handle: object) -> "tuple[str, int | bytes]":
        """Export ``alloc_handle`` as its allocation's shareable form.

        Args:
            alloc_handle: The retained ``CUmemGenericAllocationHandle``.

        Returns:
            ``("posix_fd", fd)`` or ``("fabric", blob_bytes)``.

        Raises:
            RuntimeError: If the allocation has no exportable handle type
                or the export call fails.
        """
        err, prop = _cuda.driver.cuMemGetAllocationPropertiesFromHandle(alloc_handle)
        if err != _cuda.driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemGetAllocationPropertiesFromHandle failed: {err}")
        requested = int(prop.requestedHandleTypes)

        handle_types = _cuda.driver.CUmemAllocationHandleType
        fd_type = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        fabric_type = handle_types.CU_MEM_HANDLE_TYPE_FABRIC

        if requested & int(fd_type):
            err, fd = _cuda.driver.cuMemExportToShareableHandle(
                alloc_handle, fd_type, 0
            )
            if err != _cuda.driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemExportToShareableHandle(fd) failed: {err}")
            return ("posix_fd", int(fd))

        if requested & int(fabric_type):
            err, fabric = _cuda.driver.cuMemExportToShareableHandle(
                alloc_handle, fabric_type, 0
            )
            if err != _cuda.driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuMemExportToShareableHandle(fabric) failed: {err}"
                )
            return ("fabric", bytes(fabric.data))

        raise RuntimeError(
            f"cannot export VMM allocation (requestedHandleTypes={requested}): "
            + _VMM_NOT_EXPORTABLE_HINT
        )

    def _import_and_map(self, device_index: int) -> _VmmMapping:
        """Import this wrapper's shareable handle and map it.

        Runs under ``_MAPPINGS_LOCK`` with the target device current.
        Unwinds partial work (reservation/mapping/handle) on failure.

        Args:
            device_index: Importer-local device ordinal.

        Returns:
            The fresh registry entry (refcount 0).

        Raises:
            RuntimeError: If any driver step fails.
        """
        driver = _cuda.driver
        if self._kind == "posix_fd":
            fd = _resolve_vmm_fd(self._export_id)
            try:
                err, handle = driver.cuMemImportFromShareableHandle(
                    fd,
                    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                )
            finally:
                os.close(fd)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemImportFromShareableHandle(fd) failed: {err}")
        else:
            # The binding takes the CUmemFabricHandle as a raw 64-byte
            # buffer (buffer protocol), not as the struct wrapper object.
            err, handle = driver.cuMemImportFromShareableHandle(
                self._fabric_blob,
                driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            )
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuMemImportFromShareableHandle(fabric) failed: {err}"
                )

        err, base = driver.cuMemAddressReserve(self._alloc_size, 0, 0, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemRelease(handle)
            raise RuntimeError(f"cuMemAddressReserve failed: {err}")

        (err,) = driver.cuMemMap(base, self._alloc_size, 0, handle, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(base, self._alloc_size)
            driver.cuMemRelease(handle)
            raise RuntimeError(f"cuMemMap failed: {err}")

        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        (err,) = driver.cuMemSetAccess(base, self._alloc_size, [access], 1)
        if err != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemUnmap(base, self._alloc_size)
            driver.cuMemAddressFree(base, self._alloc_size)
            driver.cuMemRelease(handle)
            raise RuntimeError(f"cuMemSetAccess failed: {err}")

        return _VmmMapping(base_ptr=int(base), alloc_handle=handle, refcount=0)
