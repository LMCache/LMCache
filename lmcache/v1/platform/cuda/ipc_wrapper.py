# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC wrapper implementations.

:class:`CudaIPCWrapper` shares tensors through PyTorch's storage IPC
(needs a shared ``/dev/shm``); :class:`RawCudaIPCWrapper` shares them
through driver-level CUDA IPC memory handles alone, which work across
fully isolated containers and also cover tensors allocated outside
PyTorch (e.g. TRT-LLM's ``cudaMalloc``'d pool).

``device_type="cuda"`` binds to one of the two via
:attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`:
:class:`CudaIPCWrapper` by default, :class:`RawCudaIPCWrapper` when the
process-global isolated-IPC switch is on (see
``lmcache/v1/platform/isolated_ipc.py``). The multiprocess adapter
dispatches through
:func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`; the TRT-LLM
adapter instantiates :class:`RawCudaIPCWrapper` directly.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cuda.utils import _cuda

_NON_IPC_MEMORY_HINT = (
    "CUDA IPC memory handles only support cudaMalloc-style allocations. "
    "Memory created through the CUDA VMM API cannot be shared this way -- "
    "common sources are PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "and vLLM's sleep mode (CuMemAllocator). Disable those or turn off "
    "isolated IPC for this deployment."
)


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

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC."""
        # Third Party
        import cupy

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        handle = _cuda.runtime.cudaIpcMemHandle_t()
        handle.reserved = self._ipc_handle_reserved
        err, ptr = _cuda.runtime.cudaIpcOpenMemHandle(
            handle, _cuda.runtime.cudaIpcMemLazyEnablePeerAccess
        )
        if err != _cuda.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")

        # Wrap as a flat ``uint8`` CuPy array at the allocation offset,
        # DLPack to torch, then view as the original dtype/shape.
        # ``uint8`` avoids dtype-conversion gaps (bfloat16, fp8 have no
        # direct CuPy/NumPy equivalent without ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(
                int(ptr), self._alloc_offset + self._nbytes, owner=self
            )
            memptr = cupy.cuda.MemoryPointer(mem, self._alloc_offset)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)
