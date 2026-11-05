# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.cuda_ops`` extension.

:class:`CudaDeviceOps` owns CUDA-pointer tensor construction and calls
:meth:`bind_native` in :meth:`ensure_native` to layer the compiled CUDA
extension on top of the torch baseline. If the extension is missing, a warning
is logged and the instance stays on the torch fallback (soft-fail, same as XPU).
It overrides :meth:`multi_layer_block_kv_transfer` to lower tensor operands to
the compiled address-taking kernel.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast
import ctypes
import math

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import synchronize_device
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops._tensor_from_ptr import _get_copy_lib
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache

_DTYPE_TO_TYPESTR = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.int8: "|i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.uint8: "|u1",
    torch.bool: "|b1",
}
_MEMCPY_D2D = 3

#: Chunks one kernel launch can address (``TORCH_CHECK(num_objects <= 4)`` in
#: ``csrc/cuda/mp_mem_kernels.cu``).
MAX_CHUNKS_PER_LAUNCH = 4

logger = init_logger(__name__)


class _CudaArrayWrapper:
    """Expose a raw pointer through the CUDA array interface."""

    def __init__(self, ptr: int, numel: int, typestr: str) -> None:
        self.__cuda_array_interface__: dict[str, object] = {
            "data": (ptr, False),
            "shape": (numel,),
            "typestr": typestr,
            "version": 3,
        }


def _from_cuda_array_interface(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    numel = math.prod(shape)
    is_bfloat16 = dtype == torch.bfloat16
    typestr = "<i2" if is_bfloat16 else _DTYPE_TO_TYPESTR.get(dtype, "|u1")
    tensor = torch.as_tensor(
        _CudaArrayWrapper(ptr, numel, typestr),
        device=device,
    )
    if is_bfloat16:
        tensor = tensor.view(torch.bfloat16)
    return tensor.view(*shape)


def _copy_from_device_pointer(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    copy_lib = _get_copy_lib()
    if copy_lib is None:
        raise RuntimeError("Failed to load libcudart/libamdhip")

    cuda_memcpy = copy_lib.cudaMemcpy
    cuda_memcpy.restype = ctypes.c_int
    cuda_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    numel = math.prod(shape)
    tensor = torch.empty(numel, dtype=dtype, device=device)
    error = cuda_memcpy(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(numel * dtype.itemsize),
        ctypes.c_int(_MEMCPY_D2D),
    )
    if error != 0:
        raise RuntimeError(f"cudaMemcpy D2D failed with error code {error}.")
    return tensor.view(*shape)


def _paged_pointer_table(
    paged_kv: DiscoverableKVCache,
    engine_kv_format: lmcache_native.EngineKVFormat,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the kernel-ordered per-layer base addresses of ``paged_kv``.

    Args:
        paged_kv: Normalized paged KV tensors.
        engine_kv_format: Layout of ``paged_kv``.
        num_layers: Number of layers to address.
        device: Device the kernel runs on.

    Returns:
        A 1-D int64 tensor on ``device`` with the addresses in the order
        ``KVFormatSpec.data_ptrs`` defines for the format.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.device_ops``, which is resolved only once the backends that
    # import this module have been imported.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec

    addresses = get_spec(paged_kv, engine_kv_format).data_ptrs(list(range(num_layers)))
    # Addresses are unsigned; the kernel reads the same bits as int64.
    as_int64 = np.array(addresses, dtype=np.uint64).view(np.int64)
    return torch.from_numpy(as_int64).to(device=device)


def allocate_pinned_chunk(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a host chunk an address-taking kernel can read and write.

    The kernel's GPU threads dereference host chunk addresses directly, which
    only works for page-locked memory.

    Args:
        shape: Canonical chunk shape.
        dtype: KV element dtype.

    Returns:
        An uninitialized pinned CPU tensor of ``shape`` and ``dtype``.
    """
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def transfer_with_address_kernel(
    kernel: Callable[..., None],
    paged_buffer_ptrs_tensor: torch.Tensor | list,
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Serve :meth:`DeviceOps.multi_layer_block_kv_transfer` on an address kernel.

    Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`. Tensor
    operands are lowered to the kernel's address form, in launches of at most
    :data:`MAX_CHUNKS_PER_LAUNCH` chunks. Unpinned host chunks are staged
    through pinned buffers for ``D2H`` and pinned into temporaries for
    ``H2D``; both cases synchronize ``device`` before returning, because the
    kernel reaches them through raw addresses that torch cannot track. Pinned
    chunks keep the asynchronous path. The legacy address form is forwarded
    to the kernel unchanged.

    Args:
        kernel: The compiled ``multi_layer_block_kv_transfer``.
        paged_buffer_ptrs_tensor: Normalized paged KV tensors on ``device``.
        lmcache_objects_ptrs: Host chunk tensors.
        block_ids: Paged-block IDs in chunk order.
        device: Device that owns the paged KV tensors.
        direction: ``D2H`` or ``H2D``.
        shape_desc: Paged-buffer shape descriptor.
        lmcache_chunk_size: Tokens per chunk.
        engine_kv_format: Engine KV layout.
        skip_prefix_n_blocks: Leading blocks to skip across the whole
            transfer; each launch skips the part that falls inside it.
    """
    if not all(isinstance(chunk, torch.Tensor) for chunk in lmcache_objects_ptrs):
        # LMCache-driven callers still pass addresses they batched
        # themselves; forward them untouched until #4465 step 3.10.
        kernel(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        return

    paged_kv = cast("DiscoverableKVCache", paged_buffer_ptrs_tensor)
    chunks = cast("list[torch.Tensor]", lmcache_objects_ptrs)
    device = torch.device(device)
    is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
    unpinned = not all(chunk.is_pinned() for chunk in chunks)
    kernel_chunks = chunks
    if unpinned and is_d2h:
        # Pinning would replace the caller's tensors and break the in-place
        # fill, so the kernel writes pinned staging buffers that are copied
        # back below.
        logger.warning(
            "Unpinned host chunks in a native D2H block transfer (likely "
            "shared memory); staging through pinned buffers adds a host "
            "copy and a device sync."
        )
        kernel_chunks = [torch.empty_like(c, pin_memory=True) for c in chunks]
    elif unpinned:
        logger.warning(
            "Unpinned host chunks in a native H2D block transfer; pinning "
            "temporaries adds a device sync."
        )
        kernel_chunks = [c if c.is_pinned() else c.pin_memory() for c in chunks]

    paged_addresses = _paged_pointer_table(
        paged_kv, engine_kv_format, int(shape_desc.nl), device
    )
    chunk_addresses = [chunk.data_ptr() for chunk in kernel_chunks]
    block_id_tensor = torch.as_tensor(block_ids, dtype=torch.int64, device=device)
    blocks_per_launch = MAX_CHUNKS_PER_LAUNCH * lmcache_chunk_size // int(shape_desc.bs)
    for launch, first in enumerate(
        range(0, len(chunk_addresses), MAX_CHUNKS_PER_LAUNCH)
    ):
        start = launch * blocks_per_launch
        launch_block_ids = block_id_tensor[start : start + blocks_per_launch]
        kernel(
            paged_addresses,
            chunk_addresses[first : first + MAX_CHUNKS_PER_LAUNCH],
            launch_block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            # A skip longer than one launch carries into the next launches.
            max(0, min(skip_prefix_n_blocks - start, len(launch_block_ids))),
        )

    if unpinned:
        # The kernel reaches these buffers through raw addresses that torch's
        # stream tracking cannot see: wait before the host reads the staging
        # copies (D2H) or the pinned temporaries are released (H2D).
        synchronize_device(device)
        if is_d2h:
            for dst, src in zip(chunks, kernel_chunks, strict=True):
                dst.copy_(src)


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    def __init__(self) -> None:
        super().__init__()
        # The compiled address-taking block kernel. It stays None only before
        # ensure_native() runs or when the extension is not installed, and the
        # torch baseline then serves block transfers.
        self._block_transfer_kernel: Callable[..., None] | None = None

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: "torch.dtype",
        device: "torch.device",
    ) -> "torch.Tensor":
        """Create a CUDA tensor from a process-local device pointer."""
        try:
            return _from_cuda_array_interface(ptr, shape, dtype, device)
        except (RuntimeError, TypeError, ValueError):
            return _copy_from_device_pointer(ptr, shape, dtype, device)

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        try:
            # First Party
            import lmcache.cuda_ops as native
        except ImportError:
            # Not installed: stay on the torch baseline without retrying.
            self._native_bound = True
            logger.warning(
                "lmcache.cuda_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops."
            )
            return
        # Every supported build exports the block kernel, so a module without
        # it is a broken install: let the AttributeError propagate rather than
        # serving the torch baseline. ``_native_bound`` stays False, so the
        # next call fails the same way.
        self._block_transfer_kernel = native.multi_layer_block_kv_transfer
        self.bind_native(native, skip={"multi_layer_block_kv_transfer"})
        self._native_bound = True

    def allocate_block_transfer_output(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate pinned host chunks when the compiled kernel will read them.

        Args:
            shape: Canonical chunk shape.
            dtype: KV element dtype.

        Returns:
            A pinned CPU tensor while the kernel is bound, otherwise the
            :class:`DeviceOps` allocation.
        """
        if self._block_transfer_kernel is None:
            return super().allocate_block_transfer_output(shape, dtype)
        return allocate_pinned_chunk(shape, dtype)

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: torch.Tensor | list,
        lmcache_objects_ptrs: list[int] | list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Run the block transfer on the compiled CUDA kernel when it is bound.

        Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`.
        If the extension is not loaded, the torch baseline serves.
        """
        args = (
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        if self._block_transfer_kernel is None:
            super().multi_layer_block_kv_transfer(*args)
            return
        transfer_with_address_kernel(self._block_transfer_kernel, *args)
