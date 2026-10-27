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
from typing import ClassVar
import ctypes
import math

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.cuda.block_transfer import (
    allocate_pinned_chunk,
    transfer_with_address_kernel,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops._tensor_from_ptr import _get_copy_lib
import lmcache.lmcache_native as lmcache_native

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
