# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.cuda_ops`` extension.

:class:`CudaDeviceOps` owns CUDA-pointer tensor construction and calls
:meth:`bind_native` in :meth:`ensure_native` to layer the compiled CUDA
extension on top of the torch baseline. If the extension is missing, a warning
is logged and the instance stays on the torch fallback (soft-fail, same as XPU).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import ctypes
import math

# Third Party
import torch

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.common.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.torch_ops._tensor_from_ptr import _get_copy_lib

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
_MEMCPY_H2D = 1
_MEMCPY_D2H = 2
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

    def lmcache_memcpy_async(
        self,
        dest: int | torch.Tensor,
        src: int | torch.Tensor,
        nbytes: int,
        direction: TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Copy bytes between host and device buffers.

        For NVIDIA CUDA pointer copies, split at host registration boundaries
        and enqueue each segment on the current CUDA stream. The caller must
        keep both buffers valid until all segments complete.

        Tensor pairs and ROCm requests delegate to the inherited fallback.
        """
        if torch.version.hip is not None or (
            isinstance(dest, torch.Tensor) or isinstance(src, torch.Tensor)
        ):
            super().lmcache_memcpy_async(
                dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments
            )
            return
        if not isinstance(dest, int) or not isinstance(src, int):
            raise TypeError("CUDA staging operands must both be pointers or tensors")
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if host_buffer_alignments <= 0 or (
            host_buffer_alignments & (host_buffer_alignments - 1)
        ):
            raise ValueError("host_buffer_alignments must be power of two")
        if int(direction) == int(TransferDirection.H2D):
            kind = _MEMCPY_H2D
        elif int(direction) == int(TransferDirection.D2H):
            kind = _MEMCPY_D2H
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        if nbytes == 0:
            return

        copy_lib = _get_copy_lib()
        if copy_lib is None or not hasattr(copy_lib, "cudaMemcpyAsync"):
            raise RuntimeError("CUDA runtime does not provide cudaMemcpyAsync")
        copy_async = copy_lib.cudaMemcpyAsync
        copy_async.restype = ctypes.c_int
        copy_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

        # The default stream can race with staging-slot writes and reuse on
        # PyTorch's non-blocking stream. Each copy must also stay within one
        # lazy allocator host registration.
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        copied = 0
        while copied < nbytes:
            bytes_to_boundary = host_buffer_alignments - (
                (host_buffer_offset + copied) % host_buffer_alignments
            )
            copy_size = min(nbytes - copied, bytes_to_boundary)
            error = copy_async(
                ctypes.c_void_p(dest + copied),
                ctypes.c_void_p(src + copied),
                ctypes.c_size_t(copy_size),
                ctypes.c_int(kind),
                stream,
            )
            if error != 0:
                raise RuntimeError(f"cudaMemcpyAsync failed with error code {error}")
            copied += copy_size

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.cuda_ops as native
        except ImportError:
            logger.warning(
                "lmcache.cuda_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops."
            )
            return
        self.bind_native(native)
