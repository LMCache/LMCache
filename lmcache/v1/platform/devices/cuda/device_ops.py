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
from lmcache.logging import init_logger
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
