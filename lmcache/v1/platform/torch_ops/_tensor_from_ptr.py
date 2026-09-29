# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional
import ctypes

# Third Party
# Third party
import torch

__all__ = [
    "_get_copy_lib",
    "_tensor_from_ptr",
    "_tensor_from_cpu_ptr",
    "_tensor_from_cuda_ptr",
    "_tensor_from_musa_ptr",
    "_contiguous_element_strides",
    "_copy_bytes_with_tensor",
]

# Cached copy library for lmcache_memcpy_async (lazy-initialized)
_copy_lib_NOT_LOADED = object()
_copy_lib: Optional[ctypes.CDLL] = _copy_lib_NOT_LOADED  # type: ignore


def _get_copy_lib() -> Optional[ctypes.CDLL]:
    """Lazily load and cache the CUDA/ROCm runtime library, or None for CPU fallback."""
    global _copy_lib
    if _copy_lib is _copy_lib_NOT_LOADED:
        # Try to load GPU runtime libraries in priority order: CUDA first, then ROCm
        # TODO: ROCm path to be validated on real device
        for name, fallback in [
            ("cudart", "libcudart.so"),  # NVIDIA CUDA Runtime
            ("amdhip64", "libamdhip64.so"),  # AMD ROCm HIP Runtime
        ]:
            try:
                path = ctypes.util.find_library(name)
                if path:
                    _copy_lib = ctypes.CDLL(path)
                else:
                    _copy_lib = ctypes.CDLL(fallback)
                break  # Successfully loaded, stop trying
            except OSError:
                continue  # Current library not available, try next
        else:
            # All GPU libraries failed to load, fall back to CPU
            _copy_lib = None
    return _copy_lib


def _tensor_from_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Create a tensor view over a raw pointer (zero-copy where possible).

    Supports CPU, CUDA, and MUSA device pointers.

    Args:
        ptr:    Raw memory pointer as int (must be non-zero).
        shape:  Desired tensor shape.
        dtype:  Desired tensor dtype, must match the memory layout.
        device: Where the pointer lives.
                - None / "cpu" / torch.device("cpu")  → CPU pointer
                - "cuda" / "cuda:N" / torch.device("cuda", N) → CUDA pointer
                - "musa" / "musa:N" / torch.device("musa", N) → MUSA pointer
                  If None and ptr looks like a CUDA/MUSA ptr, pass device explicitly.

    Returns:
        A tensor that shares memory with the original pointer.
        For CPU: always zero-copy via ctypes + torch.frombuffer.
        For CUDA: zero-copy via torch._C._construct_storage_from_data_pointer
                  (PyTorch >= 2.0) or __cuda_array_interface__, with a
                  cudaMemcpy D2D fallback.
        For MUSA: a non-owning view created from external device storage.

    Raises:
        ValueError: if ptr is 0.
        RuntimeError: If MUSA cannot construct a non-owning view for ``ptr``.

    Warning:
        The caller is responsible for keeping the underlying memory alive
        for the entire lifetime of the returned tensor.
    """
    if ptr == 0:
        raise ValueError("Pointer must be non-zero")

    # ------------------------------------------------------------------ #
    # Normalise device                                                   #
    # ------------------------------------------------------------------ #
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    assert isinstance(device, torch.device)
    # ------------------------------------------------------------------ #
    # Compute size                                                       #
    # ------------------------------------------------------------------ #
    numel = 1
    for dim in shape:
        numel *= int(dim)
    element_size = torch.empty((), dtype=dtype).element_size()
    total_bytes = numel * element_size

    # ------------------------------------------------------------------ #
    # CPU path                                                           #
    # ------------------------------------------------------------------ #
    if device.type == "cpu":
        return _tensor_from_cpu_ptr(ptr, shape, dtype, numel, total_bytes)

    # ------------------------------------------------------------------ #
    # CUDA path                                                          #
    # ------------------------------------------------------------------ #
    if device.type == "cuda":
        return _tensor_from_cuda_ptr(ptr, shape, dtype, device, numel, total_bytes)

    # ------------------------------------------------------------------ #
    # MUSA path                                                          #
    # ------------------------------------------------------------------ #
    if device.type == "musa":
        return _tensor_from_musa_ptr(ptr, shape, dtype, device, total_bytes)

    raise ValueError(
        f"Unsupported device type: {device.type!r}. Expected 'cpu', 'cuda', or 'musa'."
    )


# ====================================================================== #
#  CPU implementation                                                    #
# ====================================================================== #


def _tensor_from_cpu_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    numel: int,
    total_bytes: int,
) -> torch.Tensor:
    """
    Zero-copy CPU tensor from a raw host pointer via ctypes + torch.frombuffer.

    """
    buffer_type = ctypes.c_uint8 * total_bytes
    buf = buffer_type.from_address(ptr)
    # torch.frombuffer is zero-copy for contiguous byte buffers on CPU.
    return torch.frombuffer(buf, dtype=dtype).view(*shape)


# ====================================================================== #
#  CUDA implementation                                                   #
# ====================================================================== #
def _tensor_from_cuda_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    numel: int,
    total_bytes: int,
) -> torch.Tensor:
    """Zero-copy CUDA tensor from a raw device pointer."""

    try:
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
        is_bf16 = dtype == torch.bfloat16

        # Determine the correct typestr, smuggle bfloat16 as int16
        typestr = "<i2" if is_bf16 else _DTYPE_TO_TYPESTR.get(dtype, "|u1")

        class _CudaArrayWrapper:
            def __init__(self, ptr_int: int, shape_tuple: tuple, type_str: str):
                self.__cuda_array_interface__ = {
                    "data": (ptr_int, False),
                    "shape": shape_tuple,
                    "typestr": type_str,
                    "version": 3,
                }

        t = torch.as_tensor(_CudaArrayWrapper(ptr, (numel,), typestr), device=device)
        if is_bf16:
            t = t.view(torch.bfloat16)

        return t.view(*shape)
    except Exception:
        pass

    # Strategy 2: cudaMemcpy Device-to-Device (Fallback)
    libcudart = _get_copy_lib()
    if libcudart is None:
        raise RuntimeError("Failed to load libcudart/libamdhip")

    cudaMemcpy = libcudart.cudaMemcpy
    cudaMemcpy.restype = ctypes.c_int
    cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _MEMCPY_D2D = 3

    dst = torch.empty(numel, dtype=dtype, device=device)

    err = cudaMemcpy(
        ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(total_bytes),
        ctypes.c_int(_MEMCPY_D2D),
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpy D2D failed with error code {err}.")

    return dst.view(*shape)


# ====================================================================== #
#  MUSA implementation                                                   #
# ====================================================================== #
def _contiguous_element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return contiguous element strides for ``shape``."""
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * int(shape[index + 1])
    return tuple(strides)


def _tensor_from_musa_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    total_bytes: int,
) -> torch.Tensor:
    """Create a non-owning MUSA tensor from a raw device pointer.

    The returned tensor aliases ``ptr``. A copy fallback is intentionally not
    provided because writes through a copied tensor would not update the
    original paged buffer.
    """
    try:
        storage = torch._C._construct_storage_from_data_pointer(
            ptr,
            device,
            total_bytes,
        )
        tensor = torch.empty(0, dtype=dtype, device=storage.device)
        tensor.set_(storage, 0, shape, _contiguous_element_strides(shape))
        return tensor
    except Exception as exc:
        raise RuntimeError(
            "TorchMUSA failed to construct a non-owning tensor from a device pointer"
        ) from exc


def _copy_bytes_with_tensor(dst: int, src: int, num_bytes: int) -> None:
    """Copy raw bytes between pointers using torch tensor semantics.

    Note: This function only works for CPU-accessible memory. For device
    memory (CUDA/XPU), use lmcache_memcpy_async with the appropriate runtime
    library or PyTorch's tensor copy operations.
    """
    if num_bytes <= 0:
        return

    buffer_type = ctypes.c_uint8 * num_bytes
    dst_tensor = torch.frombuffer(buffer_type.from_address(dst), dtype=torch.uint8)
    src_tensor = torch.frombuffer(buffer_type.from_address(src), dtype=torch.uint8)
    dst_tensor.copy_(src_tensor)
