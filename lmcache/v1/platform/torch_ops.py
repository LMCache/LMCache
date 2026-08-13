# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic torch baseline for the unified ``DeviceOps`` surface.

Migrated verbatim from the former ``lmcache.python_ops_fallback`` module.
Owns the torch/CPU implementation of every device op. Internal to the platform
package -- consumers go through :class:`DeviceOps`, never this module directly.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Optional, Tuple
import ctypes
import ctypes.util
import os
import threading
import warnings

# Third Party
from numba import njit
import numpy as np
import torch

# First Party
from lmcache.lmcache_native import GPUKVFormat  # noqa: F401
from lmcache.lmcache_native import (
    EngineKVFormat,
    TransferDirection,
    is_cross_layer,
    is_kv_list,
    is_layer_list,
    is_mla,
)
from lmcache.logging import init_logger
from lmcache.v1.platform._device_detect import (
    current_device_spec,
    get_torch_device,
)
from lmcache.v1.platform.ops_types import (  # noqa: F401
    BatchStep,
    CBGroupSpec,
    KernelGroupSpec,
    LaunchVar,
    PageBufferShapeDesc,
    StagingCopy,
    set_shape_desc_dtype,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}
_pinned_ptr_registry: dict[int, int] = {}  # ptr -> size, for cudaHostUnregister

# Cached CUDA/ROCm runtime library for copy_into_buffer (lazy-initialized).
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


def _strided_tensor_from_ptr(
    ptr: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Create a zero-copy strided tensor view over a raw pointer.

    Args:
        ptr: Raw memory pointer as an integer.
        shape: Logical tensor shape.
        strides: Physical element strides for each dimension.
        dtype: Element dtype of the pointed-to storage.
        device: Device that owns the pointer.

    Returns:
        A tensor with the requested shape and physical strides.

    Raises:
        ValueError: If shape and strides have different ranks or contain
            unsupported values.
    """
    if len(shape) != len(strides):
        raise ValueError("shape and strides must have the same rank")
    if any(dim <= 0 for dim in shape):
        raise ValueError("strided raw-pointer views require positive dimensions")
    if any(stride < 0 for stride in strides):
        raise ValueError("strided raw-pointer views do not support negative strides")

    storage_numel = 1 + sum(
        (int(dim) - 1) * int(stride) for dim, stride in zip(shape, strides, strict=True)
    )
    storage = _tensor_from_ptr(ptr, (storage_numel,), dtype, device)
    return torch.as_strided(storage, shape, strides)


def _byte_tensor_from_copy_argument(
    value: int | torch.Tensor,
    nbytes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return an aliasing uint8 tensor view for a memcpy argument."""
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous():
            raise ValueError("Tensor memcpy arguments must be contiguous")
        if value.nbytes < nbytes:
            raise ValueError(
                f"Tensor contains {value.nbytes} bytes, but {nbytes} bytes were "
                "requested"
            )
        byte_tensor = value.view(torch.uint8).flatten()[:nbytes]
    else:
        if not isinstance(value, int):
            raise TypeError(
                "Memcpy arguments must be raw integer pointers or torch.Tensor values"
            )
        byte_tensor = _tensor_from_ptr(value, (nbytes,), torch.uint8, device)
        if byte_tensor.data_ptr() != value:
            raise RuntimeError(
                "Failed to create an aliasing tensor view for the raw pointer"
            )
    return byte_tensor


def _device_for_raw_pointer(ptr: int) -> torch.device:
    """Return the active accelerator device owning a PyTorch-allocated pointer."""
    torch_dev, torch_device_type = get_torch_device()
    if torch_device_type not in ("cuda", "musa") or not torch_dev.is_available():
        return torch.device("cpu")

    for segment in torch_dev.memory._snapshot()["segments"]:
        segment_start = int(segment["address"])
        segment_end = segment_start + int(segment["total_size"])
        if segment_start <= ptr < segment_end:
            return torch.device(torch_device_type, int(segment["device"]))
    return torch.device("cpu")


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


# Cuda path goes through func cudaHostAlloc, which is
# already page aligned by CUDA spec. This fallback shim mirrors that
# guarantee so consumers that require page-aligned host buffers, in
# particular the Rust raw-block backend when O_DIRECT is enabled, which
# requires page-aligned buffer pointer
try:
    _PAGE_SIZE = os.sysconf("SC_PAGESIZE")
except (AttributeError, ValueError, OSError):
    _PAGE_SIZE = 4096

logger = init_logger(__name__)

# Cached one-shot decision: pin host buffers only when an accelerator is
# present. Probed lazily on first allocation; if a pinned allocation ever
# fails at runtime we flip this to False permanently and fall back to
# pageable memory for all subsequent allocations.
_use_pinned: Optional[bool] = None


def _alloc_page_aligned_pinned_view(size: int) -> Tuple[torch.Tensor, int]:
    """
    Allocate a pinned CPU buffer whose first usable byte is page-aligned,
    and return a torch view of ``size`` bytes plus its base pointer.

    Internally over-allocates one extra page on a backing tensor, then
    slices the aligned region out. The slice shares storage with the
    backing tensor, so keeping the slice alive keeps the underlying
    memory alive (no need to track the backing tensor separately).
    """
    # Pin the host buffer when an accelerator is present (probed once).
    # StubCPUDevice.is_available returns False on CPU-only hosts.
    torch_dev, torch_device_type = get_torch_device()

    global _use_pinned
    if _use_pinned is None:
        _use_pinned = torch_dev.is_available()
    try:
        backing = torch.empty(
            size + _PAGE_SIZE, dtype=torch.uint8, pin_memory=_use_pinned
        )
    except RuntimeError:
        if not _use_pinned:
            # Pure host allocation failed (e.g. OOM); nothing to fall back to.
            raise
        logger.warning(
            "Pinned host allocation failed on device '%s'; falling back to "
            "unpinned allocation from now on.",
            torch_device_type,
        )
        _use_pinned = False
        backing = torch.empty(size + _PAGE_SIZE, dtype=torch.uint8, pin_memory=False)
    # First-touch initialization on the entire backing region
    backing.fill_(0)
    base = backing.data_ptr()
    # Distance from `base` to the next page boundary (0..PAGE_SIZE-1).
    offset = (-base) % _PAGE_SIZE
    aligned_view = backing[offset : offset + size]
    return aligned_view, aligned_view.data_ptr()


def _format_spec(engine_kv_format: EngineKVFormat) -> "type[KVFormatSpec]":
    """Return the spec class owning *engine_kv_format*'s static layout facts.

    Args:
        engine_kv_format: The format to look up.

    Returns:
        The ``KVFormatSpec`` subclass declared for the format.

    Raises:
        ValueError: If the format has no spec.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.c_ops``, whose shim is installed only once this module (the
    # torch baseline behind it) has been imported.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class

    return get_spec_class(engine_kv_format)


def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    On XPU, uses pin_memory=True (SYCL USM host allocation) for fast transfers.
    Note: NUMA node selection is not supported on non-CUDA."""

    view, aligned_ptr = _alloc_page_aligned_pinned_view(size)
    # view shares storage with its over-allocated backing tensor;
    # holding the view in the registry transitively keeps the underlying
    # memory alive.
    _tensor_registry[aligned_ptr] = view
    return aligned_ptr


def free_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory and returning pointer
    to it. On XPU, uses pin_memory=True (SYCL USM host allocation) for
    fast DMA transfers. On other non-CUDA platforms, pinning is not supported."""

    view, aligned_ptr = _alloc_page_aligned_pinned_view(size)
    _tensor_registry[aligned_ptr] = view
    return aligned_ptr


def free_pinned_ptr(ptr: int) -> None:
    """Non-CUDA equivalent of freeing a previously allocated pinned pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def batched_memcpy(src_ptrs: list[int], dst_ptrs: list[int], sizes: list[int]) -> None:
    """Non-CUDA equivalent of the native batched memcpy helper."""

    if len(src_ptrs) != len(dst_ptrs) or len(src_ptrs) != len(sizes):
        raise ValueError(
            "batched_memcpy expects equally sized src_ptrs, dst_ptrs, and sizes"
        )

    for src_ptr, dst_ptr, size in zip(src_ptrs, dst_ptrs, sizes, strict=True):
        if size <= 0:
            continue
        ctypes.memmove(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src_ptr),
            size,
        )


def alloc_shm_pinned_ptr(size: int, shm_name: str = "") -> int:
    """Non-CUDA equivalent of allocating shared memory pinned pointer.
    Uses multiprocessing.shared_memory for cross-platform POSIX shm.
    Attempts to pin the buffer via cudaHostRegister for async D2H;
    if pinning fails, continues without pinning."""

    # Strip leading '/' for SharedMemory name
    name = shm_name.lstrip("/") if shm_name else None

    # Clean up stale shm segment if it exists
    if name:
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=size)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_buffer(shm.buf)
    ptr = ctypes.addressof(buf)

    # Store references to keep them alive
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    _tensor_registry[ptr] = tensor
    _buf_registry[ptr] = buf
    _shm_registry[ptr] = shm

    # Try to pin the SHM buffer for async D2H copies
    if current_device_spec().pin_memory(ptr, size):
        _pinned_ptr_registry[ptr] = size

    return ptr


def free_shm_pinned_ptr(ptr: int, size: int = 0, shm_name: str = "") -> None:
    """Non-CUDA equivalent of freeing a shared memory
    pinned pointer. Unregisters pinned memory if it was pinned."""

    # Unpin if previously registered
    if ptr in _pinned_ptr_registry:
        current_device_spec().unpin_memory(ptr)
        _pinned_ptr_registry.pop(ptr, None)

    # Release in order: tensor -> ctypes buf -> shm
    _tensor_registry.pop(ptr, None)
    _buf_registry.pop(ptr, None)
    shm = _shm_registry.pop(ptr, None)
    if shm is not None:
        shm.close()
        shm.unlink()


# Hugepage variants: non-CUDA platforms do not support hugepages, so these
# fall back to the same regular pinned allocation.


def alloc_hugepage_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA fallback for alloc_hugepage_pinned_ptr (no hugepage support)."""
    warnings.warn(
        "Hugepages requested but not available on non-CUDA platforms; "
        "falling back to regular allocation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return alloc_pinned_ptr(size, device_id)


def free_hugepage_pinned_ptr(ptr: int, size: int = 0) -> None:
    """Non-CUDA fallback for free_hugepage_pinned_ptr (no hugepage support)."""
    free_pinned_ptr(ptr)


def alloc_hugepage_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA fallback for alloc_hugepage_pinned_numa_ptr (no hugepage support)."""
    warnings.warn(
        "Hugepages requested but not available on non-CUDA platforms; "
        "falling back to regular allocation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return alloc_pinned_numa_ptr(size, numa_id)


def free_hugepage_pinned_numa_ptr(ptr: int, size: int = 0) -> None:
    """Non-CUDA fallback for free_hugepage_pinned_numa_ptr (no hugepage support)."""
    free_pinned_numa_ptr(ptr, size)


def alloc_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating numa memory and returning pointer
    to it. Note: Numa memory is not supported on non-CUDA."""
    return alloc_pinned_numa_ptr(size, numa_id)


def free_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""
    return free_pinned_numa_ptr(ptr, size)


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    block_size: int = 0,
    head_size: int = 0,
    skip_prefix_n_tokens: int = 0,
):
    """
    Fully vectorized Python fallback for multi_layer_kv_transfer.
    Eliminates ALL token- and KV-level Python loops. For MLA layouts,
    a zero ``page_buffer_size`` derives the minimum required size from
    ``slot_mapping``, matching the native kernel behavior.
    """
    if not isinstance(key_value_ptrs, (torch.Tensor, list)):
        raise TypeError(
            f"Expected torch.Tensor or list, but got {type(key_value_ptrs).__name__}"
        )

    # 1. Filter out invalid slots.
    #    valid_mask_kv:  on key_value.device, used to index key_value
    #    valid_slots:    on paged_memory_device, used to index paged_tensor
    kv_device = key_value.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    # Skip the first skip_prefix_n_tokens tokens from transfer.
    # This matches the CUDA kernel semantics where the grid starts at
    # token_id=0 but indexes key_value/slot_mapping at
    # kv_token_id = token_id + skip_prefix_n_tokens.
    # By masking them as invalid, the vectorized indexing via valid_mask_kv
    # naturally skips them while keeping key_value indices aligned.
    if skip_prefix_n_tokens > 0:
        valid_mask_kv[:skip_prefix_n_tokens] = False
    if not valid_mask_kv.any():
        return

    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    # 2. Determine architecture variant and tensor dimensions.
    uses_mla = is_mla(engine_kv_format)
    is_blocked_scale = engine_kv_format == EngineKVFormat.NL_X_NB_BSV_BSS
    is_hnd_split = _is_hnd_split_format(engine_kv_format)
    is_hnd_split_blocks_first = _is_blocks_first_hnd_split_format(engine_kv_format)
    is_fused_packed_hnd = _is_fused_hnd_format(engine_kv_format)
    is_fused_packed_nhd = _is_fused_nhd_format(engine_kv_format)
    is_flash_infer = _is_flash_infer_format(engine_kv_format)

    if uses_mla and page_buffer_size == 0:
        page_buffer_size = int(valid_slots.max().item()) + 1

    num_layers = key_value.size(1)
    hidden_size = key_value.size(3)
    if (is_hnd_split or is_fused_packed_hnd) and head_size <= 0:
        raise ValueError(f"{engine_kv_format} requires a positive head_size")

    # Blocked layouts use these to address a logical token in physical storage.
    if is_flash_infer or is_hnd_split or is_fused_packed_hnd:
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size
    elif is_blocked_scale:
        if block_size <= 0:
            raise ValueError("NL_X_NB_BSV_BSS requires a positive block_size")
        if hidden_size < 4:
            raise ValueError(
                "NL_X_NB_BSV_BSS requires rows with values and a 4-byte scale"
            )

        value_size = hidden_size - 4
        offsets_in_block = valid_slots % block_size
        block_starts = (valid_slots // block_size) * block_size * hidden_size
        value_offsets = (
            block_starts[:, None]
            + (offsets_in_block[:, None] * value_size)
            + torch.arange(value_size, device=paged_memory_device)
        )
        scale_offsets = block_starts[:, None] + (
            block_size * value_size
            + offsets_in_block[:, None] * 4
            + torch.arange(4, device=paged_memory_device)
        )
        blocked_row_offsets = torch.cat((value_offsets, scale_offsets), dim=1)

    # Determine the physical shape of the underlying paged tensor
    # (used when wrapping a raw pointer).
    layer_shape: Tuple[int, ...]

    if is_blocked_scale:
        num_blocks = page_buffer_size // block_size
        layer_shape = (num_blocks, block_size, hidden_size)
    elif uses_mla:
        layer_shape = (page_buffer_size, hidden_size)
    elif is_flash_infer:
        num_blocks = page_buffer_size // block_size
        layer_shape = (num_blocks, 2, block_size, hidden_size)
    elif is_hnd_split:
        num_blocks = page_buffer_size // block_size
        num_heads = hidden_size // head_size
        layer_shape = (
            (num_blocks, 2, num_heads, block_size, head_size)
            if is_hnd_split_blocks_first
            else (2, num_blocks, num_heads, block_size, head_size)
        )
    elif is_fused_packed_hnd:
        num_blocks = page_buffer_size // block_size
        num_heads = hidden_size // (2 * head_size)
        layer_shape = (num_blocks, num_heads, block_size, 2 * head_size)
    elif is_fused_packed_nhd:
        layer_shape = (page_buffer_size, hidden_size)
    else:
        layer_shape = (2, page_buffer_size, hidden_size)

    # 3. Iterate over layers — the only remaining Python-level loop.
    for layer_id in range(num_layers):
        # --- A. Obtain the physical device-memory view for this layer. ---
        if isinstance(key_value_ptrs, list):
            paged_tensor = key_value_ptrs[layer_id]
        else:
            ptr = int(key_value_ptrs[layer_id].item())
            # Convert a raw device pointer into a PyTorch tensor view.
            paged_tensor = _tensor_from_ptr(
                ptr, layer_shape, key_value.dtype, paged_memory_device
            )

        # --- B. Vectorized bulk data transfer. ---
        if is_blocked_scale:
            # Paged layout: [BS x values][BS x 4-byte scales] per block.
            # key_value stores each token as the canonical [values | scale] row.
            paged_flat = paged_tensor.reshape(-1)
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[0, layer_id, valid_mask_kv, :]
                paged_flat[blocked_row_offsets] = lmc_valid.to(paged_tensor.device)
            else:
                gathered = paged_flat[blocked_row_offsets]
                key_value[0, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                )
        elif uses_mla:
            # Paged layout : [page_buffer_size, hidden_size]
            # key_value layout: [1, num_layers, num_tokens, hidden_size]
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[0, layer_id, valid_mask_kv, :]
                paged_tensor.index_copy_(
                    0, valid_slots, lmc_valid.to(paged_tensor.device)
                )
            else:
                gathered = paged_tensor.index_select(0, valid_slots)
                key_value[0, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                )
        elif is_flash_infer:
            # Paged layout : [num_blocks, 2, block_size, hidden_size]
            # key_value layout: [2, num_layers, num_tokens, hidden_size]
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[:, layer_id, valid_mask_kv, :]
                src_data = lmc_valid.transpose(0, 1).to(paged_memory_device)
                # src_data: [num_valid, 2, hidden_size]
                paged_tensor[block_indices, :, block_offsets, :] = src_data
            else:
                gathered = paged_tensor[block_indices, :, block_offsets, :]
                # gathered: [num_valid, 2, hidden_size]
                key_value[:, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                ).transpose(0, 1)
        elif is_hnd_split:
            # Paged layouts: [2, NB, NH, BS, HS] or [NB, 2, NH, BS, HS].
            # Do not reshape a permuted paged tensor: that creates a copy, so
            # H2D writes would not update the underlying paged cache.
            lmc_valid = key_value[:, layer_id, valid_mask_kv, :].reshape(
                2, -1, num_heads, head_size
            )
            if is_hnd_split_blocks_first:
                if int(direction) == int(TransferDirection.H2D):
                    paged_tensor[block_indices, :, :, block_offsets, :] = (
                        lmc_valid.permute(1, 0, 2, 3).to(paged_memory_device)
                    )
                else:
                    gathered = paged_tensor[block_indices, :, :, block_offsets, :]
                    lmc_valid = gathered.permute(1, 0, 2, 3)
            else:
                if int(direction) == int(TransferDirection.H2D):
                    paged_tensor[:, block_indices, :, block_offsets, :] = (
                        lmc_valid.permute(1, 0, 2, 3).to(paged_memory_device)
                    )
                else:
                    gathered = paged_tensor[:, block_indices, :, block_offsets, :]
                    lmc_valid = gathered.permute(1, 0, 2, 3)
            if int(direction) == int(TransferDirection.D2H):
                key_value[:, layer_id, valid_mask_kv, :] = lmc_valid.reshape(
                    2, -1, hidden_size
                ).to(kv_device, non_blocking=False)
        elif is_fused_packed_hnd:
            # Paged layout: [NB, NH, BS, 2 * HS].
            lmc_valid = key_value[0, layer_id, valid_mask_kv, :].reshape(
                -1, num_heads, 2 * head_size
            )
            if int(direction) == int(TransferDirection.H2D):
                paged_tensor[block_indices, :, block_offsets, :] = lmc_valid.to(
                    paged_memory_device
                )
            else:
                gathered = paged_tensor[block_indices, :, block_offsets, :]
                key_value[0, layer_id, valid_mask_kv, :] = gathered.reshape(
                    -1, hidden_size
                ).to(kv_device, non_blocking=False)
        elif is_fused_packed_nhd:
            # Paged layout: [NB, BS, NH, 2 * HS].
            paged_logical = paged_tensor.reshape(page_buffer_size, hidden_size)
            if int(direction) == int(TransferDirection.H2D):
                paged_logical.index_copy_(
                    0,
                    valid_slots,
                    key_value[0, layer_id, valid_mask_kv, :].to(paged_memory_device),
                )
            else:
                key_value[0, layer_id, valid_mask_kv, :] = paged_logical.index_select(
                    0, valid_slots
                ).to(kv_device, non_blocking=False)
        else:
            # Paged layout : [2, page_buffer_size, hidden_size]
            # key_value layout: [2, num_layers, num_tokens, hidden_size]
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[:, layer_id, valid_mask_kv, :]
                paged_tensor.index_copy_(
                    1, valid_slots, lmc_valid.to(paged_memory_device)
                )
            else:
                gathered = paged_tensor.index_select(1, valid_slots)
                key_value[:, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                )


def multi_layer_kv_transfer_unilateral(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
):
    """
    Python fallback for multi_layer_kv_transfer_unilateral

    Handles SGLang MHA format where K and V paged buffers are stored separately:
        ptrs = [K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
        each buffer shape: [page_buffer_size, hidden_size]

    For MLA, delegates to multi_layer_kv_transfer (same as C++ implementation).

    key_value_ptrs:
        - If torch.Tensor: int64 tensor containing raw memory pointers.
        - If list[torch.Tensor]: list of tensor objects.

    key_value layout:
        - Standard: [2, num_layers, num_tokens, hidden_size]
        - MLA:      [1, num_layers, num_tokens, hidden_size]

    direction:
        H2D = LMCache  -> PagedBuffer
        D2H = PagedBuffer -> LMCache
    """
    uses_mla = is_mla(engine_kv_format)

    # MLA case collapses back to multi_layer_kv_transfer
    # (vLLM and SGLang indexing are compatible)
    if uses_mla:
        return multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            direction,
            engine_kv_format,
            0,  # block_size unused for MLA formats
        )
    # ── Non-MLA path: unilateral (separate K/V buffers per layer) ──
    num_layers = key_value.size(1)
    hidden_size = key_value.size(3)
    layer_shape = (page_buffer_size, hidden_size)

    kv_device = key_value.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    if not valid_mask_kv.any():
        return

    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    for layer_id in range(num_layers):
        for kv_idx in range(2):  # 0 = K, 1 = V
            buffer_idx = layer_id + kv_idx * num_layers
            if isinstance(key_value_ptrs, list):
                paged_tensor = key_value_ptrs[buffer_idx]
            else:
                ptr = int(key_value_ptrs[buffer_idx].item())
                paged_tensor = _tensor_from_ptr(
                    ptr, layer_shape, key_value.dtype, paged_memory_device
                )

            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[kv_idx, layer_id, valid_mask_kv, :]
                paged_tensor.index_copy_(
                    0, valid_slots, lmc_valid.to(paged_memory_device)
                )
            else:
                gathered = paged_tensor.index_select(0, valid_slots)
                key_value[kv_idx, layer_id, valid_mask_kv, :] = gathered.to(kv_device)


def _is_hnd_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when a KV format stores heads before block tokens (HND)."""
    return _format_spec(engine_kv_format).is_hnd


def _is_fused_kv_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for formats whose K/V pair is packed in the trailing dim
    (kv_size == 1, shape_desc.hs == 2 * head_size)."""
    return _format_spec(engine_kv_format).is_fused_packed


def _is_two_major_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when the size-2 K/V axis precedes the block axis."""
    return _format_spec(engine_kv_format).is_two_major


def _is_pbs_fused_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when num_blocks and block_size are one folded PBS axis."""
    return _format_spec(engine_kv_format).is_pbs_fused


def _is_hnd_split_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for per-layer HND formats with a separate K/V axis."""
    return (
        is_layer_list(engine_kv_format)
        and _is_hnd_format(engine_kv_format)
        and not _is_fused_kv_format(engine_kv_format)
    )


def _is_blocks_first_hnd_split_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when an HND split format stores blocks before its K/V axis."""
    return _is_hnd_split_format(engine_kv_format) and not _is_two_major_format(
        engine_kv_format
    )


def _is_fused_hnd_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for per-layer HND formats with K/V packed in the content axis."""
    return (
        is_layer_list(engine_kv_format)
        and _is_fused_kv_format(engine_kv_format)
        and _is_hnd_format(engine_kv_format)
    )


def _is_fused_nhd_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for per-layer NHD formats with K/V packed in the content axis."""
    return (
        is_layer_list(engine_kv_format)
        and _is_fused_kv_format(engine_kv_format)
        and not _is_hnd_format(engine_kv_format)
    )


def _is_flash_infer_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for per-layer NHD formats with a blocks-first K/V axis."""
    return (
        is_layer_list(engine_kv_format)
        and not is_mla(engine_kv_format)
        and not _is_fused_kv_format(engine_kv_format)
        and not _is_hnd_format(engine_kv_format)
        and not _is_two_major_format(engine_kv_format)
    )


_ELEMENT_SIZE_TO_DTYPE: dict[int, torch.dtype] = {
    # Maps the byte width of a KV-cache element to a representative torch dtype.
    # Only widths that commonly appear in KV caches are listed; 1-byte entries
    # are treated as uint8 (raw bytes), 2-byte as float16, 4-byte as float32.
    # Note: bfloat16 also has element_size == 2 but cannot be distinguished here;
    # callers that need exact dtype should supply it explicitly.
    1: torch.uint8,
    2: torch.float16,
    4: torch.float32,
}


def _is_ptr_tensor(x: object) -> bool:
    """Return True when *x* is a 1-D pointer tensor (int64 or uint64)."""
    return (
        isinstance(x, torch.Tensor)
        and x.dtype in (torch.int64, torch.uint64)
        and x.ndim == 1
    )


def _per_layer_paged_shape(
    engine_kv_format: EngineKVFormat,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
) -> tuple[int, ...]:
    """Return the logical shape of a single per-layer paged buffer tensor.

    Args:
        engine_kv_format: The format enum that describes how K/V tokens are laid out.
        nb: Number of blocks in the paged buffer (``shape_desc.nb``).
        bs: Tokens per block / block size (``shape_desc.bs``).
        nh: Number of attention heads (``shape_desc.nh``).
        hs: Per-head hidden size (``shape_desc.hs``).

    Returns:
        A tuple representing the shape needed to reconstruct one layer's tensor
        from a raw pointer via :func:`_tensor_from_ptr`.
    """
    if is_mla(engine_kv_format):
        if _is_pbs_fused_format(engine_kv_format):
            return (nb * bs, 1, hs)
        return (nb, bs, hs)
    if _is_fused_hnd_format(engine_kv_format):
        # Blocks-first fused KV (HND): the desc's hs is the packed
        # 2 * head_size, so each layer is the raw [NB, NH, BS, 2 * HS].
        return (nb, nh, bs, hs)
    if _is_fused_nhd_format(engine_kv_format):
        # Blocks-first fused KV (NHD): tokens before heads.
        return (nb, bs, nh, hs)
    if _is_two_major_format(engine_kv_format):
        if _is_hnd_format(engine_kv_format):
            return (2, nb, nh, bs, hs)
        return (2, nb, bs, nh, hs)
    if _is_hnd_format(engine_kv_format):
        return (nb, 2, nh, bs, hs)
    # Covers NL_X_NB_TWO_BS_NH_HS and future NHD variants.
    return (nb, 2, bs, nh, hs)


def _infer_kv_dtype(
    paged_buffer_ptrs_tensor: object,
    lmcache_objects_ptrs: object,
    shape_desc: "PageBufferShapeDesc",
) -> torch.dtype:
    """Infer the KV element dtype from whichever inputs carry it.

    Inference order (first match wins):
    1. ``shape_desc.dtype`` — authoritative when set (requires the
       ``set_shape_desc_dtype`` helper from PR #3514; correctly distinguishes
       float16 vs bfloat16 which share ``element_size == 2``).
    2. ``paged_buffer_ptrs_tensor`` — if it is a non-pointer tensor or a list
       of tensors (including nested SGLang MHA lists), the dtype of the first
       tensor is used.
    3. ``lmcache_objects_ptrs`` — if it is a list of tensors, the dtype of the
       first chunk tensor is used.
    4. ``shape_desc.element_size`` — looked up in :data:`_ELEMENT_SIZE_TO_DTYPE`
       (ambiguous for 2-byte types; kept only as last-resort fallback).
    5. ``torch.bfloat16`` — silent default when no other source is available.
    """
    # Prefer shape_desc.dtype — it is exact and avoids the element_size ambiguity.
    if shape_desc is not None:
        sd_dtype = getattr(shape_desc, "dtype", None)
        if sd_dtype is not None:
            return sd_dtype
    if isinstance(paged_buffer_ptrs_tensor, list) and paged_buffer_ptrs_tensor:
        first = paged_buffer_ptrs_tensor[0]
        if isinstance(first, list) and first and isinstance(first[0], torch.Tensor):
            return first[0].dtype
        if isinstance(first, torch.Tensor):
            return first.dtype
    if isinstance(paged_buffer_ptrs_tensor, torch.Tensor) and not _is_ptr_tensor(
        paged_buffer_ptrs_tensor
    ):
        return paged_buffer_ptrs_tensor.dtype
    if isinstance(lmcache_objects_ptrs, list) and lmcache_objects_ptrs:
        if isinstance(lmcache_objects_ptrs[0], torch.Tensor):
            return lmcache_objects_ptrs[0].dtype
    if shape_desc is not None and shape_desc.element_size > 0:
        dtype = _ELEMENT_SIZE_TO_DTYPE.get(shape_desc.element_size)
        if dtype is None:
            raise ValueError(
                f"Unsupported element_size {shape_desc.element_size!r} in "
                "shape_desc; cannot infer KV dtype. "
                f"Supported sizes: {sorted(_ELEMENT_SIZE_TO_DTYPE)}"
            )
        return dtype
    return torch.bfloat16


def _normalize_paged_layers(
    paged_buffer_ptrs_tensor: "torch.Tensor | list",
    engine_kv_format: EngineKVFormat,
    shape_desc: "PageBufferShapeDesc | None" = None,
    device: "torch.device | str | None" = None,
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]":
    """Normalize paged buffer input based on GPU KV format.

    Accepts either tensor-form inputs (list / Tensor) or a 1-D pointer tensor
    (int64 / uint64).  When a pointer tensor is provided *shape_desc*, *device*,
    and *dtype* must be supplied so the tensors can be reconstructed via
    :func:`_tensor_from_ptr`.

    Returns:
        - Single ``torch.Tensor`` for cross-layer formats.
        - ``list[list[torch.Tensor]]`` (2 x NL) for SGLang MHA formats.
        - ``list[torch.Tensor]`` (per-layer) for all other formats.
    """
    if is_cross_layer(engine_kv_format):
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor):
            if _is_ptr_tensor(paged_buffer_ptrs_tensor):
                # 1-D pointer tensor with a single entry → reconstruct full tensor.
                if shape_desc is None or device is None or dtype is None:
                    raise ValueError(
                        "_normalize_paged_layers: shape_desc, device, and dtype are "
                        "required when paged_buffer_ptrs_tensor is a pointer tensor"
                    )
                nb = int(shape_desc.nb)
                nl = int(shape_desc.nl)
                bs = int(shape_desc.bs)
                nh = int(shape_desc.nh)
                hs = int(shape_desc.hs)
                if _is_hnd_format(engine_kv_format):
                    shape: tuple[int, ...] = (nb, nl, 2, nh, bs, hs)
                else:
                    shape = (nb, nl, 2, bs, nh, hs)
                ptr = int(paged_buffer_ptrs_tensor[0].item())
                return _tensor_from_ptr(ptr, shape, dtype, device)
            return paged_buffer_ptrs_tensor
        raise TypeError(
            "Cross-layer formats require a single torch.Tensor input; "
            "got: " + type(paged_buffer_ptrs_tensor).__name__
        )
    if is_kv_list(engine_kv_format):
        if _is_ptr_tensor(paged_buffer_ptrs_tensor):
            # 1-D pointer tensor [K_L0,...,K_LN-1, V_L0,...,V_LN-1] → nested list.
            if shape_desc is None or device is None or dtype is None:
                raise ValueError(
                    "_normalize_paged_layers: shape_desc, device, and dtype are "
                    "required when paged_buffer_ptrs_tensor is a pointer tensor"
                )
            nb = int(shape_desc.nb)
            nl = int(shape_desc.nl)
            bs = int(shape_desc.bs)
            nh = int(shape_desc.nh)
            hs = int(shape_desc.hs)
            is_flat = _is_pbs_fused_format(engine_kv_format)
            per_layer_shape: tuple[int, ...] = (
                (nb * bs, nh, hs) if is_flat else (nb, bs, nh, hs)
            )
            ptrs = [int(p.item()) for p in paged_buffer_ptrs_tensor]
            k_tensors = [
                _tensor_from_ptr(ptrs[i], per_layer_shape, dtype, device)
                for i in range(nl)
            ]
            v_tensors = [
                _tensor_from_ptr(ptrs[nl + i], per_layer_shape, dtype, device)
                for i in range(nl)
            ]
            return [k_tensors, v_tensors]
        if isinstance(paged_buffer_ptrs_tensor, list):
            # Already nested [[K tensors], [V tensors]]
            if (
                len(paged_buffer_ptrs_tensor) == 2
                and isinstance(paged_buffer_ptrs_tensor[0], list)
                and all(
                    isinstance(t, torch.Tensor)
                    for group in paged_buffer_ptrs_tensor
                    for t in group
                )
            ):
                return paged_buffer_ptrs_tensor
            # Flat list [K_L0, ..., K_LN-1, V_L0, ..., V_LN-1]
            if all(isinstance(t, torch.Tensor) for t in paged_buffer_ptrs_tensor):
                if len(paged_buffer_ptrs_tensor) % 2 != 0:
                    raise ValueError(
                        "Flat SGLang MHA list must have even length (2*NL)"
                    )
                half = len(paged_buffer_ptrs_tensor) // 2
                return [
                    paged_buffer_ptrs_tensor[:half],
                    paged_buffer_ptrs_tensor[half:],
                ]
        raise TypeError(
            "SGLang MHA formats require a list[list[torch.Tensor]], a flat "
            "list[torch.Tensor] (2*NL entries), or a 1-D pointer tensor; "
            "got: " + type(paged_buffer_ptrs_tensor).__name__
        )
    # Per-layer formats
    if _is_ptr_tensor(paged_buffer_ptrs_tensor):
        # 1-D pointer tensor [ptr_L0, ..., ptr_LN-1] → list of per-layer tensors.
        if shape_desc is None or device is None or dtype is None:
            raise ValueError(
                "_normalize_paged_layers: shape_desc, device, and dtype are "
                "required when paged_buffer_ptrs_tensor is a pointer tensor"
            )
        nb = int(shape_desc.nb)
        bs = int(shape_desc.bs)
        nh = int(shape_desc.nh)
        hs = int(shape_desc.hs)
        per_shape = _per_layer_paged_shape(engine_kv_format, nb, bs, nh, hs)
        block_stride = int(shape_desc.block_stride_elems)
        uses_padded_mla = (
            is_mla(engine_kv_format)
            and not _is_pbs_fused_format(engine_kv_format)
            and block_stride > 0
        )
        if uses_padded_mla:
            tight_block_size = bs * hs
            if block_stride < tight_block_size:
                raise ValueError(
                    "block_stride_elems must be at least bs * hs for MLA formats"
                )
            per_strides = (block_stride, hs, 1)
            return [
                _strided_tensor_from_ptr(
                    int(p.item()), per_shape, per_strides, dtype, device
                )
                for p in paged_buffer_ptrs_tensor
            ]
        return [
            _tensor_from_ptr(int(p.item()), per_shape, dtype, device)
            for p in paged_buffer_ptrs_tensor
        ]
    if isinstance(paged_buffer_ptrs_tensor, list):
        if not all(isinstance(t, torch.Tensor) for t in paged_buffer_ptrs_tensor):
            raise TypeError(
                "paged_buffer_ptrs_tensor list must contain torch.Tensor entries"
            )
        return paged_buffer_ptrs_tensor
    raise TypeError(
        "paged_buffer_ptrs_tensor must be a list[torch.Tensor] or 1-D pointer tensor; "
        "got: " + type(paged_buffer_ptrs_tensor).__name__
    )


def _normalize_lmcache_objects(
    lmcache_objects_ptrs: "list[int] | list[torch.Tensor]",
    shape_desc: "PageBufferShapeDesc | None" = None,
    lmcache_chunk_size: "int | None" = None,
    engine_kv_format: "EngineKVFormat | None" = None,
    dtype: "torch.dtype | None" = None,
) -> list[torch.Tensor]:
    """Normalize LMCache object inputs to chunk tensors.

    Accepts either a list of chunk tensors or a ``list[int]`` of raw CPU/CUDA
    pointers.
    When a pointer list is provided *shape_desc*, *lmcache_chunk_size*,
    *engine_kv_format*, and *dtype* must be supplied so the tensors can be
    reconstructed via :func:`_tensor_from_ptr`.
    """
    if not isinstance(lmcache_objects_ptrs, list):
        raise TypeError(
            "lmcache_objects_ptrs must be a list[torch.Tensor] or list[int]; "
            "got: " + type(lmcache_objects_ptrs).__name__
        )
    if not lmcache_objects_ptrs:
        return []
    if isinstance(lmcache_objects_ptrs[0], torch.Tensor):
        return lmcache_objects_ptrs  # type: ignore[return-value]
    if isinstance(lmcache_objects_ptrs[0], int):
        # Pointer mode: reconstruct each chunk on its owning device.
        if (
            shape_desc is None
            or lmcache_chunk_size is None
            or engine_kv_format is None
            or dtype is None
        ):
            raise ValueError(
                "_normalize_lmcache_objects: shape_desc, lmcache_chunk_size, "
                "engine_kv_format, and dtype are required when lmcache_objects_ptrs "
                "contains raw int pointers"
            )
        nl = int(shape_desc.nl)
        nh = int(shape_desc.nh)
        hs = int(shape_desc.hs)
        chunk_tokens = lmcache_chunk_size
        if is_mla(engine_kv_format):
            chunk_shape: tuple[int, ...] = (nl, chunk_tokens, hs)
        elif _is_fused_kv_format(engine_kv_format):
            # Single plane: hs is the packed 2 * head_size.
            chunk_shape = (nl, chunk_tokens, nh * hs)
        else:
            chunk_shape = (2, nl, chunk_tokens, nh * hs)
        return [
            _tensor_from_ptr(ptr, chunk_shape, dtype, _device_for_raw_pointer(ptr))
            for ptr in lmcache_objects_ptrs
        ]
    raise TypeError(
        "lmcache_objects_ptrs must be a list[torch.Tensor] or list[int]; "
        "got list containing: " + type(lmcache_objects_ptrs[0]).__name__
    )


def _to_block_id_list(block_ids: torch.Tensor | list[int]) -> list[int]:
    """Convert block IDs from tensor/list form into a Python ``list[int]``."""
    if isinstance(block_ids, torch.Tensor):
        return [int(x) for x in block_ids.to(dtype=torch.int64).cpu().tolist()]
    if isinstance(block_ids, list):
        return [int(x) for x in block_ids]
    raise TypeError("block_ids must be a torch.Tensor or list[int]")


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: "torch.Tensor | list",
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Python fallback implementation of block-based multi-layer KV transfer.

    Signature intentionally mirrors the C++ binding so callers can invoke
    ``lmcache.c_ops.multi_layer_block_kv_transfer`` uniformly on native and
    fallback backends.

    Args:
        paged_buffer_ptrs_tensor: Paged buffer pointers or tensors.
        lmcache_objects_ptrs: LMCache object pointers or chunk tensors.
        block_ids: Ordered engine block IDs for the transfer.
        device: Target device for the transfer.
        direction: Transfer direction (H2D or D2H).
        shape_desc: Shape descriptor of the page buffer.
        lmcache_chunk_size: Chunk size of LMCache objects.
        engine_kv_format: GPU KV cache format.
        skip_prefix_n_blocks: Number of leading blocks to skip.

    Returns:
        None

    Raises:
        ValueError: If chunk size is invalid, or transfer direction is unsupported.
        TypeError: If input types do not match expected types.
    """
    if lmcache_chunk_size <= 0:
        raise ValueError("lmcache_chunk_size must be positive")
    if int(shape_desc.bs) <= 0 or lmcache_chunk_size % int(shape_desc.bs) != 0:
        raise ValueError(
            "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
        )
    if skip_prefix_n_blocks < 0:
        raise ValueError("skip_prefix_n_blocks must be >= 0")

    is_d2h = int(direction) == int(TransferDirection.D2H)
    is_h2d = int(direction) == int(TransferDirection.H2D)
    if not (is_d2h or is_h2d):
        raise ValueError(f"Unsupported transfer direction: {direction!r}")

    kv_dtype = _infer_kv_dtype(
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs, shape_desc
    )
    normalized = _normalize_paged_layers(
        paged_buffer_ptrs_tensor,
        engine_kv_format,
        shape_desc=shape_desc,
        device=device,
        dtype=kv_dtype,
    )
    object_tensors = _normalize_lmcache_objects(
        lmcache_objects_ptrs,
        shape_desc=shape_desc,
        lmcache_chunk_size=lmcache_chunk_size,
        engine_kv_format=engine_kv_format,
        dtype=kv_dtype,
    )
    n_block_ids = (
        int(block_ids.numel())
        if isinstance(block_ids, torch.Tensor)
        else len(block_ids)
    )
    blocks_per_object = lmcache_chunk_size // int(shape_desc.bs)
    block_size = int(shape_desc.bs)

    if is_cross_layer(engine_kv_format):
        _transfer_cross_layer(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif is_kv_list(engine_kv_format):
        _transfer_sglang_mha(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif engine_kv_format == EngineKVFormat.NL_X_NB_BSV_BSS:
        _transfer_per_layer_blocked_scale(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif is_mla(engine_kv_format):
        _transfer_per_layer_mla(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif _is_fused_kv_format(engine_kv_format):
        # Before the HND branch: the fused formats are HND/NHD too, but their
        # packed K/V axis needs the single-plane path.
        _transfer_per_layer_fused(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif _is_hnd_format(engine_kv_format):
        _transfer_per_layer_hnd(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    else:
        _transfer_per_layer_nhd(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )


def _pointer_count_for_format(engine_kv_format: EngineKVFormat, num_layers: int) -> int:
    """Return the number of paged-buffer addresses used by a KV format."""
    if is_cross_layer(engine_kv_format):
        return 1
    if is_kv_list(engine_kv_format):
        return 2 * num_layers
    return num_layers


def _object_buffer_shape(
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
) -> tuple[int, ...]:
    """Return the temporary object-buffer shape for a block-transfer plan."""
    if is_mla(engine_kv_format):
        return (int(shape_desc.nl), lmcache_chunk_size, int(shape_desc.hs))
    if _is_fused_kv_format(engine_kv_format):
        return (
            int(shape_desc.nl),
            lmcache_chunk_size,
            int(shape_desc.nh) * int(shape_desc.hs),
        )
    return (
        2,
        int(shape_desc.nl),
        lmcache_chunk_size,
        int(shape_desc.nh) * int(shape_desc.hs),
    )


def _plan_dtype(
    key_scalar_type: int,
    element_size: int,
    shape_desc: PageBufferShapeDesc | None = None,
) -> torch.dtype:
    """Resolve a plan scalar type, preserving bfloat16 when it is specified."""
    scalar_types = {
        0: torch.uint8,
        1: torch.int8,
        2: torch.int16,
        3: torch.int32,
        4: torch.int64,
        5: torch.float16,
        6: torch.float32,
        7: torch.float64,
        11: torch.bool,
        15: torch.bfloat16,
    }
    dtype = scalar_types.get(key_scalar_type)
    if dtype is not None:
        return dtype
    if shape_desc is not None:
        return _infer_kv_dtype([], [], shape_desc)
    dtype = _ELEMENT_SIZE_TO_DTYPE.get(element_size)
    if dtype is None:
        raise ValueError(
            f"Unsupported element_size {element_size}; cannot infer plan dtype"
        )
    return dtype


def _plan_tensor(
    address: int | torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a plan-backed tensor, preserving a supplied tensor's ownership."""
    if isinstance(address, torch.Tensor):
        required_numel = int(np.prod(shape))
        if address.numel() < required_numel:
            raise ValueError(
                f"Plan tensor has {address.numel()} elements; expected {required_numel}"
            )
        if address.dtype != dtype:
            raise ValueError(f"Plan tensor has dtype {address.dtype}; expected {dtype}")
        return address.to(device).reshape(-1)[:required_numel].reshape(shape)
    return _tensor_from_ptr(address, shape, dtype, device)


def _plan_pointer_tensor(
    pointers: int | torch.Tensor | list[int | torch.Tensor],
    count: int,
    device: torch.device,
) -> torch.Tensor | list[torch.Tensor]:
    """Return paged-buffer pointers while preserving tensor-list plans."""
    if isinstance(pointers, list):
        if len(pointers) != count:
            raise ValueError(
                f"Plan has {len(pointers)} paged buffers; expected {count}"
            )
        if not all(isinstance(pointer, torch.Tensor) for pointer in pointers):
            raise TypeError("Tensor-list plan pointers must all be torch.Tensor")
        return [pointer.to(device) for pointer in pointers]
    return _plan_tensor(pointers, (count,), torch.int64, device)


def _validate_staging_copy(copy: StagingCopy) -> None:
    """Validate one raw-pointer staging copy before executing it."""
    if copy._nbytes < 0:
        raise ValueError("StagingCopy.nbytes must be non-negative")
    if copy._nbytes > 0 and (
        (isinstance(copy._dest, int) and copy._dest == 0)
        or (isinstance(copy._src, int) and copy._src == 0)
    ):
        raise ValueError("StagingCopy source and destination must be non-zero")


def _execute_staging_copies(
    staging: list[StagingCopy],
    direction: TransferDirection,
    host_buffer_alignment: int,
) -> None:
    """Execute staging copies serially in their declared order."""
    for copy in staging:
        if not isinstance(copy, StagingCopy):
            raise TypeError("BatchStep.staging entries must be StagingCopy instances")
        _validate_staging_copy(copy)
        lmcache_memcpy_async(
            copy._dest,
            copy._src,
            copy._nbytes,
            direction,
            copy._host_offset,
            host_buffer_alignment,
        )


def execute_object_group_transfer(
    direction: TransferDirection,
    device: torch.device | str,
    host_buffer_alignment: int,
    kernel_group_specs: list[KernelGroupSpec],
    batch_steps: list[BatchStep],
) -> None:
    """Execute an object-group transfer plan using serial torch operations.

    H2D stages each batch before its block transfers; D2H stages it after.
    This intentionally preserves the native executor's step ordering while
    trading stream overlap for portability.

    Args:
        direction: H2D for retrieval or D2H for storage.
        device: Device containing paged and temporary KV buffers.
        host_buffer_alignment: Power-of-two alignment of host staging buffers.
        kernel_group_specs: Invariant inputs for each kernel group.
        batch_steps: Ordered staging and launch work.

    Raises:
        TypeError: If a plan contains an unsupported descriptor.
        ValueError: If a plan references invalid groups, buffers, or ranges.
    """
    if int(direction) not in (int(TransferDirection.H2D), int(TransferDirection.D2H)):
        raise ValueError(f"Unsupported transfer direction: {direction!r}")
    if host_buffer_alignment <= 0 or (
        host_buffer_alignment & (host_buffer_alignment - 1) != 0
    ):
        raise ValueError("host_buffer_alignment must be power of two")

    plan_device = torch.device(device)
    is_h2d = int(direction) == int(TransferDirection.H2D)
    for step in batch_steps:
        if not isinstance(step, BatchStep):
            raise TypeError("batch_steps entries must be BatchStep instances")
        if is_h2d:
            _execute_staging_copies(step._staging, direction, host_buffer_alignment)
        for launch in step._launches:
            if not isinstance(launch, LaunchVar):
                raise TypeError(
                    "BatchStep.launches entries must be LaunchVar instances"
                )
            if launch._group_idx < 0 or launch._group_idx >= len(kernel_group_specs):
                raise ValueError(
                    f"LaunchVar.group_idx out of range: {launch._group_idx}"
                )
            group = kernel_group_specs[launch._group_idx]
            if not isinstance(group, KernelGroupSpec):
                raise TypeError("kernel_group_specs entries must be KernelGroupSpec")
            if not 1 <= launch._num_objects <= len(group._lmcache_objects_ptrs):
                raise ValueError(
                    "LaunchVar.num_objects exceeds available temporary buffers"
                )
            if launch._block_ids_offset < 0 or launch._total_blocks < 0:
                raise ValueError("LaunchVar block range must be non-negative")
            block_end = launch._block_ids_offset + launch._total_blocks
            if block_end > group._block_ids_capacity:
                raise ValueError(
                    "LaunchVar block-ID range exceeds KernelGroupSpec capacity"
                )

            dtype = _plan_dtype(
                -1, int(group._shape_desc.element_size), group._shape_desc
            )
            ptr_count = _pointer_count_for_format(
                group._engine_kv_format, int(group._shape_desc.nl)
            )
            paged_ptrs = _plan_pointer_tensor(
                group._paged_buffer_ptrs,
                ptr_count,
                plan_device,
            )
            if isinstance(group._block_ids_base, torch.Tensor):
                block_ids = group._block_ids_base.flatten()[
                    launch._block_ids_offset : block_end
                ].to(plan_device)
            else:
                block_ids = _plan_tensor(
                    group._block_ids_base
                    + launch._block_ids_offset * torch.int64.itemsize,
                    (launch._total_blocks,),
                    torch.int64,
                    plan_device,
                )
            object_shape = _object_buffer_shape(
                group._shape_desc,
                group._lmcache_chunk_size,
                group._engine_kv_format,
            )
            object_buffers = [
                _plan_tensor(ptr, object_shape, dtype, plan_device)
                for ptr in group._lmcache_objects_ptrs[: launch._num_objects]
            ]
            multi_layer_block_kv_transfer(
                paged_ptrs,
                object_buffers,
                block_ids,
                plan_device,
                direction,
                group._shape_desc,
                group._lmcache_chunk_size,
                group._engine_kv_format,
                launch._skip_prefix_n_blocks,
            )
        if not is_h2d:
            _execute_staging_copies(step._staging, direction, host_buffer_alignment)


def _cb_temp_buffer(
    group: CBGroupSpec, slot_idx: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Reconstruct one CacheBlend temporary KV slot from its raw address."""
    if slot_idx < 0 or slot_idx >= len(group._temp_buffer_ptrs):
        raise ValueError(f"CB slot index out of range: {slot_idx}")
    kv_size = (
        1
        if is_mla(group._engine_kv_format)
        or _is_fused_kv_format(group._engine_kv_format)
        else 2
    )
    return _plan_tensor(
        group._temp_buffer_ptrs[slot_idx],
        (kv_size, group._num_layers, group._slot_tokens, group._hidden_elems),
        dtype,
        device,
    )


def _validate_plan_table(table: np.ndarray, name: str, width: int) -> np.ndarray:
    """Convert a plan table to int64 and validate its two-dimensional shape."""
    converted = np.asarray(table, dtype=np.int64)
    if converted.ndim != 2 or converted.shape[1] != width:
        raise ValueError(f"{name} table must be [n, {width}]")
    return converted


def execute_cb_retrieve_plan_flat(
    device: torch.device | str,
    host_buffer_alignment: int,
    group_specs: list[CBGroupSpec],
    staging: np.ndarray | list[StagingCopy],
    ropes: np.ndarray,
    scatters: np.ndarray,
    step_offsets: np.ndarray,
) -> None:
    """Execute a flattened CacheBlend retrieve plan with torch fallbacks.

    Each step stages host data, re-applies RoPE to shifted K values, then
    scatters the temporary KV slots. The serial ordering matches the native
    executor even though it does not overlap copy and compute streams.

    Args:
        device: Device containing the paged and temporary KV buffers.
        host_buffer_alignment: Power-of-two alignment of host staging buffers.
        group_specs: Per-kernel-group transfer geometry.
        staging: ``[n, 4]`` rows of ``dest, src, nbytes, host_offset``.
        ropes: ``[n, 4]`` rows of ``group_idx, slot_idx, old_st, cur_st``.
        scatters: ``[n, 4]`` rows of
            ``group_idx, slot_idx, slot_mapping_offset, n_tok``.
        step_offsets: Cumulative ``[n_steps, 3]`` table end offsets.

    Raises:
        ValueError: If tables are malformed or a plan range is invalid.
    """
    if host_buffer_alignment <= 0 or (
        host_buffer_alignment & (host_buffer_alignment - 1) != 0
    ):
        raise ValueError("host_buffer_alignment must be power of two")
    if isinstance(staging, list):
        if not all(isinstance(copy, StagingCopy) for copy in staging):
            raise TypeError("Tensor-backed staging must contain StagingCopy values")
        staging_rows = staging
    else:
        staging_table = _validate_plan_table(staging, "staging", 4)
        staging_rows = [
            StagingCopy(int(dest), int(src), int(nbytes), int(host_offset))
            for dest, src, nbytes, host_offset in staging_table
        ]
    ropes_table = _validate_plan_table(ropes, "ropes", 4)
    scatters_table = _validate_plan_table(scatters, "scatters", 4)
    offsets_table = _validate_plan_table(step_offsets, "step_offsets", 3)

    previous = np.zeros(3, dtype=np.int64)
    limits = np.array(
        [len(staging_rows), len(ropes_table), len(scatters_table)], dtype=np.int64
    )
    for step_idx, current in enumerate(offsets_table):
        if np.any(current < previous) or np.any(current > limits):
            raise ValueError(
                f"step_offsets not monotone or out of range at step {step_idx}"
            )
        previous = current

    plan_device = torch.device(device)
    starts = np.zeros(3, dtype=np.int64)
    for ends in offsets_table:
        _execute_staging_copies(
            staging_rows[int(starts[0]) : int(ends[0])],
            TransferDirection.H2D,
            host_buffer_alignment,
        )

        step_ropes = ropes_table[starts[1] : ends[1]]
        for group_idx, group in enumerate(group_specs):
            if not isinstance(group, CBGroupSpec):
                raise TypeError("group_specs entries must be CBGroupSpec instances")
            if isinstance(group._cos_sin_cache, int) and group._cos_sin_cache == 0:
                continue
            dtype = _plan_dtype(group._key_scalar_type, group._element_size)
            group_rows = step_ropes[step_ropes[:, 0] == group_idx]
            for _, slot_idx, old_st, cur_st in group_rows:
                if old_st < 0 or cur_st < 0:
                    raise ValueError("CB RoPE positions must be non-negative")
                slot = _cb_temp_buffer(group, int(slot_idx), dtype, plan_device)
                if group._rope_head_stride <= 0 or (
                    group._hidden_elems
                    != group._rope_num_kv_heads * group._rope_head_stride
                ):
                    raise ValueError("CBGroupSpec has incompatible RoPE geometry")
                offset_elems = group._rope_base_offset // group._element_size
                if (
                    group._rope_base_offset % group._element_size != 0
                    or offset_elems < 0
                    or offset_elems >= group._rope_head_stride
                ):
                    raise ValueError("CBGroupSpec.rope_base_offset is invalid")
                positions = torch.arange(
                    group._slot_tokens, dtype=torch.long, device=plan_device
                ).repeat(group._num_layers)
                cache_rows = int(max(old_st, cur_st)) + group._slot_tokens
                cos_sin_cache = _plan_tensor(
                    group._cos_sin_cache,
                    (cache_rows, group._rot_dim),
                    dtype,
                    plan_device,
                )
                keys = slot[0].reshape(
                    group._num_layers * group._slot_tokens,
                    group._rope_num_kv_heads,
                    group._rope_head_stride,
                )
                rotary_embedding_k_fused_strided(
                    positions + int(old_st),
                    positions + int(cur_st),
                    keys[..., offset_elems:],
                    group._rope_head_stride - offset_elems,
                    group._rope_head_stride,
                    cos_sin_cache,
                    group._is_neox,
                )

        step_scatters = scatters_table[starts[2] : ends[2]]
        for group_idx, group in enumerate(group_specs):
            if not isinstance(group, CBGroupSpec):
                raise TypeError("group_specs entries must be CBGroupSpec instances")
            dtype = _plan_dtype(group._key_scalar_type, group._element_size)
            group_rows = step_scatters[step_scatters[:, 0] == group_idx]
            for _, slot_idx, mapping_offset, n_tok in group_rows:
                if n_tok < 0 or n_tok > group._slot_tokens:
                    raise ValueError("CB scatter token count exceeds slot capacity")
                mapping_end = int(mapping_offset) + int(n_tok)
                if mapping_offset < 0 or mapping_end > group.slot_mapping_capacity:
                    raise ValueError("CB scatter slot-mapping range exceeds capacity")
                if n_tok == 0:
                    continue
                if isinstance(group.slot_mapping_base, torch.Tensor):
                    slot_mapping = group.slot_mapping_base.flatten()[
                        int(mapping_offset) : mapping_end
                    ].to(plan_device)
                else:
                    slot_mapping = _plan_tensor(
                        group.slot_mapping_base
                        + int(mapping_offset) * torch.int64.itemsize,
                        (int(n_tok),),
                        torch.int64,
                        plan_device,
                    )
                slot = _cb_temp_buffer(group, int(slot_idx), dtype, plan_device)
                key_value = slot[:, :, : int(n_tok), :].contiguous()
                paged_ptrs = _plan_pointer_tensor(
                    group._paged_kv_ptrs,
                    group._num_layers,
                    plan_device,
                )
                multi_layer_kv_transfer(
                    key_value,
                    paged_ptrs,
                    slot_mapping,
                    plan_device,
                    group._page_buffer_size,
                    TransferDirection.H2D,
                    group._engine_kv_format,
                    block_size=group._block_size,
                    head_size=group._head_size,
                )
        starts = ends


def _valid_block_range(
    object_idx: int,
    block_id_list: list[int],
    blocks_per_object: int,
    block_size: int,
    skip_prefix_n_blocks: int,
) -> tuple[list[int], int] | None:
    """Return valid engine block IDs and their LMCache object token offset.

    Args:
        object_idx: Index of the LMCache object/chunk being processed.
        block_id_list: Full ordered engine block ids for the transfer.
        blocks_per_object: Number of blocks represented by one LMCache object.
        block_size: Number of tokens per block.
        skip_prefix_n_blocks: Number of leading flat block positions to skip.

    Returns:
        ``None`` if this object has no valid blocks after skip handling.
        Otherwise, a tuple of valid engine block ids and the token offset
        within this LMCache object where those blocks start.
    """
    object_flat_start = object_idx * blocks_per_object
    valid_flat_start = max(object_flat_start, skip_prefix_n_blocks)
    valid_flat_end = min(object_flat_start + blocks_per_object, len(block_id_list))
    if valid_flat_start >= valid_flat_end:
        return None
    offset_in_object = (valid_flat_start - object_flat_start) * block_size
    return block_id_list[valid_flat_start:valid_flat_end], offset_in_object


def _valid_block_range_indices(
    object_idx: int,
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    skip_prefix_n_blocks: int,
) -> tuple[int, int, int] | None:
    """Return valid [start, end) range over flat block IDs and object token offset."""
    object_flat_start = object_idx * blocks_per_object
    valid_flat_start = max(object_flat_start, skip_prefix_n_blocks)
    valid_flat_end = min(object_flat_start + blocks_per_object, n_block_ids)
    if valid_flat_start >= valid_flat_end:
        return None
    offset_in_object = (valid_flat_start - object_flat_start) * block_size
    return valid_flat_start, valid_flat_end, offset_in_object


def _transfer_cross_layer(
    paged_tensor: torch.Tensor,
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle cross-layer formats: single tensor [NB, NL, 2, ...]."""
    # NHD: [NB, NL, 2, BS, NH, HS]  HND: [NB, NL, 2, NH, BS, HS]
    is_hnd = _is_hnd_format(engine_kv_format)
    num_layers = paged_tensor.shape[1]

    if is_hnd:
        # [NB, NL, 2, NH, BS, HS]
        nh = paged_tensor.shape[3]
        hs = paged_tensor.shape[5]
    else:
        # [NB, NL, 2, BS, NH, HS]
        nh = paged_tensor.shape[4]
        hs = paged_tensor.shape[5]

    # H2D: pre-transfer objects to paged device
    if not is_d2h and object_tensors:
        objs_on_device = [obj.to(paged_tensor.device) for obj in object_tensors]
    block_ids_dev = torch.as_tensor(
        block_ids, dtype=torch.long, device=paged_tensor.device
    )

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            selected = paged_tensor.index_select(0, eff_idx)

        for layer_idx in range(num_layers):
            for kv in range(2):
                if is_d2h:
                    slice_t = selected[:, layer_idx, kv]
                    if is_hnd:
                        # N=n_valid, BS=block_size:
                        # [N, NH, BS, HS] -> [N, BS, NH, HS] -> [N*BS, NH*HS]
                        flat = slice_t.permute(0, 2, 1, 3).reshape(
                            n_valid * block_size, nh * hs
                        )
                    else:
                        # [N, BS, NH, HS] → [N*BS, NH*HS]
                        flat = slice_t.reshape(n_valid * block_size, nh * hs)
                    obj[kv, layer_idx, offset_in_object:token_end].copy_(
                        flat, non_blocking=True
                    )
                else:
                    obj_device = objs_on_device[object_idx]
                    src = obj_device[kv, layer_idx, offset_in_object:token_end]
                    if is_hnd:
                        # N=n_valid, BS=block_size:
                        # [N*BS, NH*HS] -> [N, BS, NH, HS] -> [N, NH, BS, HS]
                        src_blocks = src.reshape(n_valid, block_size, nh, hs).permute(
                            0, 2, 1, 3
                        )
                    else:
                        # N=n_valid, BS=block_size:
                        # [N*BS, NH*HS] -> [N, BS, NH, HS]
                        src_blocks = src.reshape(n_valid, block_size, nh, hs)
                    paged_tensor[:, layer_idx, kv].index_copy_(0, eff_idx, src_blocks)


def _transfer_sglang_mha(
    paged_tensors: list[list[torch.Tensor]],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle SGLang MHA formats: 2*NL tensors (list[list[Tensor]])."""
    # TWO_X_NL_X_NBBS_NH_HS: each tensor [NB*BS, NH, HS]
    # TWO_X_NL_X_NB_BS_NH_HS: each tensor [NB, BS, NH, HS]
    is_flat = _is_pbs_fused_format(engine_kv_format)
    num_layers = len(paged_tensors[0])

    # Determine target device from first tensor
    target_device = paged_tensors[0][0].device

    # H2D: pre-transfer objects
    if not is_d2h and object_tensors:
        objs_on_device = [obj.to(target_device) for obj in object_tensors]
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]
        if is_flat:
            # Flat token positions for all valid blocks:
            # block_id * block_size + token offset. Reused across layer/KV pairs.
            token_indices = (
                eff_idx[:, None] * block_size
                + torch.arange(block_size, dtype=torch.long, device=target_device)
            ).reshape(-1)

        for layer_idx in range(num_layers):
            for kv in range(2):
                layer_t = paged_tensors[kv][layer_idx]
                nh = layer_t.shape[-2]
                hs = layer_t.shape[-1]

                if is_d2h:
                    if is_flat:
                        # [NB*BS, NH, HS]
                        gathered = layer_t.index_select(0, token_indices)
                    else:
                        # [NB, BS, NH, HS]
                        gathered = layer_t.index_select(0, eff_idx).reshape(
                            n_valid * block_size, nh, hs
                        )
                    flat = gathered.reshape(n_valid * block_size, nh * hs)
                    obj[kv, layer_idx, offset_in_object:token_end].copy_(
                        flat, non_blocking=True
                    )
                else:
                    obj_device = objs_on_device[object_idx]
                    src = obj_device[kv, layer_idx, offset_in_object:token_end]
                    src_shaped = src.reshape(n_valid * block_size, nh, hs)
                    if is_flat:
                        # scatter into [NB*BS, NH, HS]
                        layer_t.index_copy_(0, token_indices, src_shaped)
                    else:
                        # N=n_valid, BS=block_size:
                        # [N*BS, NH, HS] -> [N, BS, NH, HS]
                        src_blocks = src_shaped.reshape(n_valid, block_size, nh, hs)
                        layer_t.index_copy_(0, eff_idx, src_blocks)


def _transfer_per_layer_mla(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle MLA per-layer formats: [NB, BS, HS]."""
    if not layer_tensors or not object_tensors:
        return

    is_flat = _is_pbs_fused_format(engine_kv_format)
    target_device = layer_tensors[0].device
    if is_flat:
        token_offsets = torch.arange(block_size, dtype=torch.long, device=target_device)
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]
        if is_flat:
            token_indices = (
                eff_idx[:, None] * block_size + token_offsets[None, :]
            ).reshape(-1)

        if is_d2h:
            hidden_size = layer_tensors[0].shape[-1]
            chunk_gpu = torch.empty(
                len(layer_tensors),
                n_valid * block_size,
                hidden_size,
                dtype=layer_tensors[0].dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_flat:
                    dst = chunk_gpu[layer_idx].view(
                        n_valid * block_size, 1, hidden_size
                    )
                    torch.index_select(layer, 0, token_indices, out=dst)
                else:
                    dst = chunk_gpu[layer_idx].view(n_valid, block_size, hidden_size)
                    torch.index_select(layer, 0, eff_idx, out=dst)
            obj[:, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                src = chunk_gpu[layer_idx]
                hidden_size = layer.shape[-1]
                if is_flat:
                    src_tokens = src.reshape(n_valid * block_size, 1, hidden_size)
                    layer.index_copy_(0, token_indices, src_tokens)
                else:
                    src_blocks = src.reshape(n_valid, block_size, hidden_size)
                    layer.index_copy_(0, eff_idx, src_blocks)


def _transfer_per_layer_blocked_scale(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Transfer blocked-scale pages to or from canonical token-major chunks."""
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    hidden_size = layer_tensors[0].shape[-1]
    if hidden_size <= 4:
        raise ValueError("blocked-scale rows require values and a 4-byte scale")
    value_size = hidden_size - 4
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            chunk_gpu = torch.empty(
                len(layer_tensors),
                n_valid * block_size,
                hidden_size,
                dtype=layer_tensors[0].dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                blocked = layer.index_select(0, eff_idx).reshape(n_valid, -1)
                values_end = block_size * value_size
                values = blocked[:, :values_end].reshape(
                    n_valid, block_size, value_size
                )
                scales = blocked[:, values_end:].reshape(n_valid, block_size, 4)
                chunk_gpu[layer_idx].copy_(
                    torch.cat((values, scales), dim=-1).reshape(
                        n_valid * block_size, hidden_size
                    )
                )
            obj[:, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                rows = chunk_gpu[layer_idx].reshape(n_valid, block_size, hidden_size)
                blocked = torch.cat(
                    (
                        rows[..., :value_size].reshape(n_valid, -1),
                        rows[..., value_size:].reshape(n_valid, -1),
                    ),
                    dim=-1,
                ).reshape(n_valid, block_size, hidden_size)
                layer.index_copy_(0, eff_idx, blocked)


def _transfer_per_layer_hnd(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle per-layer HND formats: heads before block tokens."""
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    # Two-major keeps K and V as separate leading planes ([2, NB, NH, BS, HS]);
    # otherwise the size-2 axis sits after the blocks ([NB, 2, NH, BS, HS]).
    is_two_major = _is_two_major_format(engine_kv_format)
    first_layer = layer_tensors[0]
    if is_two_major:
        first_k = first_layer[0]
    else:
        first_k = first_layer[:, 0]
    _nb0, nh0, _bs0, hs0 = first_k.shape

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            chunk_gpu = torch.empty(
                2,
                len(layer_tensors),
                n_valid * block_size,
                nh0 * hs0,
                dtype=first_k.dtype,
                device=target_device,
            )
            scratch = torch.empty(
                n_valid,
                nh0,
                block_size,
                hs0,
                dtype=first_k.dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                    torch.index_select(k_t, 0, eff_idx, out=scratch)
                    chunk_gpu[0, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        scratch.permute(0, 2, 1, 3)
                    )
                    torch.index_select(v_t, 0, eff_idx, out=scratch)
                    chunk_gpu[1, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        scratch.permute(0, 2, 1, 3)
                    )
                else:
                    # FlashInfer HND stores KV as [NB, 2, NH, BS, HS].
                    # Gather on dim=0 first so reads stay contiguous in memory;
                    # index_select on layer[:, 0]/layer[:, 1] non-contiguous views
                    # triggers slower element-wise gather reads.
                    selected = layer.index_select(0, eff_idx)
                    chunk_gpu[0, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        selected[:, 0].permute(0, 2, 1, 3)
                    )
                    chunk_gpu[1, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        selected[:, 1].permute(0, 2, 1, 3)
                    )
            obj[:, :, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, :, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                else:
                    k_t, v_t = layer[:, 0], layer[:, 1]
                _nb, nh, _bs, hs = k_t.shape
                k_blocks = (
                    chunk_gpu[0, layer_idx]
                    .reshape(n_valid, block_size, nh, hs)
                    .permute(0, 2, 1, 3)
                )
                v_blocks = (
                    chunk_gpu[1, layer_idx]
                    .reshape(n_valid, block_size, nh, hs)
                    .permute(0, 2, 1, 3)
                )
                if not is_two_major:
                    layer.index_copy_(
                        0, eff_idx, torch.stack([k_blocks, v_blocks], dim=1)
                    )
                else:
                    k_t.index_copy_(0, eff_idx, k_blocks)
                    v_t.index_copy_(0, eff_idx, v_blocks)


def _transfer_per_layer_fused(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle fused-K/V per-layer formats (kv_size == 1).

    The K/V pair stays packed inside each ``2 * head_size`` head, so every
    layer transfers as a single plane and the object layout is
    ``[NL, tokens, NH * 2 * HS]`` — byte-identical to the device kernel's.
    """
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    # Callers pass either the raw 4-D registration ([NB, NH, BS, 2*HS] or
    # [NB, BS, NH, 2*HS]) or the canonical 5-D split with the size-2 axis
    # second-to-last; both share storage, so flatten the trailing pair away.
    layers = [
        layer.reshape(*layer.shape[:3], -1) if layer.dim() == 5 else layer
        for layer in layer_tensors
    ]
    is_hnd = _is_hnd_format(engine_kv_format)
    first = layers[0]
    if is_hnd:
        _nb0, nh0, _bs0, hs0 = first.shape
    else:
        _nb0, _bs0, nh0, hs0 = first.shape
    nl = len(layers)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]
        # Tolerate legacy [2, NL, T, H]-shaped buffers: the flat storage is
        # reinterpreted as the single-plane layout.
        chunk_tokens = obj.numel() // (nl * nh0 * hs0)
        obj_view = obj.reshape(nl, chunk_tokens, nh0 * hs0)

        if is_d2h:
            chunk_gpu = torch.empty(
                nl,
                n_valid * block_size,
                nh0 * hs0,
                dtype=first.dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layers):
                selected = layer.index_select(0, eff_idx)
                if is_hnd:
                    # [n, NH, BS, 2*HS] -> tokens-major [n, BS, NH, 2*HS]
                    selected = selected.permute(0, 2, 1, 3)
                chunk_gpu[layer_idx].view(n_valid, block_size, nh0, hs0).copy_(selected)
            obj_view[:, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj_view[:, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layers):
                blocks = chunk_gpu[layer_idx].reshape(n_valid, block_size, nh0, hs0)
                if is_hnd:
                    blocks = blocks.permute(0, 2, 1, 3)
                layer.index_copy_(0, eff_idx, blocks)


def _transfer_per_layer_nhd(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle per-layer NHD formats: block tokens before heads."""
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    # Two-major keeps K and V as separate leading planes ([2, NB, BS, NH, HS]);
    # otherwise the size-2 axis sits after the blocks ([NB, 2, BS, NH, HS]).
    is_two_major = _is_two_major_format(engine_kv_format)
    first_layer = layer_tensors[0]
    if is_two_major:
        first_k = first_layer[0]
    else:
        first_k = first_layer[:, 0]
    _nb0, _bs0, nh0, hs0 = first_k.shape

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            chunk_gpu = torch.empty(
                2,
                len(layer_tensors),
                n_valid * block_size,
                nh0 * hs0,
                dtype=first_k.dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                    torch.index_select(
                        k_t,
                        0,
                        eff_idx,
                        out=chunk_gpu[0, layer_idx].view(n_valid, block_size, nh0, hs0),
                    )
                    torch.index_select(
                        v_t,
                        0,
                        eff_idx,
                        out=chunk_gpu[1, layer_idx].view(n_valid, block_size, nh0, hs0),
                    )
                else:
                    # FlashInfer NHD stores KV as [NB, 2, BS, NH, HS].
                    # Gather on dim=0 first to avoid index_select from
                    # non-contiguous layer[:, 0]/layer[:, 1] views, which
                    # trigger slower element-wise gather reads.
                    selected = layer.index_select(0, eff_idx)
                    chunk_gpu[0, layer_idx].copy_(
                        selected[:, 0].reshape(n_valid * block_size, nh0 * hs0)
                    )
                    chunk_gpu[1, layer_idx].copy_(
                        selected[:, 1].reshape(n_valid * block_size, nh0 * hs0)
                    )
            obj[:, :, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, :, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                    k_t.index_copy_(
                        0,
                        eff_idx,
                        chunk_gpu[0, layer_idx].reshape(n_valid, block_size, nh0, hs0),
                    )
                    v_t.index_copy_(
                        0,
                        eff_idx,
                        chunk_gpu[1, layer_idx].reshape(n_valid, block_size, nh0, hs0),
                    )
                else:
                    k_blocks = chunk_gpu[0, layer_idx].reshape(
                        n_valid, block_size, nh0, hs0
                    )
                    v_blocks = chunk_gpu[1, layer_idx].reshape(
                        n_valid, block_size, nh0, hs0
                    )
                    layer.index_copy_(
                        0, eff_idx, torch.stack([k_blocks, v_blocks], dim=1)
                    )


def single_layer_kv_transfer(
    lmc_key_value_cache: torch.Tensor,
    vllm_key_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    token_major: bool = False,
):
    """
    Vectorized Python fallback for single_layer_kv_transfer
    (eliminates per-token loops).

    Transfers KV data between LMCache buffer
    and a single vLLM paged KV cache layer.

    lmc_key_value_cache layout:
        - MLA:                    [num_tokens, aligned_head_size]
        - token_major=True:       [num_tokens, 2, num_heads * head_size]
        - token_major=False:      [2, num_tokens, num_heads * head_size]

    vllm_key_value_cache layout:
        - NL_X_TWO_NB_BS_NH_HS (flash attn):
            [2, num_blocks, block_size, num_heads, head_size]
        - NL_X_NB_TWO_BS_NH_HS (flash infer):
            [num_blocks, 2, block_size, num_heads, head_size]
        - NL_X_TWO_NB_NH_BS_HS (HND flash attn):
            [2, num_blocks, num_heads, block_size, head_size]
        - NL_X_NB_TWO_NH_BS_HS (HND flash infer):
            [num_blocks, 2, num_heads, block_size, head_size]
        - NL_X_NB_BS_HS (vLLM MLA):
            [num_blocks, block_size, head_size]

    direction:
        H2D = LMCache  -> vLLM GPU
        D2H = vLLM GPU -> LMCache
    """
    kv_device = lmc_key_value_cache.device
    paged_memory_device = vllm_key_value_cache.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0

    if not valid_mask_kv.any():
        return

    valid_token_indices = torch.nonzero(valid_mask_kv, as_tuple=True)[0]
    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    uses_mla = is_mla(engine_kv_format)

    if uses_mla:
        # ── MLA format ──
        # vllm: [num_blocks, block_size, head_size]
        # lmc:  [num_tokens, aligned_head_size]
        block_size = vllm_key_value_cache.size(1)
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        if int(direction) == int(TransferDirection.D2H):
            # vLLM -> LMCache
            lmc_key_value_cache[valid_token_indices] = vllm_key_value_cache[
                block_indices, block_offsets
            ].to(lmc_key_value_cache.device)
        else:
            # LMCache -> vLLM
            vllm_key_value_cache[block_indices, block_offsets] = lmc_key_value_cache[
                valid_token_indices
            ].to(paged_memory_device)

    else:
        # ── Non-MLA format ──
        # Determine vLLM layout and block_size
        is_two_major = _is_two_major_format(engine_kv_format)
        is_hnd = _is_hnd_format(engine_kv_format)
        # flash attn:
        #   [2, num_blocks, block_size, num_heads, head_size]
        #   -> dim2 = block_size
        # flash infer:
        #   [num_blocks, 2, block_size, num_heads, head_size]
        #   -> dim2 = block_size
        # HND layouts put num_heads before block_size.
        if is_hnd:
            num_heads = vllm_key_value_cache.size(2)
            block_size = vllm_key_value_cache.size(3)
        else:
            block_size = vllm_key_value_cache.size(2)
            num_heads = vllm_key_value_cache.size(3)
        head_size = vllm_key_value_cache.size(4)
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        for kv in range(2):
            if int(direction) == int(TransferDirection.D2H):
                if is_two_major:
                    gathered = (
                        vllm_key_value_cache[kv, block_indices, :, block_offsets]
                        if is_hnd
                        else vllm_key_value_cache[kv, block_indices, block_offsets]
                    )
                else:
                    gathered = (
                        vllm_key_value_cache[block_indices, kv, :, block_offsets]
                        if is_hnd
                        else vllm_key_value_cache[block_indices, kv, block_offsets]
                    )

                gathered_flat = gathered.reshape(-1, num_heads * head_size).to(
                    lmc_key_value_cache.device
                )
                if token_major:
                    lmc_key_value_cache[valid_token_indices, kv] = gathered_flat
                else:
                    lmc_key_value_cache[kv, valid_token_indices] = gathered_flat
            else:
                if token_major:
                    lmc_src = lmc_key_value_cache[valid_token_indices, kv]
                else:
                    lmc_src = lmc_key_value_cache[kv, valid_token_indices]
                lmc_reshaped = lmc_src.reshape(-1, num_heads, head_size).to(
                    vllm_key_value_cache.device
                )

                if is_two_major:
                    if is_hnd:
                        vllm_key_value_cache[kv, block_indices, :, block_offsets] = (
                            lmc_reshaped
                        )
                    else:
                        vllm_key_value_cache[kv, block_indices, block_offsets] = (
                            lmc_reshaped
                        )
                elif is_hnd:
                    vllm_key_value_cache[block_indices, kv, :, block_offsets] = (
                        lmc_reshaped
                    )
                else:
                    vllm_key_value_cache[block_indices, kv, block_offsets] = (
                        lmc_reshaped
                    )


def single_layer_kv_transfer_sgl(
    lmc_key_value_cache: torch.Tensor,
    sgl_key_cache: torch.Tensor,
    sgl_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    token_major: bool = False,
):
    """
    Python fallback implementation of single_layer_kv_transfer_sgl.

    Args:
        lmc_key_value_cache:
            [num_tokens, 2, num_heads*head_size] or
            [2, num_tokens, num_heads*head_size]
        sgl_key_cache: [num_blocks, block_size, num_heads, head_size]
        sgl_value_cache: [num_blocks, block_size, num_heads, head_size]
        slot_mapping: [num_tokens] - maps each token to a global slot index
        direction: False for LMCache -> SGLang, True for SGLang -> LMCache
        token_major: Boolean to determine the layout of lmc_key_value_cache
    """
    kv_device = lmc_key_value_cache.device
    paged_memory_device = sgl_key_cache.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    if not valid_mask_kv.any():
        return

    # 1. Get basic dimensions
    block_size = sgl_key_cache.size(1)
    num_heads = sgl_key_cache.size(2)
    head_size = sgl_key_cache.size(3)

    # 2. Calculate block indices and offsets within the blocks from slot_mapping
    # In SGLang/vLLM, slot_idx = block_idx * block_size + block_offset
    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)
    block_indices = valid_slots // block_size
    block_offsets = valid_slots % block_size

    # 3. Prepare LMCache views for K and V
    if token_major:
        # Layout: [num_tokens, 2, hidden_size]
        lmc_k = lmc_key_value_cache[:, 0, :]
        lmc_v = lmc_key_value_cache[:, 1, :]
    else:
        # Layout: [2, num_tokens, hidden_size]
        lmc_k = lmc_key_value_cache[0, :, :]
        lmc_v = lmc_key_value_cache[1, :, :]

    # 4. Perform the transfer
    if int(direction) == int(TransferDirection.H2D):
        # --- Direction: LMCache to SGLang (Paged Buffer) ---
        # Reshape LMC flat tensors to match SGL [num_heads, head_size]
        src_k_reshaped = (
            lmc_k[valid_mask_kv]
            .reshape(-1, num_heads, head_size)
            .to(paged_memory_device)
        )
        src_v_reshaped = (
            lmc_v[valid_mask_kv]
            .reshape(-1, num_heads, head_size)
            .to(paged_memory_device)
        )

        # Advanced indexing: update specific slots in the paged cache
        sgl_key_cache[block_indices, block_offsets] = src_k_reshaped
        sgl_value_cache[block_indices, block_offsets] = src_v_reshaped

    else:
        # --- Direction: SGLang (Paged Buffer) to LMCache ---
        # Gather tensors from paged cache based on mapping
        sampled_k = sgl_key_cache[block_indices, block_offsets].to(kv_device)
        sampled_v = sgl_value_cache[block_indices, block_offsets].to(kv_device)

        # Flatten the head dimensions and copy into LMC tensors
        lmc_k[valid_mask_kv] = sampled_k.reshape(-1, num_heads * head_size)
        lmc_v[valid_mask_kv] = sampled_v.reshape(-1, num_heads * head_size)


def load_and_reshape_flash(
    key_value: torch.Tensor,
    # Destination (Dst): Pinned CPU Tensor [2, L, T, H]
    key_cache: torch.Tensor,
    # Source (Src): GPU Cache [Blocks, BlockSize, NumHeads, HeadSize]
    value_cache: torch.Tensor,  # Source (Src): GPU Cache
    slot_mapping: torch.Tensor,  # Mapping indices [num_tokens]
    layer_idx: int,
):
    """
    Python equivalent of load_and_reshape_flash.
    Note: In the context of 'test_extract_and_load_back', this function performs
    an EXTRACT operation (Reads from GPU Cache and writes to Pinned CPU memory).
    """
    # 1. Prepare indices on the target device
    # Mapping must be on the same GPU as the cache to perform indexing
    device = key_cache.device
    slot_mapping = slot_mapping.to(device=device, dtype=torch.long)

    block_size = key_cache.size(1)

    # Calculate physical locations within the paged cache
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size

    # 2. Extract data from Cache (Gather operation)
    # The result k_out/v_out will be on the GPU
    # Shape: [num_tokens, num_heads, head_size]
    k_out = key_cache[block_indices, block_offsets]
    v_out = value_cache[block_indices, block_offsets]

    # 3. Write to the destination tensor (CPU Copy)
    # Target shape: [2, num_layers, num_tokens, hidden_dim]

    # Flatten heads into the hidden dimension: [T, NumHeads, HeadSize] -> [T, HiddenDim]
    hidden_dim = k_out.shape[1] * k_out.shape[2]

    # Assignment automatically handles the Device-to-Host (D2H) transfer
    key_value[0, layer_idx] = k_out.view(-1, hidden_dim)
    key_value[1, layer_idx] = v_out.view(-1, hidden_dim)


def reshape_and_cache_back_flash(
    key_value: torch.Tensor,
    # Source: [2, num_layer, num_tokens, num_heads * head_size]
    # (Can be on CPU/Pinned Memory or GPU)
    key_cache: torch.Tensor,
    # Destination: [num_blocks, block_size, num_heads, head_size]
    # (Must be on GPU)
    value_cache: torch.Tensor,  # Destination: (Must be on GPU)
    slot_mapping: torch.Tensor,  # Indices: [num_tokens]
    layer_idx: int,
):
    """
    Python implementation of reshape_and_cache_back_flash.

    Operation:
        Flat Tensor (Source) -> Paged Attention Cache (Destination)

    Logic:
        1. Extract the specific layer's data from key_value.
        2. Move it to the GPU (if it's on CPU).
        3. Reshape it to match the cache's head structure.
        4. Scatter (write) it into the non-contiguous cache blocks using slot_mapping.
    """

    # 1. Setup Device & Dimensions
    # The cache is on the GPU, so all indices and source data must eventually be there.
    device = key_cache.device

    block_size = key_cache.size(1)
    num_heads = key_cache.size(2)
    head_size = key_cache.size(3)

    # 2. Prepare Indices
    # slot_mapping might be on CPU, must move to GPU for indexing.
    slot_mapping = slot_mapping.to(device=device, dtype=torch.long)

    # Calculate physical block indices and offsets
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size

    # 3. Process Source Data (Key)
    # Step A: Slice the specific layer from the source tensor
    # Source shape: [2, num_layers, num_tokens, hidden_dim] -> [num_tokens, hidden_dim]
    k_src_flat = key_value[0, layer_idx]
    v_src_flat = key_value[1, layer_idx]

    # Step B: Reshape & Move to GPU
    # .to(device) handles the CPU -> GPU transfer if key_value is in pinned memory.
    # View shape: [num_tokens, num_heads, head_size]
    k_src = k_src_flat.to(device).view(-1, num_heads, head_size)
    v_src = v_src_flat.to(device).view(-1, num_heads, head_size)

    # 4. Write to Cache (Scatter)
    # Using Advanced Indexing to write data into specific blocks/offsets
    key_cache[block_indices, block_offsets] = k_src
    value_cache[block_indices, block_offsets] = v_src


def lmcache_memcpy_async(
    dest: int | torch.Tensor,
    src: int | torch.Tensor,
    nbytes: int,
    direction: TransferDirection,
    host_buffer_offset: int,
    host_buffer_alignments: int,
):
    """
    Python fallback for lmcache_memcpy_async.

    Raw pointers are converted to aliasing ``torch.uint8`` tensor views, while
    tensor arguments are viewed directly as bytes. Each resulting segment is
    copied with ``Tensor.copy_(..., non_blocking=True)``.

    Copies are split at ``host_buffer_alignments`` boundaries to keep every
    transfer within one cudaHostRegister region. This matches the native
    implementation and is required for lazy pinned allocations.

    dest:
        - If int: raw memory pointer (used for CUDA/CPU devices where we
          work with pointers).
        - If torch.Tensor: tensor object (used for non-CUDA/CPU devices
          where we operate on tensor objects directly).

    src:
        - If int: raw memory pointer (used for CUDA/CPU devices where we
          work with pointers).
        - If torch.Tensor: tensor object (used for non-CUDA/CPU devices
          where we operate on tensor objects directly).
    """
    # 1. Power of two check (kept for API compatibility)
    if host_buffer_alignments <= 0 or (
        host_buffer_alignments & (host_buffer_alignments - 1) != 0
    ):
        raise ValueError("host_buffer_alignments must be power of two")
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if host_buffer_offset < 0:
        raise ValueError("host_buffer_offset must be non-negative")

    # 2. Validate direction
    if int(direction) not in (int(TransferDirection.H2D), int(TransferDirection.D2H)):
        raise ValueError(f"Unsupported direction: {direction}")

    if nbytes == 0:
        return

    if isinstance(dest, torch.Tensor) or isinstance(src, torch.Tensor):
        if not isinstance(dest, torch.Tensor) or not isinstance(src, torch.Tensor):
            raise TypeError(
                "dest and src must both be torch.Tensor or both be raw pointers"
            )
        destination = _byte_tensor_from_copy_argument(dest, nbytes, torch.device("cpu"))
        source = _byte_tensor_from_copy_argument(src, nbytes, torch.device("cpu"))
    else:
        if not isinstance(dest, int) or not isinstance(src, int):
            raise TypeError(
                "dest and src must both be torch.Tensor or both be raw pointers"
            )
        destination = _byte_tensor_from_copy_argument(
            dest, nbytes, _device_for_raw_pointer(dest)
        )
        source = _byte_tensor_from_copy_argument(
            src, nbytes, _device_for_raw_pointer(src)
        )

    copied_bytes = 0
    alignment_mask = host_buffer_alignments - 1
    while copied_bytes < nbytes:
        aligned_region_end = (
            (copied_bytes + host_buffer_offset) & ~alignment_mask
        ) + host_buffer_alignments
        copy_end = min(host_buffer_offset + nbytes, aligned_region_end)
        copy_nbytes = copy_end - copied_bytes - host_buffer_offset
        destination[copied_bytes : copied_bytes + copy_nbytes].copy_(
            source[copied_bytes : copied_bytes + copy_nbytes],
            non_blocking=True,
        )
        copied_bytes += copy_nbytes


@njit(cache=True)
def _encode_single_channel(
    cdf_layer_c,  # np.uint32 [lp]
    sym_channel,  # np.uint8 [n_tokens]
    out_buf_lc,  # np.uint8 [buffer_size]
):
    """Core arithmetic encoding for a single (layer, channel).
    Returns number of bytes written."""
    MASK32 = 0xFFFFFFFF
    precision = 16
    max_symbol = len(cdf_layer_c) - 2
    n_tokens = len(sym_channel)

    low, high = 0, MASK32
    pending_bits = 0
    output_reg, output_reg_len = 0, 0
    ptr = 0
    buf_size = len(out_buf_lc)

    # Inline flush_bit to avoid closure (numba does not support nonlocal)
    for token_idx in range(n_tokens):
        sym = int(sym_channel[token_idx])
        c_low = int(cdf_layer_c[sym])
        c_high = 0x10000 if sym == max_symbol else int(cdf_layer_c[sym + 1])

        span = (high - low + 1) & MASK32
        if span == 0:
            span = 0x100000000

        high = (low + ((span * c_high) >> precision) - 1) & MASK32
        low = (low + ((span * c_low) >> precision)) & MASK32

        while True:
            if (high & 0x80000000) == (low & 0x80000000):
                # flush_bit(bit)
                bit = (high >> 31) & 1
                output_reg = (output_reg << 1) | bit
                output_reg_len += 1
                if output_reg_len == 8:
                    if ptr < buf_size:
                        out_buf_lc[ptr] = output_reg & 0xFF
                        ptr += 1
                    output_reg, output_reg_len = 0, 0
                # flush pending bits
                for _ in range(pending_bits):
                    output_reg = (output_reg << 1) | (1 - bit)
                    output_reg_len += 1
                    if output_reg_len == 8:
                        if ptr < buf_size:
                            out_buf_lc[ptr] = output_reg & 0xFF
                            ptr += 1
                        output_reg, output_reg_len = 0, 0
                pending_bits = 0
                low = (low << 1) & MASK32
                high = ((high << 1) | 1) & MASK32
            elif (low & 0x40000000) != 0 and (high & 0x40000000) == 0:
                pending_bits += 1
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) | 0x80000001) & MASK32
            else:
                break

    # Final flushing sequence
    pending_bits += 1
    bit = 1 if (low & 0x40000000) != 0 else 0
    output_reg = (output_reg << 1) | bit
    output_reg_len += 1
    if output_reg_len == 8:
        if ptr < buf_size:
            out_buf_lc[ptr] = output_reg & 0xFF
            ptr += 1
        output_reg, output_reg_len = 0, 0
    for _ in range(pending_bits):
        output_reg = (output_reg << 1) | (1 - bit)
        output_reg_len += 1
        if output_reg_len == 8:
            if ptr < buf_size:
                out_buf_lc[ptr] = output_reg & 0xFF
                ptr += 1
            output_reg, output_reg_len = 0, 0
    pending_bits = 0  # noqa: F841

    if output_reg_len > 0:
        if ptr < buf_size:
            out_buf_lc[ptr] = (output_reg << (8 - output_reg_len)) & 0xFF
            ptr += 1

    return ptr


def encode_fast_new(cdf, input_sym, output_buffer, output_lengths):
    """
    Python equivalent of C++ Arithmetic Encoder.
    Strictly emulates 32-bit unsigned overflow for high/low.
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    sym_np = input_sym.cpu().numpy().astype(np.uint8)

    n_layers, n_tokens, n_channels = sym_np.shape
    out_buf_np = np.zeros(output_buffer.shape, dtype=np.uint8)
    out_len_np = np.zeros(output_lengths.shape, dtype=np.int32)

    def encode_one(args):
        layer_idx, c = args
        length = _encode_single_channel(
            cdf_np[layer_idx, c],
            sym_np[layer_idx, :, c],
            out_buf_np[layer_idx, c],
        )
        out_len_np[layer_idx, c] = length

    tasks = [(layer_idx, c) for layer_idx in range(n_layers) for c in range(n_channels)]

    with ThreadPoolExecutor() as executor:
        list(executor.map(encode_one, tasks))

    output_buffer.copy_(torch.from_numpy(out_buf_np))
    output_lengths.copy_(torch.from_numpy(out_len_np))


@njit(cache=True)
def _decode_single_channel(
    cdf_layer_c,
    bs_np,
    start_off,
    end_off,
    n_tokens,
    out_layer_c,
):
    MASK32 = 0xFFFFFFFF
    precision = 16
    max_symbol = len(cdf_layer_c) - 2

    v_val = 0
    if start_off + 4 <= len(bs_np):
        v_val = (
            (int(bs_np[start_off]) << 24)
            | (int(bs_np[start_off + 1]) << 16)
            | (int(bs_np[start_off + 2]) << 8)
            | int(bs_np[start_off + 3])
        ) & MASK32

    low, high = 0, MASK32
    byte_buffer_offset = start_off + 4
    bit_idx = 1
    byte_buffer = int(bs_np[byte_buffer_offset]) if byte_buffer_offset < end_off else 0

    for i in range(n_tokens):
        span = (high - low + 1) & MASK32
        if span == 0:
            span = 0x100000000

        v_minus_l = (v_val - low) & MASK32
        count = ((v_minus_l + 1) * 0x10000 - 1) // span
        count = count & 0xFFFF

        left = 0
        right = max_symbol + 1
        while left + 1 < right:
            m = (left + right) // 2
            if int(cdf_layer_c[m]) < count:
                left = m
            elif int(cdf_layer_c[m]) > count:
                right = m
            else:
                left = m
                break

        out_layer_c[i] = left

        if i == n_tokens - 1:
            break

        sym_i = left
        c_low = int(cdf_layer_c[sym_i])
        c_high = 0x10000 if sym_i == max_symbol else int(cdf_layer_c[sym_i + 1])

        high = (low + ((span * c_high) >> precision) - 1) & MASK32
        low = (low + ((span * c_low) >> precision)) & MASK32

        while True:
            if low >= 0x80000000 or high < 0x80000000:
                v_val = ((v_val << 1) | ((byte_buffer >> (8 - bit_idx)) & 1)) & MASK32
                low = (low << 1) & MASK32
                high = ((high << 1) | 1) & MASK32
                bit_idx += 1
            elif low >= 0x40000000 and high < 0xC0000000:
                v_val = (v_val - 0x40000000) & MASK32
                v_val = ((v_val << 1) | ((byte_buffer >> (8 - bit_idx)) & 1)) & MASK32
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) | 0x80000001) & MASK32
                bit_idx += 1
            else:
                break

            if bit_idx == 9:
                bit_idx = 1
                byte_buffer_offset += 1
                byte_buffer = (
                    int(bs_np[byte_buffer_offset])
                    if byte_buffer_offset < end_off
                    else 0
                )


# Standard


def decode_fast_new(cdf, bytestreams, lengths, output):
    """
    Python implementation of Arithmetic Decoding.
    Strictly aligned with CUDA decode_with_accessor_kernel.
    bytestreams shape: [nlayers, nchannels, buffer_size]
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    bs_np = bytestreams.cpu().numpy().astype(np.uint8)
    len_np = lengths.cpu().numpy().astype(np.int32)

    n_layers, n_tokens, n_channels = output.shape
    out_np = np.zeros(output.shape, dtype=np.uint8)

    def decode_one(args):
        layer_idx, c = args
        curr_len = int(len_np[layer_idx, c])
        # For decode_fast_new, each channel has its own contiguous buffer,
        # so start_off=0 and end_off=curr_len within channel_bs
        channel_bs = bs_np[layer_idx, c]  # shape [buffer_size]
        _decode_single_channel(
            cdf_np[layer_idx, c],
            channel_bs,
            0,
            curr_len,
            n_tokens,
            out_np[layer_idx, :, c],
        )

    tasks = [(layer_idx, c) for layer_idx in range(n_layers) for c in range(n_channels)]

    with ThreadPoolExecutor() as executor:
        list(executor.map(decode_one, tasks))

    if output is not None:
        output.copy_(torch.from_numpy(out_np))


def decode_fast_prefsum(cdf, bytestreams, lengths_prefsum, output):
    """
    Python equivalent of C++ decode_fast_prefsum.
    bytestreams shape: [total_bytes] (1D, all channels packed)
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    pref_np = lengths_prefsum.cpu().numpy().astype(np.int64).flatten()

    # WA: CUDA kernel reads out-of-bound in two ways:
    # 1. max(prefsum) may equal len(bytestreams) (off-by-one on exclusive-end)
    # 2. v_val init reads 4 bytes starting at start_off, may exceed bytestreams
    # Pad with zeros to make all reads safe.
    max_prefsum = int(pref_np.max())
    pad_size = max(0, max_prefsum + 4 - bytestreams.shape[0])
    if pad_size > 0:
        bytestreams = torch.nn.functional.pad(bytestreams, (0, pad_size), value=0)

    bs_np = bytestreams.cpu().numpy().astype(np.uint8)  # must be after padding

    n_layers, n_tokens, n_channels = output.shape
    out_np = np.zeros(output.shape, dtype=np.uint8)

    def decode_one(args):
        layer_idx, c = args
        cid = layer_idx * n_channels + c
        start_off = 0 if cid == 0 else int(pref_np[cid - 1])
        end_off = int(pref_np[cid])
        _decode_single_channel(
            cdf_np[layer_idx, c],
            bs_np,
            start_off,
            end_off,
            n_tokens,
            out_np[layer_idx, :, c],
        )

    tasks = [(layer_idx, c) for layer_idx in range(n_layers) for c in range(n_channels)]

    with ThreadPoolExecutor() as executor:
        list(executor.map(decode_one, tasks))

    output.copy_(torch.from_numpy(out_np))


def calculate_cdf(input_tensor: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Equivalent to CUDA calculate_cdf.

    Calculates the CDF across tokens for each (layer, channel) pair.

    Args:
        input_tensor: 3D tensor with shape [nlayers, ntokens, nchannels].
        num_bins: Maximum number of bins (i.e., Lp - 1).

    Returns:
        int16 tensor with shape [nlayers, nchannels, num_bins + 1]
        containing normalized CDF values.
    """
    nlayers, ntokens, nchannels = input_tensor.shape
    device = input_tensor.device

    # Compute per-(layer, channel) histogram via scatter_add.
    # Permute to [nlayers, nchannels, ntokens] then flatten first two dims.
    input_perm = input_tensor.permute(0, 2, 1).reshape(-1, ntokens).long()
    src = torch.ones_like(input_perm)
    counts = torch.zeros(nlayers * nchannels, num_bins, dtype=torch.long, device=device)
    counts.scatter_add_(1, input_perm.clamp(0, num_bins - 1), src)
    counts = counts.reshape(nlayers, nchannels, num_bins)

    # Build CDF: cdf[..., 0] = 0, cdf[..., i] = sum(counts[..., 0:i])
    cdf = torch.zeros(nlayers, nchannels, num_bins + 1, dtype=torch.long, device=device)
    cdf[:, :, 1:] = torch.cumsum(counts, dim=2)

    # Total count per (layer, channel)
    total = cdf[:, :, -1:]  # [nlayers, nchannels, 1]

    # Normalize: (0xFFFF - num_bins) * cdf / total + bin_index
    max_uint16_value = 0xFFFF - num_bins
    bin_offsets = torch.arange(num_bins + 1, dtype=torch.long, device=device)

    safe_total = total.clamp(min=1)
    normalized = (max_uint16_value * cdf) // safe_total + bin_offsets

    # Where total is 0, use just the bin offsets
    normalized = torch.where(
        total > 0, normalized, bin_offsets.unsqueeze(0).unsqueeze(0)
    )

    return normalized.to(torch.int16)


def rotary_embedding_k_fused(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Apply fused rotary embedding undo/redo to key tensor in-place.

    Reverses the rotary embedding at old_positions and applies the rotary
    embedding at new_positions. head_size is unused but kept for API
    compatibility with the CUDA equivalent.

    Args:
        old_positions: Token positions whose rotary embedding to reverse.
        new_positions: Token positions whose rotary embedding to apply.
        key: Key tensor to update in-place.
        head_size: Head size (unused; kept for API compatibility).
        cos_sin_cache: Precomputed cosine/sine cache indexed by position.
        is_neox: If True, uses NeoX-style rotary (contiguous halves);
            otherwise uses GPT-J-style (interleaved).
    """
    rot_dim = cos_sin_cache.shape[1]
    half_rot = rot_dim // 2

    old_cs = cos_sin_cache[old_positions]
    new_cs = cos_sin_cache[new_positions]

    oc, os = old_cs[:, :half_rot].unsqueeze(1), old_cs[:, half_rot:].unsqueeze(1)
    nc, ns = new_cs[:, :half_rot].unsqueeze(1), new_cs[:, half_rot:].unsqueeze(1)

    if is_neox:
        x = key[..., :half_rot]
        y = key[..., half_rot:rot_dim]
    else:
        x = key[..., :rot_dim:2]
        y = key[..., 1:rot_dim:2]

    x_rev = x * oc + y * os
    y_rev = y * oc - x * os

    x_out = x_rev * nc - y_rev * ns
    y_out = y_rev * nc + x_rev * ns

    if is_neox:
        key[..., :half_rot] = x_out
        key[..., half_rot:rot_dim] = y_out
    else:
        key[..., :rot_dim:2] = x_out
        key[..., 1:rot_dim:2] = y_out


def rotary_embedding_k_fused_strided(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    head_stride: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Strided rotary_embedding_k_fused: ``key``'s last dim is ``head_stride``
    per head; rotate only the leading ``head_size`` (K) in place via a view,
    leaving the trailing V. ``head_stride == head_size`` is the contiguous case.
    """
    rotary_embedding_k_fused(
        old_positions,
        new_positions,
        key[..., :head_size],
        head_size,
        cos_sin_cache,
        is_neox,
    )


def get_gpu_pci_bus_id(device_id: int = 0) -> str | None:
    """
    Get the PCI bus ID via CUDA/ROCm runtime.
    Other backends return None.

    Args:
        device_id (int): CUDA/ROCm device index.

    Returns:
        str | None: PCI bus ID (e.g., "0000:29:00.0") or None if unavailable.
    """
    torch_dev, _ = get_torch_device()

    try:
        if torch_dev.is_available() and device_id < torch_dev.device_count():
            props = torch_dev.get_device_properties(device_id)
            # PCI function number is always 0 for GPUs
            bus_id = (
                f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
                f"{props.pci_device_id:02x}.0"
            )
            return bus_id.upper()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Completion recorder fallback (no CUDA stream ordering; enqueue immediately)
# ---------------------------------------------------------------------------

_completion_lock = threading.Lock()
_completion_buffer: list[tuple[str, bytes]] = []


def record_completion_on_stream(
    cuda_stream_ptr: int, kind: str, payload: bytes
) -> None:
    """Fallback: immediately enqueue the completion without stream ordering.

    Args:
        cuda_stream_ptr: Ignored on non-CUDA path.
        kind: Dispatch key identifying the handler (e.g. "finish_write").
        payload: Opaque msgpack-encoded bytes forwarded to the handler.
    """
    with _completion_lock:
        _completion_buffer.append((kind, payload))


def drain_recorded_completions() -> list[tuple[str, bytes]]:
    """Fallback: atomically drain and return all pending completions.

    Returns:
        List of (kind, payload) pairs recorded since the last drain.
    """
    with _completion_lock:
        items = list(_completion_buffer)
        _completion_buffer.clear()
    return items


# ---------------------------------------------------------------------------
# Event recorder fallback (no CUDA stream ordering; timestamp immediately)
# ---------------------------------------------------------------------------

_event_lock = threading.Lock()
_event_buffer: list[tuple[str, str, float, dict[str, str], dict[str, int]]] = []


def record_event_on_stream(
    cuda_stream_ptr: int,
    event_type_name: str,
    session_id: str,
    str_metadata: dict[str, str],
    int_metadata: dict[str, int],
) -> None:
    """Fallback: immediately record the event without CUDA stream ordering.

    The wall-clock timestamp is captured at call time (no host-callback).

    Args:
        cuda_stream_ptr: Ignored on non-CUDA path.
        event_type_name: Event type identifier (e.g. "mp.store.start").
        session_id: Session identifier for the event.
        str_metadata: String-valued metadata dict.
        int_metadata: Integer-valued metadata dict.
    """
    # Standard
    import time

    ts = time.time()
    with _event_lock:
        _event_buffer.append(
            (event_type_name, session_id, ts, dict(str_metadata), dict(int_metadata))
        )


def drain_recorded_events() -> list[
    tuple[str, str, float, dict[str, str], dict[str, int]]
]:
    """Fallback: atomically drain and return all pending events.

    Returns:
        List of (event_type_name, session_id, timestamp, str_metadata,
        int_metadata) tuples recorded since the last drain.
    """
    with _event_lock:
        items = list(_event_buffer)
        _event_buffer.clear()
    return items
