# SPDX-License-Identifier: Apache-2.0
"""ROCm IPC wrapper for tensors allocated outside PyTorch's caching allocator.

This is the ROCm equivalent of :class:`RawCudaIPCWrapper`.  It uses
``hipIpcGetMemHandle`` / ``hipIpcOpenMemHandle`` (via ``libamdhip64.so``
and :mod:`ctypes`) instead of the NVIDIA ``cuda.bindings`` Python wheel,
which does not exist on a ROCm-only install.

The HIP IPC API is functionally identical to CUDA IPC:

.. code-block:: c

    hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
    hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle,
                                   unsigned int flags);

``hipIpcMemHandle_t`` has the same 64-byte ``reserved`` buffer as
``cudaIpcMemHandle_t``.  Flag ``hipIpcMemLazyEnablePeerAccess`` has the
same value as its CUDA counterpart.

The reconstruction path uses CuPy (``cupy-rocm-7-0``), which exposes the
same ``UnownedMemory`` / ``MemoryPointer`` / ``ndarray`` / ``Device`` /
``from_dlpack`` API as the CUDA build.
"""

# Standard
import ctypes
import ctypes.util
import threading
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper

# ---------------------------------------------------------------------------
# ctypes bindings for libamdhip64.so IPC symbols
# ---------------------------------------------------------------------------

_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1  # same value as cudaIpcMemLazyEnablePeerAccess


class _HipIpcMemHandle(ctypes.Structure):
    """Mirror of ``hipIpcMemHandle_t`` (64-byte opaque buffer)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


_libhip_cache: ctypes.CDLL | None = None
_libhip_lock = threading.Lock()


def _load_libhip() -> ctypes.CDLL | None:
    """Lazily load ``libamdhip64.so`` and bind the HIP IPC symbols.

    Returns:
        The loaded ``ctypes.CDLL`` with bound symbols, or ``None`` if
        the library is unavailable.
    """
    global _libhip_cache
    if _libhip_cache is not None:
        return _libhip_cache

    with _libhip_lock:
        if _libhip_cache is not None:
            return _libhip_cache

        for lib_name, fallback_path in [
            ("amdhip64", "libamdhip64.so"),
            ("amdhip64", "libamdhip64.so.1"),
        ]:
            path = ctypes.util.find_library(lib_name) or fallback_path
            try:
                lib = ctypes.CDLL(path)
                lib.hipIpcGetMemHandle.restype = ctypes.c_int
                lib.hipIpcGetMemHandle.argtypes = [
                    ctypes.POINTER(_HipIpcMemHandle),
                    ctypes.c_void_p,
                ]
                lib.hipIpcOpenMemHandle.restype = ctypes.c_int
                lib.hipIpcOpenMemHandle.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    _HipIpcMemHandle,
                    ctypes.c_uint,
                ]
                _libhip_cache = lib
                return lib
            except (OSError, AttributeError):
                continue

        return None


class RocmRawIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for ROCm tensors allocated outside PyTorch's caching
    allocator.

    This is the ROCm equivalent of :class:`RawCudaIPCWrapper`.  It calls
    ``hipIpcGetMemHandle`` on the raw data pointer, then reconstructs the
    tensor on the receiving side via ``hipIpcOpenMemHandle`` plus a CuPy
    ``UnownedMemory`` → DLPack → ``torch`` round-trip.

    Like :class:`RawCudaIPCWrapper`, this sets
    ``_is_default_wrapper = False`` so auto-discovery skips it — callers
    (e.g. the TRT-LLM adapter) instantiate it directly.  The
    ``device_type`` is ``"cuda"`` because PyTorch on ROCm reports
    ``tensor.device.type == "cuda"``.
    """

    device_type: ClassVar[str] = "cuda"
    _is_default_wrapper: ClassVar[bool] = False

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        assert_contiguous(tensor)

        lib = _load_libhip()
        if lib is None:
            raise RuntimeError(
                "RocmRawIPCWrapper: libamdhip64.so not found. "
                "Ensure ROCm is installed and libamdhip64.so is in the "
                "library path."
            )

        data_ptr = tensor.data_ptr()
        handle = _HipIpcMemHandle()
        err = lib.hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(data_ptr))
        if err != 0:
            raise RuntimeError(
                f"hipIpcGetMemHandle failed: error {err} (ptr=0x{data_ptr:x})"
            )

        # Store only what's needed for reconstruction.
        self._ipc_handle_reserved = bytes(handle.reserved)
        self._nbytes = tensor.untyped_storage().nbytes()

        # DeviceIPCWrapper interface fields.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw HIP IPC."""
        # Third Party
        import cupy

        lib = _load_libhip()
        if lib is None:
            raise RuntimeError(
                "RocmRawIPCWrapper.to_tensor: libamdhip64.so not found"
            )

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        handle = _HipIpcMemHandle()
        handle.reserved = self._ipc_handle_reserved
        ptr = ctypes.c_void_p()
        err = lib.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            handle,
            ctypes.c_uint(_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
        )
        if err != 0:
            raise RuntimeError(f"hipIpcOpenMemHandle failed: error {err}")

        # Wrap as a flat uint8 CuPy array, DLPack to torch, then view
        # as the original dtype/shape.  uint8 avoids dtype-conversion
        # gaps (bfloat16, fp8 have no direct CuPy/NumPy equivalent).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(ptr.value, self._nbytes, owner=self)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)
