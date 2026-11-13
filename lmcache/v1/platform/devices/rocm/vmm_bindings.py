# SPDX-License-Identifier: Apache-2.0
"""The ``cuda.bindings.driver`` VMM calls used by the VMM IPC wrapper, on ROCm.

:class:`~lmcache.v1.platform.devices.cuda.ipc_wrapper.VmmCudaIPCWrapper`
is written against ``cuda.bindings.driver``. HIP has the same VMM API, and
``rocm.bindings.hip`` binds it, but a few binding conventions differ; this
module, the driver of
:class:`~lmcache.v1.platform.devices.rocm.ipc_wrapper.VmmRocmIPCWrapper`,
forwards the names the wrapper uses and absorbs those differences:

- ``hipMemSetAccess`` takes one descriptor, not a list.
- ``hipMemExportToShareableHandle`` writes the handle to a caller pointer.
- ``hipMemAllocationProp`` does not expose its anonymous
  ``requestedHandleTypes`` union member.

Every function returns the same tuple shape as its ``cuda.bindings`` namesake.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import ctypes

# Third Party
from rocm.bindings import hip

# hipMemAllocationProp: int type, then the requestedHandleTypes union.
_REQUESTED_HANDLE_TYPES_OFFSET = ctypes.sizeof(ctypes.c_int)


class CUresult:
    CUDA_SUCCESS = hip.hipError_t.hipSuccess


class CUmemAllocationHandleType:
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = (
        hip.hipMemAllocationHandleType.hipMemHandleTypePosixFileDescriptor
    )
    CU_MEM_HANDLE_TYPE_FABRIC = hip.hipMemAllocationHandleType.hipMemHandleTypeFabric


class CUmemLocationType:
    CU_MEM_LOCATION_TYPE_DEVICE = hip.hipMemLocationType.hipMemLocationTypeDevice


class CUmemAccess_flags:
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = (
        hip.hipMemAccessFlags.hipMemAccessFlagsProtReadWrite
    )


CUmemAccessDesc = hip.hipMemAccessDesc


@dataclass(frozen=True)
class _AllocationProperties:
    requestedHandleTypes: int


def _clear_sticky_error(result: tuple) -> tuple:
    """Return ``result`` after clearing HIP's error state if the call failed.

    A failed ``hipMem*`` call leaves its error sticky on the thread, so the
    next kernel launch (e.g. any torch op) would fail with it.
    """
    if result[0] != hip.hipError_t.hipSuccess:
        hip.hipGetLastError()
    return result


def CUdeviceptr(ptr: int) -> int:
    """HIP takes device pointers as plain integers."""
    return ptr


def cuMemRetainAllocationHandle(ptr: int) -> tuple:
    """``hipMemRetainAllocationHandle``: ``(err, handle)``."""
    return _clear_sticky_error(hip.hipMemRetainAllocationHandle(ptr))


def cuMemGetAllocationPropertiesFromHandle(handle: object) -> tuple:
    """``hipMemGetAllocationPropertiesFromHandle``: ``(err, props)``.

    Only ``props.requestedHandleTypes`` is provided, read from the union the
    binding leaves out.
    """
    err, prop = _clear_sticky_error(hip.hipMemGetAllocationPropertiesFromHandle(handle))
    if err != hip.hipError_t.hipSuccess:
        return err, None
    address = prop.as_c_void_p().value + _REQUESTED_HANDLE_TYPES_OFFSET
    return err, _AllocationProperties(ctypes.c_int.from_address(address).value)


def cuMemExportToShareableHandle(
    handle: object, handle_type: object, flags: int
) -> tuple:
    """``hipMemExportToShareableHandle`` for POSIX fds: ``(err, fd)``.

    FABRIC export is not supported on ROCm and returns
    ``hipErrorNotSupported``.
    """
    if (
        handle_type
        != CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    ):
        return hip.hipError_t.hipErrorNotSupported, None
    fd = ctypes.c_int(-1)
    (err,) = _clear_sticky_error(
        hip.hipMemExportToShareableHandle(
            ctypes.addressof(fd), handle, handle_type, flags
        )
    )
    return err, fd.value


def cuMemGetAddressRange(ptr: int) -> tuple:
    """``hipMemGetAddressRange``: ``(err, base, size)``."""
    err, base, size = _clear_sticky_error(hip.hipMemGetAddressRange(ptr))
    return err, int(base), size


def cuMemImportFromShareableHandle(os_handle: object, handle_type: object) -> tuple:
    """``hipMemImportFromShareableHandle``: ``(err, handle)``.

    A POSIX fd travels in the pointer argument by value, as on CUDA.
    """
    return _clear_sticky_error(
        hip.hipMemImportFromShareableHandle(os_handle, handle_type)
    )


def cuMemAddressReserve(size: int, alignment: int, addr: int, flags: int) -> tuple:
    """``hipMemAddressReserve``: ``(err, base)``."""
    err, base = _clear_sticky_error(
        hip.hipMemAddressReserve(size, alignment, addr or None, flags)
    )
    return err, int(base)


def cuMemMap(ptr: int, size: int, offset: int, handle: object, flags: int) -> tuple:
    """``hipMemMap``: ``(err,)``."""
    return _clear_sticky_error(hip.hipMemMap(ptr, size, offset, handle, flags))


def cuMemSetAccess(ptr: int, size: int, descs: list, count: int) -> tuple:
    """``hipMemSetAccess`` with one descriptor: ``(err,)``."""
    return _clear_sticky_error(hip.hipMemSetAccess(ptr, size, descs[0], count))


def cuMemUnmap(ptr: int, size: int) -> tuple:
    """``hipMemUnmap``: ``(err,)``."""
    return _clear_sticky_error(hip.hipMemUnmap(ptr, size))


def cuMemAddressFree(ptr: int, size: int) -> tuple:
    """``hipMemAddressFree``: ``(err,)``."""
    return _clear_sticky_error(hip.hipMemAddressFree(ptr, size))


def cuMemRelease(handle: object) -> tuple:
    """``hipMemRelease``: ``(err,)``."""
    return _clear_sticky_error(hip.hipMemRelease(handle))
