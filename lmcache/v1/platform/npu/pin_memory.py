# SPDX-License-Identifier: Apache-2.0
"""NPU host-memory pinning: try the CANN ``acl`` module first, then
``libascendcl`` via ctypes."""

# Standard
from typing import Protocol, cast
import ctypes
import ctypes.util
import glob
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import round_down, round_up
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)

#: ``aclrtHostRegisterType``: ``ACL_HOST_REGISTER_MAPPED`` is the only defined
#: value.
_ACL_HOST_REGISTER_MAPPED = 0

_ACL_SUCCESS = 0

try:
    _PAGE_SIZE = os.sysconf("SC_PAGESIZE")
except (AttributeError, ValueError, OSError):
    _PAGE_SIZE = 4096


class _AclRuntime(Protocol):
    """``acl.rt`` surface required for host-memory registration.

    Satisfied by the CANN ``acl.rt`` module and by :class:`_LibascendclRuntime`.
    ``host_register`` mirrors the C ``aclrtHostRegister`` including its
    out-param: it returns ``(dev_ptr, ret_code)``. The device-mapped alias is
    unused -- torch ``copy_`` keeps addressing the original host pointer, which
    the registration page-locks for DMA.
    """

    def host_register(self, ptr: int, size: int, flags: int) -> tuple[int, int]:
        """Register one host range; return ``(dev_ptr, ret_code)``."""
        ...

    def host_unregister(self, ptr: int) -> int:
        """Unregister one host range; return the AscendCL status code."""
        ...


class _LibascendclRuntime:
    """Adapter giving a ctypes-loaded ``libascendcl`` the ``acl.rt`` shape."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        """Adapt an already symbol-bound library handle."""
        self._lib = lib

    def host_register(self, ptr: int, size: int, flags: int) -> tuple[int, int]:
        """Call ``aclrtHostRegister``; return ``(dev_ptr, ret_code)``."""
        dev_ptr = ctypes.c_void_p(0)
        ret = self._lib.aclrtHostRegister(
            ctypes.c_void_p(ptr),
            ctypes.c_uint64(size),
            ctypes.c_int32(flags),
            ctypes.byref(dev_ptr),
        )
        return dev_ptr.value or 0, ret

    def host_unregister(self, ptr: int) -> int:
        """Call ``aclrtHostUnregister``; return the AscendCL status code."""
        return self._lib.aclrtHostUnregister(ctypes.c_void_p(ptr))


def _load_acl_rt() -> _AclRuntime | None:
    """Return the CANN ``acl.rt`` runtime module for host registration.

    The vendor module ships with CANN and shares the runtime instance
    ``torch_npu`` already loaded, so registrations issued through it are
    visible to ``copy_(..., non_blocking=True)``.

    Returns:
        ``acl.rt``, or ``None`` when the module or its register/unregister
        entry points are unavailable.
    """
    try:
        # Third Party
        import acl
    except ImportError as exc:
        logger.debug("NpuPinMemoryBackend: CANN acl module unavailable: %s", exc)
        return None

    rt = getattr(acl, "rt", None)
    if (
        rt is None
        or not callable(getattr(rt, "host_register", None))
        or not callable(getattr(rt, "host_unregister", None))
    ):
        logger.debug("NpuPinMemoryBackend: acl.rt host registration unavailable")
        return None
    return cast(_AclRuntime, rt)


def _candidate_lib_paths() -> list[str]:
    """Return ``libascendcl`` search candidates, most-specific first.

    Honors ``$ASCEND_HOME_PATH`` and the standard CANN install layout
    (``/usr/local/Ascend/cann*/<arch>-linux/lib64``). Filesystem probes only:
    :func:`ctypes.util.find_library` is intentionally not used here because it
    forks ``ldconfig``.
    """
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        return [os.path.join(home, "lib64", "libascendcl.so")]
    return sorted(glob.glob("/usr/local/Ascend/cann*/*-linux/lib64/libascendcl.so"))


def _bind_libascendcl(paths: list[str]) -> ctypes.CDLL | None:
    """dlopen the first path whose register/unregister symbols bind.

    ``dlopen`` returns the instance already loaded by ``torch_npu`` when one
    exists, so the handle shares ``torch_npu``'s device context.

    Returns:
        The bound library, or ``None`` when every candidate fails.
    """
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
            lib.aclrtHostRegister.restype = ctypes.c_int32
            lib.aclrtHostRegister.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            lib.aclrtHostUnregister.restype = ctypes.c_int32
            lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
            logger.info("NpuPinMemoryBackend: loaded libascendcl from %s", path)
            return lib
        except (OSError, AttributeError):
            continue
    return None


def _load_libascendcl() -> _AclRuntime | None:
    """Load ``libascendcl`` via ctypes as an ``acl.rt``-shaped fallback.

    The :func:`ctypes.util.find_library` probe is only paid when the cheap
    candidates fail, because it forks ``ldconfig``.

    Returns:
        The adapted runtime, or ``None`` when no usable library is found.
    """
    lib = _bind_libascendcl(_candidate_lib_paths())
    if lib is None:
        found = ctypes.util.find_library("ascendcl")
        if found:
            lib = _bind_libascendcl([found])
    if lib is None:
        logger.warning(
            "NpuPinMemoryBackend: libascendcl not available; NPU host "
            "pinning disabled, D2H/H2D will be synchronous"
        )
        return None
    return _LibascendclRuntime(lib)


class NpuPinMemoryBackend(PinMemoryBackend):
    """Pin host memory for NPU DMA via AscendCL ``aclrtHostRegister``.

    AscendCL requires a page-aligned ``ptr`` and a page-multiple ``size``, so
    every registration is widened to whole pages; the caller's original
    pointer is kept as the bookkeeping key so :meth:`unpin_memory` can reverse
    the registration when handed that same pointer.

    Attributes:
        _rt: ``acl.rt``-shaped runtime, or ``None`` when unavailable.
        _registered_bases: Maps the caller's original pointer to the
            page-aligned registered base.
        _no_npu: Latched when the NPU is unavailable or no ACL context can be
            established; makes :attr:`is_pin_supported` report ``False``
            thereafter.
        _tls: Per-thread "context ensured" flag; ACL contexts are
            thread-local.
    """

    def __init__(self) -> None:
        """Discover a runtime binding without raising on Ascend-less hosts.

        Notes:
            Discovery prefers the vendor ``acl`` module and falls back to
            ``libascendcl`` via :mod:`ctypes`. When both fail the backend
            stays unsupported and pin/unpin return ``False``.
        """
        self._rt: _AclRuntime | None = _load_acl_rt() or _load_libascendcl()
        self._registered_bases: dict[int, int] = {}
        self._no_npu: bool = False
        self._tls = threading.local()

    def _ensure_context(self) -> bool:
        """Ensure an ACL device context is current on the calling thread.

        ``aclrtHostRegister`` fails with AscendCL ``107002`` (no current
        context) when the thread has never touched the device. Worker threads
        are covered (torch_npu creates the context on first device op), but a
        cache-server thread that only manages host memory may have none, so
        the first pin attempt on each thread establishes one.

        Returns:
            True when a context is current, False when the NPU is unavailable
            (latched into ``_no_npu``).
        """
        if self._no_npu:
            return False
        if getattr(self._tls, "ensured", False):
            return True
        try:
            # Third Party
            import torch

            npu = getattr(torch, "npu", None)
            if npu is None or not npu.is_available():
                self._no_npu = True
                logger.info(
                    "NpuPinMemoryBackend: NPU unavailable; host pinning "
                    "disabled (copies will be synchronous)"
                )
                return False
            try:
                # Re-bind the current device instead of a hardcoded index so
                # worker ranks already bound to device N are not shifted.
                npu.set_device(npu.current_device())
            except Exception:
                # current_device() can raise before torch_npu initializes;
                # device 0 is the fallback.
                npu.set_device(0)
        except Exception as exc:
            # Accessing torch.npu lazily imports torch_npu, which can itself
            # raise on a broken CANN setup.
            self._no_npu = True
            logger.warning(
                "NpuPinMemoryBackend: cannot establish NPU context for "
                "pinning: %r; copies will be synchronous",
                exc,
            )
            return False
        self._tls.ensured = True
        return True

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Page-lock ``[ptr, ptr + size)`` for async NPU DMA.

        Args:
            ptr: Raw host pointer to the memory region.
            size: Region size in bytes.
            flags: Unused; AscendCL's only register type is
                ``ACL_HOST_REGISTER_MAPPED``.

        Returns:
            True if registration succeeded, False otherwise (callers degrade
            to synchronous copies).
        """
        rt = self._rt
        if rt is None or ptr == 0 or size <= 0:
            return False
        if not self._ensure_context():
            return False

        base = round_down(ptr, _PAGE_SIZE)
        reg_size = round_up(size + (ptr - base), _PAGE_SIZE)
        try:
            _, ret = rt.host_register(base, reg_size, _ACL_HOST_REGISTER_MAPPED)
        except Exception as exc:
            logger.warning(
                "aclrtHostRegister failed for ptr=%#x size=%d: %s", ptr, size, exc
            )
            return False
        if ret != _ACL_SUCCESS:
            logger.warning(
                "aclrtHostRegister failed: ptr=%#x base=%#x size=%d ret=%d; "
                "D2H/H2D will be synchronous",
                ptr,
                base,
                reg_size,
                ret,
            )
            return False
        self._registered_bases[ptr] = base
        return True

    def unpin_memory(self, ptr: int) -> bool:
        """Unregister a previously pinned region.

        The bookkeeping entry is only dropped on success, so a failed
        unregistration can be retried with the same pointer.

        Args:
            ptr: The original pointer passed to :meth:`pin_memory`.

        Returns:
            True if unregistration succeeded or nothing was registered for
            ``ptr``; False when the backend is unavailable or AscendCL
            reports an error.
        """
        rt = self._rt
        if rt is None:
            return False
        base = self._registered_bases.get(ptr)
        if base is None:
            return True
        try:
            ret = rt.host_unregister(base)
        except Exception as exc:
            logger.warning("aclrtHostUnregister failed for ptr=%#x: %s", ptr, exc)
            return False
        if ret != _ACL_SUCCESS:
            logger.warning(
                "aclrtHostUnregister failed: ptr=%#x base=%#x ret=%d", ptr, base, ret
            )
            return False
        del self._registered_bases[ptr]
        return True

    @property
    def is_pin_supported(self) -> bool:
        """Whether AscendCL host registration is usable.

        Returns:
            True when a runtime binding loaded and the NPU has not been found
            unavailable, False otherwise.
        """
        return self._rt is not None and not self._no_npu
