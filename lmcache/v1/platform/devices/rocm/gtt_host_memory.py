# SPDX-License-Identifier: Apache-2.0
"""Non-migratable GTT host memory on ROCm.

``hipHostRegister`` memory and default ``hipHostMalloc`` memory are KFD userptr
allocations whose pages the kernel may migrate; every migration evicts the
process' GPU queues. With ``HSA_USERPTR_FOR_PAGED_MEM=0``, ``hipHostMalloc``
returns driver-owned GTT memory mapped through ``/dev/dri/render*`` instead,
which cannot be migrated.
"""

# Standard
import os

# First Party
from lmcache import device_ops

GTT_ENV_VAR = "HSA_USERPTR_FOR_PAGED_MEM"

GTT_MAX_ALLOCATION_BYTES = 512 << 30
"""Exclusive upper bound of one GTT ``hipHostMalloc`` (``512 GiB - 4 KiB``
succeeds on MI355X)."""


def check_gtt_environment() -> None:
    """Require ``HSA_USERPTR_FOR_PAGED_MEM=0`` in the process environment.

    ROCr reads the variable once at initialization, so setting it from Python
    after the HIP runtime is up has no effect.

    Raises:
        RuntimeError: If the variable is not ``"0"``.
    """
    value = os.environ.get(GTT_ENV_VAR)
    if value != "0":
        raise RuntimeError(
            f"--l1-host-memory-backend gtt requires {GTT_ENV_VAR}=0 (found "
            f"{value!r}); set it in the server's environment before it starts."
        )


def _mapping_path(ptr: int) -> str:
    """Return the backing path of the ``/proc/self/maps`` entry holding ptr."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split()
            start, end = (int(x, 16) for x in fields[0].split("-"))
            if start <= ptr < end:
                return fields[5] if len(fields) > 5 else ""
    return ""


def alloc_gtt_segment(size: int) -> int:
    """Allocate one GTT host memory segment with ``hipHostMalloc``.

    Args:
        size: Bytes to allocate, below :data:`GTT_MAX_ALLOCATION_BYTES`.

    Returns:
        The host pointer of the segment.

    Raises:
        RuntimeError: If ``hipHostMalloc`` fails, or returns memory that is
            not a GTT mapping (the memory is freed first).
    """
    ptr = device_ops.alloc_pinned_ptr(size, 0)
    path = _mapping_path(ptr)
    if not path.startswith("/dev/dri/render"):
        device_ops.free_pinned_ptr(ptr)
        raise RuntimeError(
            f"hipHostMalloc returned {path or 'anonymous'} memory instead of GTT "
            f"memory; {GTT_ENV_VAR}=0 must be set before the HIP runtime starts."
        )
    return ptr


def free_gtt_segment(ptr: int) -> None:
    """Free a segment returned by :func:`alloc_gtt_segment`.

    Args:
        ptr: Host pointer returned by :func:`alloc_gtt_segment`.
    """
    device_ops.free_pinned_ptr(ptr)
