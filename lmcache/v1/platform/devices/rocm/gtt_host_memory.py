# SPDX-License-Identifier: Apache-2.0
"""Driver-owned GTT host memory segments on ROCm.

ROCm backs ``hipHostRegister`` memory and, by default, ``hipHostMalloc``
memory with KFD userptr buffer objects whose pages stay movable: compaction,
khugepaged and ``move_pages`` migrate them, and every migration makes KFD
evict the process' queues on all GPUs. When ``HSA_USERPTR_FOR_PAGED_MEM=0`` is
in the environment at ROCr initialization, ``hipHostMalloc`` instead returns a
GTT buffer object owned by the amdgpu driver and mapped through
a DRM render node (``/dev/dri/render*``); its pages cannot be migrated.
"""

# Standard
import os

# First Party
from lmcache import device_ops

GTT_ENV_VAR = "HSA_USERPTR_FOR_PAGED_MEM"

GTT_MAX_ALLOCATION_BYTES = 512 << 30
"""Exclusive upper bound of one GTT ``hipHostMalloc`` on MI355X: ``512 GiB -
4 KiB`` succeeds, ``512 GiB`` fails with ``hipErrorOutOfMemory``."""

GTT_MAPPING_PREFIX = "/dev/dri/render"

GTT_ENV_EXAMPLE = (
    "Set it in the LMCache server's environment before the process starts, "
    "for example `docker run -e HSA_USERPTR_FOR_PAGED_MEM=0 ...` or, in a "
    "Kubernetes container spec:\n"
    "  env:\n"
    "    - name: HSA_USERPTR_FOR_PAGED_MEM\n"
    '      value: "0"'
)


class GttEnvironmentError(RuntimeError):
    """The process cannot obtain driver-owned GTT host memory."""


class GttAllocationError(RuntimeError):
    """A GTT host memory segment could not be allocated."""


def check_gtt_environment() -> None:
    """Require ``HSA_USERPTR_FOR_PAGED_MEM=0`` in the process environment.

    ROCr reads the variable once at initialization, so it must come from the
    environment the server was started with; setting it from Python has no
    effect once the HIP runtime is up.

    Raises:
        GttEnvironmentError: If the variable is missing or not ``"0"``.
    """
    value = os.environ.get(GTT_ENV_VAR)
    if value != "0":
        raise GttEnvironmentError(
            "The GTT L1 host memory backend requires "
            f"{GTT_ENV_VAR}=0 (found {value!r}); without it hipHostMalloc "
            "returns movable userptr memory, whose page migrations evict the "
            "GPU queues and freeze transfers. " + GTT_ENV_EXAMPLE
        )


def default_gtt_segment_size(align_bytes: int) -> int:
    """Return the largest allowed GTT segment size for ``align_bytes``.

    Args:
        align_bytes: L1 allocation alignment, a power of two.

    Returns:
        The largest multiple of ``max(align_bytes, page size)`` strictly below
        :data:`GTT_MAX_ALLOCATION_BYTES`.
    """
    granule = max(align_bytes, os.sysconf("SC_PAGE_SIZE"))
    return (GTT_MAX_ALLOCATION_BYTES - 1) // granule * granule


def validate_gtt_segment_size(segment_size: int, align_bytes: int) -> None:
    """Check a configured GTT segment size.

    Args:
        segment_size: Requested bytes per GTT segment.
        align_bytes: L1 allocation alignment, a power of two.

    Raises:
        ValueError: If the size is not positive, not a multiple of
            ``max(align_bytes, page size)``, or not below 512 GiB.
    """
    granule = max(align_bytes, os.sysconf("SC_PAGE_SIZE"))
    if (
        segment_size <= 0
        or segment_size >= GTT_MAX_ALLOCATION_BYTES
        or segment_size % granule != 0
    ):
        raise ValueError(
            f"GTT segment size {segment_size} bytes is invalid: it must be a "
            f"positive multiple of {granule} bytes and below "
            f"{GTT_MAX_ALLOCATION_BYTES} bytes (512 GiB; one hipHostMalloc of "
            "512 GiB - 4 KiB is the measured MI355X limit). The default, "
            f"{default_gtt_segment_size(align_bytes)} bytes, is the largest "
            "valid value."
        )


def _mapping_path(ptr: int) -> str | None:
    """Return the backing path of the ``/proc/self/maps`` entry holding ptr."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split()
            start, end = (int(x, 16) for x in fields[0].split("-"))
            if start <= ptr < end:
                return fields[5] if len(fields) > 5 else ""
    return None


def alloc_gtt_segment(size: int) -> int:
    """Allocate one driver-owned GTT host memory segment.

    Args:
        size: Bytes to allocate; must be below :data:`GTT_MAX_ALLOCATION_BYTES`.

    Returns:
        The host pointer of the segment, accessible by the CPU and all GPUs.

    Raises:
        GttEnvironmentError: If the environment is not set up for GTT, or the
            runtime returned memory that is not a GTT mapping (for example
            because ``HSA_USERPTR_FOR_PAGED_MEM`` was set after ROCr started).
            The segment is freed before raising.
        GttAllocationError: If ``hipHostMalloc`` fails; the message carries
            the HIP error code.
    """
    check_gtt_environment()
    try:
        ptr = device_ops.alloc_pinned_ptr(size, 0)
    except RuntimeError as e:
        raise GttAllocationError(f"hipHostMalloc of {size} bytes failed ({e})") from e
    path = _mapping_path(ptr)
    if path is None or not path.startswith(GTT_MAPPING_PREFIX):
        device_ops.free_pinned_ptr(ptr)
        raise GttEnvironmentError(
            f"hipHostMalloc returned {path or 'anonymous'} memory instead of a "
            f"{GTT_MAPPING_PREFIX}* GTT mapping. {GTT_ENV_VAR}=0 must be in "
            "the environment before the HIP runtime initializes. " + GTT_ENV_EXAMPLE
        )
    return ptr


def free_gtt_segment(ptr: int) -> None:
    """Free a segment returned by :func:`alloc_gtt_segment` (``hipHostFree``).

    Args:
        ptr: Host pointer returned by :func:`alloc_gtt_segment`.
    """
    device_ops.free_pinned_ptr(ptr)
