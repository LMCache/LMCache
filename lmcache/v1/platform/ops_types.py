# SPDX-License-Identifier: Apache-2.0
"""Shared descriptor types for the unified ``DeviceOps`` surface.

``PageBufferShapeDesc`` and ``KernelGroupSpec`` are defined in
``lmcache.lmcache_native`` and re-exported here for callers that want the
platform-owned import path. ``StagingCopy`` / ``LaunchVar`` / ``BatchStep``
still only exist natively in the compiled ``cuda_ops`` extension, so this
module keeps placeholder stubs for those names.
"""

# Future
from __future__ import annotations

# Standard
from enum import Enum, IntEnum

# First Party
import lmcache.lmcache_native as _native

PageBufferShapeDesc = _native.PageBufferShapeDesc
KernelGroupSpec = _native.KernelGroupSpec


class TransferPhase(IntEnum):
    """Timed sections of the native object-group transfer plan executor.

    Mirrors the C++ ``TransferPhase`` enum in
    ``csrc/cuda/phase_timing_recorder.cuh``;
    values appear in ``device_ops.pop_completed_phase_timings()`` samples.
    Python-only: unlike ``TransferDirection`` (in
    :mod:`lmcache.lmcache_native`), this enum is not exposed by a native
    module.
    """

    KERNEL = 0
    STAGING = 1


class BlockTransferInputKind(Enum):
    """Operand convention of ``DeviceOps.multi_layer_block_kv_transfer``.

    Python-only: callers use it to build the arguments the resolved
    operation expects. It says nothing about whether a particular request is
    supported.
    """

    #: Paged KV tensors, chunk tensors, and ``list[int]`` block IDs.
    TENSORS = "tensors"
    #: An int64 tensor of paged-buffer pointers on the KV device, chunk data
    #: pointers as ``list[int]``, and an int64 block-ID tensor on the KV device.
    POINTERS = "pointers"


class _NativePlanType:
    """Base for object-group transfer plan types that only exist natively.

    The plan value structs (see ``csrc/cuda/mp_mem_kernels.cuh``) are built on the
    Python side and consumed by the native ``execute_object_group_transfer``.
    They have no pure-Python fallback, so constructing one without the compiled
    backend extension is unsupported. Subclasses exist only so the CPU-only
    build exposes the same names through ``lmcache.device_ops``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} requires a backend-specific native module; "
            "no generic pure-Python fallback exists."
        )


class StagingCopy(_NativePlanType):
    """Fallback stub for the native ``StagingCopy`` plan type."""


class LaunchVar(_NativePlanType):
    """Fallback stub for the native ``LaunchVar`` plan type."""


class BatchStep(_NativePlanType):
    """Fallback stub for the native ``BatchStep`` plan type."""
