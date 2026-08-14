# SPDX-License-Identifier: Apache-2.0
"""Python-only types for the unified ``DeviceOps`` surface.

The native KV-format and transfer types live in :mod:`lmcache.lmcache_native`.
This module contains the Python fallback types used by device operations.

``PageBufferShapeDesc`` and ``KernelGroupSpec`` are sourced from
``lmcache_native`` when available. ``StagingCopy`` / ``LaunchVar`` /
``BatchStep`` still only exist natively in the compiled ``cuda_ops``
extension; the pure-Python subclasses here are stubs so the CPU-only build
exposes the same names.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    import torch

# First Party
import lmcache.lmcache_native as lmcache_native


class PageBufferShapeDesc:
    """Python stand-in for the native ``PageBufferShapeDesc`` struct.

    Mirrors the pybind ``def_readwrite`` attributes in
    ``csrc/lmcache_native/pybind.cpp`` so source-only and stubbed test
    environments can construct and inspect shape descriptors without the
    compiled extension.

    ``block_stride_elems`` captures the *physical* per-block step in
    element units (= ``tensor.stride(0)``). For a tightly-packed paged
    buffer it equals ``bs * kv_size * nh * hs`` (non-MLA) /
    ``bs * nh * hs`` (MLA); for a vLLM KV pool where a group's row is
    padded to the pool's maximum row width (e.g. DeepSeek V4 compressor
    / indexer caches), it is strictly larger. Downstream kernels must
    use this value instead of recomputing a "tight" stride from the
    logical shape, otherwise they'll skip into the next block's padding
    region and read/write the wrong slots.
    """

    __slots__ = (
        "kv_size",
        "nl",
        "nb",
        "bs",
        "nh",
        "hs",
        "element_size",
        "block_stride_elems",
        "dtype",
    )

    def __init__(self) -> None:
        self.kv_size: int = 0
        self.nl: int = 0
        self.nb: int = 0
        self.bs: int = 0
        self.nh: int = 0
        self.hs: int = 0
        self.element_size: int = 0
        # 0 means "unset — fall back to tight stride"; any downstream
        # consumer that needs exact addressing must check this.
        self.block_stride_elems: int = 0
        self.dtype: torch.dtype | None = None


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


class KernelGroupSpec(_NativePlanType):
    """Fallback stub for the native ``KernelGroupSpec`` plan type."""


if not TYPE_CHECKING:
    _NATIVE_COMMON_DESCRIPTOR_NAMES = ("PageBufferShapeDesc", "KernelGroupSpec")

    if all(hasattr(lmcache_native, name) for name in _NATIVE_COMMON_DESCRIPTOR_NAMES):
        PageBufferShapeDesc = lmcache_native.PageBufferShapeDesc
        KernelGroupSpec = lmcache_native.KernelGroupSpec


def set_shape_desc_dtype(shape_desc: Any, dtype: torch.dtype) -> None:
    """Best-effort ``shape_desc.dtype = dtype``.

    The pure-Python ``PageBufferShapeDesc`` exposes a ``dtype`` slot so the
    CPU fallback kernel can disambiguate float16 vs bfloat16 (both have
    ``element_size == 2``). The native pybind class also accepts dynamic
    attributes, so call sites do not need to branch on the active backend.

    Args:
        shape_desc: A ``PageBufferShapeDesc`` instance (either the
            pure-Python fallback or the C++ pybind struct).
        dtype: The torch dtype to assign.
    """
    try:
        shape_desc.dtype = dtype
    except AttributeError:
        pass
