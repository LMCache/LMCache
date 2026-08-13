# SPDX-License-Identifier: Apache-2.0
"""Descriptor aliases for the unified ``DeviceOps`` surface.

Only ``PageBufferShapeDesc`` and ``KernelGroupSpec`` are currently treated as
backend-agnostic native types and sourced from :mod:`lmcache.lmcache_native`.
Per-batch plan descriptors such as ``StagingCopy``, ``LaunchVar``, and
``BatchStep`` stay backend-local; this module therefore exposes placeholder
Python types for them until a concrete device backend binds real native types.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.lmcache_native import KernelGroupSpec as KernelGroupSpec
    from lmcache.lmcache_native import PageBufferShapeDesc as PageBufferShapeDesc

# First Party
import lmcache.lmcache_native as lmcache_native


class _FallbackPageBufferShapeDesc:
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


class _FallbackNativePlanType:
    """Base for native-only plan types absent from the generic surface."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} requires a backend-specific native module; "
            "no generic pure-Python fallback exists."
        )


class _FallbackStagingCopy(_FallbackNativePlanType):
    """Fallback stub for the CUDA-local ``StagingCopy`` plan type."""


class _FallbackLaunchVar(_FallbackNativePlanType):
    """Fallback stub for the CUDA-local ``LaunchVar`` plan type."""


class _FallbackBatchStep(_FallbackNativePlanType):
    """Fallback stub for the CUDA-local ``BatchStep`` plan type."""


class _FallbackKernelGroupSpec(_FallbackNativePlanType):
    """Fallback stub for the native ``KernelGroupSpec`` plan type."""


if not TYPE_CHECKING:
    _NATIVE_COMMON_DESCRIPTOR_NAMES = ("PageBufferShapeDesc", "KernelGroupSpec")

    if all(hasattr(lmcache_native, name) for name in _NATIVE_COMMON_DESCRIPTOR_NAMES):
        PageBufferShapeDesc = lmcache_native.PageBufferShapeDesc
        KernelGroupSpec = lmcache_native.KernelGroupSpec
    else:
        PageBufferShapeDesc = _FallbackPageBufferShapeDesc
        KernelGroupSpec = _FallbackKernelGroupSpec

StagingCopy: TypeAlias = _FallbackStagingCopy
LaunchVar: TypeAlias = _FallbackLaunchVar
BatchStep: TypeAlias = _FallbackBatchStep


def set_shape_desc_dtype(shape_desc: Any, dtype: torch.dtype) -> None:
    """Best-effort ``shape_desc.dtype = dtype``.

    The fallback ``PageBufferShapeDesc`` exposes a ``dtype`` slot so the
    CPU fallback kernel can disambiguate float16 vs bfloat16 (both have
    ``element_size == 2``). The native pybind class now allows dynamic
    attributes too, so this helper works uniformly across both surfaces.

    Args:
        shape_desc: A ``PageBufferShapeDesc`` instance (either the
            pure-Python fallback or the C++ pybind struct).
        dtype: The torch dtype to assign.
    """
    try:
        shape_desc.dtype = dtype
    except AttributeError:
        pass
