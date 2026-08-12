# SPDX-License-Identifier: Apache-2.0
"""Shared descriptor aliases for the unified ``DeviceOps`` surface.

The canonical native descriptor types live in :mod:`lmcache.lmcache_native`.
This module re-exports those classes when available, and falls back to small
Python stand-ins in source-only / stubbed test environments that do not expose
the compiled descriptor classes.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.lmcache_native import BatchStep as BatchStep
    from lmcache.lmcache_native import KernelGroupSpec as KernelGroupSpec
    from lmcache.lmcache_native import LaunchVar as LaunchVar
    from lmcache.lmcache_native import PageBufferShapeDesc as PageBufferShapeDesc
    from lmcache.lmcache_native import StagingCopy as StagingCopy
else:
    # First Party
    import lmcache.lmcache_native as lmcache_native

    _NATIVE_DESCRIPTOR_NAMES = (
        "PageBufferShapeDesc",
        "StagingCopy",
        "LaunchVar",
        "BatchStep",
        "KernelGroupSpec",
    )

    if all(hasattr(lmcache_native, name) for name in _NATIVE_DESCRIPTOR_NAMES):
        PageBufferShapeDesc = lmcache_native.PageBufferShapeDesc
        StagingCopy = lmcache_native.StagingCopy
        LaunchVar = lmcache_native.LaunchVar
        BatchStep = lmcache_native.BatchStep
        KernelGroupSpec = lmcache_native.KernelGroupSpec
    else:

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
            """Base for native-only plan types absent from the stub module."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise NotImplementedError(
                    f"{type(self).__name__} requires lmcache.lmcache_native; "
                    "no pure-Python fallback exists."
                )

        class _FallbackStagingCopy(_FallbackNativePlanType):
            """Fallback stub for the native ``StagingCopy`` plan type."""

        class _FallbackLaunchVar(_FallbackNativePlanType):
            """Fallback stub for the native ``LaunchVar`` plan type."""

        class _FallbackBatchStep(_FallbackNativePlanType):
            """Fallback stub for the native ``BatchStep`` plan type."""

        class _FallbackKernelGroupSpec(_FallbackNativePlanType):
            """Fallback stub for the native ``KernelGroupSpec`` plan type."""

        PageBufferShapeDesc = _FallbackPageBufferShapeDesc
        StagingCopy = _FallbackStagingCopy
        LaunchVar = _FallbackLaunchVar
        BatchStep = _FallbackBatchStep
        KernelGroupSpec = _FallbackKernelGroupSpec


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
