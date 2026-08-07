# SPDX-License-Identifier: Apache-2.0
"""Shared, device-agnostic ops *types* for the unified ``DeviceOps`` surface.

These types (``TransferDirection``, ``EngineKVFormat``, ``GPUKVFormat``,
``PageBufferShapeDesc``, ``set_shape_desc_dtype``) were migrated verbatim from
the former ``lmcache.python_ops_fallback`` module. They are deliberately
self-contained (only ``torch`` + ``enum``) so call sites can import them
without pulling in the heavier torch-baseline op implementations in
:mod:`lmcache.v1.platform.torch_ops`.

The object-group transfer plan types (``_NativePlanType`` and its
``StagingCopy`` / ``LaunchVar`` / ``BatchStep`` / ``KernelGroupSpec``
subclasses) only exist natively in the compiled ``c_ops`` extension; the
pure-Python subclasses here are stubs so the CPU-only build exposes the same
names.
"""

# Future
from __future__ import annotations

# Standard
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    import torch


# The device-independent KV-format / transfer types now live in the compiled
# ``lmache_native`` module (built from c_src/storage_manager/pybind.cpp). When
# the extension is available we re-export the native definitions so the Python
# and C++ sides share one source of truth; otherwise we fall back to the
# pure-Python IntEnum declarations below so the CPU-only build still works.
try:
    # First Party
    from lmache_native import (  # type: ignore
        EngineKVFormat,
        TransferDirection,
        is_cross_layer,
        is_kv_list,
        is_layer_list,
        is_mla,
    )
except Exception:  # pragma: no cover - native ext not built
    EngineKVFormat = None  # type: ignore[assignment]
    TransferDirection = None  # type: ignore[assignment]
    is_cross_layer = None  # type: ignore[assignment]
    is_kv_list = None  # type: ignore[assignment]
    is_layer_list = None  # type: ignore[assignment]
    is_mla = None  # type: ignore[assignment]


if EngineKVFormat is None:  # pragma: no cover - native ext not built

    class TransferDirection(IntEnum):
        """Specifies the direction of a memory transfer.

        Inherits from IntEnum so that members compare equal to plain ints
        and to native pybind11 enum members with the same integer value.
        Several call sites (and the fallback ops themselves) use
        ``int(direction)`` to compare across backend / fallback boundaries.
        """

        H2D = 0
        D2H = 1

    class EngineKVFormat(IntEnum):
        """Enumeration of different engine KV cache memory layouts."""

        # used by: vLLM CROSS_LAYER mode
        NB_NL_TWO_BS_NH_HS = 0

        # used by: vLLM non-MLA flash attention
        NL_X_TWO_NB_BS_NH_HS = 1

        # used by: vLLM non-MLA flash infer
        NL_X_NB_TWO_BS_NH_HS = 2

        # used by: vLLM MLA
        NL_X_NB_BS_HS = 3

        # used by: SGLang MHA (flash attention and flash infer)
        TWO_X_NL_X_NBBS_NH_HS = 4

        # used by: SGLang MLA
        NL_X_NBBS_ONE_HS = 5

        # used by: vLLM non-MLA flash attention (HND layout)
        NL_X_TWO_NB_NH_BS_HS = 6

        # used by: vLLM non-MLA flash infer (HND layout)
        NL_X_NB_TWO_NH_BS_HS = 7

        # used by: TRT-LLM cross-layer (HND layout)
        NB_NL_TWO_NH_BS_HS = 8

        # used by: SGLang MHA via the MP daemon path
        TWO_X_NL_X_NB_BS_NH_HS = 9

        # DEPRECATED: superseded by NL_X_NB_NH_BS_CS; no longer produced by
        # detection.
        NL_X_NB_NH_BS_TWO_HS = 10

        # DEPRECATED: superseded by NL_X_NB_BS_NH_CS; no longer produced by
        # detection.
        NL_X_NB_BS_NH_TWO_HS = 11

        # used by: vLLM non-MLA blocks-first attention (HND layout, unified KV
        # cache) with K/V fused into the trailing content dim.
        NL_X_NB_NH_BS_CS = 12

        # used by: vLLM non-MLA blocks-first attention (NHD layout, unified KV
        # cache) with K/V fused into the trailing content dim.
        NL_X_NB_BS_NH_CS = 13

        # vLLM DSA indexer k-cache [NB,BS,132] u8, paged [BSxvals][BSxscales];
        # c_ops only (no pure-torch transfer path)
        NL_X_NB_BSV_BSS = 14

    def is_kv_list(engine_kv_format: EngineKVFormat) -> bool:
        """Fallback predicate: KV stored as a list of per-token KV tensors."""
        return engine_kv_format in (
            EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS,
            EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
        )

    def is_layer_list(engine_kv_format: EngineKVFormat) -> bool:
        """Fallback predicate: one list entry per layer."""
        return engine_kv_format in (
            EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
            EngineKVFormat.NL_X_NB_BS_HS,
            EngineKVFormat.NL_X_NBBS_ONE_HS,
            EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
            EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
            EngineKVFormat.NB_NL_TWO_NH_BS_HS,
            EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
            EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
            EngineKVFormat.NL_X_NB_NH_BS_CS,
            EngineKVFormat.NL_X_NB_BS_NH_CS,
            EngineKVFormat.NL_X_NB_BSV_BSS,
        )

    def is_cross_layer(engine_kv_format: EngineKVFormat) -> bool:
        """Fallback predicate: layers fused into a single tensor."""
        return engine_kv_format in (
            EngineKVFormat.NB_NL_TWO_BS_NH_HS,
            EngineKVFormat.NB_NL_TWO_NH_BS_HS,
        )

    def is_mla(engine_kv_format: EngineKVFormat) -> bool:
        """Fallback predicate: MLA variant (single latent KV head)."""
        return engine_kv_format in (
            EngineKVFormat.NL_X_NB_BS_HS,
            EngineKVFormat.NL_X_NBBS_ONE_HS,
            EngineKVFormat.NL_X_NB_BSV_BSS,
        )


# Backward-compat alias
GPUKVFormat = EngineKVFormat


class PageBufferShapeDesc:
    """Python stand-in for the C++ ``PageBufferShapeDesc`` struct.

    Mirrors the pybind ``def_readwrite`` attributes in ``csrc/pybind.cpp``
    so non-CUDA code paths can construct and inspect shape descriptors
    without the compiled extension.

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
    # TODO: Deprecate this class if it will not be used by 2+ vendors
    """Base for object-group transfer plan types that only exist natively.

    The plan value structs (see ``csrc/mp_mem_kernels.cuh``) are built on the
    Python side and consumed by the native ``execute_object_group_transfer``.
    They have no pure-Python fallback, so constructing one without the compiled
    ``c_ops`` extension is unsupported. Subclasses exist only so the CPU-only
    build exposes the same names as ``c_ops``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} requires the c_ops native extension; "
            "no pure-Python fallback exists."
        )


class StagingCopy(_NativePlanType):
    """Fallback stub for the native ``StagingCopy`` plan type."""


class LaunchVar(_NativePlanType):
    """Fallback stub for the native ``LaunchVar`` plan type."""


class BatchStep(_NativePlanType):
    """Fallback stub for the native ``BatchStep`` plan type."""


class KernelGroupSpec(_NativePlanType):
    """Fallback stub for the native ``KernelGroupSpec`` plan type."""


def set_shape_desc_dtype(shape_desc: Any, dtype: torch.dtype) -> None:
    """Best-effort ``shape_desc.dtype = dtype``.

    The pure-Python ``PageBufferShapeDesc`` exposes a ``dtype`` slot so
    the CPU fallback kernel can disambiguate float16 vs bfloat16 (both
    have ``element_size == 2``). The pybind C++ struct in
    ``csrc/pybind.cpp`` has no such field; assignment raises
    ``AttributeError`` and is silently swallowed here so call sites
    don't need to branch on the active backend.

    Args:
        shape_desc: A ``PageBufferShapeDesc`` instance (either the
            pure-Python fallback or the C++ pybind struct).
        dtype: The torch dtype to assign.
    """
    try:
        shape_desc.dtype = dtype
    except AttributeError:
        pass
