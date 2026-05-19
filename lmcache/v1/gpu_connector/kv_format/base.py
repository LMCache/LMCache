# SPDX-License-Identifier: Apache-2.0
"""
:class:`KVFormatSpec` — strategy object describing one ``GPUKVFormat``.

The previous incarnation of :mod:`lmcache.v1.gpu_connector.utils`
encoded every per-format axis lookup as a giant ``if/elif`` ladder
spread across ten module-level functions. Adding a new format meant
chasing the same enum value through ten places, and the per-format
metadata (HND? MLA? cross-layer? block-axis on dim-0?) was scattered
across half a dozen helpers. That refactor was, frankly, not great.

This module replaces the ladder with one strategy class per format:

* Class-level :class:`ClassVar`\\ s describe *static* properties of
  the format (shape descriptor, attention backend label, MLA / HND /
  cross-layer flags). They are queryable from the class, no
  ``kv_caches`` needed.
* Instance-level methods describe *dynamic* properties that need a
  concrete ``kv_caches`` value (``num_layers``, ``num_blocks``,
  ``block_size``, data pointers, ...).

A new format becomes a single new file under ``specs/``. Registration
is automatic via :meth:`__init_subclass__`; module discovery is
lazy via :mod:`lmcache.v1.gpu_connector.kv_format.registry`.
"""

# Standard
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
import lmcache.c_ops as lmc_ops


class AxisLayout(Enum):
    """Physical layout of the head/block axes inside a per-layer tensor."""

    NHD = "NHD"
    HND = "HND"
    NA = "NA"  # cross-layer / MLA / fused-PBS — order is not meaningful.


class KVFormatSpec(ABC):
    """Per-``GPUKVFormat`` strategy: shape access plus static metadata.

    Subclasses bind a single ``kv_caches`` value at construction time so
    callers can reuse one instance for many lookups (and so type
    narrowing / ``cast`` stays internal to the spec).

    Authoring a new format = subclass this (or a more specific family
    base in :mod:`families`) inside a new module under
    ``kv_format/specs/``. Registration is automatic; the
    :attr:`format_id` defaults to the class name with any trailing
    ``"Spec"`` stripped.
    """

    # ------------------------------------------------------------------
    # Class-level metadata. Subclasses override.
    # ------------------------------------------------------------------
    # Stable string identifier; defaults to ``cls.__name__`` (minus any
    # trailing ``Spec``) via ``__init_subclass__`` if not overridden.
    format_id: ClassVar[str]
    # Serving engine producing this layout. Used by the detection
    # pipeline to scope candidates per engine.
    engine: ClassVar[str]
    # Associated C++ enum value used by transfer kernels.
    gpu_kv_format: ClassVar["lmc_ops.GPUKVFormat"]
    # Human-readable shape skeleton (e.g. ``"NL x [2, NB, BS, NH, HS]"``).
    shape_desc: ClassVar[str]
    # Free-form label for ``legible_print_gpu_kv_format`` output.
    backend_label: ClassVar[str]

    # Physical head/block axis layout. Most call sites only care about
    # ``is_hnd``; the enum is here for future "list every NHD format"
    # style queries.
    layout: ClassVar[AxisLayout] = AxisLayout.NA

    # Whether the format packs every layer into one tensor.
    is_cross_layer: ClassVar[bool] = False
    # Whether the per-layer physical layout is HND (heads before BS).
    is_hnd: ClassVar[bool] = False
    # Whether the format is MLA-style (kv_size == 1, NH absorbed).
    is_mla: ClassVar[bool] = False
    # Whether per-layer tensor dim-0 is the block axis (used by
    # ``resolve_block_stride_and_log_layout`` to honour dim-0 padding).
    is_block_axis_dim0: ClassVar[bool] = False
    # Whether this format has a meaningful BS / NB dimension at all.
    # NBBS-fused formats (SGL MHA, SGL MLA) collapse num_blocks and
    # block_size into a single page_buffer_size axis.
    has_separate_block_dims: ClassVar[bool] = True

    # When True, this class is treated as an intermediate/family base
    # and is *not* registered. Set by ``families.py`` helpers and by
    # the framework on :class:`KVFormatSpec` itself.
    abstract: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Default ``format_id`` to the class name (sans trailing
        # "Spec"). Subclasses may still override explicitly.
        if "format_id" not in cls.__dict__:
            name = cls.__name__
            if name.endswith("Spec"):
                name = name[: -len("Spec")]
            cls.format_id = name
        # Concrete (non-abstract) subclasses self-register.
        if not cls.__dict__.get("abstract", False):
            # Local import to break the import cycle:
            # registry imports KVFormatSpec lazily for typing only.
            # First Party
            from lmcache.v1.gpu_connector.kv_format.registry import (
                register_spec_class,
            )

            register_spec_class(cls)

    def __init__(self, kv_caches: DiscoverableKVCache) -> None:
        self.kv_caches = kv_caches

    # ------------------------------------------------------------------
    # Required hooks. Override in subclasses.
    # ------------------------------------------------------------------
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers in the KV cache."""

    @abstractmethod
    def page_buffer_size(self) -> int:
        """Return ``num_blocks * block_size`` (or the fused PBS axis)."""

    @abstractmethod
    def num_heads(self, layer_idx: int = 0) -> int:
        """Return the number of KV heads for ``layer_idx``."""

    @abstractmethod
    def head_size(self, layer_idx: int = 0) -> int:
        """Return the per-head hidden size for ``layer_idx``."""

    @abstractmethod
    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        """Return the device data pointers for the requested layers.

        Args:
            layer_indices: Layer indices to look up.

        Returns:
            One pointer per layer for tensor-per-layer formats; for K/V
            split formats the K pointers are returned first followed by
            the V pointers; for cross-layer formats a single pointer is
            returned regardless of ``layer_indices``.
        """

    @abstractmethod
    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """Return a tensor whose shape/stride/dtype/device represent the
        physical layout of layer ``layer_idx``. Must NOT be sliced for
        K/V — it is consumed for layout introspection only.
        """

    # ------------------------------------------------------------------
    # Optional hooks with sensible defaults that handle the regular
    # (non-fused, non-cross-layer) case. Subclasses override only the
    # ones that differ.
    # ------------------------------------------------------------------
    def num_blocks(self) -> int:
        """Return the number of pre-allocated KV cache blocks.

        Raises:
            ValueError: For NBBS-fused formats that do not expose a
                separate ``num_blocks`` axis.
        """
        raise ValueError(
            "trying to access an attribute of the GPU KV Cache "
            f"that does not exist for the format {self.format_id}. "
            "A misalignment with the GPUKVFormat must be resolved"
        )

    def block_size(self, layer_idx: int = 0) -> int:
        """Return the per-block token capacity for ``layer_idx``.

        Raises:
            ValueError: For NBBS-fused formats that do not expose a
                separate ``block_size`` axis.
        """
        raise ValueError(
            "trying to access an attribute of the GPU KV Cache "
            f"that does not exist for the format {self.format_id}. "
            "A misalignment with the GPUKVFormat must be resolved"
        )

    def hidden_dim(self, layer_idx: int = 0) -> int:
        """Return ``num_heads * head_size`` for ``layer_idx``."""
        return self.num_heads(layer_idx) * self.head_size(layer_idx)

    def tokens_per_layer(self) -> int:
        """Return the number of tokens stored per layer (== PBS)."""
        return self.page_buffer_size()

    def elements_per_layer(self) -> int:
        """Return the per-layer element count (K + V for non-MLA)."""
        kv = 1 if self.is_mla else 2
        return self.tokens_per_layer() * self.num_heads() * self.head_size() * kv

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        """Return the dtype of the per-layer tensor for ``layer_idx``."""
        return self.layout_probe_tensor(layer_idx).dtype

    def concrete_shape_str(self) -> str:
        """Substitute numeric values into :attr:`shape_desc`. Default
        implementation handles the regular formats; cross-layer / fused
        specs override.
        """
        nl = self.num_layers()
        hs = self.head_size()
        if self.is_mla and not self.has_separate_block_dims:
            return f"{nl} x [{self.page_buffer_size()}, 1, {hs}]"
        if self.is_mla:
            return f"{nl} x [{self.num_blocks()}, {self.block_size()}, {hs}]"
        if not self.has_separate_block_dims:
            return f"2 x {nl} x [{self.page_buffer_size()}, {self.num_heads()}, {hs}]"
        return f"{nl} x [<see shape_desc>, {self.num_heads()}, {hs}]"

    # ------------------------------------------------------------------
    # Helpers shared by subclasses.
    # ------------------------------------------------------------------
    def _as_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.kv_caches)

    def _as_layer_list(self) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self.kv_caches)

    def _as_kv_layer_list(self) -> list[list[torch.Tensor]]:
        return cast(list[list[torch.Tensor]], self.kv_caches)
