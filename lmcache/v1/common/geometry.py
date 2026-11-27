# SPDX-License-Identifier: Apache-2.0
"""Declared page geometry: logical tokens, physical slots and byte strides.

A KV page carries three different numbers that are easy to conflate: how many
logical tokens it represents, how many bytes it occupies, and how those bytes
are laid out. For an attention page they line up (one row per token); for a
recurrent-state page they do not -- a 1,085,440-byte page covering 4,096 tokens
has 265 bytes per token row, and rounding that to a vector width reads past the
page. Today the transfer layer derives the layout instead of being told it,
which is what allows that class of bug (see #4888).

This module makes the layout explicit and validates it at construction time. It
is additive: nothing imports it yet, so it can land before the transfers that
will consume it.

Rules enforced here:

``G1`` ``declared_bytes <= physical_bytes`` when the allocation size is known;
the difference is allocator padding, not page data.

``G2`` An ``ATTENTION`` page covers whole token rows: its semantic bytes are
``logical_tokens * row_stride_bytes`` and its padding stays below the alignment.

``G3`` ``RECURRENT`` and ``MIXED`` pages have no row stride. A caller that needs
one gets ``None`` instead of a rounded guess.

``G4`` Tensor spans are ordered, disjoint, inside the declared page, and each
span's size matches its shape and dtype exactly.

``G5`` ``logical_tokens`` is the value the engine registered (DCP already
applied); this module never divides or multiplies it again.

``G6`` Padding is only the tail of the page: ``semantic + padding == declared``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.api import MemoryLayoutDesc

#: Allocator alignment used when validating how much padding a page may carry.
DEFAULT_ALIGNMENT = 4096


class PageKind(Enum):
    """How a page's bytes are laid out."""

    ATTENTION = "attention"
    """One row of ``row_stride_bytes`` per logical token."""

    RECURRENT = "recurrent"
    """Opaque state image: bytes, no per-token rows."""

    MIXED = "mixed"
    """Several tensors with their own strides inside one page."""


@dataclass(frozen=True)
class TensorSpan:
    """One tensor's slice of a page.

    Attributes:
        offset: Byte offset of the span from the start of the page.
        size: Span size in bytes; must equal ``prod(shape) * dtype.itemsize``.
        shape: The tensor's logical shape.
        dtype: The tensor's dtype; its itemsize defines ``size``.
    """

    offset: int
    size: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class PageGeometry:
    """The declared layout of one page.

    Attributes:
        kind: How the page's bytes are laid out.
        logical_tokens: Tokens this page represents, as the engine registered
            them (DCP already applied).
        declared_bytes: Bytes the page semantics cover, including tail padding;
            never more than the allocation.
        row_stride_bytes: Bytes per token row for an ``ATTENTION`` page, else
            ``None``.
        padding_bytes: Tail padding inside ``declared_bytes``.
        tensors: The page's tensors, ordered by offset.
        alignment: Allocator alignment used to bound the padding.
    """

    kind: PageKind
    logical_tokens: int
    declared_bytes: int
    row_stride_bytes: int | None
    padding_bytes: int
    tensors: tuple[TensorSpan, ...]
    alignment: int = field(default=DEFAULT_ALIGNMENT)

    def __post_init__(self) -> None:
        if self.logical_tokens <= 0:
            raise ValueError(
                f"page geometry needs a positive token count, got {self.logical_tokens}"
            )
        if self.declared_bytes < 0:
            raise ValueError(
                f"page declares {self.declared_bytes} bytes, which is negative"
            )
        if self.alignment <= 0:
            raise ValueError(f"alignment must be positive, got {self.alignment}")
        if self.padding_bytes < 0:
            raise ValueError(
                f"page padding is {self.padding_bytes} bytes, which is negative"
            )
        if self.padding_bytes >= self.alignment:
            raise ValueError(
                f"page padding is {self.padding_bytes} bytes, which is not "
                f"below the {self.alignment}-byte alignment; that is a page, "
                "not padding"
            )
        if not self.tensors:
            raise ValueError("page geometry needs at least one tensor span")
        self._validate_tensors()
        semantic = sum(span.size for span in self.tensors)
        if semantic + self.padding_bytes != self.declared_bytes:
            raise ValueError(
                f"page spans cover {semantic} bytes and declare "
                f"{self.padding_bytes} bytes of padding, but the page is "
                f"{self.declared_bytes} bytes"
            )
        if self.kind is PageKind.ATTENTION:
            self._validate_attention_rows(semantic)
        elif self.row_stride_bytes is not None:
            raise ValueError(
                f"a {self.kind.value} page has no per-token row stride, got "
                f"{self.row_stride_bytes}"
            )

    def _validate_tensors(self) -> None:
        cursor = 0
        for index, span in enumerate(self.tensors):
            if span.offset < 0 or span.size < 0:
                raise ValueError(f"tensor span {index} is negative: {span!r}")
            expected = reduce(mul, span.shape, 1) * span.dtype.itemsize
            if span.size != expected:
                raise ValueError(
                    f"tensor span {index} declares {span.size} bytes but its "
                    f"{span.shape} {span.dtype} shape needs {expected}"
                )
            if span.offset < cursor:
                raise ValueError(
                    f"tensor span {index} overlaps or is out of order: "
                    f"offset {span.offset} < {cursor}"
                )
            if span.offset + span.size > self.declared_bytes:
                raise ValueError(
                    f"tensor span {index} ends at "
                    f"{span.offset + span.size}, past the "
                    f"{self.declared_bytes}-byte page"
                )
            cursor = span.offset + span.size

    def _validate_attention_rows(self, semantic: int) -> None:
        if self.row_stride_bytes is None or self.row_stride_bytes <= 0:
            raise ValueError("an ATTENTION page needs a positive row_stride_bytes")
        rows = self.logical_tokens * self.row_stride_bytes
        if rows != semantic:
            raise ValueError(
                f"{self.logical_tokens} tokens at {self.row_stride_bytes} "
                f"bytes per row need {rows} bytes, but the spans cover "
                f"{semantic}"
            )
        if rows > self.declared_bytes:
            raise ValueError(
                f"{self.logical_tokens} tokens at {self.row_stride_bytes} "
                f"bytes per row do not fit the {self.declared_bytes}-byte page"
            )

    @classmethod
    def from_layout(
        cls,
        layout: MemoryLayoutDesc,
        *,
        logical_tokens: int,
        declared_bytes: int | None = None,
        row_stride_bytes: int | None = None,
        alignment: int = DEFAULT_ALIGNMENT,
    ) -> PageGeometry:
        """Derive a page geometry from a registered layout.

        Args:
            layout: The registered tensor shapes and dtypes, in allocation order.
            logical_tokens: Tokens the page represents, already DCP-scaled.
            declared_bytes: Bytes the page semantics cover. Defaults to the sum
                of the tensor sizes, i.e. no padding.
            row_stride_bytes: Bytes per token row for an attention page; leave
                ``None`` for recurrent or mixed pages.
            alignment: Allocator alignment used to bound the padding.

        Returns:
            The validated geometry.

        Raises:
            ValueError: The layout is empty, a tensor shape cannot be turned
                into a byte span, or the resulting geometry breaks a rule.
        """
        if not layout.shapes or len(layout.shapes) != len(layout.dtypes):
            raise ValueError(
                "page geometry needs one dtype per tensor shape; got "
                f"{len(layout.shapes)} shapes and {len(layout.dtypes)} dtypes"
            )

        spans: list[TensorSpan] = []
        offset = 0
        for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True):
            shape_tuple = tuple(int(dim) for dim in shape)
            size = reduce(mul, shape_tuple, 1) * dtype.itemsize
            spans.append(
                TensorSpan(offset=offset, size=size, shape=shape_tuple, dtype=dtype)
            )
            offset += size

        semantic = offset
        declared = semantic if declared_bytes is None else declared_bytes
        if row_stride_bytes is not None:
            kind = PageKind.ATTENTION
        elif len(spans) > 1:
            kind = PageKind.MIXED
        else:
            kind = PageKind.RECURRENT
        return cls(
            kind=kind,
            logical_tokens=logical_tokens,
            declared_bytes=declared,
            row_stride_bytes=row_stride_bytes,
            padding_bytes=declared - semantic,
            tensors=tuple(spans),
            alignment=alignment,
        )


__all__ = [
    "DEFAULT_ALIGNMENT",
    "PageGeometry",
    "PageKind",
    "TensorSpan",
]
