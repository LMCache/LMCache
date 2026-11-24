# SPDX-License-Identifier: Apache-2.0
"""A page declares its geometry instead of deriving it from token counts."""

# Standard
from dataclasses import replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.api import MemoryLayoutDesc
from lmcache.v1.common.geometry import (
    DEFAULT_ALIGNMENT,
    PageGeometry,
    PageKind,
    TensorSpan,
)

# The shape from #4888: a recurrent page of 1,085,440 bytes over 4,096 tokens.
RECURRENT_PAGE_BYTES = 1_085_440
RECURRENT_PAGE_TOKENS = 4_096
RECURRENT_ROW_BYTES = RECURRENT_PAGE_BYTES // RECURRENT_PAGE_TOKENS  # 265


def _state_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([RECURRENT_PAGE_BYTES])], dtypes=[torch.uint8]
    )


def test_an_attention_page_that_fits_exactly():
    layout = MemoryLayoutDesc(shapes=[torch.Size([82, 64])], dtypes=[torch.float16])

    geometry = PageGeometry.from_layout(
        layout, logical_tokens=82, row_stride_bytes=64 * 2
    )

    assert geometry.kind is PageKind.ATTENTION
    assert geometry.declared_bytes == 82 * 128
    assert geometry.padding_bytes == 0
    assert geometry.tensors == (
        TensorSpan(offset=0, size=82 * 128, shape=(82, 64), dtype=torch.float16),
    )


def test_attention_padding_must_stay_below_the_alignment():
    layout = MemoryLayoutDesc(shapes=[torch.Size([1, 8])], dtypes=[torch.float16])

    geometry = PageGeometry.from_layout(
        layout,
        logical_tokens=1,
        declared_bytes=16 + DEFAULT_ALIGNMENT - 1,
        row_stride_bytes=16,
        alignment=DEFAULT_ALIGNMENT,
    )

    assert geometry.padding_bytes == DEFAULT_ALIGNMENT - 1

    with pytest.raises(ValueError, match="not.*below the"):
        replace(geometry, padding_bytes=DEFAULT_ALIGNMENT)


def test_the_recurrent_page_cannot_be_expressed_as_rounded_rows():
    """265 bytes per token is a real page, not a row that rounds to 272."""
    with pytest.raises(ValueError, match="bytes per row need"):
        PageGeometry.from_layout(
            _state_layout(),
            logical_tokens=RECURRENT_PAGE_TOKENS,
            row_stride_bytes=272,
        )


def test_the_recurrent_page_is_expressible_as_an_opaque_state_page():
    geometry = PageGeometry.from_layout(
        _state_layout(), logical_tokens=RECURRENT_PAGE_TOKENS
    )

    assert geometry.kind is PageKind.RECURRENT
    assert geometry.row_stride_bytes is None
    assert geometry.declared_bytes == RECURRENT_PAGE_BYTES
    assert geometry.padding_bytes == 0
    assert RECURRENT_ROW_BYTES == 265


def test_a_recurrent_page_cannot_claim_a_row_stride():
    geometry = PageGeometry.from_layout(
        _state_layout(), logical_tokens=RECURRENT_PAGE_TOKENS
    )

    with pytest.raises(ValueError, match="no per-token row stride"):
        replace(geometry, row_stride_bytes=RECURRENT_ROW_BYTES)


def test_a_mixed_page_keeps_each_tensor_span():
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4]), torch.Size([3])],
        dtypes=[torch.float16, torch.uint8],
    )

    geometry = PageGeometry.from_layout(layout, logical_tokens=2)

    assert geometry.kind is PageKind.MIXED
    assert geometry.row_stride_bytes is None
    assert [span.offset for span in geometry.tensors] == [0, 16]
    assert [span.size for span in geometry.tensors] == [16, 3]


def test_a_span_must_match_its_shape_and_dtype():
    span = TensorSpan(offset=0, size=15, shape=(2, 4), dtype=torch.float16)

    with pytest.raises(ValueError, match="needs 16"):
        PageGeometry(
            kind=PageKind.ATTENTION,
            logical_tokens=1,
            declared_bytes=16,
            row_stride_bytes=16,
            padding_bytes=0,
            tensors=(span,),
        )


def test_spans_must_be_ordered_and_inside_the_page():
    first = TensorSpan(offset=0, size=8, shape=(4,), dtype=torch.float16)
    second = TensorSpan(offset=4, size=8, shape=(4,), dtype=torch.float16)

    with pytest.raises(ValueError, match="overlaps or is out of order"):
        PageGeometry(
            kind=PageKind.MIXED,
            logical_tokens=4,
            declared_bytes=16,
            row_stride_bytes=None,
            padding_bytes=0,
            tensors=(first, second),
        )

    outside = TensorSpan(offset=12, size=8, shape=(4,), dtype=torch.float16)
    with pytest.raises(ValueError, match="past the"):
        PageGeometry(
            kind=PageKind.MIXED,
            logical_tokens=4,
            declared_bytes=16,
            row_stride_bytes=None,
            padding_bytes=0,
            tensors=(outside,),
        )


def test_padding_is_the_tail_of_the_page():
    span = TensorSpan(offset=0, size=8, shape=(4,), dtype=torch.float16)

    with pytest.raises(ValueError, match="spans cover 8 bytes"):
        PageGeometry(
            kind=PageKind.RECURRENT,
            logical_tokens=4,
            declared_bytes=24,
            row_stride_bytes=None,
            padding_bytes=8,
            tensors=(span,),
        )


def test_an_attention_page_needs_whole_rows():
    layout = MemoryLayoutDesc(shapes=[torch.Size([4])], dtypes=[torch.float16])

    with pytest.raises(ValueError, match="bytes per row need"):
        PageGeometry.from_layout(layout, logical_tokens=4, row_stride_bytes=4)


def test_a_page_needs_a_positive_token_count():
    layout = MemoryLayoutDesc(shapes=[torch.Size([4])], dtypes=[torch.float16])

    with pytest.raises(ValueError, match="positive token count"):
        PageGeometry.from_layout(layout, logical_tokens=0)


def test_a_layout_needs_one_dtype_per_shape():
    with pytest.raises(ValueError, match="one dtype per tensor shape"):
        PageGeometry.from_layout(
            MemoryLayoutDesc(shapes=[], dtypes=[]), logical_tokens=1
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("rows", [1, 3, 7, 16])
@pytest.mark.parametrize("width", [1, 5, 64])
def test_spans_always_add_up_to_the_declared_page(dtype, rows, width):
    layout = MemoryLayoutDesc(shapes=[torch.Size([rows, width])], dtypes=[dtype])
    row_bytes = width * dtype.itemsize

    geometry = PageGeometry.from_layout(
        layout, logical_tokens=rows, row_stride_bytes=row_bytes
    )

    assert sum(span.size for span in geometry.tensors) == geometry.declared_bytes
    assert geometry.declared_bytes == rows * row_bytes
    assert all(
        span.offset + span.size <= geometry.declared_bytes for span in geometry.tensors
    )
