# SPDX-License-Identifier: Apache-2.0
"""Tests for RemoteConnector.reshape_partial_chunk with 2D/3D/4D shapes.

The ``reshape_partial_chunk`` method recalculates the token count when
only part of a chunk was read from the remote backend.  It must work
for all supported shape layouts:

* 4-D  ``[2, num_layers, num_tokens, hidden_dim]`` (standard / MLA)
* 3-D  ``[num_tokens, 2, hidden_dim]``             (layerwise MHA)
* 2-D  ``[num_tokens, hidden_dim]``                 (layerwise MLA)
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_obj(
    shape: torch.Size,
    dtype: torch.dtype = torch.float16,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> TensorMemoryObj:
    """Create a minimal TensorMemoryObj backed by a contiguous buffer."""
    numel = 1
    for d in shape:
        numel *= d
    raw_data = torch.zeros(
        numel * torch.tensor([], dtype=dtype).element_size(), dtype=torch.uint8
    )
    metadata = MemoryObjMetadata(
        shape=shape,
        dtype=dtype,
        address=0,
        phy_size=raw_data.numel(),
        ref_count=1,
        fmt=fmt,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


class _StubConnector:
    """Minimal stand-in that only exposes the fields reshape_partial_chunk
    reads, avoiding the need to satisfy RemoteConnector's abstract interface.
    """

    def __init__(self, full_chunk_size_bytes: int, single_token_size: int):
        self.full_chunk_size_bytes = full_chunk_size_bytes
        self.single_token_size = single_token_size

    # Bind the real method so we can test it on the stub.
    # First Party
    from lmcache.v1.storage_backend.connector.base_connector import (
        RemoteConnector,
    )

    reshape_partial_chunk = RemoteConnector.reshape_partial_chunk


# ---------------------------------------------------------------------------
# Parametrised test data
# ---------------------------------------------------------------------------

_4D_SHAPE = torch.Size([2, 10, 16, 128])  # token_dim = 2
_3D_SHAPE = torch.Size([16, 2, 128])  # token_dim = 0
_2D_SHAPE = torch.Size([16, 128])  # token_dim = 0

_DTYPE = torch.float16
_ELEM = torch.tensor([], dtype=_DTYPE).element_size()  # 2 bytes


def _token_size(shape: torch.Size) -> int:
    """Bytes per single token for a given shape layout."""
    if len(shape) == 4:
        # [2, L, T, D]  → per-token = 2 * L * D * elem
        return shape[0] * shape[1] * shape[3] * _ELEM
    elif len(shape) == 3:
        # [T, 2, D]  → per-token = 2 * D * elem
        return shape[1] * shape[2] * _ELEM
    else:
        # [T, D]  → per-token = D * elem
        return shape[1] * _ELEM


def _full_bytes(shape: torch.Size) -> int:
    numel = 1
    for d in shape:
        numel *= d
    return numel * _ELEM


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, expected_token_dim",
    [
        (_4D_SHAPE, 2),
        (_3D_SHAPE, 0),
        (_2D_SHAPE, 0),
    ],
    ids=["4D-standard", "3D-layerwise-MHA", "2D-layerwise-MLA"],
)
@pytest.mark.parametrize("num_tokens", [1, 8, 15])
def test_reshape_partial_chunk_token_count(shape, expected_token_dim, num_tokens):
    """Verify that the token dimension is correctly updated after a
    partial read."""
    full_bytes = _full_bytes(shape)
    tok_size = _token_size(shape)
    full_tokens = shape[expected_token_dim]

    # Only test if requested num_tokens < full tokens
    if num_tokens >= full_tokens:
        pytest.skip("num_tokens must be less than full chunk tokens")

    connector = _StubConnector(full_bytes, tok_size)
    memory_obj = _make_memory_obj(shape, _DTYPE)
    bytes_read = num_tokens * tok_size

    result = connector.reshape_partial_chunk(memory_obj, bytes_read)

    assert result.meta.shape[expected_token_dim] == num_tokens
    # Other dimensions unchanged
    for i, dim in enumerate(shape):
        if i != expected_token_dim:
            assert result.meta.shape[i] == dim


@pytest.mark.parametrize(
    "shape",
    [_4D_SHAPE, _3D_SHAPE, _2D_SHAPE],
    ids=["4D", "3D", "2D"],
)
def test_reshape_full_chunk_returns_unchanged(shape):
    """A full-chunk read should return the memory object unchanged."""
    full_bytes = _full_bytes(shape)
    tok_size = _token_size(shape)
    connector = _StubConnector(full_bytes, tok_size)
    memory_obj = _make_memory_obj(shape, _DTYPE)

    result = connector.reshape_partial_chunk(memory_obj, full_bytes)
    assert result.meta.shape == shape


@pytest.mark.parametrize(
    "shape",
    [_4D_SHAPE, _3D_SHAPE, _2D_SHAPE],
    ids=["4D", "3D", "2D"],
)
def test_reshape_partial_chunk_truncates_raw_data(shape):
    """raw_data should be sliced to bytes_read length."""
    full_bytes = _full_bytes(shape)
    tok_size = _token_size(shape)
    connector = _StubConnector(full_bytes, tok_size)
    memory_obj = _make_memory_obj(shape, _DTYPE)

    bytes_read = tok_size  # 1 token
    result = connector.reshape_partial_chunk(memory_obj, bytes_read)
    assert result.raw_data.numel() == bytes_read


@pytest.mark.parametrize(
    "shape",
    [_4D_SHAPE, _3D_SHAPE, _2D_SHAPE],
    ids=["4D", "3D", "2D"],
)
def test_reshape_partial_chunk_invalid_bytes_raises(shape):
    """Non-aligned or zero bytes_read should raise ValueError."""
    full_bytes = _full_bytes(shape)
    tok_size = _token_size(shape)
    connector = _StubConnector(full_bytes, tok_size)
    memory_obj = _make_memory_obj(shape, _DTYPE)

    with pytest.raises(ValueError):
        connector.reshape_partial_chunk(memory_obj, 0)

    with pytest.raises(ValueError):
        connector.reshape_partial_chunk(memory_obj, tok_size + 1)

    with pytest.raises(ValueError):
        connector.reshape_partial_chunk(memory_obj, full_bytes + tok_size)

    with pytest.raises(ValueError):
        connector.reshape_partial_chunk(memory_obj, -tok_size)
