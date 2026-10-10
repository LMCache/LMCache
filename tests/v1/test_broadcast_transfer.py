# SPDX-License-Identifier: Apache-2.0
"""Exercise window planning and packing for first-rank broadcast retrieval."""

# Standard
import pickle

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.broadcast_transfer import (
    CHUNK_ALIGNMENT,
    StagedChunk,
    Window,
    pack,
    plan_windows,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

pytestmark = pytest.mark.no_shared_allocator


def make_object(tensors: list[torch.Tensor], *, grouped: bool = False) -> MemoryObj:
    """Pack typed groups into the same flat format cache backends produce."""
    raw = torch.cat([tensor.reshape(-1).view(torch.uint8) for tensor in tensors])
    return TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=tensors[0].shape,
            dtype=tensors[0].dtype,
            address=0,
            phy_size=raw.numel(),
            ref_count=1,
            fmt=MemoryFormat.KV_MLA_FMT,
            shapes=[tensor.shape for tensor in tensors] if grouped else None,
            dtypes=[tensor.dtype for tensor in tensors] if grouped else None,
        ),
        None,
    )


def make_chunks(count: int, tokens: int) -> list[StagedChunk]:
    """Consecutive float32 chunks of ``tokens`` tokens with distinct contents."""
    return [
        (
            make_object(
                [
                    torch.arange(2 * tokens * 3, dtype=torch.float32).reshape(
                        1, 2, tokens, 3
                    )
                    + index * 1000
                ]
            ),
            index * tokens,
            (index + 1) * tokens,
        )
        for index in range(count)
    ]


def test_plan_windows_packs_consecutive_chunks_within_capacity() -> None:
    chunks = make_chunks(5, 2)  # 48 bytes each, each starting on an aligned offset
    capacity = 2 * CHUNK_ALIGNMENT + 48  # exactly three chunks

    groups = list(plan_windows(chunks, capacity))

    assert [len(group) for group in groups] == [3, 2]
    assert list(groups[0]) == chunks[:3]
    assert list(groups[1]) == chunks[3:]
    for group in groups:
        assert Window.from_chunks(group).num_bytes <= capacity


def test_plan_windows_isolates_a_chunk_larger_than_the_capacity() -> None:
    chunks = make_chunks(3, 2)

    groups = list(plan_windows(chunks, 10))

    assert [list(group) for group in groups] == [[chunk] for chunk in chunks]
    assert Window.from_chunks(groups[0]).num_bytes == 48


def test_plan_windows_yields_nothing_without_chunks() -> None:
    assert list(plan_windows([], 1024)) == []


def test_window_requires_at_least_one_chunk() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        Window.from_chunks([])


@pytest.mark.parametrize("grouped", [False, True])
def test_window_views_round_trip_packed_chunks(grouped: bool) -> None:
    first = [
        torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3),
        torch.arange(4 * 2, dtype=torch.float16).reshape(1, 1, 4, 2),
    ]
    second = [tensor + 7 for tensor in first]
    if not grouped:
        first, second = first[:1], second[:1]
    chunks: list[StagedChunk] = [
        (make_object(first, grouped=grouped), 0, 4),
        (make_object(second, grouped=grouped), 4, 8),
    ]
    window = Window.from_chunks(chunks)
    buffer = torch.full((window.num_bytes + 64,), 0xAB, dtype=torch.uint8)

    views = window.views(buffer)
    pack(chunks, views)

    assert window.starts == (0, 4)
    assert window.ends == (4, 8)
    assert window.offsets == (0, CHUNK_ALIGNMENT)
    assert window.num_bytes == CHUNK_ALIGNMENT + chunks[1][0].get_size()
    for (source, _, _), view, expected in zip(
        chunks, views, (first, second), strict=True
    ):
        assert view.parent() is None
        assert view.get_size() == source.get_size()
        assert view.metadata.fmt == MemoryFormat.KV_MLA_FMT
        if grouped:
            for index, tensor in enumerate(expected):
                torch.testing.assert_close(view.get_tensor(index), tensor)
        else:
            torch.testing.assert_close(view.tensor, expected[0])
        # Packing reads the source; it never takes or drops ownership.
        assert source.get_ref_count() == 1
        assert source.metadata.pin_count == 0
    # Bytes between chunks and after the window are untouched.
    assert (buffer[chunks[0][0].get_size() : CHUNK_ALIGNMENT] == 0xAB).all()
    assert (buffer[window.num_bytes :] == 0xAB).all()


def test_window_is_plain_data_that_survives_serialization() -> None:
    """Headers cross ranks through object broadcasts, so they must pickle."""
    chunks = make_chunks(2, 3)
    window = Window.from_chunks(chunks)

    restored = pickle.loads(pickle.dumps(window))

    assert restored == window
    buffer = torch.zeros(window.num_bytes, dtype=torch.uint8)
    pack(chunks, window.views(buffer))
    for (source, _, _), view in zip(chunks, restored.views(buffer), strict=True):
        torch.testing.assert_close(view.tensor, source.tensor)


def test_window_views_reject_a_buffer_that_is_too_small() -> None:
    window = Window.from_chunks(make_chunks(1, 4))

    with pytest.raises(ValueError, match="needs"):
        window.views(torch.zeros(window.num_bytes - 1, dtype=torch.uint8))


def test_pack_rejects_an_invalidated_source() -> None:
    chunks = make_chunks(1, 4)
    window = Window.from_chunks(chunks)
    views = window.views(torch.zeros(window.num_bytes, dtype=torch.uint8))
    chunks[0][0].invalidate()

    with pytest.raises(ValueError, match="valid tensor storage"):
        pack(chunks, views)
