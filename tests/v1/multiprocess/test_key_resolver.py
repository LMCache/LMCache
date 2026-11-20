# SPDX-License-Identifier: Apache-2.0
"""The node's prefetch resolver, across object groups.

Under ``--separate-object-groups`` a chunk is stored once per object group.
A prefetch that resolved group ``0`` alone warmed part of each chunk and
reported success; these cases pin the rows for every group.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    resolve_grouped_object_keys,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4
CHUNKS = 2
TOKENS = list(range(CHUNKS * CHUNK_SIZE))


def _hasher() -> TokenHasher:
    return TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")


def _layout(width: int) -> MemoryLayoutDesc:
    """A layout distinguishable by ``width``, so rows can be told apart."""
    return MemoryLayoutDesc(shapes=[torch.Size([width])], dtypes=[torch.float16])


def test_prefetch_rows_cover_every_group_with_its_own_layout_and_window():
    """One row per (object group, kv rank), each carrying its own group's
    layout and attention window -- which is why one layout for group 0
    could never serve a model stored with separate object groups."""
    layouts = {0: _layout(8), 1: _layout(16), 2: _layout(32)}
    windows = AttnWindowDesc(num_chunks_in_sw=[-1, 3, 1], world_size=2)

    rows, chunks = resolve_grouped_object_keys(
        _hasher(), "m", 2, TOKENS, "alice", layouts, windows
    )

    assert chunks == CHUNKS
    assert [row.object_group_id for row in rows] == [0, 0, 1, 1, 2, 2]
    assert [row.sliding_window_size for row in rows] == [-1, -1, 3, 3, 1, 1]
    for row in rows:
        assert row.layout_desc == layouts[row.object_group_id]
        assert len(row.keys) == CHUNKS
        assert {key.object_group_id for key in row.keys} == {row.object_group_id}


def test_a_group_with_no_layout_is_refused():
    windows = AttnWindowDesc(num_chunks_in_sw=[-1, 3], world_size=1)

    with pytest.raises(ValueError):
        resolve_grouped_object_keys(
            _hasher(), "m", 1, TOKENS, "alice", {0: _layout(8)}, windows
        )


def test_a_sub_chunk_sequence_resolves_to_nothing_in_any_group():
    windows = AttnWindowDesc(num_chunks_in_sw=[-1, 3], world_size=1)
    layouts = {0: _layout(8), 1: _layout(16)}

    assert resolve_grouped_object_keys(
        _hasher(), "m", 1, TOKENS[:2], "alice", layouts, windows
    ) == ([], 0)
