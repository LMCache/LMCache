# SPDX-License-Identifier: Apache-2.0
"""Validation and layout tests for the grouped prefetch interface types.

Pure-Python: no storage manager, GPU or native bitmap is needed.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    GroupedKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
    ipc_key_to_grouped_keys,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([2, 3])], dtypes=[torch.float16])


def _keys(n: int, gid: int = 0, kv_rank: int = 0) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="m",
            kv_rank=kv_rank,
            object_group_id=gid,
        )
        for i in range(n)
    ]


def _row(n: int, gid: int = 0, kv_rank: int = 0, window: int = -1) -> GroupedKeys:
    return GroupedKeys(
        keys=_keys(n, gid, kv_rank),
        object_group_id=gid,
        layout_desc=LAYOUT,
        sliding_window_size=window,
    )


class TestGroupedKeys:
    @pytest.mark.parametrize("window", [-1, 1, 4])
    def test_valid_windows(self, window):
        assert _row(2, window=window).sliding_window_size == window

    @pytest.mark.parametrize("window", [0, -2])
    def test_invalid_window_raises(self, window):
        with pytest.raises(ValueError):
            _row(2, window=window)


class TestPrefetchTaskSpecValidation:
    def test_defaults(self):
        spec = PrefetchTaskSpec(key_groups=[_row(3)])
        assert spec.fetching_policy == "prefix"
        assert spec.lock_mode is PrefetchLockMode.LOCK
        assert spec.num_kv_readers == 1

    def test_empty_key_groups_raises(self):
        with pytest.raises(ValueError):
            PrefetchTaskSpec(key_groups=[])

    def test_num_kv_readers_below_one_raises(self):
        with pytest.raises(ValueError):
            PrefetchTaskSpec(key_groups=[_row(1)], num_kv_readers=0)

    def test_unknown_fetching_policy_raises(self):
        with pytest.raises(ValueError):
            PrefetchTaskSpec(key_groups=[_row(1)], fetching_policy="sparse")  # type: ignore[arg-type]

    def test_non_adjacent_group_rows_raise(self):
        rows = [_row(2, gid=0, kv_rank=0), _row(2, gid=1), _row(2, gid=0, kv_rank=1)]
        with pytest.raises(ValueError):
            PrefetchTaskSpec(key_groups=rows)

    def test_prefix_requires_equal_row_lengths(self):
        rows = [_row(2, gid=0), _row(3, gid=1)]
        with pytest.raises(ValueError):
            PrefetchTaskSpec(key_groups=rows, fetching_policy="prefix")
        # "full" tolerates ragged rows.
        spec = PrefetchTaskSpec(key_groups=rows, fetching_policy="full")
        assert spec.row_lengths == (2, 3)
        assert not spec.is_uniform

    def test_prefix_requires_equal_rows_per_group(self):
        rows = [_row(2, gid=0, kv_rank=0), _row(2, gid=0, kv_rank=1), _row(2, gid=1)]
        with pytest.raises(ValueError):
            PrefetchTaskSpec(key_groups=rows, fetching_policy="prefix")
        assert PrefetchTaskSpec(
            key_groups=rows, fetching_policy="full"
        ).row_lengths == (
            2,
            2,
            2,
        )


class TestPrefetchTaskSpecShape:
    def test_grid_properties(self):
        rows = [
            _row(4, gid=0, kv_rank=0),
            _row(4, gid=0, kv_rank=1),
            _row(4, gid=2, kv_rank=0, window=2),
            _row(4, gid=2, kv_rank=1, window=2),
        ]
        spec = PrefetchTaskSpec(key_groups=rows)
        assert spec.row_lengths == (4, 4, 4, 4)
        assert spec.is_uniform
        assert spec.num_chunks == 4
        assert spec.world_size == 2
        assert spec.object_group_ids == (0, 2)
        assert spec.num_object_groups == 2
        assert spec.total_keys == 16

    def test_ragged_shape_properties_raise(self):
        spec = PrefetchTaskSpec(
            key_groups=[_row(1, gid=0), _row(3, gid=1)], fetching_policy="full"
        )
        assert spec.total_keys == 4
        with pytest.raises(ValueError):
            _ = spec.num_chunks
        # One row per group is still a well-defined fan-out.
        assert spec.world_size == 1

    def test_uneven_rows_per_group_world_size_raises(self):
        spec = PrefetchTaskSpec(
            key_groups=[
                _row(2, gid=0, kv_rank=0),
                _row(2, gid=0, kv_rank=1),
                _row(2, gid=1),
            ],
            fetching_policy="full",
        )
        with pytest.raises(ValueError):
            _ = spec.world_size


def _ipc_key(world_size: int, worker_id: int | None = None) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        worker_id=worker_id,
        token_ids=(0,),
        start=0,
        end=0,
        request_id="r",
        cache_salt="salt",
    )


class TestIpcKeyToGroupedKeys:
    def test_rows_are_group_major_rank_minor_and_chunk_ordered(self):
        hashes = [b"c0", b"c1", b"c2"]
        attn = AttnWindowDesc(num_chunks_in_sw=[-1, 2, 1], world_size=2)
        layouts = {0: LAYOUT, 1: LAYOUT, 2: LAYOUT}
        rows = ipc_key_to_grouped_keys(_ipc_key(2), hashes, [0, 2], layouts, attn)

        assert [r.object_group_id for r in rows] == [0, 0, 2, 2]
        assert [r.sliding_window_size for r in rows] == [-1, -1, 1, 1]
        assert all(r.layout_desc is LAYOUT for r in rows)
        for row in rows:
            assert [k.chunk_hash for k in row.keys] == hashes
            assert {k.object_group_id for k in row.keys} == {row.object_group_id}
            assert {k.cache_salt for k in row.keys} == {"salt"}
            assert len({k.kv_rank for k in row.keys}) == 1
        # Same rank order in every group; ranks distinct within a group.
        assert rows[0].keys[0].kv_rank != rows[1].keys[0].kv_rank
        assert [rows[0].keys[0].kv_rank, rows[1].keys[0].kv_rank] == [
            rows[2].keys[0].kv_rank,
            rows[3].keys[0].kv_rank,
        ]

    def test_matches_flat_expansion(self):
        """The rows hold exactly the keys ``ipc_key_to_object_keys`` produces."""
        hashes = [b"c0", b"c1"]
        attn = AttnWindowDesc(num_chunks_in_sw=[-1, -1], world_size=2)
        rows = ipc_key_to_grouped_keys(
            _ipc_key(2), hashes, [0, 1], {0: LAYOUT, 1: LAYOUT}, attn
        )
        flat = ipc_key_to_object_keys(_ipc_key(2), hashes, [0, 1])
        assert sorted((k for r in rows for k in r.keys), key=repr) == sorted(
            (k for g in flat for k in g), key=repr
        )

    def test_worker_specific_key_yields_one_row_per_group(self):
        attn = AttnWindowDesc(num_chunks_in_sw=[-1], world_size=2)
        rows = ipc_key_to_grouped_keys(
            _ipc_key(2, worker_id=1), [b"c0"], [0], {0: LAYOUT}, attn
        )
        assert len(rows) == 1

    def test_missing_layout_raises(self):
        attn = AttnWindowDesc(num_chunks_in_sw=[-1, -1])
        with pytest.raises(ValueError):
            ipc_key_to_grouped_keys(_ipc_key(1), [b"c0"], [0, 1], {0: LAYOUT}, attn)

    def test_group_outside_attn_desc_raises(self):
        attn = AttnWindowDesc(num_chunks_in_sw=[-1])
        with pytest.raises(ValueError):
            ipc_key_to_grouped_keys(_ipc_key(1), [b"c0"], [1], {1: LAYOUT}, attn)
