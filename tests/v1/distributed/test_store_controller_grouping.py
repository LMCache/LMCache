# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``_group_keys_by_shape``.

Each bucket the function returns is submitted as one ``submit_store_task``
call, so every key in a bucket must share a single ``(shape, dtype)``.
``object_group_id`` selects the object group whose kernel groups define that
layout (``get_layout_desc`` in ``lmcache_driven_transfer``), so keys from
different object groups describe different layouts and must not share a
bucket. Fields that do not affect the layout, such as ``cache_salt``, must
not split a bucket.
"""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.store_controller import (
    _group_keys_by_shape,
)


def make_key(
    chunk_id: int,
    model_name: str = "test_model",
    kv_rank: int = 0,
    object_group_id: int = 0,
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=kv_rank,
        object_group_id=object_group_id,
        cache_salt=cache_salt,
    )


class TestGroupKeysByShape:
    """Buckets are split by every layout-affecting field and by nothing else."""

    def test_splits_keys_from_different_object_groups(self):
        """A hybrid model emits one object group per sliding-window size, and
        each group has its own shapes, so the two must land in separate
        buckets."""
        group_0 = [make_key(i, object_group_id=0) for i in range(3)]
        group_1 = [make_key(100 + i, object_group_id=1) for i in range(2)]

        groups = _group_keys_by_shape(group_0 + group_1)

        assert len(groups) == 2, (
            f"Keys from object groups 0 and 1 describe different layouts but "
            f"landed in {len(groups)} bucket(s). A bucket is submitted as one "
            f"store task, so mixing them submits mixed-size buffers."
        )
        assert sorted(len(keys) for keys in groups.values()) == [2, 3]
        for keys in groups.values():
            assert len({key.object_group_id for key in keys}) == 1

    def test_keeps_one_object_group_in_a_single_bucket(self):
        """Keys sharing a layout stay together even when a field that does not
        affect the layout, such as ``cache_salt``, differs."""
        keys = [
            make_key(0, cache_salt=""),
            make_key(1, cache_salt="alice"),
            make_key(2, cache_salt="bob"),
        ]

        groups = _group_keys_by_shape(keys)

        assert len(groups) == 1, (
            f"All three keys share a layout, so they belong in one bucket, "
            f"got {len(groups)}. Splitting on a field that does not affect "
            f"the layout costs one store task per extra bucket."
        )
        assert len(next(iter(groups.values()))) == 3

    def test_still_splits_on_model_name_and_kv_rank(self):
        """The fields the grouping already covered keep splitting buckets."""
        keys = [
            make_key(0, model_name="model_a", kv_rank=0),
            make_key(1, model_name="model_b", kv_rank=0),
            make_key(2, model_name="model_a", kv_rank=1),
        ]

        groups = _group_keys_by_shape(keys)

        assert len(groups) == 3
        assert all(len(bucket) == 1 for bucket in groups.values())
