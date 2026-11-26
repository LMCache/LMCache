# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test of the ``gha`` L2 serde through ``StorageManager``.

Chunks of two object groups with different shapes are written to L1 in the same
batch; the store controller hands them to one serde-wrapped L2 adapter, so the
wrapper must size each temp buffer for its own group. After L1 is cleared a
prefetch decodes them back: the attention group within quantization error, the
MLA group byte for byte.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of

# Test helpers
from tests.v1.distributed.serde.test_serde_e2e import (
    clear_and_wait_drained,
    get_l2_stored_object_count,
    make_storage_manager_config,
    wait_for_condition,
)

pytest.importorskip("lmcache.gha_quant")
if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime", allow_module_level=True
    )

_MODEL = "test_model"
_HEADS, _HEAD_DIM = 4, 64
_ATTENTION_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([2, 4, 128, _HEADS * _HEAD_DIM])], dtypes=[torch.float16]
)
_MLA_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([4, 128, 128])], dtypes=[torch.float16]
)


def _key(chunk_id: int, group: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=_MODEL,
        kv_rank=0,
        object_group_id=group,
    )


def _prefetch(sm: StorageManager, keys, layout, group: int) -> int:
    handle = sm.submit_prefetch_task(
        PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(
                    keys=keys,
                    object_group_id=group,
                    layout_desc=layout,
                    num_kv_readers=1,
                )
            ]
        )
    )
    hits: list[int] = []

    def done() -> bool:
        result = sm.query_prefetch_status(handle)
        if result is None:
            return False
        hits.append(result.hit_cells[0].count_leading_ones())
        return True

    assert wait_for_condition(done, timeout=20.0), "prefetch did not finish"
    return hits[0]


def test_mixed_groups_round_trip_through_l2() -> None:
    adapter = MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=10.0)
    adapter.serde_config = SerdeConfig(
        type="gha",
        kwargs={
            "qbit": 4,
            "device": "cpu",
            "num_threads": 2,
            "groups": {
                0: [
                    {
                        "cache_category": "attention",
                        "attention_plane_layout": "split_kv",
                    }
                ],
                1: [{"cache_category": "attention", "attention_plane_layout": "mla"}],
            },
        },
    )
    sm = StorageManager(make_storage_manager_config([adapter]))
    try:
        torch.manual_seed(0)
        attention_keys = [_key(i, 0) for i in range(3)]
        mla_keys = [_key(100 + i, 1) for i in range(3)]
        originals: dict[ObjectKey, torch.Tensor] = {}
        for keys, layout in (
            (attention_keys, _ATTENTION_LAYOUT),
            (mla_keys, _MLA_LAYOUT),
        ):
            reserved = sm.reserve_write(keys, layout)
            assert len(reserved) == len(keys)
            for key, obj in reserved.items():
                tensor = obj.tensor
                assert tensor is not None
                tensor.copy_(torch.randn(tensor.shape).half())
                originals[key] = tensor.clone()
            sm.finish_write(list(reserved), l1_owners_of(reserved))

        assert wait_for_condition(
            lambda: get_l2_stored_object_count(sm) >= 6, timeout=20.0
        ), "chunks of both groups did not reach L2"
        # Only the attention chunks shrink; the MLA chunks are stored as is.
        raw_total = sum(t.numel() * t.element_size() for t in originals.values())
        stored_total = sum(
            adapter["current_size_bytes"]
            for adapter in sm.report_status()["l2_adapters"]
        )
        assert stored_total < 0.7 * raw_total
        clear_and_wait_drained(sm)

        for keys, layout, group in (
            (attention_keys, _ATTENTION_LAYOUT, 0),
            (mla_keys, _MLA_LAYOUT, 1),
        ):
            assert _prefetch(sm, keys, layout, group) == len(keys)
            with sm.read_prefetched_results(keys) as objs:
                assert objs is not None
                for key, obj in zip(keys, objs, strict=True):
                    tensor = obj.tensor
                    assert tensor is not None
                    if group == 0:
                        error = (tensor.float() - originals[key].float()).abs()
                        assert 0 < error.max() and error.mean() < 0.15
                    else:
                        assert torch.equal(
                            tensor.view(torch.uint8),
                            originals[key].view(torch.uint8),
                        )
            sm.finish_read_prefetched(keys)
    finally:
        sm.close()
