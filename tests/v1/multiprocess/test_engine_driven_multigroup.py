# SPDX-License-Identifier: Apache-2.0
"""
Tests for engine-driven multi-group (uniform coverage) transfers.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    GroupLayout,
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


def _obj_key(chunk_hash: int, group: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_hash),
        model_name="m",
        kv_rank=0,
        object_group_id=group,
    )


def _group_metadata(chunk_tokens: int = 8) -> EngineDrivenContextMetadata:
    layouts = [
        MemoryLayoutDesc(
            shapes=[torch.Size([2, 2, chunk_tokens, 16])], dtypes=[torch.float32]
        ),
        MemoryLayoutDesc(
            shapes=[torch.Size([2, 1, chunk_tokens, 4])], dtypes=[torch.float32]
        ),
    ]
    return EngineDrivenContextMetadata(
        layout_desc=layouts[0],
        block_size=4,
        use_mla=False,
        group_layouts=layouts,
    )


def test_shm_strategy_multigroup_store_and_retrieve_roundtrip() -> None:
    """Per-group reserve on store; retrieve misses until every group commits."""
    pytest.importorskip(
        "lmcache.native_storage_ops",
        reason="real StorageManager requires compiled native storage ops",
    )
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess.modules.server_transfer import (
        PickleTransferStrategy,
        ShmTransferStrategy,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=8 * 1024 * 1024,
                use_lazy=False,
                shm_name="lmcache_test_multigroup_pool",
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    sm = StorageManager(config)
    try:
        strategy = ShmTransferStrategy(
            storage_manager=sm,
            pending_writes={},
            pending_reads={},
            pending_lock=__import__("threading").Lock(),
            transfer_key_factory=lambda key, iid: (iid, key),
            fallback_strategy=PickleTransferStrategy(sm),
        )
        metadata = _group_metadata()
        key = IPCCacheServerKey.from_token_ids(
            "m", 1, 0, [1] * 16, start=0, end=16, request_id="req-mg"
        )
        group_keys = [
            [_obj_key(1, 0), _obj_key(2, 0)],
            [_obj_key(1, 1), _obj_key(2, 1)],
        ]

        prep = strategy.prepare_store(
            key=key,
            instance_id=7,
            context=metadata,
            resolve_obj_keys=lambda _k: group_keys[0],
            group_keys=group_keys,
        )
        ctx = prep.context
        assert len(ctx["slots"]) == 4
        assert ctx["group_ids"] == [0, 0, 1, 1]
        assert ctx["chunk_indices"] == [0, 1, 0, 1]
        # Group layouts differ: slot shapes must match each group's layout.
        assert ctx["slots"][0]["shape"] == [2, 2, 8, 16]
        assert ctx["slots"][2]["shape"] == [2, 1, 8, 4]

        assert (
            strategy.commit_store(
                key=key,
                instance_id=7,
                cpu_data=b"",
                context=metadata,
                resolve_obj_keys=lambda _k: group_keys[0],
            )
            is True
        )

        # prepare_retrieve reads already read-locked objects (unsafe_read); in
        # production the connector's lookup reserve-reads the matching keys
        # first. Simulate that prefetch per group before retrieving.
        for g_keys in group_keys:
            sm._l1_manager.reserve_read(g_keys)

        ret = strategy.prepare_retrieve(
            key=key,
            instance_id=7,
            resolve_obj_keys=lambda _k: group_keys[0],
            group_keys=group_keys,
        )
        assert ret.success is True
        assert ret.context["group_ids"] == [0, 0, 1, 1]
        assert strategy.commit_retrieve(key=key, instance_id=7) is True

        # A key range where group 1 has nothing must be a miss overall.
        partial_keys = [
            [_obj_key(1, 0), _obj_key(2, 0)],
            [_obj_key(9, 1), _obj_key(10, 1)],
        ]
        ret_miss = strategy.prepare_retrieve(
            key=key,
            instance_id=7,
            resolve_obj_keys=lambda _k: partial_keys[0],
            group_keys=partial_keys,
        )
        assert ret_miss.success is False
        # No leaked read locks after the miss.
        status = sm.report_status()["l1_manager"]
        assert status["read_locked_count"] == 0
    finally:
        sm.close()


class _FakeGroupedContext:
    """Grouped-context fake capturing commit calls."""

    def __init__(self, tensors, chunk_indices, group_ids):
        self._prep = (tensors, chunk_indices, group_ids)
        self.committed = False

    def prepare_store_grouped(self, _key, _iid):
        return self._prep

    def prepare_retrieve_grouped(self, _key, _iid):
        tensors, _, group_ids = self._prep
        return tensors, group_ids

    def commit_store(self, _key, _iid, _chunks):
        self.committed = True
        return True

    def commit_retrieve(self, _key, _iid):
        return True

    def close(self) -> None:
        return None


def _fanout_ctx(monkeypatch) -> tuple[EngineDrivenTransferContext, list[dict]]:
    """Worker context with two registered groups and a capturing gather."""
    ctx = EngineDrivenTransferContext()
    ctx._group_states = [
        worker_transfer._GroupState(
            layer_names=["layer_0", "layer_1"],
            engine_kv_format=MagicMock(),
            blocks_in_chunk=2,
            layout_desc=MagicMock(),
        ),
        worker_transfer._GroupState(
            layer_names=["layer_2"],
            engine_kv_format=MagicMock(),
            blocks_in_chunk=1,
            layout_desc=MagicMock(),
        ),
    ]
    calls: list[dict] = []

    def _fake_gather(kv_caches, block_ids, blocks_in_chunk, **kwargs: Any):
        calls.append(
            {
                "layers": sorted(kv_caches),
                "block_ids": block_ids,
                "blocks_in_chunk": blocks_in_chunk,
                "out": kwargs.get("out"),
                "chunk_indices": kwargs.get("chunk_indices"),
            }
        )
        return kwargs.get("out")

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    monkeypatch.setattr(
        worker_transfer, "scatter_cpu_to_paged_kv", lambda *a, **k: None
    )
    return ctx, calls


def test_submit_store_multigroup_fans_out_per_group(monkeypatch) -> None:
    ctx, calls = _fanout_ctx(monkeypatch)
    t = [torch.zeros(1) for _ in range(4)]
    fake = _FakeGroupedContext(t, [0, 1, 0, 1], [0, 0, 1, 1])
    ctx._engine_driven_context = fake  # type: ignore[assignment]

    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_store(
        "req", MagicMock(), 1, kv, [[1, 2, 3, 4], [5, 6]], MagicMock(), 2
    )

    assert future.result(timeout=1) is True
    assert fake.committed
    assert len(calls) == 2
    assert calls[0]["layers"] == ["layer_0", "layer_1"]
    assert calls[0]["block_ids"] == [1, 2, 3, 4]
    assert calls[0]["blocks_in_chunk"] == 2
    assert calls[0]["out"] == [t[0], t[1]]
    assert calls[0]["chunk_indices"] == [0, 1]
    assert calls[1]["layers"] == ["layer_2"]
    assert calls[1]["block_ids"] == [5, 6]
    assert calls[1]["blocks_in_chunk"] == 1
    assert calls[1]["out"] == [t[2], t[3]]


def test_submit_store_multigroup_group_count_mismatch(monkeypatch) -> None:
    ctx, _ = _fanout_ctx(monkeypatch)
    ctx._engine_driven_context = _FakeGroupedContext([], [], [])  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="block-id lists"):
        ctx.submit_store("req", MagicMock(), 1, {}, [[1, 2]], MagicMock(), 2)


def test_register_rejects_sliding_window_groups() -> None:
    ctx = EngineDrivenTransferContext()
    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}
    with pytest.raises(RuntimeError, match="sliding-window"):
        ctx.register(
            instance_id=1,
            kv_caches=kv,
            model_name="m",
            world_size=1,
            blocks_in_chunk=2,
            mq_client=MagicMock(),
            mq_timeout=1.0,
            send_request=MagicMock(),
            engine_group_infos=[
                EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
                EngineGroupInfo(
                    engine_group_id=1, layer_indices=(1,), sw_size_tokens=128
                ),
            ],
        )


def test_register_payload_carries_group_layouts() -> None:
    """Two full-attention groups produce per-group layouts in the payload."""
    ctx = EngineDrivenTransferContext()
    kv = {f"layer_{i}": torch.zeros(2, 6, 4, 2, 8) for i in range(3)}

    # First Party
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )

    sent: list[Any] = []

    def _send(_mq, _rt, args):
        sent.append(args[0])
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_x", pool_size=4096
        )
        return future

    ctx.register(
        instance_id=1,
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=_send,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1)),
            EngineGroupInfo(engine_group_id=1, layer_indices=(2,)),
        ],
    )
    payload: RegisterEngineDrivenContextPayload = sent[0]
    assert len(payload.group_layouts) == 2
    assert payload.group_layouts[0].num_layers == 2
    assert payload.group_layouts[1].num_layers == 1
    assert isinstance(payload.group_layouts[0], GroupLayout)
    assert len(ctx._group_states) == 2
    assert ctx._group_states[1].layer_names == ["layer_2"]
