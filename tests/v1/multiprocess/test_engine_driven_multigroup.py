# SPDX-License-Identifier: Apache-2.0
"""
Tests for engine-driven multi-group (uniform coverage) transfers.
"""

# Standard
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
import pickle as _pickle

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
from lmcache.v1.multiprocess.modules.server_transfer import PickleTransferStrategy
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContextMetadata,
    compute_kv_layout,
)
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


def _detected_block_size(kv_caches: dict[str, torch.Tensor]) -> int:
    """Return the block size registration will derive for ``kv_caches``.

    A 5-D per-layer tensor is ``[2, NB, BS, NH, HS]`` under NHD and
    ``[2, NB, NH, BS, HS]`` under HND, so the block size depends on the
    layout the detector resolves -- which is process-level (vLLM's CPU
    backend is forced to HND, GPU defaults to NHD). Tests derive it from
    the same helper registration uses instead of hard-coding one layout's
    value, so they hold wherever they run.
    """
    block_size, *_ = compute_kv_layout(kv_caches)
    return block_size


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
            blocks_per_window=1,  # sliding-window group: keeps 1 of 2 blocks/chunk
            layout_desc=MagicMock(),
        ),
        worker_transfer._GroupState(
            layer_names=["layer_2"],
            engine_kv_format=MagicMock(),
            blocks_in_chunk=1,
            blocks_per_window=1,  # full attention (window == chunk)
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
                "blocks_per_window": kwargs.get("blocks_per_window"),
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
    # Raw (full) per-group block list is passed through; the sliding-window
    # trailing-keep happens inside gather, driven by blocks_per_window.
    assert calls[0]["block_ids"] == [1, 2, 3, 4]
    assert calls[0]["blocks_in_chunk"] == 2
    assert calls[0]["blocks_per_window"] == 1
    assert calls[0]["out"] == [t[0], t[1]]
    assert calls[0]["chunk_indices"] == [0, 1]
    assert calls[1]["layers"] == ["layer_2"]
    assert calls[1]["block_ids"] == [5, 6]
    assert calls[1]["blocks_in_chunk"] == 1
    assert calls[1]["blocks_per_window"] == 1
    assert calls[1]["out"] == [t[2], t[3]]


def test_submit_store_multigroup_group_count_mismatch(monkeypatch) -> None:
    ctx, _ = _fanout_ctx(monkeypatch)
    ctx._engine_driven_context = _FakeGroupedContext([], [], [])  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="block-id lists"):
        ctx.submit_store("req", MagicMock(), 1, {}, [[1, 2]], MagicMock(), 2)


def test_register_accepts_sliding_window_groups() -> None:
    """A sliding-window group registers (window < chunk) instead of raising; the
    SW group keeps fewer blocks per chunk than its full-attention sibling, and
    the payload carries the reduced per-group window."""
    ctx = EngineDrivenTransferContext()
    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}
    # Half a chunk, so the window is always one of the two blocks per chunk.
    # A fixed token count would exceed the chunk under a layout that resolves
    # a smaller block size, and the group would silently degrade to full
    # attention -- testing nothing.
    chunk_tokens = 2 * _detected_block_size(kv)
    sw_tokens = chunk_tokens // 2

    # First Party
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )

    sent: list[Any] = []

    def _register(payload):
        sent.append(payload)
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_x", pool_size=4096
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _register

    ctx.register(
        instance_id=1,
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(1,), sw_size_tokens=sw_tokens
            ),
        ],
    )
    # Group 1's half-chunk window is 1 of the 2 blocks per chunk. Group 0
    # (full attention) keeps both.
    full, sw = ctx._group_states
    assert full.blocks_per_window == full.blocks_in_chunk == 2
    assert sw.blocks_in_chunk == 2
    assert sw.blocks_per_window == 1
    payload: RegisterEngineDrivenContextPayload = sent[0]
    assert payload.group_layouts[0].window_tokens == chunk_tokens  # full attention
    assert payload.group_layouts[1].window_tokens == sw_tokens  # sliding window


def _pickle_worker_ctx(monkeypatch):
    """Worker context with two groups over a real pickle transfer context."""
    ctx, calls = _fanout_ctx(monkeypatch)
    pctx = EngineDrivenContextPickle(
        metadata=MagicMock(), req_client=MagicMock(), mq_timeout=0.1
    )
    ctx._engine_driven_context = pctx
    return ctx, pctx, calls


def test_submit_store_multigroup_pickle_sends_group_major_payload(
    monkeypatch,
) -> None:
    """Pickle store gathers each group into fresh CPU chunks (no slots) and
    commits one group-major payload."""
    ctx, pctx, _ = _pickle_worker_ctx(monkeypatch)
    gathered = [["g0c0", "g0c1"], ["g1c0"]]
    calls: list[dict] = []
    events: list[str] = []

    def _fake_gather(*_a: Any, **kwargs: Any):
        idx = len(calls)
        calls.append(kwargs)
        events.append("gather")
        return gathered[idx]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    monkeypatch.setattr(
        worker_transfer.torch_dev, "synchronize", lambda: events.append("sync")
    )
    monkeypatch.setattr(pctx, "prepare_store", lambda _k, _i: None)
    committed: dict[str, Any] = {}

    def _fake_commit(_k: Any, _i: int, chunks: Any) -> bool:
        committed["chunks"] = chunks
        events.append("commit")
        return True

    monkeypatch.setattr(pctx, "commit_store", _fake_commit)

    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_store(
        "req", MagicMock(), 1, kv, [[1, 2, 3, 4], [5, 6]], MagicMock(), 2
    )

    assert future.result(timeout=1) is True
    assert committed["chunks"] == gathered
    # Fresh CPU gathers: no slot tensors, no chunk-index filtering.
    assert all(c.get("out") is None and c.get("chunk_indices") is None for c in calls)
    # Gather issues async device->CPU copies and commit serializes the
    # buffers, so a device synchronize must sit between the last gather and
    # the commit.
    last_gather = max(i for i, e in enumerate(events) if e == "gather")
    commit_idx = events.index("commit")
    assert "sync" in events[last_gather + 1 : commit_idx]


def test_submit_retrieve_multigroup_pickle_scatters_group_major(
    monkeypatch,
) -> None:
    ctx, pctx, _ = _pickle_worker_ctx(monkeypatch)
    payload = [[torch.ones(1)], [torch.ones(1) * 2]]
    monkeypatch.setattr(pctx, "prepare_retrieve_multigroup", lambda _k, _i: payload)
    scattered: list[dict] = []
    monkeypatch.setattr(
        worker_transfer,
        "scatter_cpu_to_paged_kv",
        lambda kv_caches, block_ids, chunks, *a, **k: scattered.append(
            {"layers": sorted(kv_caches), "chunks": chunks}
        ),
    )

    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_retrieve(
        "req", MagicMock(), 1, kv, [[1, 2], [3]], MagicMock(), 2
    )

    assert future.result(timeout=1) is True
    assert [s["layers"] for s in scattered] == [["layer_0", "layer_1"], ["layer_2"]]
    assert scattered[0]["chunks"] is payload[0]
    assert scattered[1]["chunks"] is payload[1]


def test_submit_retrieve_multigroup_pickle_group_count_mismatch(
    monkeypatch,
) -> None:
    """A payload whose group count differs from the registration is a miss."""
    ctx, pctx, _ = _pickle_worker_ctx(monkeypatch)
    monkeypatch.setattr(
        pctx, "prepare_retrieve_multigroup", lambda _k, _i: [[torch.ones(1)]]
    )
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_retrieve(
        "req", MagicMock(), 1, kv, [[1, 2], [3]], MagicMock(), 2
    )
    assert future.result(timeout=1) is False


class _FakeMemoryObj:
    def __init__(self, shape):
        self.tensor = torch.zeros(shape)


class _FakeStorageManager:
    """In-memory reserve/read double for the pickle strategy."""

    def __init__(self, shapes_by_group: dict[int, tuple[int, ...]]):
        self._shapes = shapes_by_group
        self.objs: dict[ObjectKey, _FakeMemoryObj] = {}
        self.finished_writes: list[ObjectKey] = []
        self.finished_reads: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def reserve_write(self, obj_keys, _layout, _mode):
        reserved = {}
        for obj_key in obj_keys:
            obj = _FakeMemoryObj(self._shapes[obj_key.object_group_id])
            self.objs[obj_key] = obj
            reserved[obj_key] = obj
        return reserved

    def finish_write(self, keys):
        self.finished_writes.extend(keys)

    def delete_l1_keys(self, keys, force=False):
        self.deleted.extend(keys)
        for key in keys:
            self.objs.pop(key, None)
        return len(keys), 0

    @contextmanager
    def read_prefetched_results(self, obj_keys):
        objs = [self.objs[k] for k in obj_keys if k in self.objs]
        yield objs if len(objs) == len(obj_keys) else []

    def finish_read_prefetched(self, keys):
        self.finished_reads.extend(keys)


def test_pickle_strategy_multigroup_store_retrieve_roundtrip() -> None:
    """Group-major pickle commit writes each group against its own layout;
    retrieve returns a group-major payload and is all-or-nothing."""
    shapes: dict[int, tuple[int, ...]] = {0: (2, 2, 8, 16), 1: (2, 1, 8, 4)}
    sm = _FakeStorageManager(shapes)
    strategy = PickleTransferStrategy(sm)  # type: ignore[arg-type]
    metadata = _group_metadata()
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * 16, start=0, end=16, request_id="req-pkl"
    )
    group_keys = [
        [_obj_key(1, 0), _obj_key(2, 0)],
        [_obj_key(1, 1), _obj_key(2, 1)],
    ]
    payload = [
        [torch.full(shapes[0], float(c)) for c in range(2)],
        [torch.full(shapes[1], float(10 + c)) for c in range(2)],
    ]

    assert (
        strategy.commit_store(
            key=key,
            instance_id=7,
            cpu_data=_pickle.dumps(payload),
            context=metadata,
            resolve_obj_keys=lambda _k: group_keys[0],
            group_keys=group_keys,
        )
        is True
    )
    assert len(sm.finished_writes) == 4
    assert torch.equal(sm.objs[_obj_key(2, 1)].tensor, payload[1][1])

    ret = strategy.prepare_retrieve(
        key=key,
        instance_id=7,
        resolve_obj_keys=lambda _k: group_keys[0],
        group_keys=group_keys,
    )
    assert ret.success is True
    retrieved = _pickle.loads(ret.data)
    assert len(retrieved) == 2
    assert torch.equal(retrieved[0][1], payload[0][1])
    assert len(sm.finished_reads) == 4

    # A group with a missing chunk makes the whole retrieve a miss.
    partial_keys = [group_keys[0], [_obj_key(1, 1), _obj_key(9, 1)]]
    ret_miss = strategy.prepare_retrieve(
        key=key,
        instance_id=7,
        resolve_obj_keys=lambda _k: partial_keys[0],
        group_keys=partial_keys,
    )
    assert ret_miss.success is False


def test_pickle_strategy_store_shape_mismatch_releases_reservations() -> None:
    """A chunk whose shape disagrees with the reserved layout fails the store,
    and the unwritten reservations are released and deleted -- they must not
    wedge later writes or surface as garbage cache hits."""
    shapes: dict[int, tuple[int, ...]] = {0: (2, 2, 8, 16), 1: (2, 1, 4, 4)}
    sm = _FakeStorageManager(shapes)
    strategy = PickleTransferStrategy(sm)  # type: ignore[arg-type]
    metadata = _group_metadata()
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * 16, start=0, end=16, request_id="req-pkl-mismatch"
    )
    group_keys = [
        [_obj_key(1, 0), _obj_key(2, 0)],
        [_obj_key(1, 1), _obj_key(2, 1)],
    ]
    # Group 1 chunks are chunk-sized (8 tokens) while its reserved layout is
    # window-sized (4 tokens): the exact worker/server drift this guards.
    payload = [
        [torch.full(shapes[0], float(c)) for c in range(2)],
        [torch.full((2, 1, 8, 4), float(10 + c)) for c in range(2)],
    ]

    assert (
        strategy.commit_store(
            key=key,
            instance_id=7,
            cpu_data=_pickle.dumps(payload),
            context=metadata,
            resolve_obj_keys=lambda _k: group_keys[0],
            group_keys=group_keys,
        )
        is False
    )
    # Every reservation was finished (written or released) and the two
    # mismatched group-1 objects were deleted rather than left as garbage.
    assert set(sm.finished_writes) == set(group_keys[0]) | set(group_keys[1])
    assert set(sm.deleted) == set(group_keys[1])
    assert _obj_key(1, 1) not in sm.objs
    assert _obj_key(2, 1) not in sm.objs


def test_register_payload_carries_group_layouts() -> None:
    """Two full-attention groups produce per-group layouts in the payload."""
    ctx = EngineDrivenTransferContext()
    kv = {f"layer_{i}": torch.zeros(2, 6, 4, 2, 8) for i in range(3)}

    # First Party
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )

    sent: list[Any] = []

    def _register(payload):
        sent.append(payload)
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_x", pool_size=4096
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _register

    ctx.register(
        instance_id=1,
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
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
    # Full-attention groups carry the full chunk as their window
    # (blocks_in_chunk * block_size).
    chunk_tokens = 2 * _detected_block_size(kv)
    assert payload.group_layouts[0].window_tokens == chunk_tokens
    assert payload.group_layouts[1].window_tokens == chunk_tokens
    assert len(ctx._group_states) == 2
    assert ctx._group_states[1].layer_names == ["layer_2"]
