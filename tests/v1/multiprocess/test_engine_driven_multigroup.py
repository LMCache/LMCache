# SPDX-License-Identifier: Apache-2.0
"""
Tests for engine-driven multi-group (uniform coverage) transfers.
"""

# Standard
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4
import threading

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    GroupLayout,
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.server_transfer import (
    InlineTransferStrategy,
    ShmTransferStrategy,
)
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContextMetadata,
    compute_kv_layout,
)
from lmcache.v1.multiprocess.transfer_context.inline import EngineDrivenContextInline
from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
    decode_grouped_tensors,
    encode_grouped_tensors,
    encode_tensors,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server


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


@pytest.mark.cuda
def test_shm_strategy_multigroup_store_and_retrieve_roundtrip() -> None:
    """Per-group reserve on store; retrieve misses until every group commits."""
    # Marked cuda so suites that deselect with -m "not cuda" (the XPU unit
    # lane) never collect this at all. The runtime check below still covers
    # plain CUDA-less runs, which do not filter by marker.
    if not (torch_dev.is_available() and torch_device_type == "cuda"):
        pytest.skip("CUDA is not available")

    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess.modules.server_transfer import (
        InlineTransferStrategy,
        ShmTransferStrategy,
    )

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
            fallback_strategy=InlineTransferStrategy(sm),
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
        self.skipped: Any = None

    def prepare_store_grouped(self, _key, _iid, skipped_chunks=()):
        self.skipped = skipped_chunks
        return self._prep

    def prepare_retrieve_grouped(self, _key, _iid):
        tensors, chunk_indices, group_ids = self._prep
        return tensors, group_ids, chunk_indices

    def commit_store(self, _key, _iid, _chunks):
        self.committed = True
        return True

    def commit_retrieve(self, _key, _iid):
        return True

    def close(self) -> None:
        return None


def _fanout_ctx(monkeypatch) -> tuple[EngineDrivenTransferContext, list[dict]]:
    """Worker context with two registered groups and a capturing gather."""
    ctx = EngineDrivenTransferContext(1, MagicMock())
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
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6]], MagicMock(), 2
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
        ctx.submit_store("req", MagicMock(), {}, [[1, 2]], MagicMock(), 2)


def test_register_accepts_sliding_window_groups() -> None:
    """A sliding-window group registers (window < chunk) instead of raising; the
    SW group keeps fewer blocks per chunk than its full-attention sibling, and
    the payload carries the reduced per-group window."""
    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}
    # Half a chunk, so the window is always one of the two blocks per chunk.
    # A fixed token count would exceed the chunk under a layout that resolves
    # a smaller block size, and the group would silently degrade to full
    # attention -- testing nothing.
    chunk_tokens = 2 * _detected_block_size(kv)
    sw_tokens = chunk_tokens // 2

    # First Party
    from lmcache.v1.multiprocess.custom_types import (
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

    ctx = EngineDrivenTransferContext(1, req_client)
    ctx.register(
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
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


def _inline_worker_ctx(monkeypatch):
    """Worker context with two groups over a real inline transfer context."""
    ctx, calls = _fanout_ctx(monkeypatch)
    pctx = EngineDrivenContextInline(
        metadata=MagicMock(), req_client=MagicMock(), mq_timeout=0.1
    )
    ctx._engine_driven_context = pctx
    return ctx, pctx, calls


def test_submit_store_multigroup_inline_sends_group_major_payload(
    monkeypatch,
) -> None:
    """Inline store gathers each group into fresh CPU chunks (no slots) and
    commits one group-major payload."""
    ctx, pctx, _ = _inline_worker_ctx(monkeypatch)
    gathered = [["g0c0", "g0c1"], ["g1c0", "g1c1"]]
    calls: list[dict] = []
    events: list[str] = []

    def _fake_gather(*_a: Any, **kwargs: Any):
        idx = len(calls)
        calls.append(kwargs)
        events.append("gather")
        return gathered[idx]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    monkeypatch.setattr(
        worker_transfer, "synchronize_device", lambda _d: events.append("sync")
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
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6]], MagicMock(), 2
    )

    assert future.result(timeout=1) is True
    assert committed["chunks"] == [([0, 1], gathered[0]), ([0, 1], gathered[1])]
    # Fresh CPU gathers (no slot tensors) of every non-null chunk.
    assert all(c.get("out") is None for c in calls)
    assert [c.get("chunk_indices") for c in calls] == [[0, 1], [0, 1]]
    # Gather issues async device->CPU copies and commit serializes the
    # buffers, so a device synchronize must sit between the last gather and
    # the commit.
    last_gather = max(i for i, e in enumerate(events) if e == "gather")
    commit_idx = events.index("commit")
    assert "sync" in events[last_gather + 1 : commit_idx]


def test_submit_retrieve_multigroup_inline_scatters_group_major(
    monkeypatch,
) -> None:
    ctx, pctx, _ = _inline_worker_ctx(monkeypatch)
    ctx._engine_type = "sentinel-engine"
    payload = [([0], [torch.ones(1)]), ([0], [torch.ones(1) * 2])]
    monkeypatch.setattr(pctx, "prepare_retrieve_multigroup", lambda _k, _i, _e: payload)
    order: list[str] = []
    scattered: list[dict] = []

    def _fake_scatter(kv_caches, block_ids, chunks, *a, **k):
        order.append("scatter")
        scattered.append(
            {
                "layers": sorted(kv_caches),
                "chunks": chunks,
                "engine_type": k.get("engine_type"),
            }
        )

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", _fake_scatter)
    event = MagicMock()
    event.wait.side_effect = lambda: order.append("wait")

    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_retrieve("req", MagicMock(), kv, [[1, 2], [3]], event, 2)

    assert future.result(timeout=1) is True
    assert [s["layers"] for s in scattered] == [["layer_0", "layer_1"], ["layer_2"]]
    assert scattered[0]["chunks"] is payload[0][1]
    assert scattered[1]["chunks"] is payload[1][1]
    # The engine's event gates every group's scatter, as on the single-group
    # path, and each group is scattered with the registered engine type.
    assert order == ["wait", "scatter", "scatter"]
    assert all(s["engine_type"] == "sentinel-engine" for s in scattered)


def test_submit_retrieve_multigroup_inline_group_count_mismatch(
    monkeypatch,
) -> None:
    """A payload whose group count differs from the registration is a miss."""
    ctx, pctx, _ = _inline_worker_ctx(monkeypatch)
    monkeypatch.setattr(
        pctx,
        "prepare_retrieve_multigroup",
        lambda _k, _i, _e: [([0], [torch.ones(1)])],
    )
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_retrieve("req", MagicMock(), kv, [[1, 2], [3]], MagicMock(), 2)
    assert future.result(timeout=1) is False


class _FakeMemoryObj:
    def __init__(self, shape, l1_manager_id: int = 0):
        self.tensor = torch.zeros(shape)
        self._l1_manager_id = l1_manager_id

    def get_l1_manager(self) -> int:
        return self._l1_manager_id


class _FakeStorageManager:
    """In-memory reserve/read double for the inline strategy.

    Each object group is placed in its own L1 manager, so writes must be
    completed by owner: ``finish_write(keys)`` refuses a multi-L1 setup.
    """

    def __init__(self, shapes_by_group: dict[int, tuple[int, ...]]):
        self._shapes = shapes_by_group
        self.objs: dict[ObjectKey, _FakeMemoryObj] = {}
        self.finished_writes: list[ObjectKey] = []
        self.finished_owners: set[int] = set()
        self.finished_reads: list[ObjectKey] = []
        self.read_owners: list[Any] = []
        self.deleted: list[ObjectKey] = []

    def reserve_write(self, obj_keys, _layout):
        reserved = {}
        for obj_key in obj_keys:
            gid = obj_key.object_group_id
            obj = _FakeMemoryObj(self._shapes[gid], l1_manager_id=gid)
            self.objs[obj_key] = obj
            reserved[obj_key] = obj
        return reserved

    def finish_write(self, keys):
        raise ValueError("multiple L1 managers: complete writes by owner")

    def prepare_write_completion(self, objects):
        groups: dict[int, list[ObjectKey]] = {}
        for key, obj in objects.items():
            groups.setdefault(obj.get_l1_manager(), []).append(key)
        return list(groups.items())

    def finish_write_by_owner(self, completion):
        for owner, keys in completion:
            self.finished_owners.add(owner)
            self.finished_writes.extend(keys)

    def delete_l1_keys(self, keys, force=False):
        self.deleted.extend(keys)
        for key in keys:
            self.objs.pop(key, None)
        return len(keys), 0

    @contextmanager
    def read_prefetched_results(self, obj_keys, l1_owners=None):
        self.read_owners.append(l1_owners)
        objs = [self.objs[k] for k in obj_keys if k in self.objs]
        yield objs if len(objs) == len(obj_keys) else []

    def finish_read_prefetched(self, keys, l1_owners=None):
        self.finished_reads.extend(keys)


def test_inline_strategy_multigroup_store_retrieve_roundtrip() -> None:
    """Group-major inline commit writes each group against its own layout;
    retrieve returns a group-major payload and is all-or-nothing. With the
    groups in different L1 managers, writes complete by owner and reads carry
    the request's L1 owners."""
    shapes: dict[int, tuple[int, ...]] = {0: (2, 2, 8, 16), 1: (2, 1, 8, 4)}
    sm = _FakeStorageManager(shapes)
    owners = {_obj_key(1, 0): 0, _obj_key(1, 1): 1}
    strategy = InlineTransferStrategy(
        sm,  # type: ignore[arg-type]
        read_owner_resolver=lambda _request_id: owners,
    )
    metadata = _group_metadata()
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * 16, start=0, end=16, request_id="req-inline"
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
            cpu_data=encode_grouped_tensors([([0, 1], g) for g in payload]),
            context=metadata,
            resolve_obj_keys=lambda _k: group_keys[0],
            group_keys=group_keys,
        )
        is True
    )
    assert len(sm.finished_writes) == 4
    assert sm.finished_owners == {0, 1}
    assert torch.equal(sm.objs[_obj_key(2, 1)].tensor, payload[1][1])

    ret = strategy.prepare_retrieve(
        key=key,
        instance_id=7,
        resolve_obj_keys=lambda _k: group_keys[0],
        group_keys=group_keys,
    )
    assert ret.success is True
    retrieved = decode_grouped_tensors(
        ret.data,
        expected=[
            (torch.Size(shapes[0]), torch.float32, 2),
            (torch.Size(shapes[1]), torch.float32, 2),
        ],
    )
    assert len(retrieved) == 2
    assert [indices for indices, _ in retrieved] == [[0, 1], [0, 1]]
    assert torch.equal(retrieved[0][1][1], payload[0][1])
    assert len(sm.finished_reads) == 4
    assert sm.read_owners == [owners, owners]

    # A group with a missing chunk makes the whole retrieve a miss.
    partial_keys = [group_keys[0], [_obj_key(1, 1), _obj_key(9, 1)]]
    ret_miss = strategy.prepare_retrieve(
        key=key,
        instance_id=7,
        resolve_obj_keys=lambda _k: partial_keys[0],
        group_keys=partial_keys,
    )
    assert ret_miss.success is False


def test_inline_strategy_store_shape_mismatch_releases_reservations() -> None:
    """A chunk whose shape disagrees with the reserved object fails the store,
    and the unwritten reservations are released and deleted -- they must not
    wedge later writes or surface as garbage cache hits."""
    shapes: dict[int, tuple[int, ...]] = {0: (2, 2, 8, 16), 1: (2, 1, 4, 4)}
    sm = _FakeStorageManager(shapes)
    strategy = InlineTransferStrategy(sm)  # type: ignore[arg-type]
    metadata = _group_metadata()
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * 16, start=0, end=16, request_id="req-inline-mismatch"
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
            cpu_data=encode_grouped_tensors([([0, 1], g) for g in payload]),
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


@pytest.mark.parametrize("use_shm", [False, True], ids=["inline", "shm"])
@pytest.mark.parametrize("window_tokens", [-1, 512], ids=["full", "sliding"])
def test_single_compressed_group_worker_server_roundtrip(
    use_shm: bool, window_tokens: int, unused_tcp_port: int
) -> None:
    """A single compressed group survives real RPC, storage and retrieval.

    Both transports must preserve the physical slot layout and the absolute
    chunk positions when a sliding window drops the leading chunks.
    """
    chunk_tokens = 512
    tokens_per_block = 256
    slots_per_block = 2
    num_blocks = 6
    shm_name = f"lmcache_l1_pool_test_compressed_{uuid4().hex}" if use_shm else ""
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=8 * 1024 * 1024,
                use_lazy=False,
                shm_name=shm_name,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    engine = MPCacheServerContext(config, chunk_size=chunk_tokens)
    module = EngineDrivenTransferModule(engine)
    lookup = LookupModule(engine)
    server = create_request_server(
        [module, lookup], MPServerConfig(host="127.0.0.1", port=unused_tcp_port)
    )
    server.start()
    client = RequestClientFactory.create(f"tcp://127.0.0.1:{unused_tcp_port}")
    worker = EngineDrivenTransferContext(1, client)
    source = {
        "layer_0": torch.arange(
            (2 * num_blocks + 1) * slots_per_block * 16, dtype=torch.float32
        ).reshape(-1, slots_per_block, 16)
    }
    destination = {"layer_0": torch.full_like(source["layer_0"], -1)}
    store_ids = list(range(1, num_blocks + 1))
    retrieve_ids = list(range(num_blocks + 1, 2 * num_blocks + 1))
    key = IPCCacheServerKey.from_token_ids(
        "compressed", 1, 0, [1] * 1536, start=0, end=1536, request_id="compressed"
    )
    try:
        worker.register(
            source,
            "compressed",
            1,
            2,
            5.0,
            engine_group_infos=[
                EngineGroupInfo(
                    0,
                    (0,),
                    tokens_per_block=tokens_per_block,
                    sw_size_tokens=window_tokens,
                )
            ],
            lmcache_chunk_tokens=chunk_tokens,
        )
        for _ in range(2):
            assert worker.submit_store(
                key.request_id, key, source, [store_ids], None, 2
            ).result(timeout=5)

        client.lookup(key, 1).result(timeout=5)
        assert client.wait_prefetch_status(key.request_id, 5).result(timeout=10) == 3
        assert worker.submit_retrieve(
            key.request_id, key, destination, [retrieve_ids], None, 2
        ).result(timeout=5)

        expected = torch.full_like(source["layer_0"], -1)
        first_block = (
            0 if window_tokens < 0 else num_blocks - window_tokens // tokens_per_block
        )
        expected[retrieve_ids[first_block:]] = source["layer_0"][
            store_ids[first_block:]
        ]
        assert torch.equal(destination["layer_0"], expected)
        client.end_session(key.request_id).result(timeout=5)
        unregistration = worker.unregister()
        assert unregistration is not None
        unregistration.result(timeout=5)
    finally:
        worker.close()
        client.close()
        server.close()
        lookup.close()
        module.close()
        engine.close()


def test_register_payload_carries_group_layouts() -> None:
    """Two full-attention groups produce per-group layouts in the payload."""
    kv = {f"layer_{i}": torch.zeros(2, 6, 4, 2, 8) for i in range(3)}

    # First Party
    from lmcache.v1.multiprocess.custom_types import (
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

    ctx = EngineDrivenTransferContext(1, req_client)
    ctx.register(
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
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


def _register_capturing_payload(
    kv: dict[str, torch.Tensor], engine_group_infos: list[EngineGroupInfo]
) -> tuple[RegisterEngineDrivenContextPayload, EngineDrivenTransferContext]:
    """Register ``kv`` and return the payload sent to the server."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
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

    ctx = EngineDrivenTransferContext(1, req_client)
    ctx.register(
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_timeout=1.0,
        engine_group_infos=engine_group_infos,
    )
    return sent[0], ctx


def test_register_payload_carries_per_group_kv_size() -> None:
    """A two-plane K/V group beside a one-plane (index-cache style) group must
    report each group's own plane count. The top-level ``use_mla`` describes
    only the first layer, so a server sizing every group from it reserves the
    one-plane group with two planes and inline store fails on the shape."""
    kv = {
        "layer_0": torch.zeros(2, 4, 4, 2, 8),  # separate K/V planes
        "layer_1": torch.zeros(4, 4, 16),  # single plane
    }
    payload, ctx = _register_capturing_payload(
        kv,
        [
            EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
            EngineGroupInfo(engine_group_id=1, layer_indices=(1,)),
        ],
    )

    assert payload.use_mla is False
    assert [gl.kv_size for gl in payload.group_layouts] == [2, 1]
    # The plane count the worker reports matches the layout it gathers with.
    assert [len(s.layout_desc.shapes[0]) for s in ctx._group_states] == [4, 3]


@pytest.mark.parametrize("device", sorted({"cpu", torch_device_type}))
@pytest.mark.parametrize("mixed_groups", [False, True])
@pytest.mark.parametrize("skip_tokens", [0, 768])
def test_compressed_groups_roundtrip(
    device: str, mixed_groups: bool, skip_tokens: int
) -> None:
    """Compressed, padded pages retain their bytes and logical skip boundaries."""
    chunk_tokens = 512
    # (logical tokens/block, physical slots/block, dtype, sliding window)
    geometry = [(256, 2, torch.float32, -1)]
    if mixed_groups:
        geometry += [
            (256, 64, torch.uint8, -1),
            (64, 64, torch.uint8, 256),
            (256, 2, torch.float32, 256),
        ]
    source: dict[str, torch.Tensor] = {}
    destination: dict[str, torch.Tensor] = {}
    destination_pools: list[torch.Tensor] = []
    infos: list[EngineGroupInfo] = []
    store_ids: list[list[int]] = []
    retrieve_ids: list[list[int]] = []
    expected_layouts: list[tuple[torch.Size, torch.dtype, int]] = []
    for gid, (tokens, slots, dtype, window) in enumerate(geometry):
        count = 3 * chunk_tokens // tokens
        width = slots * 16
        pool = (
            (torch.arange((2 * count + 2) * (width + 32)) % 251)
            .to(device=device, dtype=dtype)
            .reshape(2 * count + 2, width + 32)
        )
        target = torch.full_like(pool, 255)
        name = f"layer_{gid}"
        source[name] = pool[:, :width].view(-1, slots, 16)
        destination[name] = target[:, :width].view(-1, slots, 16)
        destination_pools.append(target)
        infos.append(
            EngineGroupInfo(gid, (gid,), tokens_per_block=tokens, sw_size_tokens=window)
        )
        store_ids.append(list(range(1, count + 1)))
        retrieve_ids.append(list(range(count + 1, 2 * count + 1)))
        window_tokens = chunk_tokens if window < 0 else min(chunk_tokens, window)
        expected_layouts.append(
            (torch.Size([1, window_tokens * slots // tokens, 16]), dtype, 3)
        )

    client = MagicMock()
    client.register_kv_cache_engine_driven_context.return_value.result.return_value = (
        RegisterEngineDrivenContextResponse("", 0)
    )
    client.commit_store.return_value.result.return_value = True
    ctx = EngineDrivenTransferContext(1, client)
    ctx.register(
        source,
        "compressed",
        1,
        2,
        1.0,
        engine_group_infos=infos,
        lmcache_chunk_tokens=chunk_tokens,
    )
    try:
        payload = client.register_kv_cache_engine_driven_context.call_args.args[0]
        assert [g.num_physical_slots for g in payload.group_layouts] == [
            shape[-2] for shape, _, _ in expected_layouts
        ]
        key = IPCCacheServerKey.from_token_ids(
            "compressed", 1, 0, [1] * 1536, start=0, end=1536, request_id="compressed"
        )
        assert ctx.submit_store("compressed", key, source, store_ids, None, 2).result()
        stored = decode_grouped_tensors(
            client.commit_store.call_args.args[2], expected=expected_layouts
        )
        retrieved = []
        for gid, (indices, chunks) in enumerate(stored):
            tokens, slots, _, window = geometry[gid]
            blocks_per_chunk = chunk_tokens // tokens
            keep = (
                blocks_per_chunk if window < 0 else min(window, chunk_tokens) // tokens
            )
            for index, chunk in zip(indices, chunks, strict=True):
                ids = store_ids[gid][
                    (index + 1) * blocks_per_chunk - keep : (index + 1)
                    * blocks_per_chunk
                ]
                expected = (
                    source[f"layer_{gid}"][ids].reshape(1, keep * slots, 16).cpu()
                )
                assert torch.equal(chunk, expected)
            first = 0 if window < 0 else 3 - (window + chunk_tokens - 1) // chunk_tokens
            retrieved.append((indices[first:], chunks[first:]))
        client.prepare_retrieve.return_value.result.return_value = (
            PrepareRetrieveResponse(
                success=True, data=encode_grouped_tensors(retrieved), context={}
            )
        )
        assert ctx.submit_retrieve(
            "compressed",
            key,
            destination,
            retrieve_ids,
            None,
            2,
            skip_first_n_tokens=skip_tokens,
        ).result()
        for gid, (tokens, _, _, window) in enumerate(geometry):
            expected = torch.full_like(destination[f"layer_{gid}"], 255)
            first_token = max(skip_tokens, 0 if window < 0 else 1536 - window)
            for pos, (src, dst) in enumerate(
                zip(store_ids[gid], retrieve_ids[gid], strict=True)
            ):
                if pos * tokens >= first_token:
                    expected[dst] = source[f"layer_{gid}"][src]
            assert torch.equal(destination[f"layer_{gid}"], expected)
            assert torch.all(destination_pools[gid][:, -32:] == 255)
    finally:
        ctx.close()


@pytest.mark.parametrize(
    "make_payload",
    [
        pytest.param(
            lambda shapes: encode_grouped_tensors(
                [
                    ([0, 1], [torch.zeros(shapes[0]) for _ in range(2)]),
                    # One tensor in group 1 does not match its registered layout.
                    ([0, 1], [torch.zeros(shapes[1]), torch.zeros((2, 1, 4, 4))]),
                ]
            ),
            id="group-shape-mismatch",
        ),
        pytest.param(
            lambda shapes: encode_grouped_tensors(
                [([0, 1], [torch.zeros(shapes[0]) for _ in range(2)])]
            ),
            id="group-count-mismatch",
        ),
        pytest.param(
            lambda shapes: encode_tensors([torch.zeros(shapes[0]) for _ in range(2)]),
            id="flat-envelope",
        ),
        pytest.param(
            # A legacy pickle payload (pickle.dumps([[1]]), protocol 4).
            lambda shapes: (
                b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00]\x94]\x94K\x01aa."
            ),
            id="pickle",
        ),
    ],
)
def test_inline_strategy_multigroup_rejects_payload_before_storage(
    make_payload,
) -> None:
    """A multi-group payload that does not match every group's registered
    layout -- wrong shape in one group, wrong group count, a single-group
    envelope, or a legacy pickle -- is refused while decoding, before any
    object is reserved."""
    shapes: dict[int, tuple[int, ...]] = {0: (2, 2, 8, 16), 1: (2, 1, 8, 4)}
    sm = _FakeStorageManager(shapes)
    strategy = InlineTransferStrategy(sm)  # type: ignore[arg-type]
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * 16, start=0, end=16, request_id="req-inline-reject"
    )
    group_keys = [
        [_obj_key(1, 0), _obj_key(2, 0)],
        [_obj_key(1, 1), _obj_key(2, 1)],
    ]

    with pytest.raises((ValueError, msgspec.DecodeError)):
        strategy.commit_store(
            key=key,
            instance_id=7,
            cpu_data=make_payload(shapes),
            context=_group_metadata(),
            resolve_obj_keys=lambda _k: group_keys[0],
            group_keys=group_keys,
        )
    assert sm.objs == {}
    assert sm.finished_writes == []


def test_register_uses_adapter_chunk_size_over_local_group_lcm() -> None:
    """A pipeline stage may hold only the groups with the smaller block size.
    ``blocks_in_chunk`` is counted in the engine's global (larger) block, so
    rebuilding the chunk from this stage's own groups would register half of
    it. The adapter's LMCache chunk size must win."""
    kv = {f"layer_{i}": torch.zeros(2, 8, 4, 2, 8) for i in range(2)}
    bs = _detected_block_size(kv)
    chunk_tokens = 4 * bs
    # Global block = 2 * bs (a group this stage does not hold), so the
    # adapter's blocks_in_chunk is chunk_tokens // (2 * bs) == 2.
    blocks_in_chunk = chunk_tokens // (2 * bs)
    infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0,), tokens_per_block=bs),
        EngineGroupInfo(engine_group_id=1, layer_indices=(1,), tokens_per_block=bs),
    ]

    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        RegisterEngineDrivenContextResponse,
    )

    def _register_with(**extra: Any) -> tuple[Any, EngineDrivenTransferContext]:
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
        ctx = EngineDrivenTransferContext(1, req_client)
        ctx.register(
            kv_caches=kv,
            model_name="m",
            world_size=1,
            blocks_in_chunk=blocks_in_chunk,
            mq_timeout=1.0,
            engine_group_infos=infos,
            **extra,
        )
        return sent[0], ctx

    payload, ctx = _register_with(lmcache_chunk_tokens=chunk_tokens)
    assert [gl.window_tokens for gl in payload.group_layouts] == [chunk_tokens] * 2
    assert [s.blocks_in_chunk for s in ctx._group_states] == [4, 4]

    # Without the adapter's chunk size, the stage-local reconstruction is the
    # half-chunk this guards against.
    legacy, _ = _register_with()
    assert [gl.window_tokens for gl in legacy.group_layouts] == [2 * bs] * 2

    with pytest.raises(ValueError, match="not a positive multiple"):
        _register_with(lmcache_chunk_tokens=bs + 1)


def test_register_payload_carries_group_windows_and_kinds() -> None:
    """Each group reports the trailing chunks it needs for a hit: full
    attention sends 0 (whole prefix), a sliding window rounds up to whole
    chunks, and a recurrent-state group keeps its one-chunk window."""
    kv = {f"layer_{i}": torch.zeros(2, 8, 4, 2, 8) for i in range(3)}
    bs = _detected_block_size(kv)
    payload, ctx = _register_capturing_payload(
        kv,
        [
            EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(1,), sw_size_tokens=3 * bs
            ),
            EngineGroupInfo(
                engine_group_id=2,
                layer_indices=(2,),
                sw_size_tokens=bs,
                recurrent_state=True,
            ),
        ],
    )

    # blocks_in_chunk=2, so a chunk is 2 * bs tokens: 3 * bs needs 2 chunks.
    assert [gl.window_chunks for gl in payload.group_layouts] == [0, 2, 1]
    assert [gl.recurrent for gl in payload.group_layouts] == [False, False, True]
    assert [s.window_chunks for s in ctx._group_states] == [-1, 2, 1]


def test_store_multigroup_inline_leaves_out_all_null_chunks(monkeypatch) -> None:
    """A chunk whose engine blocks are all the null block carries no KV: it is
    neither gathered nor sent, and its key is never written."""
    ctx, pctx, _ = _inline_worker_ctx(monkeypatch)
    gathered: list[dict] = []

    def _fake_gather(*_a: Any, **kwargs: Any):
        gathered.append(kwargs)
        return [f"chunk{c}" for c in kwargs["chunk_indices"]]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    prepared: list[Any] = []
    monkeypatch.setattr(pctx, "prepare_store", lambda k, i: prepared.append(k))
    committed: list[Any] = []

    def _commit(_k: Any, _i: int, chunks: Any) -> bool:
        committed.append(chunks)
        return True

    monkeypatch.setattr(pctx, "commit_store", _commit)
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}

    # Group 0 (2 blocks/chunk): chunk 0 is all null. Group 1 (1 block/chunk):
    # chunk 0 is null. A partly-null chunk is still stored.
    future = ctx.submit_store(
        "req", MagicMock(), kv, [[0, 0, 0, 4], [0, 6]], MagicMock(), 2
    )
    assert future.result(timeout=1) is True
    assert [g["chunk_indices"] for g in gathered] == [[1], [1]]
    assert committed == [[([1], ["chunk1"]), ([1], ["chunk1"])]]

    # Everything null: nothing to store, and no round trip at all.
    gathered.clear()
    committed.clear()
    prepared.clear()
    future = ctx.submit_store(
        "req", MagicMock(), kv, [[0, 0, 0, 0], [0, 0]], MagicMock(), 2
    )
    assert future.result(timeout=1) is True
    assert gathered == [] and committed == [] and prepared == []


def test_store_multigroup_shm_asks_server_to_skip_all_null_chunks(
    monkeypatch,
) -> None:
    ctx, _ = _fanout_ctx(monkeypatch)
    t = [torch.zeros(1) for _ in range(2)]
    fake = _FakeGroupedContext(t, [1, 1], [0, 1])
    ctx._engine_driven_context = fake  # type: ignore[assignment]
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}

    future = ctx.submit_store(
        "req", MagicMock(), kv, [[0, 0, 3, 4], [0, 6]], MagicMock(), 2
    )
    assert future.result(timeout=1) is True
    assert fake.skipped == ((0,), (0,))

    # No null chunks: the RPC carries no skip list at all.
    future = ctx.submit_store(
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6]], MagicMock(), 2
    )
    assert future.result(timeout=1) is True
    assert fake.skipped == ()


def _windowed_inline_ctx(monkeypatch):
    """Inline worker context whose group 1 needs only its last chunk."""
    ctx, pctx, _ = _inline_worker_ctx(monkeypatch)
    ctx._group_states[1].window_chunks = 1
    ctx._chunk_tokens = 2
    scattered: list[dict] = []

    def _fake_scatter(kv_caches, block_ids, chunks, *a, **k):
        scattered.append(
            {
                "block_ids": block_ids,
                "chunks": chunks,
                "skip_first_n_tokens": k.get("skip_first_n_tokens"),
            }
        )

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", _fake_scatter)
    return ctx, pctx, scattered


def test_retrieve_multigroup_inline_serves_each_group_window(monkeypatch) -> None:
    """A windowed group gets only its trailing chunks: they are scattered from
    that chunk's first block, with the computed-prefix skip shifted past the
    dropped chunks."""
    ctx, pctx, scattered = _windowed_inline_ctx(monkeypatch)
    requested: list[Any] = []
    a, b, c = torch.ones(1), torch.ones(1) * 2, torch.ones(1) * 3

    def _prepare(_k, _i, expected):
        requested.append([count for _, _, count in expected])
        return [([0, 1], [a, b]), ([2], [c])]

    monkeypatch.setattr(pctx, "prepare_retrieve_multigroup", _prepare)
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}

    future = ctx.submit_retrieve(
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6, 7]], None, 2, 1
    )
    assert future.result(timeout=1) is True
    assert requested == [[2, 3]]
    assert scattered[0]["block_ids"] == [1, 2, 3, 4]
    assert scattered[0]["skip_first_n_tokens"] == 1
    assert scattered[1]["block_ids"] == [7]
    assert scattered[1]["chunks"] == [c]
    assert scattered[1]["skip_first_n_tokens"] == 0


def test_retrieve_multigroup_inline_rejects_chunks_outside_the_window(
    monkeypatch,
) -> None:
    ctx, pctx, scattered = _windowed_inline_ctx(monkeypatch)
    monkeypatch.setattr(
        pctx,
        "prepare_retrieve_multigroup",
        lambda _k, _i, _e: [
            ([0, 1], [torch.ones(1), torch.ones(1)]),
            ([1, 2], [torch.ones(1), torch.ones(1)]),
        ],
    )
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}
    future = ctx.submit_retrieve(
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6, 7]], None, 2
    )
    assert future.result(timeout=1) is False
    assert scattered == []


def test_retrieve_multigroup_shm_serves_each_group_window(monkeypatch) -> None:
    ctx, _ = _fanout_ctx(monkeypatch)
    ctx._group_states[1].window_chunks = 1
    ctx._chunk_tokens = 2
    t = [torch.zeros(1) for _ in range(3)]
    scattered: list[Any] = []
    monkeypatch.setattr(
        worker_transfer,
        "scatter_cpu_to_paged_kv",
        lambda kv, block_ids, chunks, *a, **k: scattered.append((block_ids, chunks)),
    )
    kv = {name: torch.zeros(1) for name in ("layer_0", "layer_1", "layer_2")}

    ctx._engine_driven_context = _FakeGroupedContext(t, [0, 1, 2], [0, 0, 1])  # type: ignore[assignment]
    future = ctx.submit_retrieve(
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6, 7]], None, 2
    )
    assert future.result(timeout=1) is True
    assert scattered == [([1, 2, 3, 4], [t[0], t[1]]), ([7], [t[2]])]

    # Slots that are not the group's window are refused before any scatter.
    scattered.clear()
    ctx._engine_driven_context = _FakeGroupedContext(t, [0, 1, 1], [0, 0, 1])  # type: ignore[assignment]
    future = ctx.submit_retrieve(
        "req", MagicMock(), kv, [[1, 2, 3, 4], [5, 6, 7]], None, 2
    )
    assert future.result(timeout=1) is False
    assert scattered == []


class _FakeShmMemoryObj(_FakeMemoryObj):
    shm_offset = 0

    @property
    def shm_byte_length(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


class _FakeShmStorageManager(_FakeStorageManager):
    """Single-L1 double for the SHM strategy, which completes by key."""

    def __init__(self, shapes_by_group):
        super().__init__(shapes_by_group)
        self.reserved: list[ObjectKey] = []

    def reserve_write(self, obj_keys, _layout):
        reserved = {}
        for obj_key in obj_keys:
            obj = _FakeShmMemoryObj(self._shapes[obj_key.object_group_id])
            self.objs[obj_key] = obj
            reserved[obj_key] = obj
        self.reserved.extend(obj_keys)
        return reserved

    def finish_write(self, keys):
        self.finished_writes.extend(keys)

    def unsafe_read(self, keys, l1_owners=None):
        found = [k for k in keys if k in self.objs]
        return found, [self.objs[k] for k in found]


def _shm_strategy(sm) -> ShmTransferStrategy:
    return ShmTransferStrategy(
        storage_manager=sm,
        pending_writes={},
        pending_reads={},
        pending_lock=threading.Lock(),
        transfer_key_factory=lambda key, iid: (iid, key),
        fallback_strategy=InlineTransferStrategy(sm),
    )


_SHAPES: dict[int, tuple[int, ...]] = {0: (2, 2, 8, 16), 1: (2, 1, 8, 4)}
_GROUP_KEYS = [
    [_obj_key(c, 0) for c in range(3)],
    [_obj_key(c, 1) for c in range(3)],
]
_KEY = IPCCacheServerKey.from_token_ids(
    "m", 1, 0, [1] * 24, start=0, end=24, request_id="req-skip"
)


def test_inline_strategy_writes_only_the_chunks_the_payload_names() -> None:
    sm = _FakeStorageManager(_SHAPES)
    strategy = InlineTransferStrategy(sm)  # type: ignore[arg-type]
    payload = encode_grouped_tensors(
        [
            ([0, 1, 2], [torch.zeros(_SHAPES[0]) for _ in range(3)]),
            # Group 1's first two chunks were all-null and are not sent.
            ([2], [torch.ones(_SHAPES[1])]),
        ]
    )

    assert strategy.commit_store(
        key=_KEY,
        instance_id=7,
        cpu_data=payload,
        context=_group_metadata(),
        resolve_obj_keys=lambda _k: _GROUP_KEYS[0],
        group_keys=_GROUP_KEYS,
    )
    assert set(sm.objs) == set(_GROUP_KEYS[0]) | {_GROUP_KEYS[1][2]}
    assert torch.equal(sm.objs[_GROUP_KEYS[1][2]].tensor, torch.ones(_SHAPES[1]))


def test_shm_strategy_reserves_no_slot_for_skipped_chunks() -> None:
    sm = _FakeShmStorageManager(_SHAPES)
    response = _shm_strategy(sm).prepare_store(
        key=_KEY,
        instance_id=7,
        context=_group_metadata(),
        resolve_obj_keys=lambda _k: _GROUP_KEYS[0],
        group_keys=_GROUP_KEYS,
        skipped_chunks=((), (0, 1)),
    )

    assert sm.reserved == _GROUP_KEYS[0] + [_GROUP_KEYS[1][2]]
    assert response.context["group_ids"] == [0, 0, 0, 1]
    assert response.context["chunk_indices"] == [0, 1, 2, 2]


@pytest.mark.parametrize("transport", ["inline", "shm"])
def test_strategies_report_windowed_retrieve_chunk_positions(transport) -> None:
    """Given each group's in-window suffix, both transports report the
    absolute chunk positions the worker checks against its own window."""
    sm = _FakeShmStorageManager(_SHAPES)
    for keys in _GROUP_KEYS:
        sm.reserve_write(keys, None)
    narrowed = [_GROUP_KEYS[0], _GROUP_KEYS[1][2:]]
    strategy = (
        InlineTransferStrategy(sm)  # type: ignore[arg-type]
        if transport == "inline"
        else _shm_strategy(sm)
    )

    response = strategy.prepare_retrieve(
        key=_KEY,
        instance_id=7,
        resolve_obj_keys=lambda _k: _GROUP_KEYS[0],
        group_keys=narrowed,
        group_first_chunk=[0, 2],
    )

    assert response.success
    if transport == "inline":
        decoded = decode_grouped_tensors(
            response.data,
            expected=[
                (torch.Size(_SHAPES[0]), torch.float32, 3),
                (torch.Size(_SHAPES[1]), torch.float32, 3),
            ],
        )
        assert [indices for indices, _ in decoded] == [[0, 1, 2], [2]]
    else:
        assert response.context["group_ids"] == [0, 0, 0, 1]
        assert response.context["chunk_indices"] == [0, 1, 2, 2]
