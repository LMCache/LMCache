# SPDX-License-Identifier: Apache-2.0
"""Unit tests for staging KV and query tensors through the engine's Q ring.

- **Ring mapping** (``RingMapping``): opening the engine's Q rings in the SDK,
  the lock every context shares for them, and the check that vLLM took no
  ring blocks during a transfer.
- **Ring transfer** (``RingSDKTransferWrapper``): KV and query tensors staged
  through free blocks of the ring as raw L1 objects. A fake server keeps each
  rank's chunk objects as bytes and a CPU ring per rank. It answers
  ``get_q_ring`` with the rings' state and moves object bytes between its
  store and the given ring pieces in the order the server's ``sdk_retrieve``
  / ``sdk_store`` use: object groups ascending, chunks in order.
- **Context** (``LMCacheSDKContext._connect_sdk``): building the wrapper from
  the layout the server reports, for dense models only.
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
import math

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.wrapper import ring_mapping, ring_transfer
from lmcache.sdk.wrapper.ring_mapping import (
    RingMapping,
    RingStateError,
    check_ring_unchanged,
)
from lmcache.sdk.wrapper.ring_transfer import (
    RingSDKTransferWrapper,
    SDKKernelGroup,
    SDKObjectGroup,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE

# Ring mapping: a ring block is [MAP_BLOCK, MAP_HIDDEN] per layer.
MAP_BLOCK = 4
MAP_LAYERS = 2
MAP_HIDDEN = 3

# Ring transfer
CHUNK = 8
RING_LAYERS = 2
# A ring block is [4, 8] float32 per layer: 128-byte pieces.
RING_BLOCK = (4, 8)
PIECE_BYTES = 4 * 8 * 4
NUM_RING_BLOCKS = 16


def _done(value: object) -> SimpleNamespace:
    return SimpleNamespace(
        result=lambda timeout=None: value,
        to_device_future=lambda **_kw: SimpleNamespace(
            result=lambda timeout=None: value
        ),
    )


class _Wrapper:
    """IPC wrapper double: rebuilds the ring tensor it was made from."""

    def __init__(self, tensor: torch.Tensor, opened: list[int] | None = None) -> None:
        self._tensor = tensor
        self._opened = opened

    def to_tensor(self) -> torch.Tensor:
        if self._opened is not None:
            self._opened.append(1)
        return self._tensor


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrappers pin work to the ring's device; here the ring is CPU."""
    backend = SimpleNamespace(
        create_event=lambda _d: object(),
        record_event=lambda _e, _s: None,
        export_event=lambda _e, _d: b"event",
    )
    device = SimpleNamespace(
        device=lambda _d: nullcontext(),
        current_stream=lambda: None,
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(ring_transfer, "get_event_ipc_backend", lambda _d: backend)
    monkeypatch.setattr(ring_transfer, "torch_dev", device)
    monkeypatch.setattr(ring_mapping, "torch_dev", device)


# Ring mapping


class MappingServer:
    """Serves ``get_q_ring`` and ``free_lookup_locks`` for ``RingMapping``."""

    def __init__(self, ranks: int = 1, num_blocks: int = 6) -> None:
        self.rings = [
            [torch.zeros(num_blocks, MAP_BLOCK, MAP_HIDDEN) for _ in range(MAP_LAYERS)]
            for _ in range(ranks)
        ]
        self.num_alloc = 1
        # A restarted vLLM registers its rings under new instance IDs.
        self.first_instance = 100
        self.opened: list[int] = []
        self.freed: list[IPCCacheServerKey] = []

    def infos(self) -> list[QRingInfo]:
        return [
            QRingInfo(
                instance_id=self.first_instance + rank,
                worker_id=rank,
                q_caches=[
                    _Wrapper(t, self.opened)  # type: ignore[misc]
                    for t in layers
                ],
                used_blocks=[2, 3],
                num_alloc=self.num_alloc,
            )
            for rank, layers in enumerate(self.rings)
        ]

    def get_q_ring(self, model_name: str, world_size: int):
        return _done(self.infos())

    def free_lookup_locks(self, key: IPCCacheServerKey, _world_size: int):
        self.freed.append(key)
        return _done(None)


def _mapping(server: MappingServer, model_name: str = "m##query") -> RingMapping:
    mapping = RingMapping(
        server,  # type: ignore[arg-type]
        model_name,
        1,
        timeout=1.0,
    )
    mapping.connect()
    return mapping


def test_connect_opens_every_rank_in_rank_order():
    server = MappingServer(ranks=2)

    mapping = _mapping(server)

    assert [ring.info.worker_id for ring in mapping.rings] == [0, 1]
    assert mapping.rings[1].layers[0] is server.rings[1][0]
    assert mapping.rings[0].block_size == MAP_BLOCK


def test_the_same_workers_rings_are_not_reopened_but_a_restarted_ones_are():
    server = MappingServer()
    mapping = _mapping(server)
    opened = len(server.opened)

    mapping.map_rings(server.infos())
    assert len(server.opened) == opened
    server.first_instance += 10
    mapping.map_rings(server.infos())
    assert len(server.opened) == 2 * opened


def test_no_registered_ring_raises():
    with pytest.raises(RingStateError, match="no Q ring"):
        _mapping(MappingServer(ranks=0))


def test_ranks_that_disagree_on_the_layout_raise():
    server = MappingServer(ranks=2)
    server.rings[1] = [
        torch.zeros(6, MAP_BLOCK + 1, MAP_HIDDEN) for _ in range(MAP_LAYERS)
    ]

    with pytest.raises(RingStateError, match="disagree"):
        _mapping(server)


def test_every_mapping_of_a_ring_shares_one_lock():
    server = MappingServer()

    assert _mapping(server).lock is _mapping(server).lock
    assert _mapping(server).lock is not _mapping(server, "other##query").lock


def test_free_lookup_locks_releases_the_unread_range_without_a_worker():
    server = MappingServer()
    key = IPCCacheServerKey.from_token_ids("m", 1, 0, list(range(24)), 0, 24, "req")

    _mapping(server).free_lookup_locks(key, 8, 16)
    _mapping(server).free_lookup_locks(key, 16, 16)  # empty: nothing sent

    (freed,) = server.freed
    assert (freed.start, freed.end, freed.worker_id) == (8, 16, None)
    assert freed.token_bytes == key.token_bytes[: 16 * TOKEN_STRIDE]


def test_an_allocation_or_a_restarted_worker_fails_the_transfer():
    server = MappingServer()
    before = server.infos()

    check_ring_unchanged(before, server.infos(), "tensors")
    server.num_alloc += 1
    with pytest.raises(RingStateError, match="took Q ring blocks"):
        check_ring_unchanged(before, server.infos(), "tensors")
    server.num_alloc -= 1
    server.first_instance += 10
    with pytest.raises(RingStateError, match="took Q ring blocks"):
        check_ring_unchanged(before, server.infos(), "tensors")


# Ring transfer


def _dense_groups() -> list[SDKObjectGroup]:
    """One attention object per chunk: 2 x 2 x 8 x 4 float16, 256 bytes."""
    attention = SDKKernelGroup(0, (2, 2, CHUNK, 4), torch.float16)
    return [SDKObjectGroup(0, (attention,))]


class TransferServer:
    """Serves ``get_q_ring``, ``sdk_retrieve``, ``sdk_store`` and
    ``free_lookup_locks`` over a CPU ring."""

    def __init__(
        self,
        groups: list[SDKObjectGroup],
        num_blocks: int = NUM_RING_BLOCKS,
        ranks: int = 1,
    ) -> None:
        self.groups = {g.object_group_idx: g for g in groups}
        self.rings = [
            [torch.zeros(num_blocks, *RING_BLOCK) for _ in range(RING_LAYERS)]
            for _ in range(ranks)
        ]
        self.used: list[int] = []
        self.num_alloc = 1
        # (rank, token prefix bytes through the chunk, object group) -> bytes
        self.objects: dict[tuple[int, bytes, int], torch.Tensor] = {}
        self.calls: list[tuple[str, int, int, list[int]]] = []
        self.freed: list[tuple[int, int]] = []
        self.alloc_on_call = False
        self.fail = False

    def get_q_ring(self, model_name: str, world_size: int):
        return _done(self.infos())

    def sdk_retrieve(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(self._move("retrieve", key, instance_id, group_ids, piece_ids))

    def sdk_store(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(self._move("store", key, instance_id, group_ids, piece_ids))

    def infos(self) -> list[QRingInfo]:
        return [
            QRingInfo(
                instance_id=100 + rank,
                worker_id=rank,
                q_caches=[_Wrapper(t) for t in layers],  # type: ignore[misc]
                used_blocks=list(self.used),
                num_alloc=self.num_alloc,
            )
            for rank, layers in enumerate(self.rings)
        ]

    def put(self, rank: int, group_id: int, tokens: list[int], end: int, data):
        """Cache one chunk's object for a rank (as an engine store would)."""
        key = IPCCacheServerKey.from_token_ids("m", 1, rank, tokens, 0, end, "x")
        flat = data.contiguous().reshape(-1).view(torch.uint8)
        self.objects[(rank, key.token_bytes[: end * TOKEN_STRIDE], group_id)] = flat

    def _piece(self, rank: int, piece_id: int) -> torch.Tensor:
        block, layer = divmod(piece_id, RING_LAYERS)
        return self.rings[rank][layer][block].reshape(-1).view(torch.uint8)

    def _plan(self, key, group_ids):
        """(object key, group, pieces per object) in the server's order."""
        assert set(group_ids) <= set(self.groups)
        plan = []
        for group_id in sorted(group_ids):
            group = self.groups[group_id]
            for start in range(key.start, key.end, CHUNK):
                prefix = key.token_bytes[: (start + CHUNK) * TOKEN_STRIDE]
                count = math.ceil(group.num_bytes / PIECE_BYTES)
                plan.append(((key.worker_id, prefix, group_id), group, count))
        return plan

    def _move(self, name, key, instance_id, group_ids, piece_ids):
        rank = instance_id - 100
        assert key.worker_id == rank
        assert not {p // RING_LAYERS for p in piece_ids} & set(self.used)
        self.calls.append((name, key.start, key.end, list(group_ids)))
        cursor = 0
        for obj_key, group, count in self._plan(key, group_ids):
            ids = piece_ids[cursor : cursor + count]
            cursor += count
            if name == "store":
                pieces = [self._piece(rank, p) for p in ids]
                self.objects[obj_key] = torch.cat(pieces)[: group.num_bytes].clone()
            else:
                data = self.objects[obj_key]
                for index, piece_id in enumerate(ids):
                    segment = data[index * PIECE_BYTES : (index + 1) * PIECE_BYTES]
                    self._piece(rank, piece_id)[: segment.numel()] = segment
        assert cursor == len(piece_ids)
        if self.alloc_on_call:
            self.num_alloc += 1
        return not self.fail

    def free_lookup_locks(self, key: IPCCacheServerKey, _world_size: int):
        self.freed.append((key.start, key.end))
        return _done(None)


def _wrapper(server: TransferServer, writable: bool = True) -> RingSDKTransferWrapper:
    ring = RingMapping(
        server,  # type: ignore[arg-type]
        "m##query",
        len(server.rings),
        timeout=1.0,
    )
    ring.connect()
    return RingSDKTransferWrapper(
        ring, list(server.groups.values()), CHUNK, timeout=1.0, writable=writable
    )


def _key(num_tokens: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(num_tokens)), 0, num_tokens, "req"
    )


def _kv(num_tokens: int) -> torch.Tensor:
    return torch.randn(2, 2, num_tokens, 4).to(torch.float16)


def test_dense_round_trip():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    kv = _kv(3 * CHUNK)

    assert wrapper.store(_key(3 * CHUNK), 0, kv)
    back = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, kv)


def test_transfers_avoid_the_blocks_vllm_holds():
    """Pieces come only from blocks outside the reported used blocks."""
    server = TransferServer(_dense_groups())
    server.used = [0, 1, 5]
    wrapper = _wrapper(server)
    kv = _kv(2 * CHUNK)

    assert wrapper.store(_key(2 * CHUNK), 0, kv)
    back = wrapper.retrieve(_key(2 * CHUNK), 0)

    assert back is not None and torch.equal(back, kv)


def test_a_range_larger_than_the_free_blocks_is_batched():
    """Each chunk's object takes 2 pieces (one block): with 2 free blocks, 2
    chunks fit per batch."""
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    kv = _kv(3 * CHUNK)
    wrapper.store(_key(3 * CHUNK), 0, kv)
    server.calls.clear()
    server.used = [b for b in range(NUM_RING_BLOCKS) if b not in (3, 7)]

    back = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, kv)
    assert server.calls == [
        ("retrieve", 0, 2 * CHUNK, [0]),
        ("retrieve", 2 * CHUNK, 3 * CHUNK, [0]),
    ]


def test_too_few_free_blocks_raises_and_releases_the_locks():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    wrapper.store(_key(CHUNK), 0, _kv(CHUNK))
    server.used = list(range(NUM_RING_BLOCKS))

    with pytest.raises(RingStateError, match="ring_blocks"):
        wrapper.retrieve(_key(CHUNK), 0)
    assert server.freed == [(0, CHUNK)]


def test_a_vllm_allocation_during_the_retrieve_raises():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    wrapper.store(_key(CHUNK), 0, _kv(CHUNK))
    server.alloc_on_call = True

    with pytest.raises(RingStateError, match="took Q ring blocks"):
        wrapper.retrieve(_key(CHUNK), 0)


def test_a_failed_batch_returns_none_and_releases_the_rest():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    wrapper.store(_key(2 * CHUNK), 0, _kv(2 * CHUNK))
    server.used = list(range(1, NUM_RING_BLOCKS))  # one chunk per batch
    server.fail = True

    assert wrapper.retrieve(_key(2 * CHUNK), 0) is None
    assert server.freed == [(CHUNK, 2 * CHUNK)]


@pytest.mark.parametrize("num_tokens", [CHUNK + 1, CHUNK // 2])
def test_store_rejects_tensors_that_do_not_match_the_layout(num_tokens):
    wrapper = _wrapper(TransferServer(_dense_groups()))

    with pytest.raises(ValueError, match="does not match"):
        wrapper.store(_key(num_tokens), 0, _kv(num_tokens))


def test_kv_stores_go_through_exactly_one_ring():
    wrapper = _wrapper(TransferServer(_dense_groups(), ranks=2))

    with pytest.raises(RingStateError, match="exactly one Q ring"):
        wrapper.store(_key(CHUNK), 0, _kv(CHUNK))


def test_the_query_and_kv_contexts_share_the_ring_lock():
    server = TransferServer(_dense_groups())

    assert _wrapper(server, writable=False)._ring.lock is _wrapper(server)._ring.lock


# Query tensors: read-only and served by every rank.

Q_WIDTH = 6


def _query_groups() -> list[SDKObjectGroup]:
    query = SDKKernelGroup(0, (1, 2, CHUNK, Q_WIDTH), torch.float32)
    return [SDKObjectGroup(0, (query,))]


def _put_query(server: TransferServer, num_chunks: int, rank: int = 0) -> torch.Tensor:
    """Cache each chunk's query rows for a rank; return them, [1, L, T, W]."""
    tokens = list(range(num_chunks * CHUNK))
    chunks = [torch.randn(1, 2, CHUNK, Q_WIDTH) for _ in range(num_chunks)]
    for index, chunk in enumerate(chunks):
        server.put(rank, 0, tokens, (index + 1) * CHUNK, chunk)
    return torch.cat(chunks, dim=2)


def test_query_rows_come_back_in_token_order():
    server = TransferServer(_query_groups())
    rows = _put_query(server, 3)

    back = _wrapper(server, writable=False).retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, rows)


def test_ranks_retrieve_their_own_objects_and_concatenate_heads():
    server = TransferServer(_query_groups(), ranks=2)
    rank0, rank1 = _put_query(server, 2, rank=0), _put_query(server, 2, rank=1)

    back = _wrapper(server, writable=False).retrieve(_key(2 * CHUNK), 0)

    assert back is not None
    assert torch.equal(back, torch.cat([rank0, rank1], dim=-1))
    assert [call[1:3] for call in server.calls] == [(0, 2 * CHUNK)] * 2


def test_query_tensors_are_read_only():
    wrapper = _wrapper(TransferServer(_query_groups()), writable=False)

    with pytest.raises(NotImplementedError):
        wrapper.store(_key(CHUNK), 0, torch.zeros(1, 2, CHUNK, Q_WIDTH))


# Context: building the wrapper from the engine's /status.


def _bare_context(kind_name: str = "KV", world_size: int = 1):
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
    from lmcache.sdk.context import LMCacheSDKContext

    ctx = object.__new__(LMCacheSDKContext)
    ctx._kind = LMCacheSDKCacheKind[kind_name]
    ctx._model_name = ctx._kind.server_model_name("m")
    ctx._world_size = world_size
    ctx._chunk_size = CHUNK
    ctx._mq_timeout = 1.0
    groups = _dense_groups() if kind_name == "KV" else _query_groups()
    ctx._req_client = TransferServer(groups, ranks=world_size)  # type: ignore[assignment]
    ctx._transfer_ctx = None
    return ctx


def _dense_object_groups() -> list[dict]:
    """The dense model's entry in qstore's ``object_layouts`` report."""
    return [
        {
            "object_group_idx": 0,
            "kind": "attention",
            "num_chunks_in_sw": -1,
            "chunk_object_shapes": [[2, 2, CHUNK, 4]],
            "chunk_object_dtypes": ["torch.float16"],
        }
    ]


def _connect(ctx, object_groups, **kernel_group) -> None:
    """Connect as /status would: object layouts from qstore's report, the
    kernel group from the registration's layout."""
    ctx._object_layouts = [
        {
            "model_name": ctx._model_name,
            "world_size": ctx._world_size,
            "object_groups": object_groups,
        }
    ]
    ctx._connect_sdk(
        {
            "kernel_group_idx": 0,
            "object_group_idx": 0,
            "tokens_per_block": CHUNK,
            "slots_per_block": CHUNK,
            **kernel_group,
        }
    )


def test_the_context_builds_the_sdk_wrapper_from_the_reported_object_layout():
    ctx = _bare_context()

    _connect(ctx, _dense_object_groups())

    wrapper = ctx._transfer_ctx
    assert isinstance(wrapper, RingSDKTransferWrapper)
    assert wrapper._writable
    (group,) = wrapper._object_groups
    assert group.num_bytes == 2 * 2 * CHUNK * 4 * 2


def test_the_query_context_reads_the_ring_read_only_at_any_tp():
    ctx = _bare_context("QUERY", world_size=2)
    groups = _dense_object_groups()
    groups[0].update(
        chunk_object_shapes=[[1, 2, CHUNK, Q_WIDTH]],
        chunk_object_dtypes=["torch.float32"],
    )

    _connect(ctx, groups)

    assert not ctx._transfer_ctx._writable


@pytest.mark.parametrize(
    ("world_size", "mutate", "kernel_group", "match"),
    [
        (2, lambda groups: None, {}, "KV at TP 1 only"),
        (1, lambda groups: groups.clear(), {}, "no object layout"),
        (1, lambda groups: groups[0].update(num_chunks_in_sw=2), {}, "full attention"),
        (1, lambda groups: groups[0].update(kind="recurrent"), {}, "full attention"),
        (1, lambda groups: groups.append(dict(groups[0])), {}, "dense models"),
        (1, lambda groups: None, {"slots_per_block": 4}, "compressed"),
        (
            1,
            lambda groups: None,
            {"tokens_per_block": 3, "slots_per_block": 3},
            "multiple",
        ),
    ],
)
def test_the_context_rejects_layouts_the_sdk_cannot_serve(
    world_size, mutate, kernel_group, match
):
    # First Party
    from lmcache.sdk.context import LMCacheSDKError

    ctx = _bare_context(world_size=world_size)
    groups = _dense_object_groups()
    mutate(groups)

    with pytest.raises(LMCacheSDKError, match=match):
        _connect(ctx, groups, **kernel_group)
