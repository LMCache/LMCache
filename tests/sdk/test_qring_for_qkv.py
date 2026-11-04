# SPDX-License-Identifier: Apache-2.0
"""Unit tests for staging KV and query tensors through the engine's Q ring.

- **Sub-paged codec**: vLLM re-pages a hybrid model's attention block at the
  backend's kernel page size: a registered block of ``block`` tokens, viewed
  as ``[2, block, W]``, holds ``block // kernel`` kernel pages of
  ``[K(kernel, W) | V(kernel, W)]``.
- **Ring mapping** (``RingMapping``): opening the engine's Q rings in the SDK,
  the lock every context shares for them, and the check that vLLM took no
  ring blocks during a transfer.
- **Ring transfer** (``RingSDKTransferWrapper``): KV and query tensors staged
  through free blocks of the ring as raw L1 objects. A fake server keeps each
  rank's chunk objects as bytes and a CPU ring per rank. It answers
  ``get_q_ring`` with the rings' state and moves object bytes between its
  store and the given ring pieces in the order the server's ``sdk_retrieve``
  / ``sdk_store`` use: object groups ascending, chunks in order, a group read
  within its window, a recurrent group only at the range's last chunk.
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
import math

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk import context as sdk_context
from lmcache.sdk.wrapper import ring_mapping, ring_transfer
from lmcache.sdk.wrapper.ring_mapping import (
    RingMapping,
    RingStateError,
    check_ring_unchanged,
)
from lmcache.sdk.wrapper.ring_transfer import (
    RecurrentState,
    RingSDKTransferWrapper,
    SDKKernelGroup,
    SDKObjectGroup,
    kv_to_subpaged,
    subpaged_to_kv,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE

# Sub-paged codec
CODEC_BLOCK, CODEC_KERNEL, CODEC_WIDTH = 8, 4, 3

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


# Sub-paged codec


def test_decode_reads_k_and_v_from_alternating_kernel_pages():
    """One layer, one block: the block's bytes are K0 V0 K1 V1 (kernel pages)."""
    k = torch.arange(CODEC_BLOCK * CODEC_WIDTH, dtype=torch.float32).view(
        CODEC_BLOCK, CODEC_WIDTH
    )
    v = -k
    raw = torch.cat(
        [k[:CODEC_KERNEL], v[:CODEC_KERNEL], k[CODEC_KERNEL:], v[CODEC_KERNEL:]]
    )
    # The registered view splits the block's bytes into two "planes".
    pages = raw.view(2, 1, CODEC_BLOCK, CODEC_WIDTH)

    kv = subpaged_to_kv(pages, CODEC_BLOCK, CODEC_KERNEL)

    assert torch.equal(kv[0, 0], k)
    assert torch.equal(kv[1, 0], v)


def test_encode_inverts_decode_across_layers_and_blocks():
    kv = torch.randn(2, 3, 4 * CODEC_BLOCK, CODEC_WIDTH)

    pages = kv_to_subpaged(kv, CODEC_BLOCK, CODEC_KERNEL)

    assert not torch.equal(pages, kv)
    assert torch.equal(subpaged_to_kv(pages, CODEC_BLOCK, CODEC_KERNEL), kv)


def test_a_block_of_one_kernel_page_is_unchanged():
    kv = torch.randn(2, 2, 2 * CODEC_BLOCK, CODEC_WIDTH)

    assert torch.equal(kv_to_subpaged(kv, CODEC_BLOCK, CODEC_BLOCK), kv)


# Ring mapping


class MappingServer:
    """Serves ``get_q_ring`` and ``free_lookup_locks`` for ``RingMapping``."""

    def __init__(self, ranks: int = 1, num_blocks: int = 6) -> None:
        self.rings = [
            [torch.zeros(num_blocks, MAP_BLOCK, MAP_HIDDEN) for _ in range(MAP_LAYERS)]
            for _ in range(ranks)
        ]
        self.num_alloc = 1
        self.generation = 1
        self.opened: list[int] = []
        self.freed: list[IPCCacheServerKey] = []

    def infos(self) -> list[QRingInfo]:
        return [
            QRingInfo(
                instance_id=100 + rank,
                worker_id=rank,
                q_caches=[
                    _Wrapper(t, self.opened)  # type: ignore[misc]
                    for t in layers
                ],
                free_blocks=[0, 1],
                num_alloc=self.num_alloc,
                generation=self.generation,
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


def test_the_same_registration_is_not_reopened_but_a_new_generation_is():
    server = MappingServer()
    mapping = _mapping(server)
    opened = len(server.opened)

    mapping.map_rings(server.infos())
    assert len(server.opened) == opened
    server.generation += 1
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


def test_an_allocation_or_generation_change_fails_the_transfer():
    server = MappingServer()
    before = server.infos()

    check_ring_unchanged(before, server.infos(), "tensors")
    server.num_alloc += 1
    with pytest.raises(RingStateError, match="took Q ring blocks"):
        check_ring_unchanged(before, server.infos(), "tensors")
    server.num_alloc -= 1
    server.generation += 1
    with pytest.raises(RingStateError, match="took Q ring blocks"):
        check_ring_unchanged(before, server.infos(), "tensors")


# Ring transfer


def _kg(
    idx: int,
    recurrent: bool,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    kernel_block: int = CHUNK,
) -> SDKKernelGroup:
    return SDKKernelGroup(idx, recurrent, CHUNK, kernel_block, shape, dtype)


def _dense_groups() -> list[SDKObjectGroup]:
    attention = _kg(0, False, (2, 2, CHUNK, 4), torch.float16)
    return [SDKObjectGroup(0, False, (attention,))]


def _hybrid_groups() -> list[SDKObjectGroup]:
    """A recurrent object group (two kernel groups) and a sub-paged attention one."""
    recurrent = (
        _kg(0, True, (1, 2, 1, 16), torch.float32),
        _kg(1, True, (1, 1, 1, 16), torch.float32),
    )
    attention = _kg(2, False, (2, 2, CHUNK, 4), torch.bfloat16, kernel_block=4)
    return [
        SDKObjectGroup(0, True, recurrent),
        SDKObjectGroup(1, False, (attention,)),
    ]


class TransferServer:
    """Serves ``get_q_ring``, ``sdk_retrieve``, ``sdk_store`` and
    ``free_lookup_locks`` over a CPU ring."""

    def __init__(
        self, groups: list[SDKObjectGroup], num_blocks: int = 16, ranks: int = 1
    ) -> None:
        self.groups = {g.object_group_idx: g for g in groups}
        self.rings = [
            [torch.zeros(num_blocks, *RING_BLOCK) for _ in range(RING_LAYERS)]
            for _ in range(ranks)
        ]
        self.free = list(range(num_blocks))
        self.num_alloc = 1
        self.generation = 1
        # (rank, token prefix bytes through the chunk, object group) -> bytes
        self.objects: dict[tuple[int, bytes, int], torch.Tensor] = {}
        self.calls: list[tuple[str, int, int, list[int]]] = []
        self.freed: list[tuple[int, int]] = []
        self.alloc_on_call = False
        self.fail = False

    def get_q_ring(self, model_name: str, world_size: int):
        return _done(self.infos())

    def sdk_retrieve(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(
            self._move("retrieve", key, instance_id, group_ids, piece_ids, False)
        )

    def sdk_store(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(self._move("store", key, instance_id, group_ids, piece_ids, True))

    def infos(self) -> list[QRingInfo]:
        return [
            QRingInfo(
                instance_id=100 + rank,
                worker_id=rank,
                q_caches=[_Wrapper(t) for t in layers],  # type: ignore[misc]
                free_blocks=list(self.free),
                num_alloc=self.num_alloc,
                generation=self.generation,
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

    def _plan(self, key, group_ids, store):
        """(object key, group, pieces per object) in the server's order."""
        assert set(group_ids) <= set(self.groups)
        plan = []
        for group_id in sorted(group_ids):
            group = self.groups[group_id]
            chunks = list(range(key.start, key.end, CHUNK))
            if group.recurrent:
                chunks = chunks[-1:]
            elif not store and group.window_chunks >= 0:
                chunks = chunks[-group.window_chunks :]
            for start in chunks:
                prefix = key.token_bytes[: (start + CHUNK) * TOKEN_STRIDE]
                count = math.ceil(group.num_bytes / PIECE_BYTES)
                plan.append(((key.worker_id, prefix, group_id), group, count))
        return plan

    def _move(self, name, key, instance_id, group_ids, piece_ids, store):
        rank = instance_id - 100
        assert key.worker_id == rank
        assert {p // RING_LAYERS for p in piece_ids} <= set(self.free)
        self.calls.append((name, key.start, key.end, list(group_ids)))
        cursor = 0
        for obj_key, group, count in self._plan(key, group_ids, store):
            ids = piece_ids[cursor : cursor + count]
            cursor += count
            if store:
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


def _key(num_tokens: int, start: int = 0) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(num_tokens)), start, num_tokens, "req"
    )


def _kv(num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(2, 2, num_tokens, 4).to(dtype)


def _state() -> RecurrentState:
    return RecurrentState(tuple(torch.randn(1, 1, 16) for _ in range(3)))


def test_dense_round_trip():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    kv = _kv(3 * CHUNK, torch.float16)

    assert wrapper.store(_key(3 * CHUNK), 0, kv)
    back = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and back[1] is None
    assert torch.equal(back[0], kv)
    assert not wrapper.has_recurrent_state


def test_hybrid_round_trip_carries_the_state():
    server = TransferServer(_hybrid_groups())
    wrapper = _wrapper(server)
    kv, state = _kv(3 * CHUNK, torch.bfloat16), _state()

    assert wrapper.store(_key(3 * CHUNK), 0, kv, state)
    back = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None
    back_kv, back_state = back
    assert torch.equal(back_kv, kv)
    assert back_state is not None
    assert all(
        torch.equal(a, b) for a, b in zip(back_state.pages, state.pages, strict=True)
    )
    assert wrapper.has_recurrent_state


def test_the_stored_object_is_the_engines_sub_paged_layout():
    """The SDK hands true K/V; the bytes in L1 are vLLM's sub-paged rows."""
    server = TransferServer(_hybrid_groups())
    kv, state = _kv(CHUNK, torch.bfloat16), _state()

    _wrapper(server).store(_key(CHUNK), 0, kv, state)

    key = _key(CHUNK)
    stored = server.objects[(0, key.token_bytes[: CHUNK * TOKEN_STRIDE], 1)]
    expected = kv_to_subpaged(kv, CHUNK, 4).contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(stored, expected)


def test_only_the_last_chunk_holds_the_recurrent_state():
    server = TransferServer(_hybrid_groups())

    _wrapper(server).store(_key(3 * CHUNK), 0, _kv(3 * CHUNK, torch.bfloat16), _state())

    stored = {
        (len(prefix) // TOKEN_STRIDE, group) for _, prefix, group in server.objects
    }
    assert stored == {(8, 1), (16, 1), (24, 1), (24, 0)}


def test_a_range_larger_than_the_free_blocks_is_batched():
    """Each attention chunk takes 2 pieces (one block), the state 1 piece: with
    3 free blocks, 2 chunks fit per batch; the state goes with the last."""
    server = TransferServer(_hybrid_groups())
    wrapper = _wrapper(server)
    kv, state = _kv(3 * CHUNK, torch.bfloat16), _state()
    wrapper.store(_key(3 * CHUNK), 0, kv, state)
    server.calls.clear()
    server.free = [3, 7, 11]

    back = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back[0], kv)
    assert server.calls == [
        ("retrieve", 0, 2 * CHUNK, [1]),
        ("retrieve", 2 * CHUNK, 3 * CHUNK, [0, 1]),
    ]


def test_too_few_free_blocks_raises_and_releases_the_locks():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    wrapper.store(_key(CHUNK), 0, _kv(CHUNK, torch.float16))
    server.free = []

    with pytest.raises(RingStateError, match="ring_blocks"):
        wrapper.retrieve(_key(CHUNK), 0)
    assert server.freed == [(0, CHUNK)]


def test_a_vllm_allocation_during_the_retrieve_raises():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    wrapper.store(_key(CHUNK), 0, _kv(CHUNK, torch.float16))
    server.alloc_on_call = True

    with pytest.raises(RingStateError, match="took Q ring blocks"):
        wrapper.retrieve(_key(CHUNK), 0)


def test_a_failed_batch_returns_none_and_releases_the_rest():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    wrapper.store(_key(2 * CHUNK), 0, _kv(2 * CHUNK, torch.float16))
    server.free = [0]  # one chunk per batch
    server.fail = True

    assert wrapper.retrieve(_key(2 * CHUNK), 0) is None
    assert server.freed == [(CHUNK, 2 * CHUNK)]


def test_retrieve_from_an_offset_releases_the_prefix_locks():
    server = TransferServer(_dense_groups())
    wrapper = _wrapper(server)
    kv = _kv(2 * CHUNK, torch.float16)
    wrapper.store(_key(2 * CHUNK), 0, kv)

    back = wrapper.retrieve(_key(2 * CHUNK, start=CHUNK), 0)

    assert back is not None and torch.equal(back[0], kv[:, :, CHUNK:])
    assert server.freed == [(0, CHUNK)]


@pytest.mark.parametrize(
    ("kv_tokens", "state", "match"),
    [
        (CHUNK + 1, _state(), "does not match"),
        (CHUNK, None, "one page per recurrent layer"),
        (CHUNK, RecurrentState((torch.zeros(1, 1, 16),)), "one page per recurrent"),
    ],
)
def test_store_rejects_inputs_that_do_not_match_the_layout(kv_tokens, state, match):
    wrapper = _wrapper(TransferServer(_hybrid_groups()))
    key = _key(kv_tokens)

    with pytest.raises(ValueError, match=match):
        wrapper.store(key, 0, _kv(kv_tokens, torch.bfloat16), state)


def test_the_query_and_kv_contexts_share_the_ring_lock():
    server = TransferServer(_dense_groups())

    assert _wrapper(server, writable=False)._ring.lock is _wrapper(server)._ring.lock


# The context's SDK path: building the wrapper from the engine's /status.


def _bare_context(world_size: int = 1):
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
    from lmcache.sdk.context import LMCacheSDKContext

    ctx = object.__new__(LMCacheSDKContext)
    ctx._kind = LMCacheSDKCacheKind.KV
    ctx._model_name = "m"
    ctx._world_size = world_size
    ctx._chunk_size = CHUNK
    ctx._mq_timeout = 1.0
    ctx._mp_conf = {
        "separate_object_groups": True,
        "runtime_plugin_config": {"extra_config": {"sdk.kernel_block_size": 4}},
    }
    ctx._req_client = TransferServer(_hybrid_groups())  # type: ignore[assignment]
    ctx._transfer_ctx = None
    return ctx


def _hybrid_object_groups() -> list[dict]:
    """The hybrid model's entry in qstore's ``object_layouts`` report."""
    return [
        {
            "object_group_idx": 0,
            "kind": "recurrent",
            "num_chunks_in_sw": 1,
            "chunk_object_shapes": [[1, 2, 1, 16], [1, 1, 1, 16]],
            "chunk_object_dtypes": ["torch.float32", "torch.float32"],
        },
        {
            "object_group_idx": 1,
            "kind": "attention",
            "num_chunks_in_sw": -1,
            "chunk_object_shapes": [[2, 2, CHUNK, 4]],
            "chunk_object_dtypes": ["torch.bfloat16"],
        },
    ]


# The registered attention page: one synthetic head of 2 kv heads x 2 dims,
# as vLLM's hybrid allocator registers it; the model's heads come from its
# Hugging Face config (faked by ``model_heads``).
SUBPAGED_ATTENTION = {
    "engine_kv_shape": "NL x [NB, 2, BS, NH, HS]",
    "engine_kv_concrete_shape": f"2 x [9, 2, {CHUNK}, 1, 4]",
}


@pytest.fixture
def model_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk_context, "_model_kv_heads", lambda _name, _ws: (2, 2))


def _connect(ctx, object_groups, owners: list[int]) -> None:
    """Connect as /status would: object layouts from qstore's report, and the
    kernel groups (membership, block sizes, registered heads) from the
    registration's layout."""
    ctx._object_layouts = [
        {
            "model_name": ctx._model_name,
            "world_size": ctx._world_size,
            "object_groups": object_groups,
        }
    ]
    ctx._connect_sdk(
        [
            {
                "kernel_group_idx": index,
                "object_group_idx": owner,
                "tokens_per_block": CHUNK,
                "slots_per_block": CHUNK,
                **SUBPAGED_ATTENTION,
            }
            for index, owner in enumerate(owners)
        ]
    )


def test_the_context_builds_the_sdk_wrapper_from_the_reported_object_layout(
    model_heads,
):
    ctx = _bare_context()

    _connect(ctx, _hybrid_object_groups(), owners=[0, 0, 1])

    wrapper = ctx._transfer_ctx
    assert isinstance(wrapper, RingSDKTransferWrapper)
    assert wrapper.has_recurrent_state
    assert [og.num_bytes for og in wrapper._object_groups] == [192, 256]
    assert wrapper._attention[0].sub_paged


@pytest.mark.parametrize(
    ("world_size", "mutate", "owners", "match"),
    [
        (2, lambda groups: None, [0, 0, 1], "KV at TP 1 only"),
        (1, lambda groups: groups.clear(), [0, 0, 1], "no object layout"),
        (
            1,
            lambda groups: groups[1].update(num_chunks_in_sw=2),
            [0, 0, 1],
            "cannot be served",
        ),
        (1, lambda groups: None, [0, 1, 1], "packs 2 tensors"),
    ],
)
def test_the_context_rejects_layouts_the_sdk_cannot_serve(
    model_heads, world_size, mutate, owners, match
):
    # First Party
    from lmcache.sdk.context import LMCacheSDKError

    ctx = _bare_context(world_size)
    groups = _hybrid_object_groups()
    mutate(groups)

    with pytest.raises(LMCacheSDKError, match=match):
        _connect(ctx, groups, owners)


# Query tensors: read-only, windowed, and served by every rank.

Q_ROWS = 4  # rows each chunk keeps: a sub-chunk window
Q_WIDTH = 6


def _query_groups(window_chunks: int = -1) -> list[SDKObjectGroup]:
    query = _kg(0, False, (1, 2, Q_ROWS, Q_WIDTH), torch.float32)
    return [SDKObjectGroup(0, False, (query,), window_chunks)]


def _put_query(server: TransferServer, num_chunks: int, rank: int = 0) -> torch.Tensor:
    """Cache each chunk's kept query rows for a rank; return them, [1, L, T, W]."""
    tokens = list(range(num_chunks * CHUNK))
    chunks = [torch.randn(1, 2, Q_ROWS, Q_WIDTH) for _ in range(num_chunks)]
    for index, chunk in enumerate(chunks):
        server.put(rank, 0, tokens, (index + 1) * CHUNK, chunk)
    return torch.cat(chunks, dim=2)


def test_query_rows_come_back_as_each_chunk_keeps_them():
    server = TransferServer(_query_groups())
    rows = _put_query(server, 3)

    back = _wrapper(server, writable=False).retrieve(_key(3 * CHUNK), 0)

    assert back is not None and back[1] is None
    assert torch.equal(back[0], rows)


def test_a_windowed_group_reads_only_the_trailing_chunks():
    """As the server does: each call reads the group's last window chunks."""
    server = TransferServer(_query_groups(window_chunks=2))
    rows = _put_query(server, 3)

    back = _wrapper(server, writable=False).retrieve(_key(3 * CHUNK), 0)

    assert back is not None
    assert torch.equal(back[0], rows[:, :, Q_ROWS:])


def test_ranks_retrieve_their_own_objects_and_concatenate_heads():
    server = TransferServer(_query_groups(), ranks=2)
    rank0, rank1 = _put_query(server, 2, rank=0), _put_query(server, 2, rank=1)

    back = _wrapper(server, writable=False).retrieve(_key(2 * CHUNK), 0)

    assert back is not None
    assert torch.equal(back[0], torch.cat([rank0, rank1], dim=-1))
    assert [call[1:3] for call in server.calls] == [(0, 2 * CHUNK)] * 2


def test_query_tensors_are_read_only():
    wrapper = _wrapper(TransferServer(_query_groups()), writable=False)

    with pytest.raises(NotImplementedError):
        wrapper.store(_key(CHUNK), 0, torch.zeros(1, 2, CHUNK, Q_WIDTH))


def test_recurrent_state_is_not_split_over_ranks():
    server = TransferServer(_hybrid_groups(), ranks=2)

    with pytest.raises(RingStateError, match="ranks"):
        _wrapper(server).retrieve(_key(CHUNK), 0)
    assert server.freed == [(0, CHUNK)]


def _query_object_groups(window: int = 1) -> list[dict]:
    return [
        {
            "object_group_idx": 0,
            "kind": "attention",
            "num_chunks_in_sw": window,
            "chunk_object_shapes": [[1, 2, Q_ROWS, Q_WIDTH]],
            "chunk_object_dtypes": ["torch.float32"],
        }
    ]


def test_the_query_context_reads_the_ring_read_only_with_its_window():
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind

    ctx = _bare_context(world_size=2)
    ctx._kind = LMCacheSDKCacheKind.QUERY
    ctx._model_name = "m##query"
    ctx._req_client = TransferServer(_query_groups(), ranks=2)  # type: ignore[assignment]

    _connect(ctx, _query_object_groups(1), owners=[0])

    wrapper = ctx._transfer_ctx
    assert isinstance(wrapper, RingSDKTransferWrapper)
    assert not wrapper._writable
    assert wrapper._object_groups[0].window_chunks == 1
    # windowed_range now reads the layout's window: the last chunk, Q_ROWS each.
    assert (ctx._window_chunks, ctx._rows_per_chunk) == (1, Q_ROWS)
    assert ctx.windowed_range(3 * CHUNK) == (2 * CHUNK, Q_ROWS)


def test_the_query_context_rejects_recurrent_groups():
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
    from lmcache.sdk.context import LMCacheSDKError

    ctx = _bare_context()
    ctx._kind = LMCacheSDKCacheKind.QUERY
    groups = _query_object_groups()
    groups[0]["kind"] = "recurrent"

    with pytest.raises(LMCacheSDKError, match="cannot be served"):
        _connect(ctx, groups, owners=[0])
