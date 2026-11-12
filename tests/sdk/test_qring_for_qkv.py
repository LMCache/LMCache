# SPDX-License-Identifier: Apache-2.0
"""Unit tests for staging KV and query tensors through the engine's Q ring.

A fake server keeps each chunk's object as bytes and a GPU ring. It answers
``get_q_ring`` with the ring's state and moves object bytes between its store
and the given ring pieces in the order the server's ``sdk_retrieve`` /
``sdk_store`` use: chunks in order.

Reservations are shared by every mapping of the process and keyed by the
ring's instance ID, so every fake ring takes a fresh instance ID.
"""

# Standard
from types import SimpleNamespace
from typing import Any
import itertools
import math
import threading
import time

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
    RecurrentState,
    RingSDKTransferWrapper,
    SDKObjectLayout,
)
from lmcache.v1.common.token_codec import TOKEN_STRIDE
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo

CHUNK = 8
RING_LAYERS = 2
# A ring block is [4, 8] float32 per layer: 128-byte pieces.
RING_BLOCK = (4, 8)
PIECE_BYTES = 4 * 8 * 4
NUM_RING_BLOCKS = 16
Q_WIDTH = 6

_INSTANCE_IDS = itertools.count(1000)

pytestmark = pytest.mark.skipif(
    not ring_transfer._COPY_ENGINE, reason="needs the copy engine"
)


def _done(value: object) -> SimpleNamespace:
    return SimpleNamespace(
        result=lambda timeout=None: value,
        to_device_future=lambda **_kw: SimpleNamespace(
            result=lambda timeout=None: value
        ),
    )


@pytest.fixture(autouse=True)
def fake_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake server needs no CUDA IPC events."""
    backend = SimpleNamespace(
        create_event=lambda _d: object(),
        record_event=lambda _e, _s: None,
        export_event=lambda _e, _d: b"event",
    )
    monkeypatch.setattr(ring_transfer, "get_event_ipc_backend", lambda _d: backend)


def _kv_layout() -> SDKObjectLayout:
    """One attention object per chunk: 2 x 2 x 8 x 4 float16, 256 bytes."""
    return SDKObjectLayout(0, (2, 2, CHUNK, 4), torch.float16)


def _query_layout(rows: int = CHUNK) -> SDKObjectLayout:
    return SDKObjectLayout(0, (1, 2, rows, Q_WIDTH), torch.float32)


class FakeServer:
    """Serves ``get_q_ring``, ``sdk_retrieve``, ``sdk_store`` and
    ``free_lookup_locks`` over one GPU ring. With ``state_bytes``, object
    group 1 holds a hybrid model's recurrent state: one object, for the last
    chunk of a one-chunk range."""

    def __init__(
        self, layout: SDKObjectLayout | None = None, state_bytes: int = 0
    ) -> None:
        self.layout = layout or _kv_layout()
        self.state_bytes = state_bytes
        self.instance_id = next(_INSTANCE_IDS)
        self.ring = [
            torch.zeros(NUM_RING_BLOCKS, *RING_BLOCK).cuda() for _ in range(RING_LAYERS)
        ]
        self.used: list[int] = []
        self.num_alloc = 1
        # Token prefix bytes through the chunk (and "state" for the
        # recurrent object) -> the object's bytes
        self.objects: dict[Any, torch.Tensor] = {}
        self.calls: list[tuple[int, int]] = []
        self.groups: list[list[int]] = []
        self.freed: list[IPCCacheServerKey] = []
        self.alloc_on_call = False
        self.fail = False

    def info(self) -> QRingInfo:
        return QRingInfo(
            instance_id=self.instance_id,
            worker_id=0,
            q_caches=[
                SimpleNamespace(to_tensor=lambda t=t: t)  # type: ignore[misc]
                for t in self.ring
            ],
            used_blocks=list(self.used),
            num_alloc=self.num_alloc,
        )

    def get_q_ring(self, model_name: str, world_size: int):
        return _done([self.info()])

    def sdk_retrieve(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(self._move(False, key, instance_id, group_ids, piece_ids))

    def sdk_store(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(self._move(True, key, instance_id, group_ids, piece_ids))

    def free_lookup_locks(self, key: IPCCacheServerKey, _world_size: int):
        self.freed.append(key)
        return _done(None)

    def _piece(self, piece_id: int) -> torch.Tensor:
        block, layer = divmod(piece_id, RING_LAYERS)
        return self.ring[layer][block].reshape(-1).view(torch.uint8)

    def _move(self, store, key, instance_id, group_ids, piece_ids):
        assert instance_id == self.instance_id
        assert not {p // RING_LAYERS for p in piece_ids} & set(self.used)
        self.calls.append((key.start, key.end))
        self.groups.append(list(group_ids))
        count = math.ceil(self.layout.num_bytes / PIECE_BYTES)
        for index, start in enumerate(range(key.start, key.end, CHUNK)):
            prefix = key.token_bytes[: (start + CHUNK) * TOKEN_STRIDE]
            ids = piece_ids[index * count : (index + 1) * count]
            self._copy(store, prefix, ids, self.layout.num_bytes)
        if len(group_ids) > 1:  # the recurrent object, after the last chunk
            ids = piece_ids[(index + 1) * count :]
            self._copy(store, (prefix, "state"), ids, self.state_bytes)
        if self.alloc_on_call:
            self.num_alloc += 1
        return not self.fail

    def _copy(self, store, obj_key, ids, num_bytes):
        if store:
            data = torch.cat([self._piece(p) for p in ids])
            self.objects[obj_key] = data[:num_bytes].clone()
        else:
            data = self.objects[obj_key]
            for i, piece_id in enumerate(ids):
                segment = data[i * PIECE_BYTES : (i + 1) * PIECE_BYTES]
                self._piece(piece_id)[: segment.numel()] = segment


def _mapping(server: FakeServer, timeout: float = 5.0) -> RingMapping:
    mapping = RingMapping(server, "m##query", 1, timeout)  # type: ignore[arg-type]
    mapping.connect()
    return mapping


def _wrapper(server: FakeServer, writable: bool = True) -> RingSDKTransferWrapper:
    return RingSDKTransferWrapper(
        _mapping(server), server.layout, CHUNK, timeout=1.0, writable=writable
    )


def _key(num_tokens: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(num_tokens)), 0, num_tokens, "req"
    )


def _kv(num_tokens: int) -> torch.Tensor:
    return torch.randn(2, 2, num_tokens, 4).to(torch.float16)


def _ranges(server: FakeServer) -> list[tuple[int, int]]:
    return [(key.start, key.end) for key in server.freed]


# Ring mapping


def test_no_registered_ring_raises():
    """connect() raises when vLLM registered no Q ring for the model."""
    server = FakeServer()
    server.get_q_ring = lambda *_a: _done([])  # type: ignore[method-assign]

    with pytest.raises(RingStateError, match="no Q ring"):
        _mapping(server)


def test_free_lookup_locks_releases_the_unread_range_without_a_worker():
    """Frees only [start, end) under a worker-less key; an empty range sends nothing."""
    server = FakeServer()
    key = _key(3 * CHUNK)

    _mapping(server).free_lookup_locks(key, 8, 16)
    _mapping(server).free_lookup_locks(key, 16, 16)  # empty: nothing sent

    (freed,) = server.freed
    assert (freed.start, freed.end, freed.worker_id) == (8, 16, None)
    assert freed.token_bytes == key.token_bytes[: 16 * TOKEN_STRIDE]


def test_an_allocation_or_a_restarted_worker_fails_the_transfer():
    """The ring check fails if vLLM allocated blocks or re-registered the ring."""
    server = FakeServer()
    before = server.info()

    check_ring_unchanged(before, server.info(), "tensors")
    server.num_alloc += 1
    with pytest.raises(RingStateError, match="took Q ring blocks"):
        check_ring_unchanged(before, server.info(), "tensors")
    server.num_alloc -= 1
    server.instance_id = next(_INSTANCE_IDS)
    with pytest.raises(RingStateError, match="took Q ring blocks"):
        check_ring_unchanged(before, server.info(), "tensors")


def test_reservations_take_whole_chunks_of_disjoint_free_pieces():
    """Reservations take whole chunks of free, non-overlapping pieces only."""
    server = FakeServer()
    server.used = list(range(3, NUM_RING_BLOCKS))  # 3 free blocks: 6 pieces
    mapping = _mapping(server)

    first = mapping.reserve(pieces_per_chunk=4, max_chunks=10)
    second = mapping.reserve(pieces_per_chunk=2, max_chunks=10)

    assert (len(first.pieces), len(second.pieces)) == (4, 2)
    assert set(first.pieces) | set(second.pieces) == set(range(6))


def test_reserve_waits_for_a_release_when_no_chunk_fits():
    """reserve() blocks while another transfer holds the pieces, until it releases."""
    mapping = _mapping(FakeServer())
    whole_ring = NUM_RING_BLOCKS * RING_LAYERS
    held = mapping.reserve(pieces_per_chunk=whole_ring, max_chunks=1)
    got: list = []
    waiter = threading.Thread(
        target=lambda: got.append(mapping.reserve(pieces_per_chunk=1, max_chunks=1))
    )

    waiter.start()
    time.sleep(0.3)
    assert not got  # still waiting
    mapping.release(held)
    waiter.join(timeout=5)

    assert len(got) == 1


def test_set_max_transfers_changes_the_shared_slots():
    """Every mapping shares the transfer slots, whose count set_max_transfers sets."""
    first, second = _mapping(FakeServer()), _mapping(FakeServer())
    try:
        RingMapping.set_max_transfers(1)
        assert first.transfer_slot is second.transfer_slot
        assert first.transfer_slot.acquire(blocking=False)
        assert not second.transfer_slot.acquire(blocking=False)
        first.transfer_slot.release()
        with pytest.raises(ValueError, match="at least 1"):
            RingMapping.set_max_transfers(0)
    finally:
        RingMapping.set_max_transfers(2)


# Ring transfer


def test_kv_round_trip_avoids_the_blocks_vllm_holds():
    """Stored KV comes back unchanged, via free blocks only, with pieces released."""
    server = FakeServer()
    server.used = [0, 1, 5]
    wrapper = _wrapper(server)
    kv = _kv(3 * CHUNK)

    assert wrapper.store(_key(3 * CHUNK), 0, kv)
    back, _ = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, kv)
    assert not ring_mapping._RESERVED[server.instance_id]


def test_a_range_larger_than_the_free_blocks_is_batched():
    """A range larger than the free blocks is retrieved in several batches."""
    server = FakeServer()
    wrapper = _wrapper(server)
    kv = _kv(3 * CHUNK)
    wrapper.store(_key(3 * CHUNK), 0, kv)
    server.calls.clear()
    server.used = [b for b in range(NUM_RING_BLOCKS) if b not in (3, 7)]

    back, _ = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, kv)
    assert server.calls == [(0, 2 * CHUNK), (2 * CHUNK, 3 * CHUNK)]


def test_too_few_free_blocks_raises_and_releases_the_locks():
    """No room for one chunk raises and frees the lookup locks of the whole range."""
    server = FakeServer()
    wrapper = _wrapper(server)
    server.used = list(range(NUM_RING_BLOCKS))

    with pytest.raises(RingStateError, match="ring_blocks"):
        wrapper.retrieve(_key(CHUNK), 0)
    assert _ranges(server) == [(0, CHUNK)]


@pytest.mark.parametrize("failure", ["alloc", "fail"])
def test_a_failed_batch_releases_the_rest(failure):
    """A failed batch (vLLM allocation or server failure) frees the unread locks."""
    server = FakeServer()
    wrapper = _wrapper(server)
    wrapper.store(_key(2 * CHUNK), 0, _kv(2 * CHUNK))
    server.used = list(range(1, NUM_RING_BLOCKS))  # one chunk per batch
    server.alloc_on_call = failure == "alloc"
    server.fail = failure == "fail"

    if failure == "alloc":
        with pytest.raises(RingStateError, match="took Q ring blocks"):
            wrapper.retrieve(_key(2 * CHUNK), 0)
    else:
        assert wrapper.retrieve(_key(2 * CHUNK), 0) is None
    assert _ranges(server) == [(CHUNK, 2 * CHUNK)]


@pytest.mark.parametrize("num_tokens", [CHUNK + 1, CHUNK // 2])
def test_store_rejects_tensors_that_do_not_match_the_layout(num_tokens):
    """store() rejects tensors that are not whole chunks of the layout."""
    wrapper = _wrapper(FakeServer())

    with pytest.raises(ValueError, match="does not match"):
        wrapper.store(_key(num_tokens), 0, _kv(num_tokens))


@pytest.mark.parametrize("rows", [CHUNK, 4])  # 4: a sub-chunk sliding window
def test_query_rows_come_back_in_token_order_and_are_read_only(rows):
    """Query chunks come back concatenated in token order, and cannot be stored."""
    server = FakeServer(_query_layout(rows))
    chunks = [torch.randn(1, 2, rows, Q_WIDTH) for _ in range(3)]
    for index, chunk in enumerate(chunks):
        end = (index + 1) * CHUNK
        server.objects[_key(3 * CHUNK).token_bytes[: end * TOKEN_STRIDE]] = (
            chunk.reshape(-1).view(torch.uint8)
        )
    wrapper = _wrapper(server, writable=False)

    back, _ = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, torch.cat(chunks, dim=2))
    with pytest.raises(NotImplementedError):
        wrapper.store(_key(CHUNK), 0, torch.zeros(1, 2, CHUNK, Q_WIDTH))


# Context: building the wrapper from the engine's /status.


def _connect(
    world_size: int = 1, mutate=lambda groups: None, kind: str = "KV", **kernel_group
):
    """Connect a bare context as /status would: object layouts from qstore's
    report, a kernel group per object group from the registration's layout
    (``kernel_group`` overrides the first)."""
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
    from lmcache.sdk.context import LMCacheSDKContext

    ctx = object.__new__(LMCacheSDKContext)
    ctx._kind = LMCacheSDKCacheKind[kind]
    ctx._window_chunks = -1
    ctx._model_name = ctx._kind.server_model_name("m")
    ctx._world_size = world_size
    ctx._chunk_size = CHUNK
    ctx._mq_timeout = 1.0
    ctx._mp_conf = {}
    ctx._req_client = FakeServer()  # type: ignore[assignment]
    ctx._transfer_ctx = None
    groups = [
        {
            "object_group_idx": 0,
            "kind": "attention",
            "num_chunks_in_sw": -1,
            "chunk_object_shapes": [[2, 2, CHUNK, 4]],
            "chunk_object_dtypes": ["torch.float16"],
        }
    ]
    mutate(groups)
    ctx._object_layouts = [
        {
            "model_name": ctx._model_name,
            "world_size": world_size,
            "object_groups": groups,
        }
    ]
    ctx._connect_sdk(
        [
            {
                "kernel_group_idx": index,
                "object_group_idx": group["object_group_idx"],
                "tokens_per_block": CHUNK,
                "slots_per_block": CHUNK,
                **(kernel_group if index == 0 else {}),
            }
            for index, group in enumerate(groups)
        ]
    )
    return ctx


def test_the_context_builds_the_wrapper_from_the_reported_object_layout():
    """The context builds a writable KV wrapper from the reported object layout."""
    wrapper = _connect()._transfer_ctx

    assert isinstance(wrapper, RingSDKTransferWrapper)
    assert wrapper._writable
    assert wrapper._layout == _kv_layout()


@pytest.mark.parametrize(
    ("world_size", "mutate", "kernel_group", "match"),
    [
        (2, lambda groups: None, {}, "TP 1 only"),
        (1, lambda groups: groups.clear(), {}, "no object layout"),
        (1, lambda groups: groups[0].update(kind="recurrent"), {}, "full attention"),
        (1, lambda groups: groups[0].update(num_chunks_in_sw=2), {}, "full attention"),
        (1, lambda groups: groups.append(dict(groups[0])), {}, "dense models"),
        (1, lambda groups: None, {"slots_per_block": 4}, "compressed"),
    ],
)
def test_the_context_rejects_layouts_the_sdk_cannot_serve(
    world_size, mutate, kernel_group, match
):
    """The context rejects TP > 1, missing, non-dense or compressed layouts."""
    # First Party
    from lmcache.sdk.context import LMCacheSDKError

    with pytest.raises(LMCacheSDKError, match=match):
        _connect(world_size, mutate, **kernel_group)


@pytest.mark.parametrize(
    ("window", "rows", "expected"),
    [
        # sw_size_tokens 12 (chunk 8): the last 2 chunks, whole.
        (2, CHUNK, (3 * CHUNK, 2 * CHUNK)),
        # sw_size_tokens 4: the last chunk, its last 4 rows.
        (1, 4, (4 * CHUNK, 4)),
    ],
)
def test_the_query_context_reads_only_a_sliding_windows_trailing_rows(
    window, rows, expected
):
    """vLLM's lmcache.mp.q.sw_size_tokens, as the server reports it."""

    def windowed(groups):
        groups[0].update(
            num_chunks_in_sw=window, chunk_object_shapes=[[1, 2, rows, Q_WIDTH]]
        )

    ctx = _connect(mutate=windowed, kind="QUERY")

    assert not ctx._transfer_ctx._writable
    assert ctx.windowed_range(5 * CHUNK + 3) == expected


# Hybrid models: attention KV plus a recurrent state (object group 1).

STATE = SDKObjectLayout(1, (1, 3, 2, 4), torch.float32)  # 3 layers, 96 bytes


def test_hybrid_round_trip_carries_the_state_with_the_last_chunk():
    """The last chunk goes in its own batch, with the recurrent object; a
    sub-paged attention layout comes back as true K and V."""
    layout = SDKObjectLayout(0, (2, 2, CHUNK, 4), torch.float16, CHUNK, 4)
    server = FakeServer(layout, state_bytes=STATE.num_bytes)
    wrapper = RingSDKTransferWrapper(
        _mapping(server), layout, CHUNK, timeout=1.0, recurrent=[STATE]
    )
    kv = _kv(3 * CHUNK)
    state = RecurrentState(tuple(torch.randn(1, 2, 4) for _ in range(3)))

    with pytest.raises(ValueError, match="one page per recurrent layer"):
        wrapper.store(_key(3 * CHUNK), 0, kv)
    assert wrapper.store(_key(3 * CHUNK), 0, kv, state)
    result = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert result is not None
    back, back_state = result
    assert torch.equal(back, kv)
    assert back_state is not None
    assert all(map(torch.equal, back_state.pages, state.pages))
    assert server.calls[-2:] == [(0, 2 * CHUNK), (2 * CHUNK, 3 * CHUNK)]
    assert server.groups[-2:] == [[0], [0, 1]]


def test_the_wrapper_needs_the_copy_engine(monkeypatch):
    """Without the copy engine, the wrapper refuses to build."""
    monkeypatch.setattr(ring_transfer, "_COPY_ENGINE", False)
    with pytest.raises(RuntimeError, match="CUDA >= 12.8"):
        _wrapper(FakeServer())


@pytest.mark.parametrize(
    ("heads", "override", "expected"),
    [("2, 4", 0, CHUNK), ("1, 8", 4, 4), ("1, 8", 0, "sdk.kernel_block_size")],
)
def test_the_attention_kernel_block_size_detects_sub_paging(heads, override, expected):
    """The model's 2 KV heads x 4 registered as is, or as one synthetic head
    of 8 (sub-paged), whose kernel page size the server must set."""
    # First Party
    from lmcache.sdk.context import LMCacheSDKError, _attention_kernel_block_size

    group = {
        "kernel_group_idx": 0,
        "tokens_per_block": CHUNK,
        "engine_kv_shape": "NL x [NB, 2, BS, NH, HS]",
        "engine_kv_concrete_shape": f"2 x [6, 2, {CHUNK}, {heads}]",
    }
    mp_conf = {
        "runtime_plugin_config": {"extra_config": {"sdk.kernel_block_size": override}}
    }
    if isinstance(expected, str):
        with pytest.raises(LMCacheSDKError, match=expected):
            _attention_kernel_block_size(group, 2, 4, mp_conf)
    else:
        assert _attention_kernel_block_size(group, 2, 4, mp_conf) == expected


def test_the_kv_context_takes_a_hybrid_layout(monkeypatch):
    # First Party
    from lmcache.sdk import context

    monkeypatch.setattr(context, "_model_kv_heads", lambda *_: (2, 4))

    def hybrid(groups):
        groups.append(
            {
                "object_group_idx": 1,
                "kind": "recurrent",
                "num_chunks_in_sw": 1,
                "chunk_object_shapes": [list(STATE.shape)],
                "chunk_object_dtypes": ["torch.float32"],
            }
        )

    ctx = _connect(
        mutate=hybrid,
        engine_kv_shape="NL x [NB, 2, BS, NH, HS]",
        engine_kv_concrete_shape=f"2 x [6, 2, {CHUNK}, 2, 4]",
    )

    assert ctx.is_hybrid
    assert ctx._transfer_ctx._recurrent == (STATE,)
