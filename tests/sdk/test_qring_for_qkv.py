# SPDX-License-Identifier: Apache-2.0
"""Unit tests for staging KV and query tensors through the engine's Q ring.

A fake server keeps each chunk's object as bytes and a CPU ring. It answers
``get_q_ring`` with the ring's state and moves object bytes between its store
and the given ring pieces in the order the server's ``sdk_retrieve`` /
``sdk_store`` use: chunks in order.

Reservations are shared by every mapping of the process and keyed by the
ring's instance ID, so every fake ring takes a fresh instance ID.
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
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
from lmcache.sdk.wrapper.ring_transfer import RingSDKTransferWrapper, SDKObjectLayout
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.token_codec import TOKEN_STRIDE
from lmcache.v1.transport.qstore import QRingInfo

CHUNK = 8
RING_LAYERS = 2
# A ring block is [4, 8] float32 per layer: 128-byte pieces.
RING_BLOCK = (4, 8)
PIECE_BYTES = 4 * 8 * 4
NUM_RING_BLOCKS = 16
Q_WIDTH = 6

_INSTANCE_IDS = itertools.count(1000)


def _done(value: object) -> SimpleNamespace:
    return SimpleNamespace(
        result=lambda timeout=None: value,
        to_device_future=lambda **_kw: SimpleNamespace(
            result=lambda timeout=None: value
        ),
    )


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


def _kv_layout() -> SDKObjectLayout:
    """One attention object per chunk: 2 x 2 x 8 x 4 float16, 256 bytes."""
    return SDKObjectLayout(0, (2, 2, CHUNK, 4), torch.float16)


def _query_layout() -> SDKObjectLayout:
    return SDKObjectLayout(0, (1, 2, CHUNK, Q_WIDTH), torch.float32)


class FakeServer:
    """Serves ``get_q_ring``, ``sdk_retrieve``, ``sdk_store`` and
    ``free_lookup_locks`` over one CPU ring."""

    def __init__(self, layout: SDKObjectLayout | None = None) -> None:
        self.layout = layout or _kv_layout()
        self.instance_id = next(_INSTANCE_IDS)
        self.ring = [
            torch.zeros(NUM_RING_BLOCKS, *RING_BLOCK) for _ in range(RING_LAYERS)
        ]
        self.used: list[int] = []
        self.num_alloc = 1
        # Token prefix bytes through the chunk -> the chunk object's bytes
        self.objects: dict[bytes, torch.Tensor] = {}
        self.calls: list[tuple[int, int]] = []
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
        return _done(self._move(False, key, instance_id, piece_ids))

    def sdk_store(self, key, instance_id, group_ids, piece_ids, _handle):
        return _done(self._move(True, key, instance_id, piece_ids))

    def free_lookup_locks(self, key: IPCCacheServerKey, _world_size: int):
        self.freed.append(key)
        return _done(None)

    def _piece(self, piece_id: int) -> torch.Tensor:
        block, layer = divmod(piece_id, RING_LAYERS)
        return self.ring[layer][block].reshape(-1).view(torch.uint8)

    def _move(self, store, key, instance_id, piece_ids):
        assert instance_id == self.instance_id
        assert not {p // RING_LAYERS for p in piece_ids} & set(self.used)
        self.calls.append((key.start, key.end))
        count = math.ceil(self.layout.num_bytes / PIECE_BYTES)
        for index, start in enumerate(range(key.start, key.end, CHUNK)):
            prefix = key.token_bytes[: (start + CHUNK) * TOKEN_STRIDE]
            ids = piece_ids[index * count : (index + 1) * count]
            if store:
                data = torch.cat([self._piece(p) for p in ids])
                self.objects[prefix] = data[: self.layout.num_bytes].clone()
            else:
                data = self.objects[prefix]
                for i, piece_id in enumerate(ids):
                    segment = data[i * PIECE_BYTES : (i + 1) * PIECE_BYTES]
                    self._piece(piece_id)[: segment.numel()] = segment
        if self.alloc_on_call:
            self.num_alloc += 1
        return not self.fail


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
    back = wrapper.retrieve(_key(3 * CHUNK), 0)

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

    back = wrapper.retrieve(_key(3 * CHUNK), 0)

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


def test_query_rows_come_back_in_token_order_and_are_read_only():
    """Query chunks come back concatenated in token order, and cannot be stored."""
    server = FakeServer(_query_layout())
    chunks = [torch.randn(1, 2, CHUNK, Q_WIDTH) for _ in range(3)]
    for index, chunk in enumerate(chunks):
        end = (index + 1) * CHUNK
        server.objects[_key(3 * CHUNK).token_bytes[: end * TOKEN_STRIDE]] = (
            chunk.reshape(-1).view(torch.uint8)
        )
    wrapper = _wrapper(server, writable=False)

    back = wrapper.retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back, torch.cat(chunks, dim=2))
    with pytest.raises(NotImplementedError):
        wrapper.store(_key(CHUNK), 0, torch.zeros(1, 2, CHUNK, Q_WIDTH))


# Context: building the wrapper from the engine's /status.


def _connect(world_size: int = 1, mutate=lambda groups: None, **kernel_group):
    """Connect a bare KV context as /status would: object layouts from
    qstore's report, the kernel group from the registration's layout."""
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
    from lmcache.sdk.context import LMCacheSDKContext

    ctx = object.__new__(LMCacheSDKContext)
    ctx._kind = LMCacheSDKCacheKind.KV
    ctx._model_name = ctx._kind.server_model_name("m")
    ctx._world_size = world_size
    ctx._chunk_size = CHUNK
    ctx._mq_timeout = 1.0
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
        {"tokens_per_block": CHUNK, "slots_per_block": CHUNK, **kernel_group}
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
