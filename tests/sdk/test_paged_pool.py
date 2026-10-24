# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``PagedPoolTransferWrapper``.

A fake server stands in for LMCache, playing the lmcache-driven STORE and
RETRIEVE handlers per kernel group against a registered pool: it copies each
chunk's blocks into its storage and back, does not store a chunk whose block
ids are all the null block, and reads a recurrent group's last chunk only.
The pools are small CPU tensors in vLLM's formats, so the real gather/scatter
helpers run:

- dense: one fused-K/V attention group, ``[NB, BS, NH, CS]``;
- hybrid: two recurrent (GDN) groups and one sub-paged attention group, all
  ``[NB, 2, BS, 1, W]`` as for Qwen3.5.
"""

# Standard
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.wrapper import paged_pool
from lmcache.sdk.wrapper.paged_pool import (
    NULL_BLOCK_ID,
    PagedPoolTransferWrapper,
    PoolCapacityError,
    PoolGroup,
    RecurrentState,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

CHUNK = 8
INSTANCE = 7

# Dense: fused K/V, two blocks per chunk, a two-chunk pool.
NUM_LAYERS = 2
BLOCK = 4
NUM_HEADS = 2
CS = 8  # fused K/V: 2 * head_size
WIDTH = NUM_HEADS * CS
POOL_CHUNKS = 2

# Hybrid: one block per chunk (544 / 544 on Qwen3.5), two kernel pages each.
KERNEL = 4
HYBRID_WIDTH = 6  # NH * HS of the real heads == HS of the one-head view
HYBRID_POOL_CHUNKS = 3
HYBRID_GROUPS = (
    PoolGroup(("layer.0", "layer.1"), CHUNK, recurrent=True),
    PoolGroup(("layer.2",), CHUNK, recurrent=True),
    PoolGroup(("layer.3", "layer.4"), CHUNK, kernel_block_size=KERNEL),
)


def _done(value: object) -> SimpleNamespace:
    return SimpleNamespace(result=lambda timeout=None: value)


class FakeServer:
    """Plays the lmcache-driven STORE/RETRIEVE handlers per kernel group."""

    def __init__(
        self,
        pool: dict[str, torch.Tensor],
        groups: Sequence[PoolGroup],
        tokens_per_chunk: int = CHUNK,
    ) -> None:
        self.pool = pool
        self.groups = groups
        # Sub-chunk windows: the server keeps each chunk's last tokens only.
        self.tokens_per_chunk = tokens_per_chunk
        self.storage: dict[tuple[tuple[int, ...], int], list[torch.Tensor]] = {}
        self.retrieves: list[tuple[int, int]] = []
        self.block_ids: list[list[list[int]]] = []
        self.fail_retrieve_at: int | None = None
        self.closed = False

    def create_recorded_event(self) -> object:
        return object()

    def _chunk_blocks(self, group: PoolGroup, ids: list[int], chunk: int) -> list[int]:
        """A chunk's blocks of ``group``, the trailing kept ones for attention."""
        per_chunk = CHUNK // group.tokens_per_block
        blocks = ids[chunk * per_chunk : (chunk + 1) * per_chunk]
        if group.recurrent:
            return blocks
        return blocks[-(self.tokens_per_chunk // group.tokens_per_block) :]

    def submit_store(self, _rid, key, _kv, block_ids, _event, _bpc):
        self.block_ids.append(block_ids)
        for g, (group, ids) in enumerate(zip(self.groups, block_ids, strict=True)):
            for c, start in enumerate(range(key.start, key.end, CHUNK)):
                blocks = self._chunk_blocks(group, ids, c)
                if all(block == NULL_BLOCK_ID for block in blocks):
                    continue  # all-null chunk: not stored
                self.storage[(key.token_ids[: start + CHUNK], g)] = [
                    self.pool[name][blocks].clone() for name in group.layer_names
                ]
        return _done(True)

    def submit_retrieve(self, _rid, key, _kv, block_ids, _event, _bpc):
        self.retrieves.append((key.start, key.end))
        self.block_ids.append(block_ids)
        if key.start == self.fail_retrieve_at:
            return _done(False)
        starts = list(range(key.start, key.end, CHUNK))
        for g, (group, ids) in enumerate(zip(self.groups, block_ids, strict=True)):
            # A recurrent group's window is its last chunk.
            wanted = starts[-1:] if group.recurrent else starts
            for start in wanted:
                stored = self.storage.get((key.token_ids[: start + CHUNK], g))
                if stored is None:
                    return _done(False)
                blocks = self._chunk_blocks(group, ids, starts.index(start))
                for name, pages in zip(group.layer_names, stored, strict=True):
                    self.pool[name][blocks] = pages
        # Widen the window between the server filling the pool and the
        # wrapper draining it, where an unguarded caller would overwrite it.
        time.sleep(0.002)
        return _done(True)

    def unregister(self):
        return _done(None)

    def close(self) -> None:
        self.closed = True


class FakeClient:
    """Records the lookup-lock releases the wrapper sends."""

    def __init__(self) -> None:
        self.freed: list[tuple[int, int]] = []

    def free_lookup_locks(self, key: IPCCacheServerKey, _world_size: int):
        assert key.worker_id is None
        self.freed.append((key.start, key.end))
        return _done(None)


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper pins work to the pool's accelerator; here the pool is CPU."""
    stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(
        paged_pool,
        "torch_dev",
        SimpleNamespace(device=lambda _d: nullcontext(), current_stream=lambda: stream),
    )


Made = tuple[PagedPoolTransferWrapper, FakeServer, FakeClient]


def _wrapper(
    pool: dict[str, torch.Tensor],
    groups: Sequence[PoolGroup],
    num_chunks: int,
    num_planes: int,
    tokens_per_chunk: int = CHUNK,
) -> Made:
    server = FakeServer(pool, groups, tokens_per_chunk)
    client = FakeClient()
    wrapper = PagedPoolTransferWrapper(
        server,  # type: ignore[arg-type]
        INSTANCE,
        pool,
        groups,
        LayoutHints(kv_layout="NHD"),
        CHUNK,
        num_chunks=num_chunks,
        num_planes=num_planes,
        tokens_per_chunk=tokens_per_chunk,
        req_client=client,  # type: ignore[arg-type]
        timeout=1.0,
    )
    return wrapper, server, client


def _dense(tokens_per_chunk: int = CHUNK) -> Made:
    """A two-chunk fused-K/V pool; block 0 is the null block."""
    num_blocks = 1 + POOL_CHUNKS * CHUNK // BLOCK
    pool = {
        f"layer.{i}": torch.zeros(num_blocks, BLOCK, NUM_HEADS, CS)
        for i in range(NUM_LAYERS)
    }
    groups = [PoolGroup(tuple(pool), BLOCK)]
    return _wrapper(pool, groups, POOL_CHUNKS, 1, tokens_per_chunk)


def _hybrid() -> Made:
    """A hybrid pool: recurrent groups keep one chunk, attention three."""
    pool = {}
    for group in HYBRID_GROUPS:
        # Null block, data, and a spare (a 2-block pool reads as [2, NB, ...]).
        num_blocks = 1 + (1 if group.recurrent else HYBRID_POOL_CHUNKS) + 1
        for name in group.layer_names:
            pool[name] = torch.zeros(num_blocks, 2, CHUNK, 1, HYBRID_WIDTH)
    return _wrapper(pool, HYBRID_GROUPS, HYBRID_POOL_CHUNKS, 2)


@pytest.fixture
def setup() -> Made:
    return _dense()


def _key(num_tokens: int, start: int = 0, offset: int = 100) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(offset, offset + num_tokens)), start, num_tokens, "req"
    )


def _kv(num_tokens: int) -> torch.Tensor:
    return torch.randn(1, NUM_LAYERS, num_tokens, WIDTH)


def _hybrid_kv(num_tokens: int) -> torch.Tensor:
    return torch.randn(2, 2, num_tokens, HYBRID_WIDTH)


def _state() -> RecurrentState:
    return RecurrentState(
        tuple(torch.randn(1, 2, CHUNK, 1, HYBRID_WIDTH) for _ in range(3))
    )


def test_round_trip_spans_several_pool_batches(setup):
    """5 chunks through a 2-chunk pool: batches [0,16), [16,32), [32,40)."""
    wrapper, server, client = setup
    kv = _kv(5 * CHUNK)

    assert wrapper.store(_key(5 * CHUNK), INSTANCE, kv)
    back = wrapper.retrieve(_key(5 * CHUNK), INSTANCE)

    assert back is not None and torch.equal(back[0], kv) and back[1] is None
    assert server.retrieves == [(0, 16), (16, 32), (32, 40)]
    assert client.freed == []


def test_retrieve_from_an_offset_releases_the_unread_prefix(setup):
    """The lookup locked [0, end); a retrieve from 16 must free [0, 16)."""
    wrapper, _, client = setup
    kv = _kv(4 * CHUNK)
    wrapper.store(_key(4 * CHUNK), INSTANCE, kv)

    back = wrapper.retrieve(_key(4 * CHUNK, start=2 * CHUNK), INSTANCE)

    assert back is not None and torch.equal(back[0], kv[:, :, 2 * CHUNK :, :])
    assert client.freed == [(0, 2 * CHUNK)]


def test_failed_batch_releases_the_batches_after_it(setup):
    wrapper, server, client = setup
    wrapper.store(_key(5 * CHUNK), INSTANCE, _kv(5 * CHUNK))
    server.fail_retrieve_at = 16

    assert wrapper.retrieve(_key(5 * CHUNK), INSTANCE) is None
    assert server.retrieves == [(0, 16), (16, 32)]
    assert client.freed == [(32, 40)]


def test_sub_chunk_window_returns_the_tail_of_every_chunk():
    """With a one-block window per chunk, the server fills only each chunk's
    last block, and the wrapper returns exactly those rows."""
    wrapper, _, _ = _dense(tokens_per_chunk=BLOCK)
    kv = _kv(3 * CHUNK)
    wrapper.store(_key(3 * CHUNK), INSTANCE, kv)

    back = wrapper.retrieve(_key(3 * CHUNK), INSTANCE)

    tails = [
        kv[:, :, start + CHUNK - BLOCK : start + CHUNK, :]
        for start in range(0, 3 * CHUNK, CHUNK)
    ]
    assert back is not None and back[0].shape[2] == 3 * BLOCK
    assert torch.equal(back[0], torch.cat(tails, dim=2))


def test_concurrent_callers_share_the_pool_safely(setup):
    """batch.modify drives one context from a thread per stream: batches of
    different calls must never overwrite each other in the shared pool."""
    wrapper, _, _ = setup
    keys = [_key(5 * CHUNK, offset=1000 * i) for i in range(8)]
    kvs = [_kv(5 * CHUNK) for _ in keys]

    with ThreadPoolExecutor(len(keys)) as ex:
        assert all(ex.map(lambda i: wrapper.store(keys[i], INSTANCE, kvs[i]), range(8)))
    with ThreadPoolExecutor(len(keys)) as ex:
        backs = list(ex.map(lambda i: wrapper.retrieve(keys[i], INSTANCE), range(8)))

    assert all(
        b is not None and torch.equal(b[0], kv)
        for b, kv in zip(backs, kvs, strict=True)
    )


def test_hybrid_round_trip_restores_kv_and_the_last_chunk_state():
    wrapper, _, client = _hybrid()
    kv, state = _hybrid_kv(3 * CHUNK), _state()

    assert wrapper.store(_key(3 * CHUNK), INSTANCE, kv, state)
    back = wrapper.retrieve(_key(3 * CHUNK), INSTANCE)

    assert back is not None
    back_kv, back_state = back
    assert torch.equal(back_kv, kv)
    assert back_state is not None
    assert all(
        torch.equal(a, b) for a, b in zip(back_state.pages, state.pages, strict=True)
    )
    assert client.freed == []


def test_hybrid_block_ids_skip_the_null_block_and_null_all_but_the_last_state():
    wrapper, server, _ = _hybrid()
    wrapper.store(_key(3 * CHUNK), INSTANCE, _hybrid_kv(3 * CHUNK), _state())

    recurrent, _, attention = server.block_ids[0]
    assert recurrent == [NULL_BLOCK_ID, NULL_BLOCK_ID, 1]
    assert attention == [1, 2, 3]
    # Only the last chunk's state is stored for each recurrent group.
    stored = {(len(prefix), g) for prefix, g in server.storage}
    assert stored == {(24, 0), (24, 1), (8, 2), (16, 2), (24, 2)}


def test_hybrid_state_is_keyed_at_the_end_of_the_stored_prefix():
    """The carried state must be found under the new prefix's last chunk."""
    wrapper, _, _ = _hybrid()
    wrapper.store(_key(3 * CHUNK), INSTANCE, _hybrid_kv(3 * CHUNK), _state())

    assert wrapper.retrieve(_key(2 * CHUNK), INSTANCE) is None  # none at 16


def test_hybrid_retrieve_past_capacity_releases_its_locks_and_raises():
    """A recurrent state cannot be batched, so a range past the pool fails."""
    wrapper, server, client = _hybrid()

    with pytest.raises(PoolCapacityError, match="exceeds the pool"):
        wrapper.retrieve(_key(4 * CHUNK), INSTANCE)

    assert client.freed == [(0, 4 * CHUNK)]
    assert server.block_ids == []


def test_hybrid_store_past_capacity_raises():
    wrapper, _, _ = _hybrid()
    with pytest.raises(PoolCapacityError):
        wrapper.store(_key(4 * CHUNK), INSTANCE, _hybrid_kv(4 * CHUNK), _state())


def test_hybrid_store_rejects_a_state_of_the_wrong_size():
    wrapper, _, _ = _hybrid()
    with pytest.raises(ValueError, match="per recurrent layer"):
        wrapper.store(
            _key(CHUNK),
            INSTANCE,
            _hybrid_kv(CHUNK),
            RecurrentState(_state().pages[:2]),
        )


@pytest.mark.parametrize(
    ("make", "num_tokens", "kv", "with_state"),
    [
        # Split K/V is [2, L, T, D]; this single-plane pool takes [1, L, T, D].
        (_dense, CHUNK, torch.randn(2, NUM_LAYERS, CHUNK, WIDTH), False),
        # The tensor must cover the key's whole range.
        (_dense, 2 * CHUNK, torch.randn(1, NUM_LAYERS, CHUNK, WIDTH), False),
        # A hybrid tensor holds the attention layers only.
        (_hybrid, CHUNK, torch.randn(2, 3, CHUNK, HYBRID_WIDTH), True),
    ],
    ids=["planes", "tokens", "attention-layers"],
)
def test_store_rejects_a_tensor_of_the_wrong_shape(make, num_tokens, kv, with_state):
    wrapper, _, _ = make()
    state = _state() if with_state else None
    with pytest.raises(ValueError, match="does not match"):
        wrapper.store(_key(num_tokens), INSTANCE, kv, state)


def test_calls_for_another_instance_are_rejected(setup):
    wrapper, _, _ = setup
    with pytest.raises(ValueError, match="does not own this pool"):
        wrapper.retrieve(_key(CHUNK), INSTANCE + 1)


def test_sub_keys_chain_from_token_zero():
    """A batch's key keeps the tokens up to its end, so its hashes match."""
    key = _key(4 * CHUNK)
    sub = PagedPoolTransferWrapper._sub_key(key, CHUNK, 2 * CHUNK)
    assert sub.token_ids == key.token_ids[: 2 * CHUNK]
    assert (sub.start, sub.end) == (CHUNK, 2 * CHUNK)
    assert sub.request_id == key.request_id


def test_close_unregisters_the_pool(setup):
    wrapper, server, _ = setup
    wrapper.close()
    assert server.closed
