# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``QRingTransferWrapper``: retrieving query tensors through
free blocks of the engine's Q ring.

A fake server holds each rank's query chunks and a CPU ring per rank. It
answers ``get_q_ring`` with the ring's state and, on ``retrieve_q``, writes
each chunk's kept rows into the given ring blocks, as the server's
``retrieve_q`` does through the ring's IPC mapping.
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.wrapper import q_ring
from lmcache.sdk.wrapper.q_ring import QRingStateError, QRingTransferWrapper
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo

CHUNK = 8
BLOCK = 4  # two ring blocks per chunk
LAYERS = 2
HIDDEN = 3


def _done(value: object) -> SimpleNamespace:
    return SimpleNamespace(
        result=lambda timeout=None: value,
        to_device_future=lambda **_kw: SimpleNamespace(
            result=lambda timeout=None: value
        ),
    )


class _Wrapper:
    """IPC wrapper double: rebuilds the ring tensor it was made from."""

    def __init__(self, tensor: torch.Tensor, opened: list[int]) -> None:
        self._tensor = tensor
        self._opened = opened

    def to_tensor(self) -> torch.Tensor:
        self._opened.append(1)
        return self._tensor


class FakeServer:
    """Serves ``get_q_ring``, ``retrieve_q`` and ``free_lookup_locks``."""

    def __init__(
        self, num_blocks: int = 6, ranks: int = 1, sw_size_tokens: int = -1
    ) -> None:
        self.sw_size_tokens = sw_size_tokens
        self.rings = [
            [torch.zeros(num_blocks, BLOCK, HIDDEN) for _ in range(LAYERS)]
            for _ in range(ranks)
        ]
        self.free = [list(range(num_blocks)) for _ in range(ranks)]
        self.step = 1
        self.generation = 1
        self.opened: list[int] = []
        # (worker_id, chunk start) -> [L, CHUNK, HIDDEN]
        self.storage: dict[tuple[int, int], torch.Tensor] = {}
        self.retrieves: list[tuple[int, int, int, list[int]]] = []
        self.freed: list[tuple[int, int]] = []
        self.step_on_retrieve = False

    def put(self, worker_id: int, num_tokens: int) -> torch.Tensor:
        """Cache ``num_tokens`` of query rows for a rank; return them."""
        rows = torch.randn(LAYERS, num_tokens, HIDDEN)
        for start in range(0, num_tokens, CHUNK):
            self.storage[(worker_id, start)] = rows[:, start : start + CHUNK]
        return rows

    def get_q_ring(self, model_name: str, world_size: int):
        return _done(
            [
                QRingInfo(
                    instance_id=100 + rank,
                    worker_id=rank,
                    q_caches=[_Wrapper(t, self.opened) for t in layers],  # type: ignore[misc]
                    sw_size_tokens=self.sw_size_tokens,
                    free_blocks=list(self.free[rank]),
                    step=self.step,
                    generation=self.generation,
                )
                for rank, layers in enumerate(self.rings)
            ]
        )

    def retrieve_q(self, key, instance_id, ring_block_ids, _handle):
        rank = instance_id - 100
        assert set(ring_block_ids) <= set(self.free[rank])
        self.retrieves.append((rank, key.start, key.end, list(ring_block_ids)))
        kept_tokens = CHUNK if self.sw_size_tokens < 0 else self.sw_size_tokens
        kept_blocks = kept_tokens // BLOCK
        per_chunk = CHUNK // BLOCK
        for c, start in enumerate(range(key.start, key.end, CHUNK)):
            rows = self.storage[(key.worker_id, start)]
            ids = ring_block_ids[c * per_chunk : (c + 1) * per_chunk][-kept_blocks:]
            for j, block_id in enumerate(ids):
                offset = CHUNK - kept_tokens + j * BLOCK
                for layer, ring_layer in enumerate(self.rings[rank]):
                    ring_layer[block_id] = rows[layer, offset : offset + BLOCK]
        if self.step_on_retrieve:
            self.step += 1
        return _done(True)

    def free_lookup_locks(self, key: IPCCacheServerKey, _world_size: int):
        self.freed.append((key.start, key.end))
        return _done(None)


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper pins work to the ring's device; here the rings are CPU."""
    backend = SimpleNamespace(
        create_event=lambda _d: object(),
        record_event=lambda _e, _s: None,
        export_event=lambda _e, _d: b"event",
    )
    monkeypatch.setattr(q_ring, "get_event_ipc_backend", lambda _d: backend)
    monkeypatch.setattr(
        q_ring,
        "torch_dev",
        SimpleNamespace(
            device=lambda _d: nullcontext(),
            current_stream=lambda: None,
            empty_cache=lambda: None,
        ),
    )


def _wrapper(server: FakeServer, world_size: int = 1) -> QRingTransferWrapper:
    wrapper = QRingTransferWrapper(
        server,  # type: ignore[arg-type]
        "m##query",
        world_size,
        CHUNK,
        timeout=1.0,
    )
    wrapper.connect()
    return wrapper


def _key(num_tokens: int, start: int = 0) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m##query", 1, 0, list(range(num_tokens)), start, num_tokens, "req"
    )


def test_round_trip_through_free_ring_blocks():
    server = FakeServer()
    rows = server.put(0, 2 * CHUNK)

    back = _wrapper(server).retrieve(_key(2 * CHUNK), 0)

    assert back is not None and back[1] is None
    assert torch.equal(back[0], rows.unsqueeze(0))
    assert server.retrieves == [(0, 0, 2 * CHUNK, [0, 1, 2, 3])]


def test_only_reported_free_blocks_are_used():
    server = FakeServer()
    server.free = [[5, 1, 3, 4]]
    rows = server.put(0, 2 * CHUNK)

    back = _wrapper(server).retrieve(_key(2 * CHUNK), 0)

    assert back is not None and torch.equal(back[0], rows.unsqueeze(0))
    assert server.retrieves[0][3] == [1, 3, 4, 5]


def test_a_range_larger_than_the_free_blocks_is_batched():
    server = FakeServer()
    server.free = [[0, 1]]  # one chunk at a time
    rows = server.put(0, 3 * CHUNK)

    back = _wrapper(server).retrieve(_key(3 * CHUNK), 0)

    assert back is not None and torch.equal(back[0], rows.unsqueeze(0))
    assert [(s, e) for _, s, e, _ in server.retrieves] == [
        (0, 8),
        (8, 16),
        (16, 24),
    ]


def test_too_few_free_blocks_raises_and_releases_the_locks():
    server = FakeServer()
    server.free = [[0]]
    server.put(0, CHUNK)

    with pytest.raises(QRingStateError, match="ring_blocks"):
        _wrapper(server).retrieve(_key(CHUNK), 0)
    assert server.freed == [(0, CHUNK)]


def test_a_vllm_step_during_the_retrieve_raises():
    server = FakeServer()
    server.put(0, CHUNK)
    wrapper = _wrapper(server)
    server.step_on_retrieve = True

    with pytest.raises(QRingStateError, match="runs no steps"):
        wrapper.retrieve(_key(CHUNK), 0)


def test_a_sub_chunk_window_returns_each_chunks_tail():
    server = FakeServer(sw_size_tokens=BLOCK)
    rows = server.put(0, 2 * CHUNK)

    back = _wrapper(server).retrieve(_key(2 * CHUNK), 0)

    tails = torch.cat([rows[:, c + CHUNK - BLOCK : c + CHUNK] for c in (0, CHUNK)], 1)
    assert back is not None and torch.equal(back[0], tails.unsqueeze(0))


def test_a_new_generation_reopens_the_ring():
    server = FakeServer()
    server.put(0, CHUNK)
    wrapper = _wrapper(server)
    opened = len(server.opened)

    wrapper.retrieve(_key(CHUNK), 0)
    assert len(server.opened) == opened
    server.generation += 1
    wrapper.retrieve(_key(CHUNK), 0)
    assert len(server.opened) == 2 * opened


def test_ranks_retrieve_their_own_keys_and_concatenate_heads():
    server = FakeServer(ranks=2)
    rank0 = server.put(0, CHUNK)
    rank1 = server.put(1, CHUNK)

    back = _wrapper(server, world_size=2).retrieve(_key(CHUNK), 0)

    assert back is not None
    assert torch.equal(back[0], torch.cat([rank0, rank1], dim=-1).unsqueeze(0))
    assert [rank for rank, *_ in server.retrieves] == [0, 1]


def test_retrieve_from_an_offset_releases_the_prefix_locks():
    server = FakeServer()
    rows = server.put(0, 2 * CHUNK)

    back = _wrapper(server).retrieve(_key(2 * CHUNK, start=CHUNK), 0)

    assert back is not None and torch.equal(back[0], rows[:, CHUNK:].unsqueeze(0))
    assert server.freed == [(0, CHUNK)]


def test_store_is_not_supported():
    with pytest.raises(NotImplementedError):
        _wrapper(FakeServer()).store()


def test_no_registered_ring_raises():
    server = FakeServer()
    server.rings = []
    server.free = []
    with pytest.raises(QRingStateError, match="no Q ring"):
        _wrapper(server)


@pytest.mark.parametrize(
    ("kind", "source", "match"),
    [
        ("KV", "ring", "only the QUERY kind"),
        ("QUERY", "shared", "source must be"),
    ],
)
def test_context_validates_the_query_source(kind: str, source: str, match: str):
    # First Party
    from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
    from lmcache.sdk.context import LMCacheSDKContext, LMCacheSDKError

    with pytest.raises(LMCacheSDKError, match=match):
        LMCacheSDKContext(
            "tcp://localhost:1",
            "http://localhost:1",
            "m",
            kind=LMCacheSDKCacheKind[kind],
            source=source,  # type: ignore[arg-type]
        )
