# SPDX-License-Identifier: Apache-2.0
"""CPU KV events for chunks that a LOOKUP loads into the local L1.

A chunk can become readable in a server's L1 without that server's engine
ever storing it: a LOOKUP prefetches it from a tier another server filled (a
shared L2, or a P2P peer attached as an L2 adapter). These tests drive the
real LookupModule, prefetch controller, L1 manager, filesystem L2 adapter,
event bus and KVEventModule on CPU.

Simulated parts: there is no engine and no GPU, so object payloads are filled
CPU buffers, and the server that originally stored the chunks is a second
storage manager writing through its own L1 into the same L2 directory. P2P
transport is not exercised here; a P2P peer reaches the same prefetch
admission as any other L2 adapter.
"""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.attn_window import AttnWindowDesc
from lmcache.v1.common.custom_types import IPCCacheServerKey, KVEventRecord
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    L2AdaptersConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.mp_observability import event_bus as event_bus_module
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lookup import LookupModule

MODEL = "model"
CHUNK = 16
HASH_ALGORITHM = "blake3"
LAYOUT = MemoryLayoutDesc([torch.Size([CHUNK, 8])], [torch.float16])
WAIT_SECONDS = 30.0


def _tokens(num_chunks: int, first: int = 1) -> list[int]:
    return list(range(first, first + num_chunks * CHUNK))


def _lookup_key(
    tokens: list[int], request_id: str, start: int = 0
) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        MODEL, 1, None, tokens, start=start, end=len(tokens), request_id=request_id
    )


def _chunk_hashes(tokens: list[int]) -> list[bytes]:
    return TokenHasher(CHUNK, HASH_ALGORITHM).compute_packed_chunk_hashes(
        _lookup_key(tokens, "hashes").token_bytes
    )


def _object_keys(tokens: list[int]) -> list[ObjectKey]:
    return ipc_key_to_object_keys(
        _lookup_key(tokens, "keys"),
        _chunk_hashes(tokens),
        [0],
        DEFAULT_SHARD_REPLICATION_DESC,
    )[0]


def _storage_config(
    l2_dir: Path | None, prefetch_policy: str = "default"
) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4 * 1024 * 1024, use_lazy=False, shm_name=""
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[] if l2_dir is None else [FSL2AdapterConfig(str(l2_dir))]
        ),
        prefetch_policy=prefetch_policy,
    )


def _flush(bus: EventBus) -> None:
    """Dispatch every queued event, then resume the drain thread."""
    bus.stop()
    bus.start()


@dataclass
class _Server:
    """The modules of one MP server that issues LOOKUPs and never STOREs."""

    bus: EventBus
    ctx: MPCacheServerContext
    lookup: LookupModule
    events: KVEventModule | None

    def load(self, tokens: list[int], request_id: str, start: int = 0) -> int | None:
        """LOOKUP ``tokens`` and wait for the prefetch; return the hit tokens."""
        self.lookup.lookup(_lookup_key(tokens, request_id, start), tp_size=1)
        hit = self.lookup.wait_prefetch_status(request_id, WAIT_SECONDS)
        _flush(self.bus)
        return hit

    def release(self, tokens: list[int], request_id: str, start: int = 0) -> None:
        """Release the lookup's read locks, as an engine does after its retrieve."""
        self.lookup.free_lookup_locks(_lookup_key(tokens, request_id, start), tp_size=1)
        _flush(self.bus)

    def records(self) -> list[KVEventRecord]:
        assert self.events is not None
        batch = self.events.read_kv_events(MODEL, 0, 256)
        assert not batch.lost
        return batch.events


@dataclass
class _Harness:
    """Servers sharing one event bus and one L2 directory."""

    bus: EventBus
    l2_dir: Path
    contexts: list[MPCacheServerContext] = field(default_factory=list)

    def seed_l2(self, tokens: list[int]) -> None:
        """Store ``tokens`` into the shared L2 through another server's L1.

        Stands in for the engine that originally computed the chunks. The
        writer is closed before any server under test subscribes, so its L1
        notifications cannot be mistaken for the reader's.
        """
        keys = _object_keys(tokens)
        in_l2: set[ObjectKey] = set()
        all_in_l2 = threading.Event()

        def on_l2_stored(event: Event) -> None:
            in_l2.update(event.metadata["keys"])
            if in_l2.issuperset(keys):
                all_in_l2.set()

        self.bus.subscribe(EventType.L2_KEYS_STORED, on_l2_stored)
        writer = StorageManager(_storage_config(self.l2_dir))
        try:
            reserved = writer.reserve_write(keys, LAYOUT)
            assert set(reserved) == set(keys)
            for memory in reserved.values():
                tensor = memory.get_tensor(0)
                assert tensor is not None
                tensor.fill_(1)
            writer.finish_write(keys, l1_owners_of(reserved))
            assert all_in_l2.wait(WAIT_SECONDS), "the writer never reached the L2"
        finally:
            writer.close()
            _flush(self.bus)

    def server(
        self,
        prefetch_policy: str = "default",
        kv_events: bool = True,
        shares_l2: bool = True,
    ) -> _Server:
        """Start a server with its own L1, by default on the shared L2 directory."""
        ctx = MPCacheServerContext(
            _storage_config(self.l2_dir if shares_l2 else None, prefetch_policy),
            chunk_size=CHUNK,
        )
        self.contexts.append(ctx)
        ctx.layout_desc_registry.register(
            MODEL,
            1,
            LAYOUT,
            chunk_size=CHUNK,
            hash_algorithm=HASH_ALGORITHM,
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1]),
        )
        events = KVEventModule(ctx, kv_event_log_size=256) if kv_events else None
        return _Server(self.bus, ctx, LookupModule(ctx), events)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Harness]:
    """A live event bus installed as the process singleton, plus an L2 directory."""
    bus = EventBus(EventBusConfig())
    monkeypatch.setattr(event_bus_module, "_global_bus", bus)
    l2_dir = tmp_path / "shared-l2"
    l2_dir.mkdir()
    bus.start()
    built = _Harness(bus, l2_dir)
    try:
        yield built
    finally:
        for ctx in reversed(built.contexts):
            ctx.close()
        bus.stop()


@pytest.mark.parametrize("covered_chunks", [0, 1])
def test_load_into_retained_l1_is_announced_without_a_local_store(
    harness: _Harness, covered_chunks: int
) -> None:
    """A LOOKUP-only server announces the chunks its prefetch retained in L1.

    ``covered_chunks`` models a prefix the engine already holds: the lookup
    starts after it, and the first loaded chunk still names its real parent.
    """
    tokens = _tokens(4)
    harness.seed_l2(tokens)
    server = harness.server()
    hashes = _chunk_hashes(tokens)
    start = covered_chunks * CHUNK
    loaded = range(covered_chunks, 4)

    # A lookup that finds nothing binds its chunks, which is not a placement.
    assert server.load(_tokens(4, first=1000), "miss") == 0
    assert server.records() == []

    assert server.load(tokens, "load", start=start) == len(tokens)

    assert [
        (r.kind, r.medium, r.block_hashes, r.parent_block_hash, r.token_ids)
        for r in server.records()
    ] == [
        (
            "stored",
            "CPU",
            [hashes[index]],
            hashes[index - 1] if index else None,
            tokens[index * CHUNK : (index + 1) * CHUNK],
        )
        for index in loaded
    ]

    # A request that ends without retrieving, as a cancelled one does, leaves
    # the retained copies readable, so their placements stay.
    server.release(tokens, "load", start=start)
    assert [r.kind for r in server.records()] == ["stored"] * len(loaded)

    # Losing the server's only local copies withdraws exactly those chunks.
    keys = _object_keys(tokens)
    assert server.ctx.storage_manager.delete_l1_keys(keys) == (len(loaded), 0)
    _flush(harness.bus)
    assert [(r.kind, r.block_hashes) for r in server.records()[len(loaded) :]] == [
        ("removed", [hashes[index]]) for index in loaded
    ]


def test_temporary_prefetch_copy_is_withdrawn_when_its_reader_finishes(
    harness: _Harness,
) -> None:
    """A buffer-only prefetch leaves no placement behind its temporary copy."""
    tokens = _tokens(2)
    harness.seed_l2(tokens)
    server = harness.server(prefetch_policy="skip_l1")
    hashes = _chunk_hashes(tokens)

    assert server.load(tokens, "temporary") == len(tokens)
    assert [r.kind for r in server.records()] == ["stored", "stored"]

    server.release(tokens, "temporary")
    assert [(r.kind, r.block_hashes) for r in server.records()[2:]] == [
        ("removed", [hashes[0]]),
        ("removed", [hashes[1]]),
    ]


@pytest.mark.parametrize(
    "kv_events, shares_l2", [(False, True), (True, False)], ids=["no-consumer", "no-l2"]
)
def test_lookup_publishes_no_binding_when_it_cannot_lead_to_an_announcement(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    kv_events: bool,
    shares_l2: bool,
) -> None:
    """A lookup unpacks no tokens when nothing could use the binding.

    With the event channel off the lookup still loads. A server without an L2
    adapter has nothing to load, so it has nothing to bind.
    """
    tokens = _tokens(2)
    harness.seed_l2(tokens)
    server = harness.server(kv_events=kv_events, shares_l2=shares_l2)
    published: list[EventType] = []
    publish = harness.bus.publish

    def recording_publish(event: Event) -> None:
        published.append(event.event_type)
        publish(event)

    monkeypatch.setattr(harness.bus, "publish", recording_publish)

    assert server.load(tokens, "quiet") == (len(tokens) if shares_l2 else 0)

    assert EventType.MP_LOOKUP_PREFETCH_START in published
    assert EventType.MP_TOKENS not in published
