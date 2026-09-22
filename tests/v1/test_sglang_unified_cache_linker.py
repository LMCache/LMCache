# SPDX-License-Identifier: Apache-2.0
"""Linker contract tests with real GPU page views and a controlled MP peer."""

# Standard
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Third Party
import pytest
import torch

pytest.importorskip("sglang.srt.mem_cache.unified_cache.linker_factory")

# Third Party
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)

# First Party
from lmcache.integration.sglang import unified_cache_linker as module
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.futures import MessagingFuture

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA page views"
    ),
]


def done(value: Any) -> MessagingFuture:
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


@dataclass
class Arguments:
    model_path: str = "test/model"
    revision: str = "immutable-revision"
    tp_size: int = 1


@dataclass
class Parameters:
    token_to_kv_pool_allocator: Any
    page_size: int = 2
    attn_tp_cache_group: Any = None
    tp_cache_group: Any = None
    pp_rank: int = 0
    pp_size: int = 1
    attn_cp_rank: int = 0
    attn_cp_size: int = 1


class Allocator:
    def get_kvcache(self) -> None:
        return None


@dataclass
class Peer:
    objects: dict = field(default_factory=dict)
    locks: Counter = field(default_factory=Counter)
    sessions: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    closed: bool = False
    chunk_size: int = 1

    def get_chunk_size(self):
        return done(self.chunk_size)

    def lookup(self, key, tp_size):
        assert key.chunk_hashes and not key.token_ids
        assert key.num_kv_readers == tp_size == 1
        count = 0
        for digest in key.chunk_hashes:
            identity = (key.model_name, digest)
            if identity not in self.objects:
                break
            self.locks[identity] += 1
            count += 1
        self.sessions[key.request_id] = count
        return done(None)

    def query_prefetch_status(self, rid):
        return done(self.sessions[rid])

    def free_lookup_locks(self, key, tp_size):
        for digest in key.chunk_hashes[key.start : key.end]:
            identity = (key.model_name, digest)
            assert self.locks[identity] > 0
            self.locks[identity] -= 1
        return done(None)

    def end_session(self, rid):
        self.sessions.pop(rid, None)
        return done(None)

    def ping(self, instance_id):
        return done(True)

    def close(self):
        self.closed = True

    def complete(self, success=True):
        pending, self.pending = self.pending, []
        for callback, future in pending:
            callback(success)
            future.set_result(success)


class Context:
    def __init__(self, peer):
        self.peer = peer
        self.registered = False
        self.closed = False
        peer.contexts.append(self)

    def register(self, *args, **kwargs):
        self.registered = True

    def unregister(self):
        self.registered = False
        return done(None)

    def close(self):
        self.closed = True

    def create_recorded_event(self):
        return object()

    def submit_store(self, rid, key, tensors, groups, event, blocks):
        future = MessagingFuture()

        def write(success):
            if success:
                for digest, block in zip(key.chunk_hashes, groups[0], strict=True):
                    self.peer.objects[(key.model_name, digest)] = [
                        tensor[block].clone() for tensor in tensors.values()
                    ]

        self.peer.pending.append((write, future))
        return future

    def submit_retrieve(self, rid, key, tensors, groups, event, blocks):
        future = MessagingFuture()

        def read(success):
            for digest, block in zip(
                key.chunk_hashes[key.start : key.end], groups[0], strict=True
            ):
                identity = (key.model_name, digest)
                assert self.peer.locks[identity] > 0
                if success:
                    for tensor, stored in zip(
                        tensors.values(), self.peer.objects[identity], strict=True
                    ):
                        tensor[block].copy_(stored)
                self.peer.locks[identity] -= 1

        self.peer.pending.append((read, future))
        return future


@pytest.fixture
def factory(monkeypatch):
    instances = []
    peer = Peer()
    monkeypatch.setattr(module.RequestClientFactory, "create", lambda url: peer)
    monkeypatch.setattr(
        module, "create_transfer_context", lambda *a, **kw: Context(peer)
    )

    def build(extra_pool=None, **options):
        entries = []
        for name in [PoolName.KV] + ([extra_pool] if extra_pool else []):
            slots = 1 if name == PoolName.MAMBA else 2
            buffers = [
                torch.arange(32 * width, device="cuda", dtype=torch.float32).view(
                    32, width
                )
                for width in (16, 32)
            ]
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=name,
                    device_pool=None,
                    components=[buffers],
                    layer_mapping={0: 0, 1: 1},
                    page_size=slots,
                    rows_are_pages=name == PoolName.MAMBA,
                )
            )
        pool_group = DevicePoolGroup(entries, 2, 2)
        monkeypatch.setattr(
            module, "resolve_hybrid_device_pool_group", lambda **kw: pool_group
        )
        linker = module.LMCacheLinker(
            Arguments(),
            Parameters(Allocator()),
            components=set(),
            extra_config={"heartbeat_interval": 3600, **options},
        )
        instances.append(linker)
        return linker, peer

    yield build
    peer.complete()
    for linker in instances:
        linker.close()


def transfer(name, keys, first_page=None, window=1):
    page = 1 if name == PoolName.MAMBA else 2
    return PoolTransfer(
        name=name,
        keys=list(keys),
        device_indices=(
            None
            if first_page is None
            else torch.arange(
                first_page * page, (first_page + len(keys)) * page, device="cuda"
            )
        ),
        hit_policy=PoolHitPolicy.ALL_PAGES
        if name == PoolName.KV
        else PoolHitPolicy.TRAILING_PAGES,
    )


def store(linker, peer, transfers):
    assert linker.offload(transfers)
    assert linker.num_completed_offloads() == 0
    peer.complete()
    assert linker.num_completed_offloads() == 1
    assert linker.pop_completed_offload()


def test_cold_store_flush_restore_into_different_slots(factory):
    linker, peer = factory()
    keys = ["a", "b", "c", "d"]
    assert linker.lookup("cold", [transfer(PoolName.KV, keys)]) == []
    store(linker, peer, [transfer(PoolName.KV, keys, 1)])
    linker.reset()
    assert linker.lookup("warm", [transfer(PoolName.KV, keys)]) == [1, 2, 3, 4]
    assert linker.load("warm", [transfer(PoolName.KV, keys, 8)])
    assert not peer.pending, "load must queue rather than start GPU writes"
    index = linker.start_layer_wise_loading()
    linker.layer_done_counter.set_consumer(index)
    assert linker.num_completed_loads() == 0
    peer.complete()
    linker.layer_done_counter.wait_until(0)
    assert linker.pop_completed_load() == ["warm"]
    for tensor in linker.pools[PoolName.KV].tensors.values():
        torch.testing.assert_close(tensor[8:12], tensor[1:5])
    assert not +peer.locks
    assert not peer.sessions


@pytest.mark.parametrize("side", [PoolName.SWA, PoolName.MAMBA])
def test_sparse_trailing_state_and_partial_load_release_unused_pins(factory, side):
    linker, peer = factory(side)
    keys = ["a", "b", "c", "d"]
    # Checkpoints exist at boundaries 2 and 4, but not 1 and 3.
    store(linker, peer, [transfer(PoolName.KV, keys[:2], 1), transfer(side, ["b"], 1)])
    store(linker, peer, [transfer(PoolName.KV, keys[2:], 3), transfer(side, ["d"], 2)])
    assert linker.lookup("r", [transfer(PoolName.KV, keys), transfer(side, ["d"])]) == [
        2,
        4,
    ]
    # Another request already installed FULL. Only the missing state is loaded.
    assert linker.load("r", [transfer(side, ["d"], 8)])
    linker.start_layer_wise_loading()
    peer.complete()
    assert linker.pop_completed_load() == ["r"]
    assert not +peer.locks
    assert not peer.sessions


def test_sliding_window_requires_every_page_at_boundary(factory):
    linker, peer = factory(PoolName.SWA)
    keys = ["a", "b", "c", "d"]
    store(
        linker,
        peer,
        [transfer(PoolName.KV, keys, 1), transfer(PoolName.SWA, keys[-2:], 1)],
    )
    assert linker.lookup(
        "r", [transfer(PoolName.KV, keys), transfer(PoolName.SWA, keys[-2:])]
    ) == [4]
    linker.release_request("r")
    assert not +peer.locks
    assert not peer.sessions


def test_local_hit_bypass_and_repeated_lookup_release_old_reservations(factory):
    linker, peer = factory()
    store(linker, peer, [transfer(PoolName.KV, ["a", "b"], 1)])
    for _ in range(3):
        assert linker.lookup("r", [transfer(PoolName.KV, ["a", "b"])]) == [1, 2]
        assert sum(peer.locks.values()) == 2
    linker.release_request("r")
    assert not +peer.locks
    assert not peer.sessions


def test_release_preserves_queued_load_but_explicit_cancel_releases(factory):
    linker, peer = factory()
    store(linker, peer, [transfer(PoolName.KV, ["a"], 1)])
    linker.lookup("r", [transfer(PoolName.KV, ["a"])])
    assert linker.load("r", [transfer(PoolName.KV, ["a"], 8)])
    linker.release_request("r")
    assert sum(peer.locks.values()) == 1
    assert linker.cancel_queued_load("r")
    assert not linker.cancel_queued_load("r")
    assert not +peer.locks
    assert not peer.sessions


def test_failed_store_is_not_acknowledged_as_persistent(factory):
    linker, peer = factory()
    assert linker.offload([transfer(PoolName.KV, ["a"], 1)])
    peer.complete(False)
    assert not linker.pop_completed_offload()
    assert linker.lookup("r", [transfer(PoolName.KV, ["a"])]) == []


def test_failed_retrieve_never_releases_tree_completion(factory):
    linker, peer = factory()
    store(linker, peer, [transfer(PoolName.KV, ["a"], 1)])
    linker.lookup("r", [transfer(PoolName.KV, ["a"])])
    linker.load("r", [transfer(PoolName.KV, ["a"], 8)])
    linker.start_layer_wise_loading()
    peer.complete(False)
    with pytest.raises(RuntimeError, match="retrieve failed"):
        linker.num_completed_loads()


def test_namespace_separates_cache_identity(factory):
    first, peer = factory(namespace="model-A")
    store(first, peer, [transfer(PoolName.KV, ["a"], 1)])
    second, _ = factory(namespace="model-B")
    assert second.lookup("r", [transfer(PoolName.KV, ["a"])]) == []


def test_close_unregisters_all_pools(factory):
    linker, peer = factory(PoolName.MAMBA)
    linker.close()
    linker.close()
    assert peer.closed
    assert all(c.closed and not c.registered for c in peer.contexts)
    with pytest.raises(RuntimeError, match="closed"):
        linker.lookup("r", [transfer(PoolName.KV, ["a"])])


def test_timed_out_lookup_retains_lease_until_close(factory, monkeypatch):
    linker, peer = factory(timeout=0.01)
    store(linker, peer, [transfer(PoolName.KV, ["a"], 1)])
    response: MessagingFuture[Any] = MessagingFuture()
    monkeypatch.setattr(peer, "query_prefetch_status", lambda rid: response)
    with pytest.raises(LMCacheTimeoutError):
        linker.lookup("r", [transfer(PoolName.KV, ["a"])])
    assert sum(peer.locks.values()) == 1
    response.set_result(1)
    linker.close()
    assert not +peer.locks
    assert not peer.sessions


def test_timed_out_unlock_is_not_submitted_twice(factory, monkeypatch):
    linker, peer = factory(timeout=0.01)
    store(linker, peer, [transfer(PoolName.KV, ["a"], 1)])
    linker.lookup("r", [transfer(PoolName.KV, ["a"])])
    response: MessagingFuture[Any] = MessagingFuture()
    unlock = peer.free_lookup_locks
    calls = []

    def delayed_ack(key, tp_size):
        calls.append(key)
        unlock(key, tp_size)
        return response

    monkeypatch.setattr(peer, "free_lookup_locks", delayed_ack)
    with pytest.raises(LMCacheTimeoutError):
        linker.release_request("r")
    response.set_result(None)
    linker.close()
    assert len(calls) == 1
    assert not +peer.locks
    assert not peer.sessions
