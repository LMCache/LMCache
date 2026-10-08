# SPDX-License-Identifier: Apache-2.0
"""Multi-process (gloo) harness for LMCacheEngine in PCP shard store mode.

Run: python engine_harness.py <scenario> <world_size>
Spawns world_size processes; each builds a real LMCacheEngine (LocalCPUBackend
L1, fake GPU connector, gloo broadcast functions) and runs the scenario.
Exit code 0 only if every rank passes. Every rank prints one
"RESULT <rank> <json>" line.
"""

# Standard
from datetime import timedelta
import json
import os
import random
import sys
import threading
import traceback

os.environ.setdefault("LMCACHE_TRACK_USAGE", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

# Third Party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CHUNK = 16
HEAD = 8  # kv_shape[3] * kv_shape[4]
LAYERS = 2


# ------------------------------------------------------------------ fakes
def kv_for(seq: int, start: int, shape) -> torch.Tensor:
    """Deterministic 'GPU KV' of tokens [start, start+n) of sequence seq."""
    n = 1
    for d in shape:
        n *= d
    return (torch.arange(n, dtype=torch.float32) + seq * 1e6 + start * 1e3).reshape(
        shape
    )


class FakeGPUConnector:
    """batched_from_gpu fills chunks with kv_for(); batched_to_gpu records them.
    No load_stream attribute (the engine then synchronizes its own stream)."""

    def __init__(self):
        self.loaded = {}  # start -> (end, tensor)
        self.to_gpu_calls = 0

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        seq = kwargs["seq"]
        for mo, s, _ in zip(memory_objs, starts, ends, strict=True):
            t = mo.tensor
            t.copy_(kv_for(seq, s, t.shape))

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        self.to_gpu_calls += 1
        for mo, s, e in zip(memory_objs, starts, ends, strict=True):
            self.loaded[s] = (e, mo.tensor.detach().cpu().clone())

    def get_shape(self, num_tokens):
        return torch.Size([1, LAYERS, num_tokens, HEAD])


def bcast(tensor, src):
    dist.broadcast(tensor, src)


def bcast_obj(obj, src):
    box = [obj]
    dist.broadcast_object_list(box, src)
    return box[0]


def make_engine(
    rank,
    world,
    shard=True,
    max_cpu_gb=0.05,
    extra=None,
    unfull=False,
    async_loading=False,
):
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK,
        local_cpu=True,
        max_local_cpu_size=max_cpu_gb,
        lmcache_instance_id="pcp_shard_test",
    )
    config.save_unfull_chunk = unfull
    config.enable_async_loading = async_loading
    config.extra_config = {"save_only_first_rank": True}
    if shard:
        config.extra_config["pcp_shard_store"] = True
    if extra:
        config.extra_config.update(extra)
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=world,
        local_world_size=world,
        worker_id=rank,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(LAYERS, 1, CHUNK, 1, HEAD),
        use_mla=True,
        role="worker",
    )
    connector = FakeGPUConnector()
    engine = LMCacheEngine(
        config,
        metadata,
        ChunkedTokenDatabase(config, metadata),
        connector,
        bcast,
        bcast_obj,
    )
    engine.post_init()
    return engine, connector


def tokens_for(seq: int, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seq)
    return torch.randint(0, 50000, (n,), generator=g)


def keys_of(engine, tokens):
    return [
        (s, e, k) for s, e, k in engine.token_database.process_tokens(tokens=tokens)
    ]


def combined_lookup(engine, tokens, lookup_id, pin=True):
    """What the scheduler sees: min over every rank's lookup server answer."""
    mine = engine.lookup(tokens=tokens, lookup_id=lookup_id, pin=pin)
    allres = [None] * dist.get_world_size()
    dist.all_gather_object(allres, mine)
    return min(allres), allres


def check_loaded(connector, seq, ranges):
    """Every chunk in ranges loaded with the right bytes."""
    for s, e in ranges:
        assert s in connector.loaded, f"chunk at {s} not loaded"
        le, t = connector.loaded[s]
        assert le == e, (s, le, e)
        exp = kv_for(seq, s, t.shape)
        assert torch.equal(t, exp), f"chunk at {s}: wrong data"


def chunk_ranges(n_tokens, start=0, stop=None):
    stop = n_tokens if stop is None else stop
    return [(s, min(s + CHUNK, n_tokens)) for s in range(start, stop, CHUNK)]


def l1_keys(engine):
    be = engine.storage_manager.storage_backends["LocalCPUBackend"]
    return set(be.hot_cache.keys())


# ------------------------------------------------------------------ scenarios
def sc_full(rank, world):
    """Store -> each rank holds only owned chunks -> lookup prefix = all ->
    retrieve loads every chunk on every rank with the right bytes."""
    engine, conn = make_engine(rank, world, unfull=True)
    n = 10 * CHUNK + 7  # 11 chunks, last one partial (owner 10 % world)
    toks = tokens_for(1, n)
    engine.store(toks, seq=1)
    stored = {k.chunk_hash for k in l1_keys(engine)}
    chunks = keys_of(engine, toks)
    owned = {k.chunk_hash for s, _, k in chunks if (s // CHUNK) % world == rank}
    assert stored == owned, (len(stored), len(owned))

    hit, per_rank = combined_lookup(engine, toks, "req-1")
    assert hit == n, (hit, per_rank)
    assert sum(len(v) for v in engine.lookup_pins["req-1"].values()) == len(owned)

    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="req-1")
    assert bool(ret.all()), ret
    check_loaded(conn, 1, chunk_ranges(n))
    engine.lookup_unpin("req-1")
    assert "req-1" not in engine.lookup_pins
    be = engine.storage_manager.storage_backends["LocalCPUBackend"]
    for k, mo in be.hot_cache.items():
        assert mo.get_ref_count() == 1 and not mo.is_pinned, (
            k,
            mo.get_ref_count(),
            mo.metadata.pin_count,
        )
    return {"lookup": per_rank, "retrieved": int(ret.sum())}


def sc_mask(rank, world):
    """vLLM already holds the first 2 chunks (mask False): only the rest load."""
    engine, conn = make_engine(rank, world)
    n = 9 * CHUNK
    toks = tokens_for(2, n)
    engine.store(toks, seq=2)
    mask = torch.ones(n, dtype=torch.bool)
    mask[: 2 * CHUNK] = False
    ret = engine.retrieve(toks, mask, req_id="req-2")
    assert not bool(ret[: 2 * CHUNK].any())
    assert bool(ret[2 * CHUNK :].all())
    check_loaded(conn, 2, chunk_ranges(n, 2 * CHUNK))
    assert 0 not in conn.loaded and CHUNK not in conn.loaded
    return {"retrieved": int(ret.sum())}


def sc_evict(rank, world):
    """Owner of chunk 5 loses it: lookup and retrieve both stop at chunk 5."""
    engine, conn = make_engine(rank, world)
    n = 12 * CHUNK
    toks = tokens_for(3, n)
    engine.store(toks, seq=3)
    chunks = keys_of(engine, toks)
    if rank == 5 % world:
        assert engine.storage_manager.remove(chunks[5][2]) > 0
    dist.barrier()
    hit, per_rank = combined_lookup(engine, toks, "req-3")
    assert hit == 5 * CHUNK, (hit, per_rank)
    # The adapter would ask for tokens[:hit]; ask for everything to check
    # that ranks still agree when owners beyond the hole have their chunks.
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="req-3")
    assert int(ret.sum()) == 5 * CHUNK and bool(ret[: 5 * CHUNK].all())
    check_loaded(conn, 3, chunk_ranges(n, 0, 5 * CHUNK))
    assert 5 * CHUNK not in conn.loaded
    engine.lookup_unpin("req-3")
    return {"lookup": per_rank, "retrieved": int(ret.sum())}


def sc_fetch_failure(rank, world):
    """contains() says present but the owner's get fails (None) or raises:
    every rank agrees on the same shorter prefix, nobody hangs."""
    engine, conn = make_engine(rank, world)
    n = 12 * CHUNK
    toks = tokens_for(4, n)
    engine.store(toks, seq=4)
    chunks = keys_of(engine, toks)
    sm = engine.storage_manager
    orig_get = sm.batched_get

    # round 1: owner of chunk 6 gets None for chunk 6
    bad = chunks[6][2]

    def get_none(keys, location=None):
        res = orig_get(keys, location)
        out = []
        for k, mo in zip(keys, res, strict=True):
            if k == bad and mo is not None:
                mo.ref_count_down()
                mo = None
            out.append(mo)
        return out

    if rank == 6 % world:
        sm.batched_get = get_none
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="r4a")
    sm.batched_get = orig_get
    assert int(ret.sum()) == 6 * CHUNK, int(ret.sum())
    check_loaded(conn, 4, chunk_ranges(n, 0, 6 * CHUNK))

    # round 2: owner of chunk 3 raises
    conn.loaded.clear()

    def get_raise(keys, location=None):
        raise RuntimeError("injected storage failure")

    if rank == 3 % world:
        sm.batched_get = get_raise
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="r4b")
    sm.batched_get = orig_get
    assert int(ret.sum()) == 3 * CHUNK, int(ret.sum())
    check_loaded(conn, 4, chunk_ranges(n, 0, 3 * CHUNK))

    # round 3: everything back to normal, full load (collectives still aligned)
    conn.loaded.clear()
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="r4c")
    assert bool(ret.all())
    check_loaded(conn, 4, chunk_ranges(n))
    for mo in engine.storage_manager.storage_backends[
        "LocalCPUBackend"
    ].hot_cache.values():
        assert mo.get_ref_count() == 1, mo.get_ref_count()
    return {"ok": True}


def sc_unhealthy_stage(rank, world):
    """An unhealthy rank and a rank whose device staging fails still join the
    collectives: everyone agrees on the shorter prefix, nobody hangs."""
    # First Party
    from lmcache.v1.memory_management import MemoryObjMetadata

    engine, conn = make_engine(rank, world)
    n = 12 * CHUNK
    toks = tokens_for(7, n)
    engine.store(toks, seq=7)
    ones = torch.ones(n, dtype=torch.bool)

    # round 1: rank 2 unhealthy -> its first owned chunk (2) is the limit
    if rank == 2:
        engine.is_healthy = lambda: False
    ret = engine.retrieve(toks, ones, req_id="r7a")
    if rank == 2:
        del engine.is_healthy
    assert int(ret.sum()) == 2 * CHUNK, int(ret.sum())
    check_loaded(conn, 7, chunk_ranges(n, 0, 2 * CHUNK))

    # round 2: rank 1 fails to stage its 2nd owned chunk (chunk 1 + world)
    conn.loaded.clear()
    orig_to_dict = MemoryObjMetadata.to_dict
    calls = {"n": 0}

    def flaky_to_dict(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected staging failure")
        return orig_to_dict(self)

    if rank == 1:
        MemoryObjMetadata.to_dict = flaky_to_dict
    ret = engine.retrieve(toks, ones, req_id="r7b")
    MemoryObjMetadata.to_dict = orig_to_dict
    assert int(ret.sum()) == (1 + world) * CHUNK, int(ret.sum())
    check_loaded(conn, 7, chunk_ranges(n, 0, (1 + world) * CHUNK))

    # round 3: back to normal
    conn.loaded.clear()
    ret = engine.retrieve(toks, ones, req_id="r7c")
    assert bool(ret.all())
    check_loaded(conn, 7, chunk_ranges(n))
    for mo in engine.storage_manager.storage_backends[
        "LocalCPUBackend"
    ].hot_cache.values():
        assert mo.get_ref_count() == 1, mo.get_ref_count()
    return {"ok": True}


def sc_mismatch(rank, world):
    """Ranks asked to load different chunk lists: all load nothing, no hang,
    and the next retrieve works."""
    engine, conn = make_engine(rank, world)
    n = 8 * CHUNK
    toks = tokens_for(5, n)
    engine.store(toks, seq=5)
    other = toks.clone()
    other[3 * CHUNK] += 1  # changes hashes of chunks 3..7
    mine = other if rank == 1 else toks
    ret = engine.retrieve(mine, torch.ones(n, dtype=torch.bool), req_id="r5a")
    assert int(ret.sum()) == 0
    assert conn.to_gpu_calls == 0
    # shorter list on one rank
    mine = toks[: 4 * CHUNK] if rank == 0 else toks
    ret = engine.retrieve(mine, torch.ones(len(mine), dtype=torch.bool), req_id="r5b")
    assert int(ret.sum()) == 0
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="r5c")
    assert bool(ret.all())
    check_loaded(conn, 5, chunk_ranges(n))
    return {"ok": True}


def sc_random(rank, world):
    """Random prompts sharing prefixes, random evictions on random ranks.
    Each step: lookup min == reference prefix computed from global presence;
    retrieve of tokens[:hit] loads exactly that prefix with right bytes on
    every rank."""
    engine, conn = make_engine(rank, world, max_cpu_gb=0.2)
    rng = random.Random(1234)  # same stream on every rank
    base = tokens_for(100, 40 * CHUNK)
    seqs = []
    for i in range(8):
        n = rng.randint(1, 40 * CHUNK)
        toks = base[:n].clone()
        if i % 2:
            cut = rng.randint(0, n - 1)
            toks[cut:] = tokens_for(200 + i, n - cut)
        seqs.append(toks)
    # KV must be a function of the token prefix for shared prefixes to match:
    # store all with seq id 0 (kv_for depends only on start)
    for toks in seqs:
        engine.store(toks, seq=0)
    hits = []
    for step in range(25):
        toks = seqs[rng.randrange(len(seqs))]
        chunks = keys_of(engine, toks)
        # random eviction of one chunk of this prompt on its owner
        if chunks and rng.random() < 0.5:
            j = rng.randrange(len(chunks))
            if (chunks[j][0] // CHUNK) % world == rank:
                engine.storage_manager.remove(chunks[j][2])
        dist.barrier()
        mine_present = [
            engine.storage_manager.contains(k) is not None
            and bool(engine.storage_manager.contains(k))
            for s, _, k in chunks
            if (s // CHUNK) % world == rank
        ]
        gathered = [None] * world
        dist.all_gather_object(gathered, mine_present)
        present = []
        idx = [0] * world
        for s, _, _ in chunks:
            o = (s // CHUNK) % world
            present.append(gathered[o][idx[o]])
            idx[o] += 1
        expect = 0
        for (s, e, _), ok in zip(chunks, present, strict=True):
            if not ok:
                break
            expect = e
        rid = f"rand-{step}"
        hit, per_rank = combined_lookup(engine, toks, rid)
        assert hit == expect, (step, hit, expect, per_rank)
        hits.append((len(toks), hit))
        conn.loaded.clear()
        if hit > 0:
            ret = engine.retrieve(
                toks[:hit], torch.ones(hit, dtype=torch.bool), req_id=rid
            )
            assert int(ret.sum()) == hit, (step, int(ret.sum()), hit)
            check_loaded(conn, 0, chunk_ranges(hit))
        engine.lookup_unpin(rid)
    return {"len_hit": hits}


def sc_cpu_budget(rank, world):
    """L1 per rank = max_local_cpu_size / world_size (or the per-rank key)."""
    engine, _ = make_engine(rank, world, max_cpu_gb=0.08)
    be = engine.storage_manager.storage_backends["LocalCPUBackend"]
    alloc = be.memory_allocator
    size = _alloc_size(alloc)
    expect = int(0.08 / world * 1024**3)
    assert abs(size - expect) <= 4096 * 1024, (size, expect)
    return {"l1_bytes": size}


def _alloc_size(alloc):
    for attr in ("pin_allocator", "allocator"):
        sub = getattr(alloc, attr, None)
        if sub is not None and sub is not alloc:
            return _alloc_size(sub)
    for attr in ("total_size", "size", "capacity", "buffer_size"):
        v = getattr(alloc, attr, None)
        if isinstance(v, int):
            return v
    buf = getattr(alloc, "buffer", None)
    if buf is not None:
        return buf.numel() * buf.element_size()
    raise AssertionError(f"cannot find allocator size: {type(alloc)} {vars(alloc)}")


def sc_default(rank, world):
    """Mode off: rank 0 stores everything, other ranks nothing; the original
    broadcast path loads every chunk on every rank."""
    engine, conn = make_engine(rank, world, shard=False, unfull=True)
    assert engine._pcp_shard is False
    n = 7 * CHUNK + 3
    toks = tokens_for(6, n)
    engine.store(toks, seq=6)
    if rank == 0:
        assert len(l1_keys(engine)) == 8
        hit = engine.lookup(tokens=toks, lookup_id="r6", pin=True)
        assert hit == n
    else:
        assert engine.storage_manager is None
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="r6")
    assert bool(ret.all())
    check_loaded(conn, 6, chunk_ranges(n))
    if rank == 0:
        engine.lookup_unpin("r6")
    return {"ok": True}


SCENARIOS = {
    "full": sc_full,
    "mask": sc_mask,
    "evict": sc_evict,
    "fetch_failure": sc_fetch_failure,
    "mismatch": sc_mismatch,
    "unhealthy_stage": sc_unhealthy_stage,
    "random": sc_random,
    "cpu_budget": sc_cpu_budget,
    "default": sc_default,
}


# ------------------------------------------------------------ async loading
class _ReplyStub:
    """Stands in for LMCacheAsyncLookupServer: records what this rank would
    send to the scheduler's async lookup client."""

    def __init__(self):
        self.replies = {}
        self.cv = threading.Condition()

    def send_response_to_scheduler(self, lookup_id, num_hit_tokens):
        with self.cv:
            self.replies[lookup_id] = num_hit_tokens
            self.cv.notify_all()

    def wait(self, lookup_id, timeout=30):
        with self.cv:
            assert self.cv.wait_for(lambda: lookup_id in self.replies, timeout), (
                f"no async lookup reply for {lookup_id}"
            )
            return self.replies[lookup_id]


def make_async_engine(rank, world, **kw):
    engine, conn = make_engine(rank, world, async_loading=True, **kw)
    stub = _ReplyStub()
    engine.storage_manager.async_lookup_server = stub
    return engine, conn, stub


def async_combined_lookup(engine, stub, tokens, lookup_id):
    """What LMCacheAsyncLookupClient sees: min over every rank's async reply."""
    engine.async_lookup_and_prefetch(lookup_id=lookup_id, tokens=tokens, pin=True)
    mine = stub.wait(lookup_id)
    allres = [None] * dist.get_world_size()
    dist.all_gather_object(allres, mine)
    return min(allres), allres


def _assert_l1_released(engine):
    be = engine.storage_manager.storage_backends["LocalCPUBackend"]
    for k, mo in be.hot_cache.items():
        assert mo.get_ref_count() == 1 and not mo.is_pinned, (
            k,
            mo.get_ref_count(),
            mo.metadata.pin_count,
        )


def sc_async_full(rank, world):
    """Async lookup + prefetch: each rank prefetches its owned chunks, the
    min over ranks is the whole prompt, the load is byte-exact, and nothing
    stays pinned or referenced afterwards."""
    engine, conn, stub = make_async_engine(rank, world, unfull=True)
    n = 10 * CHUNK + 7
    toks = tokens_for(11, n)
    engine.store(toks, seq=11)
    hit, per_rank = async_combined_lookup(engine, stub, toks, "areq-1")
    assert hit == n, (hit, per_rank)
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="areq-1")
    assert bool(ret.all()), ret
    check_loaded(conn, 11, chunk_ranges(n))
    engine.lookup_unpin("areq-1")
    _assert_l1_released(engine)
    return {"lookup": per_rank, "retrieved": int(ret.sum())}


def sc_async_evict(rank, world):
    """Owner of chunk 5 lost it: every rank agrees on 5 chunks."""
    engine, conn, stub = make_async_engine(rank, world)
    n = 12 * CHUNK
    toks = tokens_for(12, n)
    engine.store(toks, seq=12)
    chunks = keys_of(engine, toks)
    if rank == 5 % world:
        assert engine.storage_manager.remove(chunks[5][2]) > 0
    dist.barrier()
    hit, per_rank = async_combined_lookup(engine, stub, toks, "areq-2")
    assert hit == 5 * CHUNK, (hit, per_rank)
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="areq-2")
    assert int(ret.sum()) == 5 * CHUNK and bool(ret[: 5 * CHUNK].all())
    check_loaded(conn, 12, chunk_ranges(n, 0, 5 * CHUNK))
    engine.lookup_unpin("areq-2")
    _assert_l1_released(engine)
    return {"lookup": per_rank, "retrieved": int(ret.sum())}


def sc_async_owner_miss(rank, world):
    """Rank 1 has no hit at all (its first owned chunk, chunk 1, is gone).
    It must answer the start of chunk 1, not 0, so the prefix is chunk 0."""
    engine, conn, stub = make_async_engine(rank, world)
    n = 6 * CHUNK
    toks = tokens_for(13, n)
    engine.store(toks, seq=13)
    chunks = keys_of(engine, toks)
    if rank == 1:
        for pos in range(1, len(chunks), world):
            assert engine.storage_manager.remove(chunks[pos][2]) > 0
    dist.barrier()
    hit, per_rank = async_combined_lookup(engine, stub, toks, "areq-3")
    assert hit == CHUNK, (hit, per_rank)
    if rank == 1:
        assert per_rank[1] == CHUNK, per_rank
    ret = engine.retrieve(toks, torch.ones(n, dtype=torch.bool), req_id="areq-3")
    assert int(ret.sum()) == CHUNK and bool(ret[:CHUNK].all())
    check_loaded(conn, 13, chunk_ranges(n, 0, CHUNK))
    engine.lookup_unpin("areq-3")
    _assert_l1_released(engine)
    return {"lookup": per_rank, "retrieved": int(ret.sum())}


SCENARIOS.update(
    {
        "async_full": sc_async_full,
        "async_evict": sc_async_evict,
        "async_owner_miss": sc_async_owner_miss,
    }
)


def _worker(rank, world, scenario, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60)
        )
        res = SCENARIOS[scenario](rank, world)
        print(f"RESULT {rank} {json.dumps(res)}", flush=True)
        q.put((rank, "ok", ""))
    except BaseException:
        q.put((rank, "fail", traceback.format_exc()))
    finally:
        # The engine's storage threads are not daemons: leave hard once the
        # result is flushed.
        sys.stdout.flush()
        sys.stderr.flush()
        q.close()
        q.join_thread()
        os._exit(0)


def main():
    scenario, world = sys.argv[1], int(sys.argv[2])
    port = 29500 + random.Random(os.getpid()).randint(0, 2000)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, scenario, port, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = []
    for _ in range(world):
        results.append(q.get(timeout=240))
    for p in procs:
        p.join(timeout=30)
    failed = [r for r in results if r[1] != "ok"]
    for rank, _, tb in sorted(failed):
        print(f"rank {rank} FAILED:\n{tb}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
