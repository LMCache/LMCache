# SPDX-License-Identifier: Apache-2.0
"""A blend lookup sees every store acknowledged before it.

A store's L1 commit and fingerprint enqueue run as stream-ordered callbacks
after the STORE RPC has returned, and the fingerprints are indexed on a
background thread after that. These tests force each window
deterministically -- the stream callbacks are recorded and fired by hand, and
the background drainer is held on an event -- and check that a lookup in the
window defers instead of completing with a miss.

Runs a real BlendModule (real matcher, sessions and token hasher) under a
mocked transfer module and storage manager; no CUDA.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import time

# Third Party
import msgspec
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchResult
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.blend import store as store_mod
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

_CHUNK = 4
_N_CHUNKS = 4
_STORED = list(range(1000, 1000 + _N_CHUNKS * _CHUNK))
# The stored chunks shifted by one chunk: chunk 0 of a position-0 store is the
# prefix leg's, so the other chunks are the fingerprint matches.
_QUERY = list(range(50_000, 50_000 + _CHUNK)) + _STORED
_EXPECTED_MATCHES = _N_CHUNKS - 1


class _AllFoundStorageManager:
    """Every prefetched key is found; read locks are not tracked."""

    def submit_prefetch_task(self, spec, external_request_id=None):
        handle = MagicMock()
        handle.key_groups = list(spec.key_groups)
        handle.total_requested_keys = 0
        return handle

    def query_prefetch_status(self, handle):
        def rows():
            return [Bitmap(len(row.keys), len(row.keys)) for row in handle.key_groups]

        return PrefetchResult(
            hit_cells=rows(),
            l1_hit_cells=rows(),
            l2_hit_cells=[Bitmap(len(row.keys)) for row in handle.key_groups],
        )

    def finish_read_prefetched(self, keys, read_locks: int = 1, l1_owners=None):
        pass


class _Rig:
    """A BlendModule whose stream callbacks are recorded, not run."""

    def __init__(self, monkeypatch, stop_drainer: bool) -> None:
        ctx = MagicMock()
        ctx.chunk_size = _CHUNK
        ctx.storage_manager = _AllFoundStorageManager()
        ctx.event_bus.has_subscribers.return_value = False
        # Prefix leg: no full chunk hashes -> 0 prefix coverage.
        ctx.token_hasher.compute_chunk_hashes.return_value = []
        ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
        ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
            num_chunks_in_sw=[-1], world_size=1, group_kinds=("attention",)
        )
        ctx.session_manager = SessionManager(
            hasher=TokenHasher(chunk_size=_CHUNK), cleanup_interval=None
        )
        self.transfer = MagicMock()
        self.transfer.store_with_chunk_mask.return_value = (
            b"evt",
            True,
            [True] * _N_CHUNKS,
        )
        self.transfer.get_and_touch_context_entry.return_value = SimpleNamespace(
            cache_context=SimpleNamespace(cupy_stream=object())
        )
        self.recorded: list[tuple[str, bytes]] = []
        monkeypatch.setattr(
            store_mod,
            "submit_callback_to_stream",
            lambda stream, kind, payload: self.recorded.append(
                (kind, msgspec.msgpack.encode(payload))
            ),
        )
        self.blend = BlendModule(ctx, lmcache_driven_transfer=self.transfer)
        if stop_drainer:
            self.stop_drainer()

    def stop_drainer(self) -> None:
        self.blend._fingerprint_stop.set()
        self.blend._fingerprint_worker.join(timeout=5.0)
        assert not self.blend._fingerprint_worker.is_alive()

    def close(self) -> None:
        self.blend.close()
        self.blend._fingerprint_worker.join(timeout=5.0)

    def store(self, request_id: str, worker_id: int = 0) -> None:
        key = IPCCacheServerKey(
            model_name="m",
            world_size=1,
            num_kv_readers=1,
            worker_id=worker_id,
            token_ids=tuple(_STORED),
            start=0,
            end=len(_STORED),
            request_id=request_id,
        )
        assert self.blend.store(key, 1, [[0]], b"") == (b"evt", True)

    def fire(self, count: int | None = None) -> None:
        """Run recorded callbacks in stream order, as the dispatcher does:
        decoded with the registered payload type, handed to its handler."""
        handlers = {
            c.args[0]: (c.args[1], c.args[2])
            for c in self.transfer.register_host_func.call_args_list
        }
        n = len(self.recorded) if count is None else count
        fired, self.recorded = self.recorded[:n], self.recorded[n:]
        for kind, payload in fired:
            handler, payload_type = handlers[kind]
            handler(msgspec.msgpack.decode(payload, type=payload_type))

    def lookup(self, request_id: str):
        key = IPCCacheServerKey(
            model_name="m",
            world_size=1,
            num_kv_readers=1,
            worker_id=None,
            token_ids=tuple(_QUERY),
            start=0,
            end=len(_QUERY),
            request_id=request_id,
        )
        return self.blend.cb_unified_lookup(key, tp_size=1)


@pytest.fixture
def rig(monkeypatch):
    made: list[_Rig] = []

    def make(stop_drainer: bool = True) -> _Rig:
        made.append(_Rig(monkeypatch, stop_drainer))
        return made[-1]

    yield make
    for r in made:
        r.close()


def _assert_all_matched(result) -> None:
    assert result is not None
    assert result.prefix_coverage_tokens == 0
    assert len(result.non_prefix_segments) == _EXPECTED_MATCHES


def test_lookup_defers_until_an_acknowledged_store_commits(rig):
    """The STORE RPC has returned but its stream callbacks (L1 commit,
    fingerprint enqueue) have not run: the lookup defers rather than
    completing with no match, and finds every chunk once they run."""
    r = rig()
    r.store("req-store")
    assert r.recorded, "the store must submit stream-ordered callbacks"

    assert r.lookup("req-blend") is None

    r.fire()
    _assert_all_matched(r.lookup("req-blend"))


def test_lookup_defers_while_the_drainer_holds_a_fingerprint_job(rig):
    """The background drainer has dequeued the store's fingerprint job but
    not indexed it yet: the queue is empty, so draining it at lookup time is
    not enough. The lookup must still wait for the job."""
    r = rig(stop_drainer=False)
    matcher = r.blend._token_range_matcher
    index = matcher.on_new_token_hashes
    entered, release = threading.Event(), threading.Event()

    def held_index(*args, **kwargs):
        if threading.current_thread() is r.blend._fingerprint_worker:
            entered.set()
            assert release.wait(timeout=30.0)
        return index(*args, **kwargs)

    matcher.on_new_token_hashes = held_index
    r.store("req-store")
    r.fire()
    assert entered.wait(timeout=30.0), "the drainer never picked up the job"
    assert r.blend._fingerprint_queue.empty()

    assert r.lookup("req-blend") is None

    release.set()
    deadline = time.monotonic() + 30.0
    result = r.lookup("req-blend")
    while result is None and time.monotonic() < deadline:
        time.sleep(0.01)
        result = r.lookup("req-blend")
    _assert_all_matched(result)


def test_lookup_defers_until_every_rank_s_store_commits(rig):
    """Only worker 0 registers fingerprints, but a chunk is found only when
    every rank's key is in L1: the lookup also waits for another rank's
    store to commit."""
    r = rig()
    r.store("req-store", worker_id=0)
    r.fire()
    r.store("req-store", worker_id=1)
    assert r.recorded, "a worker-1 store must also mark its landing"

    assert r.lookup("req-blend") is None

    r.fire()
    _assert_all_matched(r.lookup("req-blend"))


def test_lookup_does_not_wait_for_stores_after_it(rig):
    """The lookup's ticket is fixed at its first call: a store acknowledged
    later never delays it."""
    r = rig()
    r.store("req-store")
    assert r.lookup("req-blend") is None
    n_first = len(r.recorded)

    r.store("req-later")
    r.fire(n_first)  # only the first store lands

    _assert_all_matched(r.lookup("req-blend"))
    assert r.blend._unlanded_stores, "the later store is still in flight"


def test_lookup_proceeds_past_the_visibility_timeout(rig):
    """A store that never lands delays a lookup by the bound, not forever."""
    r = rig()
    r.blend._STORE_VISIBILITY_TIMEOUT_S = 0.0
    r.store("req-store")

    result = r.lookup("req-blend")

    assert result is not None
    assert result.non_prefix_segments == []
    assert r.blend._cb_landing_tickets == {}


def test_store_without_device_work_lands_at_once(rig):
    """A store rejected before any device work (unregistered instance)
    submits no callback, so nothing would ever release it later."""
    r = rig()
    r.transfer.store_with_chunk_mask.return_value = (b"", False, [])
    key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=1,
        worker_id=0,
        token_ids=tuple(_STORED),
        start=0,
        end=len(_STORED),
        request_id="req-rejected",
    )

    assert r.blend.store(key, 1, [[0]], b"") == (b"", False)

    assert r.recorded == []
    assert r.blend._unlanded_stores == {}


def test_failed_store_releases_its_landing(rig):
    """A store that raises still releases its landing."""
    r = rig()
    r.transfer.store_with_chunk_mask.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        r.store("req-failed")

    assert r.blend._unlanded_stores == {}
    assert r.lookup("req-blend") is not None
