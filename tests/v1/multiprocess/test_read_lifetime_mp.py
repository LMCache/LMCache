# SPDX-License-Identifier: Apache-2.0
"""Focused unit & transport tests for PR #2: MP lookup_id session isolation."""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.common.custom_types import READ_LIFETIME_V1, IPCCacheServerKey
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import ActiveReadCompletion
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.read_completion import ActiveReadCompletionTracker
from lmcache.v1.platform.stream import CompletionEvent
from tests.v1.multiprocess.transport_test_utils import (
    request_server_url,
    start_lookup_request_server,
)


def _make_lookup_env(read_ttl: int = 5):
    ctx = MPCacheServerContext(
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=1024 * 1024, use_lazy=False, shm_name=""
                ),
                read_ttl_seconds=read_ttl,
                write_ttl_seconds=5,
            ),
            eviction_config=EvictionConfig("noop"),
        ),
        chunk_size=4,
        hash_algorithm="blake3",
        session_ttl_seconds=60.0,
    )
    layout = MemoryLayoutDesc(shapes=[torch.Size([16])], dtypes=[torch.uint8])
    ctx.layout_desc_registry.register(
        "m", 1, layout, chunk_size=4, hash_algorithm="blake3"
    )
    return LookupModule(ctx), ctx, ctx.storage_manager


def _seed(sm, ctx, tokens: list[int]) -> list[ObjectKey]:
    l1 = sm.l1_managers()[0]
    hasher = ctx.layout_desc_registry.get_token_hasher("m", 1)
    hashes = hasher.compute_chunk_hashes(tokens)
    keys = [ObjectKey(h, "m", ObjectKey.ComputeKVRank(1, 0, 1, 0), 0) for h in hashes]
    layout = MemoryLayoutDesc(shapes=[torch.Size([16])], dtypes=[torch.uint8])
    res = l1.reserve_write(keys, [False] * len(keys), layout, "seed")
    for _, obj in res.values():
        assert obj is not None
        obj.tensor.fill_(7)
    l1.finish_write(keys, "seed")
    return keys


def test_named_lookup_isolation_and_tombstone_fence() -> None:
    """Named lookups isolate same/different requests and fence delayed LOOKUP."""
    lookup_mod, ctx, sm = _make_lookup_env(read_ttl=1)
    try:
        tokens = list(range(8))
        keys = _seed(sm, ctx, tokens)
        l1 = sm.l1_managers()[0]

        # Lookup A expires; Lookup B stays isolated and intact.
        k_a = IPCCacheServerKey.from_token_ids(
            "m", 1, None, tokens, start=0, end=8, request_id="req-A", lookup_id="lkp-A"
        )
        lookup_mod.lookup(k_a, 1)
        assert lookup_mod.wait_prefetch_status("req-A", 2.0, "lkp-A") == 8
        time.sleep(1.1)

        k_b = IPCCacheServerKey.from_token_ids(
            "m", 1, None, tokens, start=0, end=8, request_id="req-B", lookup_id="lkp-B"
        )
        lookup_mod.lookup(k_b, 1)
        assert lookup_mod.wait_prefetch_status("req-B", 2.0, "lkp-B") == 8

        # Tearing down expired A never affects B's named leases.
        lookup_mod.end_session("req-A", "lkp-A")
        assert all(not l1.is_key_evictable(k) for k in keys)

        # B promotes its leases cleanly, then releasing frees them.
        owners_b = ctx.get_read_owners(k_b.session_key)
        objs, comp = sm.acquire_read(
            keys, read_owner=k_b.read_owner, consumer_id=0, l1_owners=owners_b
        )
        assert objs is not None and len(objs) == 2
        lookup_mod.end_session("req-B", "lkp-B")
        assert all(not l1.is_key_evictable(k) for k in keys)
        sm.finish_active_reads(comp)
        assert all(l1.is_key_evictable(k) for k in keys)

        # Delayed LOOKUP arriving AFTER END_SESSION is fenced without leaking.
        lookup_mod.end_session("req-late", "lkp-late")
        k_late = IPCCacheServerKey.from_token_ids(
            "m",
            1,
            None,
            tokens,
            start=0,
            end=8,
            request_id="req-late",
            lookup_id="lkp-late",
        )
        lookup_mod.lookup(k_late, 1)
        assert ctx.session_manager.get(k_late.session_key) is None
        assert all(l1.is_key_evictable(k) for k in keys)
    finally:
        lookup_mod.close()
        ctx.close()


def test_engine_driven_rejects_lookup_id() -> None:
    """Engine-driven prepare_retrieve fails fast when lookup_id is non-empty."""
    _lookup, ctx, _sm = _make_lookup_env()
    mod = EngineDrivenTransferModule(ctx)
    try:
        key = IPCCacheServerKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_bytes=pack_token_ids([1, 2, 3, 4]),
            start=0,
            end=4,
            request_id="req-ed",
            lookup_id="lkp-1",
        )
        with pytest.raises(ValueError, match="lmcache_driven"):
            mod.prepare_retrieve(key, 1)
    finally:
        mod.close()
        ctx.close()


def test_active_read_completion_tracker_poll_and_close_drain() -> None:
    """ActiveReadCompletionTracker polls events and synchronously drains on close."""
    released: list[ActiveReadCompletion] = []
    tracker = ActiveReadCompletionTracker(
        on_complete=released.append, poll_interval_seconds=60.0
    )
    done_flag = False
    spec = MagicMock()
    spec.is_stream_event_complete.side_effect = lambda _ev: done_flag
    fake_event = CompletionEvent(spec, object())
    fake_dev = MagicMock(type="cuda")
    fake_stream = object()

    c1 = ActiveReadCompletion(((0, (1,)),))
    c2 = ActiveReadCompletion(((0, (2,)),))
    with (
        patch(
            "lmcache.v1.multiprocess.read_completion.record_completion_event",
            return_value=fake_event,
        ),
        patch(
            "lmcache.v1.multiprocess.read_completion.synchronize_stream"
        ) as mock_sync,
    ):
        tracker.defer(c1, fake_dev, fake_stream)
        assert tracker.pending_count() == 1
        assert tracker.poll() == 0

        done_flag = True
        assert tracker.poll() == 1
        assert released == [c1]

        done_flag = False
        tracker.defer(c2, fake_dev, fake_stream)
        tracker.close()
        mock_sync.assert_called_once_with(fake_dev, fake_stream)
        assert released == [c1, c2]
        assert tracker.pending_count() == 0


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_lookup_id_round_trips_over_zmq_and_grpc(transport: str) -> None:
    """lookup_id round-trips cleanly over both ZMQ and gRPC transports."""
    lookup_mod, ctx, sm = _make_lookup_env()
    tokens = list(range(8))
    _seed(sm, ctx, tokens)
    url = request_server_url(transport, 15689 if transport == "zmq" else 15690)  # type: ignore[arg-type]
    server = start_lookup_request_server(transport, url, lookup_mod)  # type: ignore[arg-type]
    client = RequestClientFactory.create(url)
    try:
        key = IPCCacheServerKey.from_token_ids(
            "m", 1, None, tokens, start=0, end=8, request_id="req-w", lookup_id="lkp-w"
        )
        client.lookup(key, 1).result(timeout=5.0)
        assert (
            client.wait_prefetch_status("req-w", 2.0, "lkp-w").result(timeout=5.0) == 8
        )
        client.end_session("req-w", "lkp-w").result(timeout=5.0)
        assert ctx.session_manager.get(key.session_key) is None
        # Default empty lookup_id still works on legacy call path.
        client.end_session("req-w").result(timeout=5.0)
        assert READ_LIFETIME_V1 == "read_lifetime_v1"
    finally:
        client.close()
        server.close()
        lookup_mod.close()
        ctx.close()
