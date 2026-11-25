# SPDX-License-Identifier: Apache-2.0
"""The ``MP_LOOKUP_HIT_TAIL`` event a finished lookup publishes, which lets
the coordinator recover a request served entirely from cache."""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.common.attn_window import AttnWindowDesc
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    ipc_key_to_object_keys,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.kv_load_policy import KVLoadContext, KVLoadPolicy
from lmcache.v1.multiprocess.modules.lookup import LookupModule

CHUNK_SIZE = 256
CHUNK_HASHES = [b"h0", b"h1", b"h2"]


class _Decline(KVLoadPolicy):
    def __init__(self) -> None:
        super().__init__({})

    def should_load(self, ctx: KVLoadContext) -> bool:
        return False


def _ctx(
    subscribed: bool = True, hit_chunks: int = 0, looked_up: int = len(CHUNK_HASHES)
) -> MagicMock:
    """A mock engine context whose prefetch over ``looked_up`` chunks (the
    range past ``start``) hits its first ``hit_chunks``."""
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus.has_subscribers.side_effect = lambda event_type: (
        subscribed and event_type is EventType.MP_LOOKUP_HIT_TAIL
    )
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
        DEFAULT_SHARD_REPLICATION_DESC
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    token_hasher = MagicMock()
    token_hasher.compute_packed_chunk_hashes.return_value = CHUNK_HASHES
    ctx.layout_desc_registry.get_chunk_size.return_value = CHUNK_SIZE
    ctx.layout_desc_registry.get_token_hasher.return_value = token_hasher
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req-1",
        total_requested_keys=len(CHUNK_HASHES),
        submit_time=time.monotonic(),
    )
    ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
        hit_cells=[Bitmap(looked_up, hit_chunks)],
        l1_hit_cells=[Bitmap(looked_up, hit_chunks)],
        l2_hit_cells=[Bitmap(looked_up)],
    )
    return ctx


def _key(start_chunks: int = 0, cache_salt: str = "") -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=range(len(CHUNK_HASHES) * CHUNK_SIZE),
        start=start_chunks * CHUNK_SIZE,
        end=len(CHUNK_HASHES) * CHUNK_SIZE,
        request_id="req-1",
        cache_salt=cache_salt,
    )


def _hit_tails(ctx: MagicMock) -> list[Event]:
    return [
        call.args[0]
        for call in ctx.event_bus.publish.call_args_list
        if call.args[0].event_type is EventType.MP_LOOKUP_HIT_TAIL
    ]


def _expected_key(key: IPCCacheServerKey, chunk_hash: bytes) -> ObjectKey:
    """The store path's key for ``chunk_hash``: group 0, rank 0."""
    stored = ipc_key_to_object_keys(
        key, [chunk_hash], [0], DEFAULT_SHARD_REPLICATION_DESC
    )[0][0]
    return stored


def test_a_hit_reports_its_last_chunk_in_the_store_paths_namespace():
    ctx = _ctx(hit_chunks=2)
    module = LookupModule(ctx)
    key = _key(cache_salt="tenant")

    module.lookup(key, tp_size=1)
    assert module.query_prefetch_status("req-1") == 2 * CHUNK_SIZE

    [event] = _hit_tails(ctx)
    assert event.session_id == "req-1"
    assert event.metadata["tail_key"] == _expected_key(key, b"h1")
    assert event.metadata["token_offset"] == CHUNK_SIZE


@pytest.mark.parametrize(
    ("start_chunks", "hit_chunks", "tail"),
    [
        pytest.param(1, 1, 1, id="covered-prefix-then-hit"),
        pytest.param(3, 0, 2, id="all-covered"),
    ],
)
def test_an_engine_covered_prefix_counts_toward_the_tail(
    start_chunks, hit_chunks, tail
):
    """``found_count`` is absolute: the chunks below ``start`` were never
    looked up but still belong to the request's sequence."""
    ctx = _ctx(hit_chunks=hit_chunks, looked_up=len(CHUNK_HASHES) - start_chunks)
    module = LookupModule(ctx)
    key = _key(start_chunks=start_chunks)

    module.lookup(key, tp_size=1)
    module.query_prefetch_status("req-1")

    [event] = _hit_tails(ctx)
    assert event.metadata["tail_key"] == _expected_key(key, CHUNK_HASHES[tail])
    assert event.metadata["token_offset"] == tail * CHUNK_SIZE


@pytest.mark.parametrize(
    ("subscribed", "hit_chunks", "policy"),
    [
        pytest.param(True, 0, None, id="miss"),
        pytest.param(False, 2, None, id="no-subscriber"),
        pytest.param(True, 2, _Decline(), id="declined"),
    ],
)
def test_no_tail_is_reported(subscribed, hit_chunks, policy):
    ctx = _ctx(subscribed=subscribed, hit_chunks=hit_chunks)
    module = LookupModule(ctx, policy) if policy else LookupModule(ctx)

    module.lookup(_key(), tp_size=1)
    module.query_prefetch_status("req-1")

    assert _hit_tails(ctx) == []
