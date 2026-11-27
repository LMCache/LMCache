# SPDX-License-Identifier: Apache-2.0
"""Tests for a lookup whose request ends before the engine asks for its
result: its read locks are released when the session is destroyed."""

# Standard
from typing import Callable
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager

REQUEST_ID = "req-abandoned"
CHUNK_SIZE = 256
NUM_CHUNKS = 3
OWNERS = {
    ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0): 11
    for i in range(2)
}


def _module_with_unconsumed_lookup(
    session_ttl: float = 600.0,
) -> tuple[LookupModule, MagicMock]:
    """Send one LOOKUP through a lookup module and leave its result unqueried.

    Returns:
        The module and its context: a mock, except for a real session manager.
    """
    token_hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    ctx = MagicMock()
    ctx.event_bus.has_subscribers.return_value = False
    registry = ctx.layout_desc_registry
    registry.get_chunk_size.return_value = CHUNK_SIZE
    registry.get_token_hasher.return_value = token_hasher
    registry.find_attn_desc.return_value = MagicMock(
        num_chunks_in_sw=[-1], num_object_groups=1
    )
    registry.find_shard_replication_desc.return_value = DEFAULT_SHARD_REPLICATION_DESC
    registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.session_manager = SessionManager(
        token_hasher, ttl=session_ttl, cleanup_interval=None
    )
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=7,
        external_request_id=REQUEST_ID,
        total_requested_keys=NUM_CHUNKS,
        submit_time=0.0,
    )
    module = LookupModule(ctx)
    num_tokens = NUM_CHUNKS * CHUNK_SIZE
    key = IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(num_tokens)),
        start=0,
        end=num_tokens,
        request_id=REQUEST_ID,
    )
    module.lookup(key, 1)
    return module, ctx


def _end_session(module: LookupModule, ctx: MagicMock) -> None:
    module.end_session(REQUEST_ID)


def _expire_session(module: LookupModule, ctx: MagicMock) -> None:
    assert ctx.session_manager.cleanup_expired() == 1


@pytest.mark.parametrize(
    ("destroy_session", "session_ttl"),
    [(_end_session, 600.0), (_expire_session, -1.0)],
    ids=["end_session", "session_ttl"],
)
def test_ended_request_releases_the_locks_of_its_unconsumed_lookup(
    destroy_session: Callable[[LookupModule, MagicMock], None], session_ttl: float
) -> None:
    """The locks the storage manager hands over are released on the L1 that
    holds them, and the lookup no longer counts as active."""
    module, ctx = _module_with_unconsumed_lookup(session_ttl)
    ctx.storage_manager.abandon_prefetch_task.return_value = OWNERS
    active_before = module.report_status()["active_prefetch_jobs"]

    destroy_session(module, ctx)

    ctx.storage_manager.finish_read_prefetched.assert_called_once_with(
        list(OWNERS), read_locks=1, l1_owners=OWNERS
    )
    assert active_before == 1
    assert module.report_status()["active_prefetch_jobs"] == 0


def test_ended_request_releases_nothing_the_storage_manager_keeps() -> None:
    """A lookup still in progress, or one whose locks are past their TTL, is
    abandoned but not released here: the storage manager hands over no lock."""
    module, ctx = _module_with_unconsumed_lookup()
    ctx.storage_manager.abandon_prefetch_task.return_value = {}

    module.end_session(REQUEST_ID)

    ctx.storage_manager.abandon_prefetch_task.assert_called_once_with(
        ctx.storage_manager.submit_prefetch_task.return_value
    )
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
    assert module.report_status()["active_prefetch_jobs"] == 0


def test_ended_request_with_a_consumed_lookup_releases_nothing() -> None:
    """Once the engine has the result, the retrieve or FREE_LOOKUP_LOCKS owns
    the release; releasing again would strip another reader's lock."""
    module, ctx = _module_with_unconsumed_lookup()
    ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
        hit_cells=[Bitmap(NUM_CHUNKS, 2)],
        l1_hit_cells=[Bitmap(NUM_CHUNKS, 2)],
        l2_hit_cells=[Bitmap(NUM_CHUNKS)],
        l1_owners=OWNERS,
    )
    assert module.query_prefetch_status(REQUEST_ID) == 2 * CHUNK_SIZE

    module.end_session(REQUEST_ID)

    ctx.storage_manager.abandon_prefetch_task.assert_not_called()
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
