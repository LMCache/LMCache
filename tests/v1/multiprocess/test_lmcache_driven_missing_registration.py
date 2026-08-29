# SPDX-License-Identifier: Apache-2.0
"""Missing-registration responses for LMCache-driven GPU transfers."""

# Standard
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import (
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.session import SessionManager


class _TestTokenHasher:
    """Small deterministic hasher for lock-ownership tests."""

    chunk_size = 2

    def compute_chunk_hashes(
        self,
        token_ids: list[int],
        prefix_hash: object = None,
        start: int = 0,
        end: int | None = None,
    ) -> list[bytes]:
        del prefix_hash
        effective_end = len(token_ids) if end is None else end
        return [
            f"h{i // self.chunk_size}".encode() for i in range(start, effective_end, 2)
        ]


class _CountingStorageManager:
    """Track anonymous L1 read locks and fail on an over-release."""

    def __init__(self) -> None:
        self.readers: Counter[ObjectKey] = Counter()
        self.finish_calls: list[tuple[list[ObjectKey], int]] = []

    def acquire(self, keys: list[ObjectKey], read_locks: int = 1) -> None:
        for key in keys:
            self.readers[key] += read_locks

    def finish_read_prefetched(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> None:
        self.finish_calls.append((list(keys), read_locks))
        for key in keys:
            assert self.readers[key] >= read_locks
            self.readers[key] -= read_locks


def _cache_key(
    *,
    world_size: int,
    worker_id: int | None,
    request_id: str,
    start: int = 0,
    end: int = 4,
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="test-model",
        world_size=world_size,
        worker_id=worker_id,
        token_ids=(1, 2, 3, 4),
        start=start,
        end=end,
        request_id=request_id,
    )


@pytest.mark.parametrize("method_name", ["store", "retrieve"])
def test_missing_registration_returns_terminal_false(method_name: str) -> None:
    """An absent context returns an event-free response instead of raising.

    The MQ blocking-handler exception path does not send an error response.
    Returning a normal response therefore ensures the caller's future reaches
    a terminal state during the restart-before-registration window. The empty
    handle indicates that the server submitted no device work.
    """
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module.get_and_touch_context_entry = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    module._ctx = MagicMock()
    module._ctx.session_manager.get.return_value = None
    producer_event = b"worker-producer-event"
    key = _cache_key(world_size=1, worker_id=0, request_id="request")

    result = getattr(module, method_name)(
        key,
        42,
        [[0]],
        producer_event,
    )

    assert result == (b"", False)
    module.get_and_touch_context_entry.assert_called_once_with(42)


@pytest.mark.parametrize("mla", [False, True], ids=["sharded-kv", "mla-shared-kv"])
def test_tp_failed_worker_releases_only_its_reader_share_once(mla: bool) -> None:
    """A failed TP worker must preserve its peer and a concurrent reader.

    Two lookups of the same content are represented: the request under test
    and a concurrent request.  One worker's missing registration releases one
    read lock, the successful worker releases its own queued read lock, and the
    concurrent request must still retain its complete lock allocation.  The
    duplicate failed response verifies exactly-once cleanup.
    """
    tp_size = 2
    world_size = 1 if mla else tp_size
    failed_worker = 0
    successful_worker = 0 if mla else 1
    request_id = "failed-request"

    hasher = _TestTokenHasher()
    sessions = SessionManager(hasher, cleanup_interval=None)  # type: ignore[arg-type]
    lookup_key = _cache_key(
        world_size=world_size, worker_id=None, request_id=request_id
    )
    session = sessions.get_or_create(request_id)
    session.begin_lookup(lookup_key, (-1,))
    session.record_prefetch_result(2, (0,))

    storage = _CountingStorageManager()
    layout_registry = MagicMock()
    layout_registry.find_attn_desc.side_effect = AssertionError(
        "failed-retrieve cleanup must use the lookup session's layout"
    )
    ctx = SimpleNamespace(
        chunk_size=hasher.chunk_size,
        token_hasher=hasher,
        session_manager=sessions,
        layout_desc_registry=layout_registry,
        storage_manager=storage,
    )

    hashes = hasher.compute_chunk_hashes(list(lookup_key.token_ids), end=lookup_key.end)
    all_rank_keys = ipc_key_to_object_keys(lookup_key, hashes, [0])[0]
    lookup_read_locks = tp_size if mla else 1
    # The request under test and a concurrent reader both own read locks.
    storage.acquire(all_rank_keys, read_locks=lookup_read_locks)
    storage.acquire(all_rank_keys, read_locks=lookup_read_locks)

    # vLLM already owns the first chunk locally, so the scheduler legitimately
    # releases that prefix for every rank.  The worker RETRIEVEs only the
    # remaining partial-prefix range.
    request_prefix_keys = ipc_key_to_object_keys(lookup_key, hashes[:1], [0])[0]
    storage.finish_read_prefetched(request_prefix_keys, read_locks=lookup_read_locks)

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = ctx  # type: ignore[assignment]
    module.get_and_touch_context_entry = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    failed_key = _cache_key(
        world_size=world_size,
        worker_id=failed_worker,
        request_id=request_id,
        start=2,
        end=4,
    )

    args = (failed_key, 101, [[0]], b"producer-event")
    assert module.retrieve(*args) == (b"", False)
    assert module.retrieve(*args) == (b"", False)

    retrieve_hashes = hashes[1:]
    failed_keys = ipc_key_to_object_keys(failed_key, retrieve_hashes, [0])[0]
    assert storage.finish_calls == [
        (request_prefix_keys, lookup_read_locks),
        (failed_keys, 1),
    ]

    # The peer succeeded and queued its normal device-completion release.
    successful_key = _cache_key(
        world_size=world_size,
        worker_id=successful_worker,
        request_id=request_id,
        start=2,
        end=4,
    )
    successful_keys = ipc_key_to_object_keys(successful_key, retrieve_hashes, [0])[0]
    storage.finish_read_prefetched(successful_keys, read_locks=1)

    # Only the concurrent lookup's read locks remain.  Releasing those must
    # neither underflow nor encounter a lock already consumed by the failure.
    expected_concurrent_read_locks = lookup_read_locks
    assert all(
        storage.readers[key] == expected_concurrent_read_locks for key in all_rank_keys
    )
    storage.finish_read_prefetched(all_rank_keys, read_locks=lookup_read_locks)
    assert all(storage.readers[key] == 0 for key in all_rank_keys)
    layout_registry.find_attn_desc.assert_not_called()


def test_cleanup_exception_does_not_suppress_terminal_false() -> None:
    """A resolution failure must not consume the claim or strand the caller."""
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module.get_and_touch_context_entry = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    module._ctx = MagicMock()
    session = MagicMock()
    session.prepare_failed_retrieve_release.return_value = (2, (0,), (-1,), 7)
    module._ctx.session_manager.get.return_value = session
    module._ctx.token_hasher.compute_chunk_hashes.side_effect = RuntimeError(
        "cleanup failed"
    )

    result = module.retrieve(
        _cache_key(world_size=1, worker_id=0, request_id="request"),
        42,
        [[0]],
        b"worker-producer-event",
    )

    assert result == (b"", False)
    session.claim_failed_retrieve_release.assert_not_called()
