# SPDX-License-Identifier: Apache-2.0
"""Reject stale lookup reservations before device access or anonymous release."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import Session, SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def _key() -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=(1, 2, 3, 4),
        start=0,
        end=4,
        request_id="queued",
    )


@pytest.mark.parametrize("ttl", [1, 30, 300, 600])
def test_lookup_age_starts_at_submission_and_resets_on_new_lookup(ttl: int) -> None:
    """Prefetch completion must not restart the lifetime of its earliest locks."""
    session = Session("queued", TokenHasher(chunk_size=4, hash_algorithm="blake3"))
    assert session.lookup_expired(ttl)
    with patch("lmcache.v1.multiprocess.session.time.monotonic", return_value=10.0):
        session.begin_lookup(_key(), (-1,))
    with patch(
        "lmcache.v1.multiprocess.session.time.monotonic", return_value=10.0 + ttl / 2
    ):
        session.record_prefetch_result(1, (0,))
        assert not session.lookup_expired(ttl)
    with patch(
        "lmcache.v1.multiprocess.session.time.monotonic", return_value=10.0 + ttl
    ):
        assert session.lookup_expired(ttl)
        session.begin_lookup(_key(), (-1,))
        assert not session.lookup_expired(ttl)


@pytest.mark.parametrize("ttls", [(1,), (600,), (600, 30)])
def test_context_uses_shortest_configured_l1_ttl(ttls: tuple[int, ...]) -> None:
    """The context checks configured TTL and rejects a missing session."""
    hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")
    sessions = SessionManager(hasher, cleanup_interval=None)
    storage = MagicMock(spec=StorageManager)
    storage.read_ttl_seconds = min(ttls)
    with (
        patch(
            "lmcache.v1.multiprocess.engine_context.StorageManager",
            return_value=storage,
        ),
        patch(
            "lmcache.v1.multiprocess.engine_context.SessionManager",
            return_value=sessions,
        ),
    ):
        config = MagicMock()
        config.l1_manager_configs = [MagicMock(), MagicMock()]
        ctx = MPCacheServerContext(config, chunk_size=4)
    assert ctx.lookup_expired("unknown")
    session = sessions.get_or_create("queued")
    with patch("lmcache.v1.multiprocess.session.time.monotonic", return_value=1000.0):
        session.begin_lookup(_key(), (-1,))
    with patch(
        "lmcache.v1.multiprocess.session.time.monotonic",
        return_value=1000.0 + min(ttls),
    ):
        assert ctx.lookup_expired("queued")
    sessions.remove("queued")
    assert ctx.lookup_expired("queued")


@pytest.mark.parametrize("expired", [False, True])
def test_gpu_rejects_stale_lookup_before_device_work(expired: bool) -> None:
    """A stale request gets a normal failed reply and cannot release peer locks."""
    ctx = MagicMock(spec=MPCacheServerContext)
    ctx.storage_manager = MagicMock()
    ctx.lookup_expired.return_value = expired
    with patch(
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.DeviceHostFuncDispatcher"
    ):
        module = LMCacheDrivenTransferModule(ctx)
    with patch.object(module, "get_and_touch_context_entry", return_value=None) as get:
        assert module.retrieve(_key(), 42, [[0]], b"producer") == (b"", False)
        assert get.call_count == (0 if expired else 1)
    if expired:
        ctx.storage_manager.finish_read_prefetched.assert_not_called()
        ctx.resolve_obj_keys.assert_not_called()


@pytest.mark.parametrize("expired", [False, True])
def test_cpu_and_shm_reject_stale_lookup_before_strategy(expired: bool) -> None:
    """The common engine-driven entry point protects inline and SHM reads."""
    ctx = MagicMock(spec=MPCacheServerContext)
    ctx.shm_pool_info = {"shm_name": "", "pool_size": 0}
    ctx.lookup_expired.return_value = expired
    module = EngineDrivenTransferModule(ctx)
    if expired:
        response = module.prepare_retrieve(_key(), 42)
        assert not response.success and response.data == b"" and response.context == {}
    else:
        with pytest.raises(ValueError, match="not registered"):
            module.prepare_retrieve(_key(), 42)
    ctx.resolve_obj_keys.assert_not_called()


def test_expired_cleanup_does_not_decrement_new_reader() -> None:
    """Cancellation of an old lookup leaves anonymous counts to TTL cleanup."""
    ctx = MagicMock(spec=MPCacheServerContext)
    ctx.storage_manager = MagicMock()
    ctx.lookup_expired.return_value = True
    module = LookupModule(ctx)
    module.free_lookup_locks(_key(), 1)
    module.free_lookup_locks(_key(), 1)
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
    ctx.session_manager.get_or_create.assert_not_called()
