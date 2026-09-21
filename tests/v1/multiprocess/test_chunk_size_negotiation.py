# SPDX-License-Identifier: Apache-2.0
"""Tests for MP server chunk-size negotiation."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.server import MPCacheServer


def _storage_manager_config() -> MagicMock:
    config = MagicMock()
    config.l1_manager_config.gds_l1_config = None
    memory_config = config.l1_manager_config.memory_config
    memory_config.shm_name = ""
    memory_config.use_lazy = False
    memory_config.devdax_path = None
    memory_config.size_in_bytes = 0
    return config


def _context(chunk_size: int = 256) -> MPCacheServerContext:
    return MPCacheServerContext(_storage_manager_config(), chunk_size=chunk_size)


def test_chunk_size_negotiation_raises_configured_minimum_to_alignment() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher") as token_hasher,
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        session_mgr.return_value.active_count.return_value = 0
        ctx = _context(chunk_size=256)

        assert ctx.negotiate_chunk_size(640) == 640

        assert ctx.chunk_size == 640
        assert token_hasher.call_args_list[-1].kwargs == {
            "chunk_size": 640,
            "hash_algorithm": "blake3",
        }
        assert session_mgr.call_count == 2


def test_chunk_size_negotiation_rounds_minimum_up_to_alignment() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        session_mgr.return_value.active_count.return_value = 0
        ctx = _context(chunk_size=1024)

        assert ctx.negotiate_chunk_size(544) == 1088


def test_chunk_size_negotiation_never_shrinks_below_configured_minimum() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher") as token_hasher,
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)

        assert ctx.negotiate_chunk_size(32) == 256

        assert ctx.chunk_size == 256
        token_hasher.assert_called_once_with(chunk_size=256, hash_algorithm="blake3")


def test_negotiated_chunk_size_rejects_later_incompatible_alignment() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)

        assert ctx.negotiate_chunk_size(32) == 256
        with pytest.raises(ValueError, match="must be a multiple"):
            ctx.negotiate_chunk_size(400)


def test_chunk_size_negotiation_rejects_resize_after_sessions_start() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)
        ctx.session_manager.get_or_create("request-1")

        with pytest.raises(ValueError, match="after sessions have started"):
            ctx.negotiate_chunk_size(640)


def test_get_chunk_size_finalizes_configured_value() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)
        module = ManagementModule(ctx)

        assert module.get_chunk_size() == 256
        with pytest.raises(ValueError, match="must be a multiple"):
            module.negotiate_chunk_size(640)


def test_management_module_negotiates_chunk_size() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        session_mgr.return_value.active_count.return_value = 0
        ctx = _context(chunk_size=256)
        module = ManagementModule(ctx)

        assert module.negotiate_chunk_size(400) == 400
        assert module.get_chunk_size() == 400


def test_chunk_size_bind_listener_runs_after_resize() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        session_mgr.return_value.active_count.return_value = 0
        ctx = _context(chunk_size=256)
        listener = MagicMock()
        ctx.add_chunk_size_bind_listener(listener)

        assert ctx.negotiate_chunk_size(640) == 640

        listener.assert_called_once_with(640)


def test_server_status_reports_current_chunk_size() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager") as storage_mgr,
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        storage_mgr.return_value.report_status.return_value = {"is_healthy": True}
        session_mgr.return_value.active_count.return_value = 0
        ctx = _context(chunk_size=256)
        server = MPCacheServer(ctx, modules=[])

        status = server.report_status()

        assert status["chunk_size"] == 256
        assert status["hash_algorithm"] == "blake3"
        assert status["active_sessions"] == 0
