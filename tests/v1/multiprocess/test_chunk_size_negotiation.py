# SPDX-License-Identifier: Apache-2.0
"""Tests for MP server chunk-size negotiation."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.model_chunk_profile import ModelChunkProfileManager
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.server import MPCacheServer
from lmcache.v1.multiprocess.session import SessionManager


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


def test_model_chunk_profiles_keep_chunk_sizes_and_hashers_independent() -> None:
    registry = ModelChunkProfileManager(
        default_chunk_size=256,
        hash_algorithm="blake3",
        model_chunk_sizes={"model-a": 128},
        model_hash_algorithms={"model-a": "builtin"},
    )

    registry.ensure_model_profile("model-b")
    default_hasher = registry.get_token_hasher()
    model_a_hasher = registry.get_token_hasher("model-a")
    model_b_hasher = registry.get_token_hasher("model-b")

    assert registry.get_chunk_size() == 256
    assert registry.get_chunk_size("model-a") == 128
    assert registry.get_chunk_size("model-b") == 256
    assert registry.get_configured_chunk_size("model-a") == 128
    assert registry.get_hash_algorithm_name() == "blake3"
    assert registry.get_hash_algorithm_name("model-a") == "builtin"
    assert registry.get_hash_algorithm_name("model-b") == "blake3"
    assert model_a_hasher.chunk_size == 128
    assert model_a_hasher.hash_algorithm_name == "builtin"
    assert model_b_hasher.chunk_size == 256

    rebound_hasher = registry.bind_chunk_size(384, "model-a")

    assert registry.get_chunk_size("model-a") == 384
    assert rebound_hasher.chunk_size == 384
    assert rebound_hasher.hash_algorithm_name == "builtin"
    assert rebound_hasher is registry.get_token_hasher("model-a")
    assert registry.get_token_hasher() is default_hasher
    assert registry.get_token_hasher("model-b") is model_b_hasher


def test_chunk_size_negotiation_raises_configured_size_to_alignment() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch(
            "lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"
        ) as token_hasher,
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)

        assert ctx.negotiate_chunk_size(640) == 640

        assert ctx.chunk_size == 640
        assert token_hasher.call_args_list[-1].kwargs == {
            "chunk_size": 640,
            "hash_algorithm": "blake3",
        }
        session_mgr.assert_called_once_with(ttl=SessionManager.DEFAULT_SESSION_TTL)


def test_chunk_size_negotiation_preserves_configured_session_ttl() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheServerContext(
            _storage_manager_config(),
            chunk_size=256,
            session_ttl_seconds=7200,
        )

        assert ctx.negotiate_chunk_size(640) == 640
        assert session_mgr.call_args_list[-1].kwargs == {"ttl": 7200}


def test_chunk_size_negotiation_rounds_minimum_up_to_alignment() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=1024)

        assert ctx.negotiate_chunk_size(544) == 1088


def test_chunk_size_negotiation_never_shrinks_below_configured_size() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch(
            "lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"
        ) as token_hasher,
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
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)

        assert ctx.negotiate_chunk_size(32) == 256
        assert ctx.negotiate_chunk_size(64) == 256
        with pytest.raises(ValueError, match="separate MP servers"):
            ctx.negotiate_chunk_size(400)


def test_get_chunk_size_finalizes_configured_value() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
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
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)
        module = ManagementModule(ctx)

        assert module.negotiate_chunk_size(400) == 400
        assert module.get_chunk_size() == 400


def test_chunk_size_bind_listener_runs_after_resize() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)
        listener = MagicMock()
        ctx.add_chunk_size_bind_listener(listener)

        assert ctx.negotiate_chunk_size(640) == 640

        listener.assert_called_once_with(640)


def test_chunk_size_bind_listener_failure_is_retried() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)
        completed_listener = MagicMock()
        retried_listener = MagicMock(side_effect=[RuntimeError("transient"), None])
        ctx.add_chunk_size_bind_listener(completed_listener)
        ctx.add_chunk_size_bind_listener(retried_listener)

        with pytest.raises(RuntimeError, match="transient"):
            ctx.negotiate_chunk_size(640)
        assert ctx.chunk_size == 640

        with pytest.raises(ValueError, match="separate MP servers"):
            ctx.negotiate_chunk_size(400)

        assert ctx.negotiate_chunk_size(640) == 640
        completed_listener.assert_called_once_with(640)
        assert retried_listener.call_count == 2


def test_token_hasher_access_notifies_listener_before_returning() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = _context(chunk_size=256)
        listener = MagicMock()
        ctx.add_chunk_size_bind_listener(listener)

        token_hasher = ctx.token_hasher

        assert ctx.token_hasher is token_hasher
        assert ctx.finalized_chunk_size == 256
        listener.assert_called_once_with(256)


def test_server_status_reports_current_chunk_size() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager") as storage_mgr,
        patch("lmcache.v1.multiprocess.model_chunk_profile.TokenHasher"),
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
