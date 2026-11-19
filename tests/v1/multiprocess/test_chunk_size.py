# SPDX-License-Identifier: Apache-2.0
"""Tests for MP server chunk-size reporting and validation."""

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


def _context(
    chunk_size: int = 256,
    *,
    requires_global_chunk_size: bool = False,
) -> MPCacheServerContext:
    return MPCacheServerContext(
        _storage_manager_config(),
        chunk_size=chunk_size,
        requires_global_chunk_size=requires_global_chunk_size,
    )


def test_management_get_chunk_size_always_returns_server_minimum() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        module = ManagementModule(_context(chunk_size=256))

        assert module.get_chunk_size() == 256


def test_global_chunk_size_mode_rejects_model_specific_size() -> None:
    """CacheBlend and Coordinator accept only the server chunk size."""
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        context = _context(chunk_size=256, requires_global_chunk_size=True)

        context.validate_chunk_size(256)
        with pytest.raises(ValueError, match="requires one global chunk size"):
            context.validate_chunk_size(512)


def test_server_status_reports_default_chunk_size() -> None:
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager") as storage_mgr,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        storage_mgr.return_value.report_status.return_value = {"is_healthy": True}
        ctx = _context(chunk_size=256)
        server = MPCacheServer(ctx, modules=[])

        status = server.report_status()

        assert status["chunk_size"] == 256
        assert status["hash_algorithm"] == "blake3"
        assert status["active_sessions"] == 0
