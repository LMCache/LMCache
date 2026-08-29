# SPDX-License-Identifier: Apache-2.0
"""Regression: HTTP-layer passthroughs kept through the service refactor."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.server import MPCacheServer
from lmcache.v1.multiprocess.services.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferService,
)
from lmcache.v1.multiprocess.services.management import ManagementService


def _engine(
    ctx: MagicMock,
    *,
    management: MagicMock | None = None,
    lmcache_driven_transfer: MagicMock | None = None,
) -> MPCacheServer:
    return MPCacheServer(
        ctx,
        status_reporters=[],
        closeables=[],
        management=management or MagicMock(spec=ManagementService),
        lmcache_driven_transfer=lmcache_driven_transfer,
    )


def test_storage_manager_returns_context_storage_manager() -> None:
    sm = MagicMock(name="storage_manager")
    ctx = MagicMock()
    ctx.storage_manager = sm

    engine = _engine(ctx)
    assert engine.storage_manager is sm


def test_cache_contexts_unwraps_entries_from_gpu_transfer_service() -> None:
    gpu0, gpu1 = MagicMock(name="gpu_ctx_0"), MagicMock(name="gpu_ctx_1")
    gpu_transfer = MagicMock(spec=LMCacheDrivenTransferService)
    gpu_transfer.context_entries_snapshot.return_value = {
        0: ContextEntry(cache_context=gpu0, model_name="m", world_size=1),
        7: ContextEntry(cache_context=gpu1, model_name="m", world_size=1),
    }

    engine = _engine(MagicMock(), lmcache_driven_transfer=gpu_transfer)
    # Values must be unwrapped GPUCacheContexts.
    assert engine.cache_contexts == {0: gpu0, 7: gpu1}


def test_cache_contexts_returns_none_in_engine_driven_mode() -> None:
    engine = _engine(MagicMock())
    assert engine.cache_contexts is None


def test_clear_delegates_to_management_service() -> None:
    mgmt = MagicMock(spec=ManagementService)
    engine = _engine(MagicMock(), management=mgmt)
    engine.clear()
    mgmt.clear.assert_called_once_with()
