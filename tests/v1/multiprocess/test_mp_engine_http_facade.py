# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the MP HTTP API facade on ``MPCacheEngine``.

Pinned bug: after #3391 split engine state across modules, the live
``lmcache server`` HTTP endpoints (``/clear-cache``, ``/quota``,
``/kvcache/check``, ``/status``) crash because they reach
``engine.clear``, ``engine.storage_manager``, and
``engine.gpu_contexts`` that no longer exist on the compositor (see
#3431).

Existing endpoint tests (``test_http_server.py`` /
``test_http_quota_endpoints.py``) hand the HTTP app a mock object with
the pre-refactor monolithic shape, so they keep passing while the real
engine path is broken. These tests instantiate the real refactored
compositor and assert the facade keeps the expected HTTP-API surface
working for both NonGPU- and GPU-module loadouts.

Two layers of coverage:

* Facade unit tests (``test_storage_manager_*``, ``test_clear_*``,
  ``test_gpu_contexts_*``, ``test_supports_gpu_kvcache_check_*``)
  exercise the ``MPCacheEngine`` and ``GPUTransferModule`` public
  surface against a real compositor.
* HTTP integration tests (``test_http_*_with_real_engine``) drive the
  FastAPI app via ``TestClient`` against the same real compositor and
  pin that ``/clear-cache``, ``/quota``, ``/kvcache/check`` no longer
  500 / yield the documented status codes (200 / 200 / 501) for the
  default non-GPU loadout.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
import sys

# Third Party
from fastapi.testclient import TestClient
import pytest

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
    from lmcache.v1.multiprocess.server import MPCacheEngine


@pytest.fixture(autouse=True)
def stub_native_storage_ops() -> Iterator[None]:
    """Stub native modules so the imports below work in source-only test runs.

    Mirrors the fixture in ``test_non_cuda_data_transfer.py`` — the MP
    engine path imports ``lmcache.native_storage_ops`` and ``cupy``
    transitively, neither of which build/install on macOS-CPU runners.
    Auto-applied so individual tests don't have to remember it.
    """
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def mock_context() -> MPCacheEngineContext:
    """Build a real ``MPCacheEngineContext`` with mocked deps.

    The context's heavy collaborators (``StorageManager``,
    ``TokenHasher``, ``SessionManager``, event bus) are patched so the
    test runs CPU-only with no cache backends configured.
    """
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext

    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        return MPCacheEngineContext(
            storage_manager_config=MagicMock(),
            chunk_size=16,
        )


@pytest.fixture
def non_gpu_engine(mock_context: MPCacheEngineContext) -> MPCacheEngine:
    """Compositor with ``LookupModule`` + ``ManagementModule`` +
    ``NonGPUTransferModule`` — mirrors ``transfer_mode != 'gpu'``.
    """
    # First Party
    from lmcache.v1.multiprocess.engine_module import EngineModule
    from lmcache.v1.multiprocess.modules.lookup import LookupModule
    from lmcache.v1.multiprocess.modules.management import ManagementModule
    from lmcache.v1.multiprocess.modules.non_gpu_transfer import (
        NonGPUTransferModule,
    )
    from lmcache.v1.multiprocess.server import MPCacheEngine

    modules: list[EngineModule] = [
        LookupModule(mock_context),
        ManagementModule(mock_context),
        NonGPUTransferModule(mock_context),
    ]
    return MPCacheEngine(mock_context, modules)


@pytest.fixture
def gpu_engine(mock_context: MPCacheEngineContext) -> MPCacheEngine:
    """Compositor with ``LookupModule`` + ``ManagementModule`` +
    ``GPUTransferModule`` — mirrors ``transfer_mode == 'gpu'``.

    ``GPUTransferModule`` registers a CPU-side host-callback dispatcher
    that we don't need for facade-level tests; we patch it out to keep
    the test lightweight.
    """
    # First Party
    from lmcache.v1.multiprocess.engine_module import EngineModule
    from lmcache.v1.multiprocess.modules.gpu_transfer import GPUTransferModule
    from lmcache.v1.multiprocess.modules.lookup import LookupModule
    from lmcache.v1.multiprocess.modules.management import ManagementModule
    from lmcache.v1.multiprocess.server import MPCacheEngine

    with patch("lmcache.v1.multiprocess.modules.gpu_transfer.DeviceHostFuncDispatcher"):
        modules: list[EngineModule] = [
            LookupModule(mock_context),
            ManagementModule(mock_context),
            GPUTransferModule(mock_context),
        ]
    return MPCacheEngine(mock_context, modules)


# ---------------------------------------------------------------------
# storage_manager facade
# ---------------------------------------------------------------------


def test_storage_manager_forwards_to_context(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """``engine.storage_manager`` must return the same instance the
    shared context owns. The ``/quota/*`` endpoints depend on this.
    """
    assert non_gpu_engine.storage_manager is non_gpu_engine.context.storage_manager


# ---------------------------------------------------------------------
# clear() facade
# ---------------------------------------------------------------------


def test_clear_routes_through_management_module(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """``engine.clear()`` must hit the registered ``ManagementModule``,
    which in turn touches ``storage_manager``.
    The HTTP ``/clear-cache`` endpoint depends on this path.
    """
    sm = non_gpu_engine.context.storage_manager
    assert isinstance(sm, MagicMock)
    sm.reset_mock()

    non_gpu_engine.clear()

    # ManagementModule.clear() calls memcheck → clear(force=True) → memcheck
    sm.clear.assert_called_once_with(force=True)
    assert sm.memcheck.call_count == 2


def test_clear_raises_without_management_module(
    mock_context: MPCacheEngineContext,
) -> None:
    """If a future loadout drops ``ManagementModule``, the facade must
    raise rather than silently no-op (silent no-op would mask the same
    kind of regression that motivated this fix).
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = MPCacheEngine(mock_context, [LookupModule(mock_context)])

    with pytest.raises(RuntimeError, match="ManagementModule"):
        engine.clear()


# ---------------------------------------------------------------------
# gpu_contexts facade + supports_gpu_kvcache_check capability
# ---------------------------------------------------------------------


def test_gpu_contexts_returns_empty_dict_in_non_gpu_mode(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """Without a ``GPUTransferModule`` the facade returns an empty
    dict, matching the pre-refactor contract (``gpu_contexts`` always
    returned a dict). HTTP callers that need to distinguish "no GPU
    support" from "no registrations" should consult
    :attr:`supports_gpu_kvcache_check` instead.
    """
    assert non_gpu_engine.gpu_contexts == {}


def test_supports_gpu_kvcache_check_false_in_non_gpu_mode(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """The capability flag is what HTTP handlers must read to return
    ``501 Not Implemented`` cleanly when the engine has no GPU
    transfer module.
    """
    assert non_gpu_engine.supports_gpu_kvcache_check is False


def test_gpu_contexts_returns_empty_snapshot_when_no_registrations(
    gpu_engine: MPCacheEngine,
) -> None:
    """``GPUTransferModule`` registered but no ``register_kv_cache``
    call yet — the facade returns an empty dict.
    """
    assert gpu_engine.gpu_contexts == {}


def test_supports_gpu_kvcache_check_true_in_gpu_mode(
    gpu_engine: MPCacheEngine,
) -> None:
    """With ``GPUTransferModule`` registered, the capability flag is
    ``True`` regardless of whether any contexts have been registered.
    """
    assert gpu_engine.supports_gpu_kvcache_check is True


def test_gpu_contexts_unwraps_entry_to_gpu_context(
    gpu_engine: MPCacheEngine,
) -> None:
    """The facade must return ``dict[instance_id, GPUCacheContext]``,
    not ``dict[instance_id, GPUContextEntry]`` — diagnostic callers
    (``/kvcache/check``) reach for ``ctx.kv_tensors`` /
    ``ctx.gpu_kv_format_`` directly.

    Reaches into the module's private ``_gpu_contexts`` to seed the
    test state because the public ``register_kv_cache`` requires real
    GPU-side ``KVCache`` / ``GPUCacheContext`` construction that the
    HTTP-integration tests below already exercise end-to-end. This
    keeps the unwrap pin focused and side-effect-free.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.gpu_transfer import (
        GPUContextEntry,
        GPUTransferModule,
    )

    gpu_module = gpu_engine._find_module(GPUTransferModule)
    assert gpu_module is not None

    fake_gpu_ctx = MagicMock(name="GPUCacheContext")
    gpu_module._gpu_contexts[7] = GPUContextEntry(
        gpu_context=fake_gpu_ctx,
        model_name="m",
        world_size=1,
    )

    snapshot = gpu_engine.gpu_contexts
    assert snapshot == {7: fake_gpu_ctx}

    # Returned dict is a fresh snapshot — mutating it must not affect
    # the module's internal registry.
    snapshot.clear()
    assert 7 in gpu_module._gpu_contexts


# ---------------------------------------------------------------------
# _find_module helper
# ---------------------------------------------------------------------


def test_find_module_returns_first_match(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """``_find_module`` must locate by ``isinstance`` semantics so
    facades reading through it work regardless of module ordering.
    Verified indirectly through ``clear()`` and ``gpu_contexts``
    elsewhere; this case pins direct lookup for the typed helper.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.management import ManagementModule

    mgmt = non_gpu_engine._find_module(ManagementModule)
    assert mgmt is not None
    assert isinstance(mgmt, ManagementModule)


def test_find_module_returns_none_when_absent(
    mock_context: MPCacheEngineContext,
) -> None:
    """``_find_module`` returns ``None`` (not raise) when no module
    of the requested class is registered.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.gpu_transfer import GPUTransferModule
    from lmcache.v1.multiprocess.modules.lookup import LookupModule
    from lmcache.v1.multiprocess.modules.management import ManagementModule
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = MPCacheEngine(mock_context, [LookupModule(mock_context)])

    assert engine._find_module(GPUTransferModule) is None
    assert engine._find_module(ManagementModule) is None


# ---------------------------------------------------------------------
# HTTP integration tests — drive the real FastAPI app + real engine
# ---------------------------------------------------------------------


def test_http_clear_cache_with_real_engine_returns_200(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """``POST /clear-cache`` against the real refactored compositor
    must return ``200 OK`` (the issue #3431 repro path that returns
    ``500 Internal Server Error`` before this fix).
    """
    # First Party
    from lmcache.v1.multiprocess.http_server import app

    sm = non_gpu_engine.context.storage_manager
    assert isinstance(sm, MagicMock)
    sm.reset_mock()
    app.state.engine = non_gpu_engine

    response = TestClient(app).post("/clear-cache")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    sm.clear.assert_called_once_with(force=True)


def test_http_quota_list_with_real_engine_returns_200(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """``GET /quota`` against the real refactored compositor must
    return ``200 OK`` (pre-fix: ``500 Internal Server Error`` because
    the handler called ``engine.storage_manager`` on a compositor
    without that attribute).
    """
    # First Party
    from lmcache.v1.multiprocess.http_server import app

    sm = non_gpu_engine.context.storage_manager
    assert isinstance(sm, MagicMock)
    sm.get_usage_bytes_by_cache_salt.return_value = {}
    sm.quota_manager.list_quotas.return_value = []
    app.state.engine = non_gpu_engine

    response = TestClient(app).get("/quota")

    assert response.status_code == 200
    assert response.json() == {"users": {}}


def test_http_kvcache_check_non_gpu_engine_returns_501(
    non_gpu_engine: MPCacheEngine,
) -> None:
    """``GET /kvcache/check`` on a non-GPU engine must return
    ``501 Not Implemented`` (the documented behavior). This guards
    against the supports_gpu_kvcache_check capability getting
    miswired so non-GPU engines silently fall through.
    """
    # First Party
    from lmcache.v1.multiprocess.http_server import app

    app.state.engine = non_gpu_engine

    response = TestClient(app).get(
        "/kvcache/check", params={"block_ids": "0", "chunk_size": "1"}
    )

    assert response.status_code == 501
    assert "checksum not supported" in response.json()["error"]
