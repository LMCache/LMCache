# SPDX-License-Identifier: Apache-2.0
"""Parametrized real-wiring tests for the current platform registry.

This file verifies the *current real* platform wiring: each production base
class is discovered and its concrete device subclasses are indexed precisely.

Chains are added or removed by editing the ``@pytest.mark.parametrize`` list —
each entry is a ``(base_path, {device_type: impl_path})`` tuple of dotted-path
strings. Precise ``==`` equality forces explicit mapping updates in review, so
adding a new device implementation requires updating the expected mapping and a
reviewer must confirm the new wiring.

Heavy imports happen *inside* the test body (via ``_resolve``), not at module
top level, and the ``stub_c_ops`` fixture installs a fake ``lmcache.c_ops`` in
``sys.modules`` before the body runs. Together these keep the file collectable
and runnable on CPU-only runners with no compiled native ``lmcache.c_ops``.
"""

# Standard
from types import ModuleType
import importlib
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _registry as platform_registry


class _EnumNamespace:
    """Minimal attribute namespace for import-only enum lookups."""

    def __getattr__(self, name: str) -> str:
        return name


class _StubCOpsModule(ModuleType):
    EngineKVFormat: _EnumNamespace
    PageBufferShapeDesc: type[object]


@pytest.fixture
def stub_c_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight ``lmcache.c_ops`` stub for import-only tests."""
    stub = _StubCOpsModule("lmcache.c_ops")
    stub.EngineKVFormat = _EnumNamespace()
    stub.PageBufferShapeDesc = type("PageBufferShapeDesc", (), {})
    monkeypatch.setitem(sys.modules, "lmcache.c_ops", stub)


def _resolve(dotted: str) -> type:
    """Import the module named by ``dotted`` and return its final attribute.

    Args:
        dotted: A fully-qualified dotted path, e.g.
            ``"lmcache.v1.platform.base.cache_context.BaseCacheContext"``.

    Returns:
        The class object referenced by ``dotted``.
    """
    module_path, _, attr = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


# ---------------------------------------------------------------------------
# TEST 2 — Real-wiring discovery (parametrized per base chain)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_path, expected_paths",
    [
        pytest.param(
            "lmcache.v1.platform.base.cache_context.BaseCacheContext",
            {
                "cpu": "lmcache.v1.platform.cpu.cache_context.CPUCacheContext",
                "cuda": "lmcache.v1.platform.cuda.cache_context.GPUCacheContext",
            },
            id="cache_context",
        ),
        pytest.param(
            "lmcache.v1.platform.base.ipc_wrapper.DeviceIPCWrapper",
            {
                "cpu": "lmcache.v1.platform.cpu.shm.CpuShmTensorWrapper",
                "cuda": "lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper",
            },
            id="ipc_wrapper",
        ),
        pytest.param(
            "lmcache.v1.platform.base.pin_memory.PinMemoryBackend",
            # pin_memory has ONLY a CUDA concrete backend. Verified against the
            # code: base/pin_memory.py::PinMemoryBackend has no device_type
            # (no-op fallback, skipped by the empty-device_type rule) and there
            # is NO cpu/pin_memory.py, so the only indexed impl is the CUDA one.
            {"cuda": "lmcache.v1.platform.cuda.pin_memory.CudaPinMemoryBackend"},
            id="pin_memory",
        ),
    ],
)
def test_real_wiring_discovery(
    stub_c_ops: None,
    base_path: str,
    expected_paths: dict[str, str],
) -> None:
    """Verify the current real platform wiring: each production base class is
    discovered and its concrete device subclasses are indexed precisely.

    To add or remove a real chain edit the ``@pytest.mark.parametrize`` list
    above — each entry is one ``(base_path, {device_type: impl_path})`` tuple of
    dotted-path strings. Precise ``==`` equality is intentional: adding a new
    device implementation requires updating the expected mapping so reviewers
    must confirm the new wiring.

    Real production classes are resolved *inside* this test body (after the
    ``stub_c_ops`` fixture is active), so every heavy import happens while the
    fake ``lmcache.c_ops`` stub is installed and nothing heavy is imported at
    collection time. This keeps the test runnable on CPU-only runners with no
    compiled native extension.
    """
    saved = platform_registry.snapshot()
    try:
        # Resolve real classes now that stub_c_ops is active (in-body imports).
        base_cls = _resolve(base_path)
        expected = {dt: _resolve(path) for dt, path in expected_paths.items()}

        # Force a clean real rescan from scratch
        platform_registry.reset_for_tests()

        # The base class must be discovered by the base-package scanner
        assert base_cls in platform_registry._collect_base_classes()

        # Precise equality: the indexed mapping must exactly match 'expected'.
        assert platform_registry.get_all_impls(base_cls) == expected

        # Per-device lookup via get_impl
        for device_type, cls in expected.items():
            assert platform_registry.get_impl(base_cls, device_type) is cls

    finally:
        platform_registry.restore(saved)
