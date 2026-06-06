# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GPU transfer layout registration lifetime."""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock, patch
import sys
import types

# Third Party
import pytest
import torch


class _FakeGPUContext:
    """Small stand-in for GPUCacheContext used by registration tests."""

    num_layers: int = 2


class _FakeDeviceHostFuncDispatcher:
    """No-op dispatcher to avoid starting native completion threads."""

    def register(self, kind: str, handler: object, payload_type: object) -> None:
        """Record no native callback registration."""

    def start(self) -> None:
        """Start no background thread."""

    def stop(self) -> None:
        """Stop no background thread."""


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so MP server imports work in source-only test runs."""
    module = types.ModuleType("lmcache.native_storage_ops")
    module_any = cast(Any, module)
    module_any.TTLLock = type("TTLLock", (), {})
    module_any.Bitmap = type("Bitmap", (), {})
    module_any.PeriodicEventNotifier = type("PeriodicEventNotifier", (), {})
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


def test_unregister_one_shared_gpu_layout_keeps_registry_until_last_instance(
    monkeypatch: pytest.MonkeyPatch,
    stub_native_storage_ops: Any,
) -> None:
    """Unregistering one shared GPU instance must not remove the shared layout."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
    from lmcache.v1.multiprocess.modules import gpu_transfer as gpu_transfer_mod

    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 16, 32])],
        dtypes=[torch.float32],
    )
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.layout_desc_registry = LayoutDescRegistry()

    def fake_create_cache_context(
        kv_caches: object,
        lmcache_logical_chunk_size: int,
        layout_hints: object = None,
        group_views: object = (),
        engine_type: object = None,
    ) -> _FakeGPUContext:
        """Return a fake cache context without touching CUDA or wrappers."""
        return _FakeGPUContext()

    def fake_layout_desc(
        gpu_context: _FakeGPUContext,
        num_tokens: int,
    ) -> MemoryLayoutDesc:
        """Return the shared layout descriptor used by both registrations."""
        return layout_desc

    monkeypatch.setattr(
        gpu_transfer_mod,
        "DeviceHostFuncDispatcher",
        _FakeDeviceHostFuncDispatcher,
    )
    monkeypatch.setattr(
        gpu_transfer_mod, "create_cache_context", fake_create_cache_context
    )
    monkeypatch.setattr(gpu_transfer_mod, "get_layout_desc", fake_layout_desc)
    monkeypatch.setattr(
        gpu_transfer_mod.torch_dev,
        "empty_cache",
        lambda: None,
        raising=False,
    )

    module = gpu_transfer_mod.GPUTransferModule(ctx)
    module.register_kv_cache(1, [], "shared-model", 1, EngineType.VLLM, {}, [])
    module.register_kv_cache(2, [], "shared-model", 1, EngineType.VLLM, {}, [])
    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(1)

    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(2)
    assert ctx.layout_desc_registry.find("shared-model", 1) is None
