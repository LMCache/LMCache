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


class _FakeKVLayerGroupsManager:
    """Minimal manager stub with two hybrid object groups."""

    num_object_groups: int = 2

    def get_attn_desc(self) -> Any:
        """One full-attention group and one sliding-window group."""
        # First Party
        from lmcache.v1.distributed.api import AttnWindowDesc

        return AttnWindowDesc(num_chunks_in_sw=[-1, 2])


class _FakeGPUContext:
    """Small stand-in for GPUCacheContext used by registration tests."""

    device: torch.device = torch.device("cpu")
    num_layers: int = 2
    kv_layer_groups_manager: _FakeKVLayerGroupsManager = _FakeKVLayerGroupsManager()

    def close(self) -> None:
        """No-op teardown (real GPUCacheContext.close deregisters its GDS buffer)."""


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
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as lmcache_driven_transfer_mod,
    )

    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 16, 32])],
        dtypes=[torch.float32],
    )
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.layout_desc_registry = LayoutDescRegistry()
    requested_object_groups: list[int] = []

    def fake_create_cache_context(
        kv_caches: object,
        lmcache_tokens_per_chunk: int,
        layout_hints: object = None,
        engine_group_infos: object = (),
        engine_type: object = None,
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
    ) -> _FakeGPUContext:
        """Return a fake cache context without touching CUDA or wrappers."""
        return _FakeGPUContext()

    def fake_layout_desc(
        gpu_context: _FakeGPUContext,
        num_tokens: int,
        object_group_id: int = 0,
    ) -> MemoryLayoutDesc:
        """Return the shared layout descriptor used by both registrations."""
        requested_object_groups.append(object_group_id)
        return layout_desc

    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "DeviceHostFuncDispatcher",
        _FakeDeviceHostFuncDispatcher,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "create_cache_context",
        fake_create_cache_context,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "get_layout_desc",
        fake_layout_desc,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod.torch_dev,
        "empty_cache",
        lambda: None,
        raising=False,
    )

    module = lmcache_driven_transfer_mod.LMCacheDrivenTransferModule(ctx)
    module.register_kv_cache(1, [], "shared-model", 1, EngineType.VLLM, {}, [])
    module.register_kv_cache(2, [], "shared-model", 1, EngineType.VLLM, {}, [])
    assert requested_object_groups == [0, 1, 0, 1]
    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(1)

    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(2)
    assert ctx.layout_desc_registry.find("shared-model", 1) is None


def _layout() -> Any:
    """A minimal layout descriptor for registry tests."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    return MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.float16])


def test_registry_attn_desc_roundtrip() -> None:
    """register stores the attention-window descriptor; find_attn_desc reads it."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register(
        "m", 2, _layout(), attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2])
    )

    assert registry.find_attn_desc("m", 2).num_chunks_in_sw == [-1, 2]


def test_registry_object_group_layouts_roundtrip() -> None:
    """Each object group's distinct layout remains available to lookup."""
    # First Party
    from lmcache.v1.distributed.api import (
        AttnWindowDesc,
        MemoryLayoutDesc,
        ObjectGroupLayoutDesc,
    )
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    layouts = ObjectGroupLayoutDesc(
        layouts=(
            MemoryLayoutDesc(
                shapes=[torch.Size([2, 4])],
                dtypes=[torch.float16],
            ),
            MemoryLayoutDesc(
                shapes=[torch.Size([4, 4])],
                dtypes=[torch.float16],
            ),
        )
    )
    registry = LayoutDescRegistry()
    registry.register(
        "hybrid",
        1,
        layouts,
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[2, -1]),
    )

    assert registry.find_object_group_layouts("hybrid", 1) is layouts
    assert registry.find("hybrid", 1) is layouts.layouts[0]


def test_registry_expands_uniform_layout_for_attention_groups() -> None:
    """A singular layout remains compatible with multi-group registrations."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    layout = _layout()
    registry = LayoutDescRegistry()
    registry.register(
        "hybrid",
        1,
        layout,
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[2, -1]),
    )

    layouts = registry.find_object_group_layouts("hybrid", 1)
    assert layouts is not None
    assert layouts.num_object_groups == 2
    assert layouts.get_layout(0) is layout
    assert layouts.get_layout(1) is layout


def test_registry_rejects_mismatched_groups_without_mutation() -> None:
    """A rejected registration must not leave a partial registry entry."""
    # First Party
    from lmcache.v1.distributed.api import (
        AttnWindowDesc,
        ObjectGroupLayoutDesc,
    )
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    layouts = ObjectGroupLayoutDesc(layouts=(_layout(), _layout()))
    registry = LayoutDescRegistry()

    with pytest.raises(ValueError, match="same number of object groups"):
        registry.register(
            "hybrid",
            1,
            layouts,
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1]),
        )

    assert registry.find_object_group_layouts("hybrid", 1) is None


def test_registry_rejects_empty_attention_groups_for_uniform_layout() -> None:
    """A uniform layout still requires at least one object group."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()

    with pytest.raises(ValueError, match="num_object_groups must be positive"):
        registry.register(
            "hybrid",
            1,
            _layout(),
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[]),
        )

    assert registry.find_object_group_layouts("hybrid", 1) is None


def test_registry_attn_desc_raises_when_unregistered() -> None:
    """find_attn_desc raises for an unknown (model, world_size) pair."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()

    with pytest.raises(ValueError, match="No attention-window descriptor"):
        registry.find_attn_desc("missing", 1)


def test_registry_windows_default_single_group_when_omitted() -> None:
    """A registration without windows resolves to a single full-attention group."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register("m", 1, _layout())

    assert registry.find_attn_desc("m", 1).num_chunks_in_sw == [-1]


def test_registry_windows_updated_on_reregister() -> None:
    """Re-registering the same pair refreshes the stored windows."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register(
        "m", 1, _layout(), attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1])
    )
    registry.register(
        "m", 1, _layout(), attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 4])
    )

    assert registry.find_attn_desc("m", 1).num_chunks_in_sw == [-1, 4]
