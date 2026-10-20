# SPDX-License-Identifier: Apache-2.0
"""Native profile rejection must release its candidate cache context."""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.custom_types import KVEventProfile
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule


@pytest.mark.parametrize(
    "failure", ["geometry", "profile", "layout", "capability", "none"]
)
def test_native_registration_releases_only_its_candidate(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Reject before publication, preserving registered layouts and live contexts."""
    bus = EventBus(EventBusConfig())
    registry = LayoutDescRegistry()
    context = SimpleNamespace(
        event_bus=bus,
        chunk_size=4,
        layout_desc_registry=registry,
        validate_native_event_storage=lambda: None,
        storage_manager=MagicMock(),
        separate_object_groups=False,
        full_sw_kv=False,
    )
    events = KVEventModule(cast(MPCacheServerContext, context), kv_event_log_size=16)
    accepted = MagicMock(name="accepted_context", num_layers=1)
    candidate = MagicMock(name="candidate_context", num_layers=1)
    for cache_context in (accepted, candidate):
        cache_context.kv_layer_groups_manager.num_object_groups = 1
        cache_context.kv_layer_groups_manager.get_attn_desc.return_value = (
            AttnWindowDesc([-1])
        )
    original_layout = MemoryLayoutDesc([], [])
    monkeypatch.setattr(transfer, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(
        transfer, "create_cache_context", MagicMock(side_effect=[accepted, candidate])
    )
    monkeypatch.setattr(transfer, "get_event_ipc_backend", MagicMock())
    monkeypatch.setattr(
        transfer, "get_layout_desc", lambda *args, **kwargs: original_layout
    )
    monkeypatch.setattr(transfer, "torch_dev", MagicMock())
    module = transfer.LMCacheDrivenTransferModule(cast(MPCacheServerContext, context))
    profile = KVEventProfile("native-test", 2)
    try:
        module.register_kv_cache(1, [], "model", 1, EngineType.VLLM, {}, [], profile)
        if failure == "geometry":
            profile = KVEventProfile("native-test", 3)
        elif failure == "profile":
            profile = KVEventProfile("changed-profile", 2)
        elif failure == "layout":
            candidate.kv_layer_groups_manager.get_attn_desc.return_value = (
                AttnWindowDesc([2])
            )
        elif failure == "capability":
            context.kv_event_module = None
        if failure == "none":
            module.register_kv_cache(
                2, [], "model", 1, EngineType.VLLM, {}, [], profile
            )
            assert module.tracked_instance_count() == 2
            candidate.close.assert_not_called()
            module.unregister_kv_cache(2)
        else:
            with pytest.raises(ValueError):
                module.register_kv_cache(
                    2, [], "model", 1, EngineType.VLLM, {}, [], profile
                )
        assert module.tracked_instance_count() == 1
        assert registry.find("model", 1) is original_layout
        accepted.close.assert_not_called()
        candidate.close.assert_called_once_with()
    finally:
        module.close()
        events.close()
        bus.stop()
    accepted.close.assert_called_once_with()
    candidate.close.assert_called_once_with()
    assert registry.find("model", 1) is None
