# SPDX-License-Identifier: Apache-2.0
"""Tests for the maru register_kv_layout engine hook in register_kv_cache.

The hook brings up the maru CXL pool once the KV layout is known. These tests
exercise it with a bare transfer module (bypassing the CUDA dispatcher in
__init__) and mocked context creation, mirroring test_worker_liveness.py.
"""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as gpu_mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)


def _bare_module() -> LMCacheDrivenTransferModule:
    """A transfer module with only the state register_kv_cache touches."""
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = MagicMock(name="ctx")
    module._cache_contexts = {}
    module._lock = threading.Lock()
    return module


def _register(module: LMCacheDrivenTransferModule) -> None:
    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])


def test_hook_forwards_kv_2ltd_for_non_mla(monkeypatch):
    monkeypatch.setattr(
        gpu_mod, "create_cache_context", lambda *a, **kw: MagicMock(num_layers=2)
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(gpu_mod, "is_mla", lambda fmt: False)
    module = _bare_module()

    _register(module)

    call = module._ctx.storage_manager.register_kv_layout.call_args
    # (layout_desc, fmt, chunk_size, num_object_groups)
    assert call.args[1] == MemoryFormat.KV_2LTD


def test_hook_forwards_kv_mla_fmt_for_mla(monkeypatch):
    monkeypatch.setattr(
        gpu_mod, "create_cache_context", lambda *a, **kw: MagicMock(num_layers=2)
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(gpu_mod, "is_mla", lambda fmt: True)
    module = _bare_module()

    _register(module)

    call = module._ctx.storage_manager.register_kv_layout.call_args
    assert call.args[1] == MemoryFormat.KV_MLA_FMT


def test_hook_failure_closes_context_and_does_not_register(monkeypatch):
    """A rejected layout (e.g. maru >1 object group) must not half-register."""
    cache_ctx = MagicMock(num_layers=2, name="cache_ctx")
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: cache_ctx)
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(gpu_mod, "is_mla", lambda fmt: False)
    module = _bare_module()
    module._ctx.storage_manager.register_kv_layout.side_effect = ValueError(
        "maru L1 supports a single object group only"
    )

    with pytest.raises(ValueError, match="single object group"):
        _register(module)

    cache_ctx.close.assert_called_once()
    assert 1 not in module._cache_contexts  # not left half-registered


def test_hook_is_noop_style_for_stock_backend(monkeypatch):
    """For stock, register_kv_layout is a no-op; the instance still registers."""
    monkeypatch.setattr(
        gpu_mod, "create_cache_context", lambda *a, **kw: MagicMock(num_layers=2)
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(gpu_mod, "is_mla", lambda fmt: False)
    module = _bare_module()
    # Stock StorageManager.register_kv_layout returns None (no-op).
    module._ctx.storage_manager.register_kv_layout.return_value = None

    _register(module)

    assert 1 in module._cache_contexts  # registration completed normally
