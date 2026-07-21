# SPDX-License-Identifier: Apache-2.0
"""Tests for the KV-layout bring-up hook in register_kv_cache.

The hook forwards the layout and the raw group-0 engine KV format to
``StorageManager.register_kv_layout`` unconditionally — a silent no-op for the
stock L1 backend, a CXL pool bring-up for maru (which maps the engine format
to its memory format internally). These tests exercise the hook with a bare
transfer module (bypassing the CUDA dispatcher in __init__) and mocked context
creation, mirroring test_worker_liveness.py. The format mapping itself is
covered at the maru level in test_maru_l1_manager.py.
"""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
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


def test_hook_forwards_layout_and_raw_engine_format(monkeypatch):
    """The hook forwards the layout and the group-0 engine format verbatim."""
    engine_fmt = MagicMock(name="engine_kv_format")
    cache_ctx = MagicMock(num_layers=2, name="cache_ctx")
    cache_ctx.get_engine_kv_format.return_value = engine_fmt
    layout = MagicMock(name="layout_desc")
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: cache_ctx)
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: layout)
    module = _bare_module()

    _register(module)

    cache_ctx.get_engine_kv_format.assert_called_once_with(0)
    call = module._ctx.storage_manager.register_kv_layout.call_args
    # (layout_desc, engine_kv_format, chunk_size, num_object_groups)
    assert call.args[0] is layout
    assert call.args[1] is engine_fmt
    assert call.args[2] is module._ctx.chunk_size
    assert 1 in module._cache_contexts  # registration completed normally


def test_hook_failure_closes_context_and_does_not_register(monkeypatch):
    """A rejected layout (e.g. maru >1 object group) must not half-register."""
    cache_ctx = MagicMock(num_layers=2, name="cache_ctx")
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: cache_ctx)
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module = _bare_module()
    module._ctx.storage_manager.register_kv_layout.side_effect = ValueError(
        "maru L1 supports a single object group only"
    )

    with pytest.raises(ValueError, match="single object group"):
        _register(module)

    cache_ctx.close.assert_called_once()
    assert 1 not in module._cache_contexts  # not left half-registered


def test_hook_is_inert_under_a_fully_mocked_context(monkeypatch):
    """Upstream-style tests mock the whole ctx; the hook must stay harmless.

    Regression guard for the pybind ``is_mla()`` TypeError class: the hook
    must never feed mocked values into native code. Tests that mock the whole
    module context (test_worker_liveness.py, test_ipc_memory_reclaim.py, and
    future upstream tests alike) rely on this.
    """
    monkeypatch.setattr(
        gpu_mod, "create_cache_context", lambda *a, **kw: MagicMock(num_layers=2)
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module = _bare_module()

    _register(module)  # must not raise

    assert 1 in module._cache_contexts
