# SPDX-License-Identifier: Apache-2.0

"""Tests for MaruMemoryAllocator (mocked maru runtime, no CXL required)."""

# Standard
from unittest.mock import MagicMock
import sys

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.v1.distributed.config import MaruL1Config
    from lmcache.v1.distributed.memory_manager.maru_memory_allocator import (
        MaruMemoryAllocator,
        _to_tcp,
    )
    from lmcache.v1.memory_management import MemoryFormat
except ImportError:
    pytest.skip("maru allocator deps unavailable", allow_module_level=True)


def _cfg() -> MaruL1Config:
    return MaruL1Config(server_url="maru://localhost:9000", pool_size_bytes=1 << 30)


# 1 group, shape (4 tokens, 8 feats) fp16 -> 4*8*2 = 64 bytes; 4 tokens/chunk.
_LAYOUT = ([torch.Size([4, 8])], [torch.float16], MemoryFormat.KV_2LTD, 4)


@pytest.fixture
def maru_mocks(monkeypatch):
    """Inject fake maru / maru_lmcache modules for init_layout's lazy imports."""
    handler = MagicMock()
    handler.connect.return_value = True
    handler.get_chunk_size.return_value = 64
    adapter = MagicMock()
    maru_mod = MagicMock()
    maru_mod.MaruHandler.return_value = handler
    lmcache_mod = MagicMock()
    lmcache_mod.CxlMemoryAdapter.return_value = adapter
    monkeypatch.setitem(sys.modules, "maru", maru_mod)
    monkeypatch.setitem(sys.modules, "maru_lmcache", lmcache_mod)
    return handler, adapter, maru_mod


def test_to_tcp():
    assert _to_tcp("maru://h:1") == "tcp://h:1"
    assert _to_tcp("tcp://h:1") == "tcp://h:1"


def test_methods_before_init_raise():
    alloc = MaruMemoryAllocator(_cfg())
    assert not alloc.is_initialized
    with pytest.raises(RuntimeError):
        alloc.allocate([torch.Size([4, 8])], [torch.float16])
    with pytest.raises(RuntimeError):
        _ = alloc.handler
    with pytest.raises(RuntimeError):
        _ = alloc.single_token_size


def test_init_layout_builds_pool(maru_mocks):
    handler, adapter, maru_mod = maru_mocks
    alloc = MaruMemoryAllocator(_cfg())
    alloc.init_layout(*_LAYOUT)

    assert alloc.is_initialized
    handler.connect.assert_called_once()
    assert alloc.handler is handler
    assert alloc.single_token_size == 16  # 64 bytes / 4 tokens

    _, kwargs = maru_mod.MaruConfig.call_args
    assert kwargs["server_url"] == "tcp://localhost:9000"
    assert kwargs["pool_size"] == 1 << 30
    assert kwargs["chunk_size_bytes"] == 64
    assert kwargs["auto_connect"] is False


def test_init_layout_same_layout_is_noop(maru_mocks):
    handler, _, _ = maru_mocks
    alloc = MaruMemoryAllocator(_cfg())
    alloc.init_layout(*_LAYOUT)
    alloc.init_layout(*_LAYOUT)
    handler.connect.assert_called_once()  # not reconnected


def test_init_layout_mismatch_raises(maru_mocks):
    alloc = MaruMemoryAllocator(_cfg())
    alloc.init_layout(*_LAYOUT)
    with pytest.raises(ValueError):
        alloc.init_layout(
            [torch.Size([8, 8])], [torch.float16], MemoryFormat.KV_2LTD, 4
        )


def test_init_layout_bad_chunk_raises(maru_mocks):
    alloc = MaruMemoryAllocator(_cfg())
    with pytest.raises(ValueError):
        alloc.init_layout(
            [torch.Size([4, 8])], [torch.float16], MemoryFormat.KV_2LTD, 0
        )


def test_init_layout_non_divisible_chunk_raises(maru_mocks):
    alloc = MaruMemoryAllocator(_cfg())
    # 64 bytes is not a multiple of 5 tokens.
    with pytest.raises(ValueError):
        alloc.init_layout(
            [torch.Size([4, 8])], [torch.float16], MemoryFormat.KV_2LTD, 5
        )


def test_connect_failure_raises(maru_mocks):
    handler, _, _ = maru_mocks
    handler.connect.return_value = False
    alloc = MaruMemoryAllocator(_cfg())
    with pytest.raises(RuntimeError):
        alloc.init_layout(*_LAYOUT)


def test_delegation_and_lifecycle(maru_mocks):
    _, adapter, _ = maru_mocks
    alloc = MaruMemoryAllocator(_cfg())
    alloc.init_layout(*_LAYOUT)
    obj = MagicMock()

    adapter.allocate.return_value = obj
    assert alloc.allocate([torch.Size([4, 8])], [torch.float16]) is obj

    alloc.get_by_location(1, 2, 64)
    adapter.get_by_location.assert_called_once_with(
        region_id=1, page_index=2, actual_size=64, single_token_size=16
    )

    alloc.create_store_handle(obj)
    adapter.create_store_handle.assert_called_once_with(obj)

    # abort_alloc returns the page via the adapter's real free
    alloc.abort_alloc(obj)
    adapter.free.assert_called_once_with(obj)

    # free/batched_free are no-ops (lifecycle owned by MaruServer)
    adapter.free.reset_mock()
    alloc.free(obj)
    alloc.batched_free([obj])
    adapter.free.assert_not_called()


def test_close_is_idempotent(maru_mocks):
    handler, adapter, _ = maru_mocks
    alloc = MaruMemoryAllocator(_cfg())
    alloc.init_layout(*_LAYOUT)
    alloc.close()
    adapter.close.assert_called_once()
    handler.close.assert_called_once()
    assert not alloc.is_initialized
    alloc.close()  # safe to call again
