# SPDX-License-Identifier: Apache-2.0
"""Exercise streaming retrieval and source ownership through LMCacheEngine."""

# Standard
from typing import Any
from unittest.mock import MagicMock
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus, EventType
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import TokenDatabase

# Local
from .test_broadcast_transfer import make_object

pytestmark = pytest.mark.no_shared_allocator


class RecordingConnector(MockGPUConnector):
    """Write real tensor data and optionally fail after queuing a write."""

    def __init__(self, destination: torch.Tensor, fail_on_write: int | None) -> None:
        super().__init__((2, 1, 4, 1, 3))
        self.destination = destination
        self.fail_on_write = fail_on_write
        self.write_count = 0
        self.pending: list[tuple[MemoryObj, int, int]] = []
        self.fail_fence = False

    def batched_to_gpu(
        self,
        memory_objs: Any = None,
        starts: Any = None,
        ends: Any = None,
        **kwargs: Any,
    ) -> None:
        """Queue writes, then optionally inject a recoverable allocation failure."""
        self.write_count += 1
        self.pending.extend(zip(memory_objs, starts, ends, strict=True))
        if self.write_count == self.fail_on_write:
            raise torch.OutOfMemoryError("injected connector allocation failure")

    def synchronize_load(self) -> None:
        """Consume pending source views to expose premature reuse/release bugs."""
        if self.pending and self.fail_fence:
            raise RuntimeError("injected device failure")
        for obj, start, end in self.pending:
            assert obj.tensor is not None
            self.destination[:, :, start:end].copy_(obj.tensor)
        self.pending.clear()


@pytest.mark.parametrize("async_loading", [False, True])
@pytest.mark.parametrize("failure", [None, 1, 2, 3, "read", "fence"])
@pytest.mark.parametrize("budget", [128, 256])
def test_engine_releases_sources_and_only_publishes_complete_loads(
    async_loading: bool,
    failure: int | str | None,
    budget: int,
) -> None:
    count = 5
    chunk_tokens = 4
    expected = torch.arange(2 * count * chunk_tokens * 3, dtype=torch.float32).reshape(
        1, 2, count * chunk_tokens, 3
    )
    destination = torch.full_like(expected, -1)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_tokens,
        retrieve_buffer_size=budget,
        enable_async_loading=async_loading,
        py_enable_gc=True,
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(2, 1, chunk_tokens, 1, 3),
        use_mla=True,
    )
    keys = [
        CacheEngineKey("test", 1, 0, index, torch.float32) for index in range(count)
    ]
    sources = [
        make_object([expected[:, :, start : start + chunk_tokens]])
        for start in range(0, count * chunk_tokens, chunk_tokens)
    ]
    database = MagicMock(spec=TokenDatabase)
    database.process_tokens.side_effect = lambda **kwargs: iter(
        [
            (index * chunk_tokens, (index + 1) * chunk_tokens, key)
            for index, key in enumerate(keys)
        ]
    )
    connector = RecordingConnector(
        destination, failure if isinstance(failure, int) else None
    )
    connector.fail_fence = failure == "fence"
    engine = LMCacheEngine(
        config,
        metadata,
        database,
        connector,
        lambda tensor, src: None,
        lambda obj, src: obj,
    )
    storage = MagicMock(spec=StorageManager)
    engine.storage_manager = storage
    fetched: list[MemoryObj] = []

    def get(keys: list[CacheEngineKey], location: str) -> list[MemoryObj | None]:
        assert 1 <= len(keys) <= budget // 96
        # Do not prefetch another batch before releasing the preceding one.
        if fetched:
            assert all(obj.get_ref_count() == 0 for obj in fetched)
        batch = [
            None
            if failure == "read" and key.chunk_hash == 2
            else sources[key.chunk_hash]
            for key in keys
        ]
        fetched.extend(obj for obj in batch if obj is not None)
        return batch

    storage.batched_get.side_effect = get
    storage.get_block_mapping.return_value = {
        "LocalCPUBackend": [
            (key, index * chunk_tokens, (index + 1) * chunk_tokens)
            for index, key in enumerate(keys)
        ]
    }
    event_loop = asyncio.new_event_loop()
    if async_loading:
        future = event_loop.create_future()
        if failure == "read":
            future.set_exception(OSError("injected prefetch failure"))
        else:
            future.set_result([list(zip(keys, sources, strict=True))])
        engine.event_manager.add_event(EventType.LOADING, "test", future)
        engine.event_manager.update_event_status(
            EventType.LOADING, "test", EventStatus.DONE
        )
    try:
        if failure == "fence":
            with pytest.raises(RuntimeError, match="synchronization failed"):
                engine.retrieve(list(range(count * chunk_tokens)), req_id="test")
            assert not engine.is_healthy()
            assert connector.pending
            # Admission is now disabled, without releasing held input objects.
            held_sources = sources if async_loading else fetched
            assert any(obj.get_ref_count() > 0 for obj in held_sources)
            assert not engine.retrieve(
                list(range(count * chunk_tokens)), req_id="test"
            ).any()
            connector.fail_fence = False
            engine.close()
            assert all(obj.get_ref_count() == 0 for obj in held_sources)
            return
        result = engine.retrieve(list(range(count * chunk_tokens)), req_id="test")
        assert not connector.pending
        if failure is None:
            assert result.all()
            torch.testing.assert_close(destination, expected)
        else:
            assert not result.any()
        owned_sources = sources if async_loading and failure != "read" else fetched
        assert all(source.get_ref_count() == 0 for source in owned_sources)
        engine.lookup_unpin("test")
        assert all(source.get_ref_count() == 0 for source in owned_sources)
        if not async_loading:
            storage.batched_get.assert_called()
    finally:
        connector.fail_fence = False
        engine.close()
        event_loop.close()


@pytest.mark.parametrize("budget", [0, -1])
def test_retrieval_budget_must_be_positive(budget: int) -> None:
    config = LMCacheEngineConfig.from_defaults(retrieve_buffer_size=budget)
    with pytest.raises(ValueError, match="retrieve_buffer_size"):
        config.validate()


def test_retrieval_budget_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMCACHE_RETRIEVE_BUFFER_SIZE", "4096")
    assert LMCacheEngineConfig.from_env().retrieve_buffer_size == 4096


def test_unhealthy_engine_does_not_read_sources() -> None:
    config = LMCacheEngineConfig.from_defaults(
        retrieve_buffer_size=128, py_enable_gc=True
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(1, 1, 4, 1, 3),
        use_mla=True,
    )
    database = MagicMock(spec=TokenDatabase)
    engine = LMCacheEngine(
        config,
        metadata,
        database,
        RecordingConnector(torch.empty(1, 1, 4, 3), None),
        lambda tensor, src: None,
        lambda obj, src: obj,
    )
    try:
        engine.mark_init_failed("test")
        assert not engine.retrieve([1, 2, 3, 4]).any()
        database.process_tokens.assert_not_called()
    finally:
        engine.close()
