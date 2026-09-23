# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RESP connector integration.

These tests verify the RESP protocol client implementation, including:
- Basic operations (exists, get, set)
- Batch operations (batch_get, batch_set, batch_exists)
- Error handling
- Different chunk sizes per operation
"""

# Standard
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any
from unittest.mock import patch
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector import CreateConnector

# Local
from ...conftest import MockRESPClient
from ..utils import (
    check_mem_obj_equal,
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)

#: Default deadline (seconds) for synchronous waits on futures submitted to
#: the test event-loop thread. A stalled loop must fail fast with diagnostics
#: instead of hanging CI until the whole job is cancelled.
_FUTURE_RESULT_TIMEOUT_S = 60.0


def bounded_result(
    future: Future[Any],
    op: str,
    async_loop: asyncio.AbstractEventLoop,
    async_thread: threading.Thread,
    timeout: float = _FUTURE_RESULT_TIMEOUT_S,
) -> Any:
    """Wait for a test future with a deadline, failing fast with diagnostics.

    Args:
        future: Future submitted to the test event-loop thread.
        op: Operation label (e.g. "exists", "put", "get") for the message.
        async_loop: Event loop the future was submitted to.
        async_thread: Thread running ``async_loop``.
        timeout: Maximum seconds to wait before failing.

    Returns:
        The future's result.

    Raises:
        AssertionError: On deadline expiry. The message reports the operation
            plus loop/thread/future liveness; the future is cancelled first.
    """
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        diagnosis = (
            f"RESP test operation {op!r} timed out after {timeout}s "
            f"(loop_running={async_loop.is_running()} "
            f"loop_closed={async_loop.is_closed()} "
            f"thread_alive={async_thread.is_alive()} "
            f"future_done={future.done()} "
            f"future_cancelled={future.cancelled()})"
        )
        future.cancel()
        raise AssertionError(diagnosis) from exc


async def _never_completes() -> None:
    """Coroutine that never completes (stall stand-in for regression test)."""
    await asyncio.Future()


async def _return_hello() -> str:
    """Trivial coroutine proving the loop still works after a stall."""
    return "hello"


@pytest.fixture(autouse=True)
def mock_resp_client():
    """Use in-memory MockRESPClient so tests never hit real Redis (no 6379)."""
    with patch(
        "lmcache.v1.storage_backend.connector.redis_connector.RESPClient",
        MockRESPClient,
    ):
        yield


def _get_metadata(use_mla: bool = False):
    """Helper to create test metadata."""
    kv_shape = (32, 1 if use_mla else 2, 256, 1 if use_mla else 8, 128)
    dtype = torch.bfloat16
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
    )
    return metadata


def _create_local_cpu_backend(memory_allocator, use_mla=False, config=None):
    """Helper to create a local CPU backend for testing."""
    if config is None:
        config = LMCacheEngineConfig.from_defaults(
            extra_config={
                "save_chunk_meta": False,  # RESP requires this
                "resp_num_threads": 4,
            }
        )
    metadata = _get_metadata(use_mla)
    return LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )


@pytest.fixture
def resp_url():
    """RESP URL for testing; mock only, no real port (no 6379)."""
    return "resp://mock.local:0"


@pytest.fixture
def resp_config():
    """Create a config for RESP connector testing."""
    return LMCacheEngineConfig.from_defaults(
        extra_config={
            "save_chunk_meta": False,  # RESP requires this
            "resp_num_threads": 4,
        }
    )


@pytest.fixture
def local_backend():
    """Create a local CPU backend for testing."""
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)  # 1GB
    backend = _create_local_cpu_backend(memory_allocator, use_mla=False)
    yield backend
    backend.close()


def test_resp_connector_basic_operations(
    resp_url, local_backend, resp_config, autorelease_v1
):
    """Test basic RESP operations: exists, put, get."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, resp_config)
        )

        random_key = dumb_cache_engine_key()

        # Test 1: Key doesn't exist initially
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert not bounded_result(
            future, op="exists", async_loop=async_loop, async_thread=async_thread
        ), "Key should not exist initially"

        # Test 2: Create and store test data
        num_tokens = 256  # Full chunk
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        # Fill with deterministic test data
        torch.manual_seed(42)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Test 3: Put data
        future = asyncio.run_coroutine_threadsafe(
            connector.put(random_key, memory_obj), async_loop
        )
        bounded_result(
            future, op="put", async_loop=async_loop, async_thread=async_thread
        )

        # Test 4: Key exists after put
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert bounded_result(
            future, op="exists", async_loop=async_loop, async_thread=async_thread
        ), "Key should exist after put"
        assert memory_obj.get_ref_count() == 1

        # Test 5: Get and verify data
        future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
        retrieved_memory_obj = bounded_result(
            future, op="get", async_loop=async_loop, async_thread=async_thread
        )

        check_mem_obj_equal([retrieved_memory_obj], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_resp_connector_batch_operations(
    resp_url, local_backend, resp_config, autorelease_v1
):
    """Test RESP batch operations: batch_put, batch_get, batch_exists."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, resp_config)
        )

        # Create multiple keys (unique chunk_hash per key so each put/get pair matches)
        num_keys = 10
        keys = [dumb_cache_engine_key(i) for i in range(num_keys)]

        # Test 1: Batch exists - all should be False initially
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        count = bounded_result(
            future, op="batch_exists", async_loop=async_loop, async_thread=async_thread
        )
        assert count == 0, "No keys should exist initially"

        # Test 2: Create memory objects
        num_tokens = 256  # Full chunk
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_objs = []

        for i in range(num_keys):
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            # Fill with unique deterministic data
            torch.manual_seed(42 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
            memory_objs.append(memory_obj)

        # Test 3: Batch put
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, memory_objs), async_loop
        )
        bounded_result(
            future, op="put", async_loop=async_loop, async_thread=async_thread
        )

        # Test 4: Batch exists - all should be True now
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        count = bounded_result(
            future, op="batch_exists", async_loop=async_loop, async_thread=async_thread
        )
        assert count == num_keys, "All keys should exist after batch_put"

        # Test 5: Batch get and verify
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys), async_loop
        )
        retrieved_objs = bounded_result(
            future, op="batch_get", async_loop=async_loop, async_thread=async_thread
        )

        assert len(retrieved_objs) == num_keys
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_resp_connector_different_chunk_sizes(resp_url, autorelease_v1):
    """Test that different operations can use different chunk sizes."""
    async_loop, async_thread = init_asyncio_loop()

    # Create backend with 4MB chunks
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "save_chunk_meta": False,
            "resp_num_threads": 4,
        }
    )

    # Metadata with larger chunks
    # (chunk_size must match kv_shape token dim for get_shapes())
    kv_shape = (32, 2, 512, 8, 128)  # Larger chunk size
    dtype = torch.bfloat16
    metadata = LMCacheMetadata(
        model_name="test-model-large",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=kv_shape,
        use_mla=False,
        chunk_size=512,
    )
    local_backend = LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, config)
        )

        # Test with larger chunk
        key1 = dumb_cache_engine_key()
        mem_obj_shape = torch.Size([2, 32, 512, 1024])
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(100)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Put and get with larger chunk
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key1, memory_obj), async_loop
        )
        bounded_result(
            future, op="put", async_loop=async_loop, async_thread=async_thread
        )

        future = asyncio.run_coroutine_threadsafe(connector.get(key1), async_loop)
        retrieved_obj = bounded_result(
            future, op="get", async_loop=async_loop, async_thread=async_thread
        )

        check_mem_obj_equal([retrieved_obj], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


def test_resp_connector_nonexistent_key(
    resp_url, local_backend, resp_config, autorelease_v1
):
    """Test getting a non-existent key returns None."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, resp_config)
        )

        nonexistent_key = dumb_cache_engine_key()

        # Test exists returns False
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(nonexistent_key), async_loop
        )
        assert not bounded_result(
            future, op="exists", async_loop=async_loop, async_thread=async_thread
        )

        # Test get returns None (RESP protocol should handle this gracefully)
        # Note: This might throw an error depending on how RESP handles missing keys
        future = asyncio.run_coroutine_threadsafe(
            connector.get(nonexistent_key), async_loop
        )

        try:
            result = bounded_result(
                future, op="get", async_loop=async_loop, async_thread=async_thread
            )
            assert result is None, "Getting non-existent key should return None"
        except AssertionError:
            raise
        except Exception:
            # RESP might throw an error for missing keys, which is also acceptable
            pass

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_resp_connector_sequential_operations(
    resp_url, local_backend, resp_config, autorelease_v1
):
    """Test multiple sequential put/get operations work correctly."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, resp_config)
        )

        # Perform 5 sequential put/get cycles
        num_ops = 5
        for i in range(num_ops):
            key = dumb_cache_engine_key()

            # Create unique data
            num_tokens = 256
            mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
            dtype = torch.bfloat16
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            torch.manual_seed(1000 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

            # Put
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            bounded_result(
                future, op="put", async_loop=async_loop, async_thread=async_thread
            )

            # Get and verify
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            retrieved_obj = bounded_result(
                future, op="get", async_loop=async_loop, async_thread=async_thread
            )

            check_mem_obj_equal([retrieved_obj], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_resp_connector_concurrent_operations(
    resp_url, local_backend, resp_config, autorelease_v1
):
    """Test that concurrent operations are handled correctly."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, resp_config)
        )

        # Submit multiple operations concurrently
        # (unique keys so each get matches its put)
        num_concurrent = 5
        keys = [dumb_cache_engine_key(i) for i in range(num_concurrent)]
        memory_objs = []

        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16

        # Create and submit all puts concurrently
        put_futures = []
        for i, key in enumerate(keys):
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            torch.manual_seed(2000 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
            memory_objs.append(memory_obj)

            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            put_futures.append(future)

        # Wait for all puts to complete
        for future in put_futures:
            bounded_result(
                future, op="put", async_loop=async_loop, async_thread=async_thread
            )

        # Submit all gets concurrently
        get_futures = []
        for key in keys:
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            get_futures.append(future)

        # Verify all gets
        retrieved_objs = [
            bounded_result(
                f, op="get", async_loop=async_loop, async_thread=async_thread
            )
            for f in get_futures
        ]
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize("num_threads", [1, 4, 8, 16])
def test_resp_connector_thread_scaling(resp_url, num_threads, autorelease_v1):
    """Test RESP connector with different numbers of worker threads."""
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "save_chunk_meta": False,
            "resp_num_threads": num_threads,
        }
    )
    local_backend = _create_local_cpu_backend(memory_allocator, False, config)

    try:
        connector = autorelease_v1(
            CreateConnector(resp_url, async_loop, local_backend, config)
        )

        # Test basic operation with different thread counts
        key = dumb_cache_engine_key()
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(3000)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Put
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        bounded_result(
            future, op="put", async_loop=async_loop, async_thread=async_thread
        )

        # Get and verify
        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_obj = bounded_result(
            future, op="get", async_loop=async_loop, async_thread=async_thread
        )

        check_mem_obj_equal([retrieved_obj], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


def test_resp_connector_stalled_operation_fails_fast() -> None:
    """A stalled loop operation must fail fast with actionable diagnostics."""
    async_loop, async_thread = init_asyncio_loop()
    try:
        stalled = asyncio.run_coroutine_threadsafe(_never_completes(), async_loop)
        with pytest.raises(AssertionError, match="timed out"):
            bounded_result(
                stalled,
                op="get",
                async_loop=async_loop,
                async_thread=async_thread,
                timeout=5.0,
            )
        assert stalled.done(), "Stalled future should be done after cancel"
        healthy = asyncio.run_coroutine_threadsafe(_return_hello(), async_loop)
        assert (
            bounded_result(
                healthy,
                op="get",
                async_loop=async_loop,
                async_thread=async_thread,
            )
            == "hello"
        )
    finally:
        close_asyncio_loop(async_loop, async_thread)
