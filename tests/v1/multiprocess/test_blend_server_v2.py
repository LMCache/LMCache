# SPDX-License-Identifier: Apache-2.0
"""
Unit and integration tests for BlendTokenRangeMatcher and BlendEngineV2.

Structure
---------
Part 1 – BlendTokenRangeMatcher (pure unit tests, no GPU/server needed)
    Tests the rolling-hash sub-sequence matching logic in isolation.

Part 2 – BlendEngineV2 integration tests (two-process ZMQ architecture)
    Uses CB_LOOKUP_PRE_COMPUTED_V2 / CB_RETRIEVE_PRE_COMPUTED_V2, which
    return/accept list[CBMatchResult] instead of list[tuple[int, int]].

Tests cover:
1. BlendTokenRangeMatcher – empty, exact, sub-sequence, partial, no-match
2. Server startup and basic connectivity
3. CB KV cache registration/unregistration
4. CB Store Pre-Computed
5. CB Lookup Pre-Computed V2 (returns list[CBMatchResult]; sub-sequence tests)
6. CB Retrieve Pre-Computed V2 (accepts list[CBMatchResult]; data-correctness tests)
7. CB Store Final (bridge to normal operations)
"""

# Standard
from typing import Generator
import multiprocessing as mp
import os
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_PROMETHEUS_CONFIG
from lmcache.v1.multiprocess.blend_server_v2 import BlendTokenRangeMatcher
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_response_class,
)

# Configuration constants
SERVER_HOST = "localhost"
SERVER_PORT = (
    5557  # Different from test_cache_server (5555) and test_blend_server (5556)
)
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0
DEFAULT_TIMEOUT = 10.0


# =============================================================================
# Part 1: BlendTokenRangeMatcher unit tests
# =============================================================================


class TestBlendTokenRangeMatcher:
    """Pure unit tests for BlendTokenRangeMatcher – no GPU or server needed."""

    def test_empty_matcher_returns_empty(self):
        """match_sub_sequence on a fresh matcher always returns []."""
        matcher = BlendTokenRangeMatcher(chunk_size=4)
        assert matcher.match_sub_sequence([1, 2, 3, 4, 5]) == []

    def test_query_shorter_than_chunk_size_returns_empty(self):
        """Queries with fewer tokens than chunk_size cannot contain a full chunk."""
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        matcher.on_new_token_hashes(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [ObjectKey.IntHash2Bytes(1001), ObjectKey.IntHash2Bytes(1002)],
        )
        assert matcher.match_sub_sequence([1, 2, 3]) == []

    def test_on_new_token_hashes_with_partial_chunk_does_not_crash(self):
        """If token_ids has fewer than chunk_size tokens, no chunk is registered."""
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        # Only 3 tokens – not enough for even one chunk
        matcher.on_new_token_hashes([1, 2, 3], [])
        assert matcher.match_sub_sequence([1, 2, 3, 4, 5]) == []

    def test_exact_single_chunk_match_at_position_zero(self):
        """Register one chunk; query with exact same tokens → match at cur_st=0."""
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        stored = [10, 20, 30, 40]
        th = 42
        matcher.on_new_token_hashes(stored, [ObjectKey.IntHash2Bytes(th)])

        # Query must be at least chunk_size long
        query = stored + [99]
        results = matcher.match_sub_sequence(query)

        assert len(results) == 1
        r = results[0]
        assert r.old_st == 0
        assert r.old_ed == chunk_size
        assert r.cur_st == 0
        assert r.cur_ed == chunk_size
        assert isinstance(r.hash, bytes)

    def test_exact_multi_chunk_match(self):
        """Register two consecutive chunks; exact query finds both."""
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        stored = [1, 2, 3, 4, 5, 6, 7, 8]
        matcher.on_new_token_hashes(
            stored, [ObjectKey.IntHash2Bytes(100), ObjectKey.IntHash2Bytes(200)]
        )

        results = matcher.match_sub_sequence(stored)

        assert len(results) == 2
        by_old_st = sorted(results, key=lambda r: r.old_st)
        assert by_old_st[0].old_st == 0
        assert by_old_st[0].old_ed == chunk_size
        assert by_old_st[0].cur_st == 0
        assert by_old_st[0].cur_ed == chunk_size
        assert by_old_st[1].old_st == chunk_size
        assert by_old_st[1].old_ed == 2 * chunk_size
        assert by_old_st[1].cur_st == chunk_size
        assert by_old_st[1].cur_ed == 2 * chunk_size

    def test_sub_sequence_match_at_nonzero_position(self):
        """
        Key V2 feature: a stored chunk that appears at a non-zero, non-aligned
        position in the query is still found.

        Stored: [1, 2, 3, 4]  (one chunk at old_st=0)
        Query:  [100, 200, 1, 2, 3, 4, 300]
        The stored chunk starts at position 2 in the query → cur_st=2.
        """
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        stored = [1, 2, 3, 4]
        matcher.on_new_token_hashes(stored, [ObjectKey.IntHash2Bytes(777)])

        query = [100, 200, 1, 2, 3, 4, 300]
        results = matcher.match_sub_sequence(query)

        assert len(results) == 1
        r = results[0]
        assert r.old_st == 0
        assert r.old_ed == chunk_size
        assert r.cur_st == 2
        assert r.cur_ed == 2 + chunk_size

    def test_sub_sequence_second_chunk_only(self):
        """
        Two chunks are registered, but the query only contains the second chunk
        at a non-aligned position.

        Stored: [1,2,3,4, 5,6,7,8]  (two chunks)
        Query:  [100, 200, 300, 5, 6, 7, 8, 400]
        Only the second stored chunk ([5,6,7,8]) appears → one match, old_st=4.
        """
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        stored = [1, 2, 3, 4, 5, 6, 7, 8]
        matcher.on_new_token_hashes(
            stored, [ObjectKey.IntHash2Bytes(101), ObjectKey.IntHash2Bytes(202)]
        )

        query = [100, 200, 300, 5, 6, 7, 8, 400]
        results = matcher.match_sub_sequence(query)

        assert len(results) == 1
        r = results[0]
        # Second stored chunk
        assert r.old_st == chunk_size
        assert r.old_ed == 2 * chunk_size
        # Appears at position 3 in the query rolling window
        assert r.cur_st == 3
        assert r.cur_ed == 3 + chunk_size

    def test_no_match_returns_empty(self):
        """Completely disjoint tokens → empty result."""
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        matcher.on_new_token_hashes([1, 2, 3, 4], [ObjectKey.IntHash2Bytes(999)])

        results = matcher.match_sub_sequence([5, 6, 7, 8, 9])
        assert results == []

    def test_hash_field_encodes_token_hash(self):
        """
        CBMatchResult.hash must equal the bytes hash passed
        to on_new_token_hashes.
        """
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        stored = [10, 20, 30, 40]
        th_bytes = ObjectKey.IntHash2Bytes(12345)
        matcher.on_new_token_hashes(stored, [th_bytes])

        results = matcher.match_sub_sequence(stored + [0])
        assert len(results) == 1
        assert results[0].hash == th_bytes

    def test_multiple_registrations_accumulate(self):
        """
        Two separate on_new_token_hashes calls; query containing both
        stored chunks finds both.
        """
        chunk_size = 4
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)
        matcher.on_new_token_hashes([1, 2, 3, 4], [ObjectKey.IntHash2Bytes(1001)])
        matcher.on_new_token_hashes([5, 6, 7, 8], [ObjectKey.IntHash2Bytes(1002)])

        query = [1, 2, 3, 4, 5, 6, 7, 8]
        results = matcher.match_sub_sequence(query)
        assert len(results) == 2

    def test_large_chunk_size(self):
        """
        Verify that BlendTokenRangeMatcher works correctly with a larger
        chunk_size (CHUNK_SIZE = 256), using actual-production-sized tokens.
        """
        chunk_size = CHUNK_SIZE
        matcher = BlendTokenRangeMatcher(chunk_size=chunk_size)

        # Two non-overlapping production-sized chunks
        chunk_a = list(range(1000, 1000 + chunk_size))
        chunk_b = list(range(2000, 2000 + chunk_size))
        stored = chunk_a + chunk_b
        matcher.on_new_token_hashes(
            stored, [ObjectKey.IntHash2Bytes(5001), ObjectKey.IntHash2Bytes(5002)]
        )

        # Query: some prefix, then chunk_a, then some suffix
        prefix = list(range(9000, 9000 + chunk_size))
        query = prefix + chunk_a
        results = matcher.match_sub_sequence(query)

        assert len(results) == 1
        r = results[0]
        assert r.old_st == 0
        assert r.old_ed == chunk_size
        assert r.cur_st == chunk_size  # chunk_a starts after the prefix
        assert r.cur_ed == 2 * chunk_size


# =============================================================================
# Helper Functions and Classes for integration tests
# =============================================================================


def initialize_plain_kv_cache(
    device: torch.device,
    num_layers: int = 32,
    num_tokens: int = 4096,
    hidden_dim: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Initialize a plain [2, L, T, D] KV cache tensor for PlainGPUCacheContext."""
    torch.random.manual_seed(42)
    return torch.rand(
        (2, num_layers, num_tokens, hidden_dim),
        dtype=dtype,
        device=device,
    )


def initialize_paged_kv_cache(
    device: torch.device,
    num_pages: int = 1024,
    num_layers: int = 32,
    page_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Initialize paged KV cache tensors for standard GPUCacheContext."""
    torch.random.manual_seed(42)
    return [
        torch.rand(
            (2, num_pages, page_size, num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]


class CBClientContext:
    """Client context for CB operations with plain [2, L, T, D] GPU buffer."""

    def __init__(
        self,
        device: torch.device,
        num_layers: int = 32,
        num_tokens: int = 4096,
        hidden_dim: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.gpu_kv_cache = initialize_plain_kv_cache(
            device, num_layers, num_tokens, hidden_dim, dtype
        )

    def get_kv_cache(self) -> KVCache:
        return [CudaIPCWrapper(self.gpu_kv_cache)]

    def get_tensor_slice(self, start_token: int, num_tokens: int) -> torch.Tensor:
        return self.gpu_kv_cache[:, :, start_token : start_token + num_tokens, :]

    def set_tensor_slice(self, start_token: int, num_tokens: int, value: float) -> None:
        self.gpu_kv_cache[:, :, start_token : start_token + num_tokens, :] = value


class ClientContext:
    """Client context for standard (non-CB) operations with paged GPU buffer."""

    def __init__(
        self,
        device: torch.device,
        num_pages: int = 1024,
        num_layers: int = 32,
        page_size: int = 16,
        num_heads: int = 8,
        head_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.num_pages = num_pages
        self.num_layers = num_layers
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype
        self.gpu_kv_caches = initialize_paged_kv_cache(
            device, num_pages, num_layers, page_size, num_heads, head_size, dtype
        )

    def get_kv_cache(self) -> KVCache:
        return [CudaIPCWrapper(tensor) for tensor in self.gpu_kv_caches]

    def get_tensor_slice(
        self, layer: int, start_page: int, num_pages: int
    ) -> torch.Tensor:
        return self.gpu_kv_caches[layer][:, start_page : start_page + num_pages]


def create_cb_cache_key(
    token_ids: tuple[int, ...],
    model: str = "testmodel",
    request_id: str = "test-request",
    worker_id: int | None = 0,
) -> IPCCacheEngineKey:
    """Create a cache key for CB testing."""
    return IPCCacheEngineKey(
        model_name=model,
        world_size=1,
        worker_id=worker_id,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id=request_id,
    )


def create_cache_key(
    token_ids: tuple[int, ...],
    model: str = "testmodel",
    request_id: str = "test-request-norm",
    worker_id: int | None = 0,
) -> IPCCacheEngineKey:
    return IPCCacheEngineKey(
        model_name=model,
        world_size=1,
        worker_id=worker_id,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id=request_id,
    )


def expected_full_chunks(num_tokens: int, chunk_size: int = CHUNK_SIZE) -> int:
    """Number of tokens covered by complete chunks only."""
    return (num_tokens // chunk_size) * chunk_size


# =============================================================================
# Server Process Runner (BlendEngineV2)
# =============================================================================


def server_process_runner_v2(
    host: str, port: int, chunk_size: int, cpu_buffer_size: float
):
    """Entry point for the server process running BlendEngineV2."""
    # First Party
    from lmcache.v1.multiprocess.blend_server_v2 import run_cache_server

    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=int(cpu_buffer_size * 1024**3),
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    run_cache_server(
        storage_manager_config=storage_manager_config,
        prometheus_config=DEFAULT_PROMETHEUS_CONFIG,
        host=host,
        port=port,
        chunk_size=chunk_size,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_process() -> Generator[mp.Process, None, None]:
    """Start the BlendEngineV2 server in a separate process for the module."""
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=server_process_runner_v2,
        args=(SERVER_HOST, SERVER_PORT, CHUNK_SIZE, CPU_BUFFER_SIZE),
        daemon=True,
    )
    process.start()
    time.sleep(3)
    yield process

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="module")
def zmq_context() -> Generator[zmq.Context, None, None]:
    context = zmq.Context.instance()
    yield context


@pytest.fixture(scope="function")
def client(
    server_process: mp.Process, zmq_context: zmq.Context
) -> Generator[MessageQueueClient, None, None]:
    c = MessageQueueClient(server_url=SERVER_URL, context=zmq_context)
    yield c
    c.close()


@pytest.fixture(scope="function")
def cb_client_context() -> Generator[CBClientContext, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda:0")
    ctx = CBClientContext(device=device)
    yield ctx
    del ctx.gpu_kv_cache
    torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def client_context() -> Generator[ClientContext, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda:0")
    ctx = ClientContext(device=device)
    yield ctx
    del ctx.gpu_kv_caches
    torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def cb_registered_instance(
    client: MessageQueueClient, cb_client_context: CBClientContext
) -> Generator[int, None, None]:
    """Register a CB KV cache instance; unregister and clear after the test."""
    instance_id = os.getpid() + 1000

    future = client.submit_request(
        RequestType.CB_REGISTER_KV_CACHE,
        [instance_id, cb_client_context.get_kv_cache(), "testmodel", 1],
        get_response_class(RequestType.CB_REGISTER_KV_CACHE),
    )
    assert future.result(timeout=DEFAULT_TIMEOUT) is None

    yield instance_id

    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        client.submit_request(
            RequestType.CB_UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        print(f"Error during CB unregister: {e}")


@pytest.fixture(scope="function")
def registered_instance(
    client: MessageQueueClient, client_context: ClientContext
) -> Generator[int, None, None]:
    """Register a standard KV cache instance; unregister and clear after test."""
    instance_id = os.getpid()

    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [instance_id, client_context.get_kv_cache(), "testmodel", 1],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    )
    assert future.result(timeout=DEFAULT_TIMEOUT) is None

    yield instance_id

    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.UNREGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        print(f"Error during unregister: {e}")


# =============================================================================
# Part 2: BlendEngineV2 Integration Tests
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Server startup and basic connectivity
# ---------------------------------------------------------------------------


def test_server_running_v2(server_process: mp.Process):
    """Server process should be alive."""
    assert server_process.is_alive(), "BlendEngineV2 server process should be running"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NOOP request requires CUDA server"
)
def test_noop_request_v2(client: MessageQueueClient):
    """NOOP request should return 'OK'."""
    result = client.submit_request(
        RequestType.NOOP, [], get_response_class(RequestType.NOOP)
    ).result(timeout=DEFAULT_TIMEOUT)
    assert result == "OK"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GET_CHUNK_SIZE requires CUDA server"
)
def test_get_chunk_size_v2(client: MessageQueueClient):
    """Server should report the configured chunk size."""
    chunk_size = client.submit_request(
        RequestType.GET_CHUNK_SIZE, [], get_response_class(RequestType.GET_CHUNK_SIZE)
    ).result(timeout=DEFAULT_TIMEOUT)
    assert chunk_size == CHUNK_SIZE


# ---------------------------------------------------------------------------
# 2. CB KV cache registration / unregistration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB register/unregister requires CUDA"
)
def test_cb_register_unregister_kv_cache_v2(
    client: MessageQueueClient, cb_client_context: CBClientContext
):
    """Register then unregister a CB KV cache; both should return None."""
    instance_id = os.getpid() + 2000

    result = client.submit_request(
        RequestType.CB_REGISTER_KV_CACHE,
        [instance_id, cb_client_context.get_kv_cache(), "testmodel", 1],
        get_response_class(RequestType.CB_REGISTER_KV_CACHE),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert result is None

    result = client.submit_request(
        RequestType.CB_UNREGISTER_KV_CACHE,
        [instance_id],
        get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert result is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB register multiple instances requires CUDA"
)
def test_cb_register_multiple_instances_v2(
    client: MessageQueueClient, cb_client_context: CBClientContext
):
    """Register and unregister multiple distinct CB instances."""
    base_id = os.getpid() + 3000
    instance_ids = [base_id + i for i in range(3)]

    for iid in instance_ids:
        result = client.submit_request(
            RequestType.CB_REGISTER_KV_CACHE,
            [iid, cb_client_context.get_kv_cache(), "testmodel", 1],
            get_response_class(RequestType.CB_REGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
        assert result is None

    for iid in instance_ids:
        result = client.submit_request(
            RequestType.CB_UNREGISTER_KV_CACHE,
            [iid],
            get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
        assert result is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB unregister nonexistent requires CUDA"
)
def test_cb_unregister_nonexistent_v2(client: MessageQueueClient):
    """Unregistering a non-existent instance should not raise; returns None."""
    result = client.submit_request(
        RequestType.CB_UNREGISTER_KV_CACHE,
        [999999],
        get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert result is None


# ---------------------------------------------------------------------------
# 3. CB Store Pre-Computed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Store Pre-Computed requires CUDA"
)
def test_cb_store_pre_computed_basic_v2(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """Storing one chunk should return True."""
    token_ids = tuple(range(1000, 1000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="store-basic-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    result = (
        client.submit_request(
            RequestType.CB_STORE_PRE_COMPUTED,
            [key, 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )

    assert result is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Pre-Computed multi-chunk requires CUDA",
)
def test_cb_store_pre_computed_multiple_chunks_v2(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """Storing a document with multiple full chunks should succeed."""
    num_tokens = CHUNK_SIZE * 3
    token_ids = tuple(range(2000, 2000 + num_tokens))
    key = create_cb_cache_key(token_ids, request_id="store-multi-chunk-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    result = (
        client.submit_request(
            RequestType.CB_STORE_PRE_COMPUTED,
            [key, 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )

    assert result is True


# ---------------------------------------------------------------------------
# 4. CB Lookup Pre-Computed V2
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Lookup V2 after store requires CUDA"
)
def test_cb_lookup_v2_returns_cb_match_results(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store one chunk then lookup with the same tokens.
    The V2 response must be a list[CBMatchResult] (not list[tuple]).
    """
    token_ids = tuple(range(3000, 3000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="lookup-type-check-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_result = (
        client.submit_request(
            RequestType.CB_STORE_PRE_COMPUTED,
            [key, 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert store_result is True

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) > 0
    assert all(isinstance(r, CBMatchResult) for r in cb_results)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Lookup V2 exact match requires CUDA"
)
def test_cb_lookup_v2_exact_match_fields(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store one chunk then lookup with identical tokens.
    CBMatchResult fields must reflect an exact (position 0) match.
    """
    token_ids = tuple(range(4000, 4000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="lookup-fields-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert len(cb_results) == 1
    r = cb_results[0]
    assert r.old_st == 0
    assert r.old_ed == CHUNK_SIZE
    assert r.cur_st == 0
    assert r.cur_ed == CHUNK_SIZE
    assert isinstance(r.hash, bytes) and len(r.hash) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Lookup V2 sub-sequence requires CUDA"
)
def test_cb_lookup_v2_sub_sequence_match(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Key V2 feature: store a chunk, then look up with a query where the stored
    tokens appear at a non-zero, non-aligned position.

    Stored tokens: [5000 .. 5000+CHUNK_SIZE)   (at GPU offset 0)
    Query tokens:  [9000 .. 9000+CHUNK_SIZE) ++ [5000 .. 5000+CHUNK_SIZE)
                   ^--- dummy prefix --------^   ^--- stored chunk ------^

    Expected match: cur_st = CHUNK_SIZE (stored chunk found in second half).
    """
    stored_tokens = tuple(range(5000, 5000 + CHUNK_SIZE))
    store_key = create_cb_cache_key(stored_tokens, request_id="sub-seq-store-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_result = (
        client.submit_request(
            RequestType.CB_STORE_PRE_COMPUTED,
            [store_key, 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert store_result is True

    # Query: prefix (different tokens) followed by the stored chunk
    prefix_tokens = tuple(range(9000, 9000 + CHUNK_SIZE))
    query_tokens = prefix_tokens + stored_tokens
    lookup_key = create_cb_cache_key(query_tokens, request_id="sub-seq-lookup-v2")

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 1, (
        "Should find exactly one match (the stored chunk embedded in the query)"
    )
    r = cb_results[0]
    # The stored chunk was at old_st=0 in its original sequence
    assert r.old_st == 0
    assert r.old_ed == CHUNK_SIZE
    # It appears at position CHUNK_SIZE (after the prefix) in the query
    assert r.cur_st == CHUNK_SIZE
    assert r.cur_ed == 2 * CHUNK_SIZE


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Lookup V2 no match requires CUDA"
)
def test_cb_lookup_v2_no_match(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """Lookup with tokens never stored → empty list."""
    token_ids = tuple(range(80000, 80000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="no-match-v2")

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Lookup V2 partial match requires CUDA"
)
def test_cb_lookup_v2_partial_match(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store two chunks (A and B).  Query contains A but not B.
    Only one CBMatchResult should be returned.
    """
    chunk_a = tuple(range(6000, 6000 + CHUNK_SIZE))
    chunk_b = tuple(range(7000, 7000 + CHUNK_SIZE))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Store chunk A
    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_a, request_id="partial-A-v2"),
            0,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Store chunk B at a different offset
    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_b, request_id="partial-B-v2"),
            CHUNK_SIZE,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Query contains only chunk A (not chunk B)
    dummy = tuple(range(8000, 8000 + CHUNK_SIZE))
    query_tokens = chunk_a + dummy
    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [create_cb_cache_key(query_tokens, request_id="partial-lookup-v2")],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 1
    assert cb_results[0].cur_st == 0
    assert cb_results[0].cur_ed == CHUNK_SIZE


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup V2 isolation from normal store requires CUDA",
)
def test_cb_lookup_v2_cannot_find_normal_store(
    client: MessageQueueClient,
    client_context: ClientContext,
    cb_client_context: CBClientContext,
    registered_instance: int,
    cb_registered_instance: int,
):
    """
    ISOLATION: data stored via normal STORE must NOT be found by
    CB_LOOKUP_PRE_COMPUTED_V2 (returns []).
    """
    token_ids = tuple(range(CHUNK_SIZE))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_key = create_cache_key(token_ids, request_id="isolation-normal-v2")
    client.submit_request(
        RequestType.STORE,
        [store_key, registered_instance, list(range(16)), event.ipc_handle()],
        get_response_class(RequestType.STORE),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [create_cb_cache_key(token_ids, request_id="isolation-cb-v2")],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 0, (
        "CB_LOOKUP_PRE_COMPUTED_V2 must not see data stored via normal STORE"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup V2 multiple chunks aligned requires CUDA",
)
def test_cb_lookup_v2_multiple_chunks_found(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store two independent chunks then look up with a query containing both at
    aligned positions.  Both CBMatchResults should be returned.

    Stored:  chunk_A (30000..30256), chunk_B (31000..31256) – separate docs.
    Query:   [chunk_A || chunk_B]  (2 × CHUNK_SIZE tokens, aligned)
    Expected: 2 results – cur_st=0 for chunk_A, cur_st=CHUNK_SIZE for chunk_B.
    """
    chunk_a = tuple(range(30000, 30000 + CHUNK_SIZE))
    chunk_b = tuple(range(31000, 31000 + CHUNK_SIZE))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_a, request_id="multi-aligned-store-a"),
            0,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_b, request_id="multi-aligned-store-b"),
            CHUNK_SIZE,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    query_tokens = chunk_a + chunk_b
    lookup_key = create_cb_cache_key(query_tokens, request_id="multi-aligned-lookup-v2")

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 2, "Both stored chunks must be found"
    sorted_results = sorted(cb_results, key=lambda r: r.cur_st)
    assert sorted_results[0].cur_st == 0
    assert sorted_results[0].cur_ed == CHUNK_SIZE
    assert sorted_results[1].cur_st == CHUNK_SIZE
    assert sorted_results[1].cur_ed == 2 * CHUNK_SIZE


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup V2 multiple chunks unaligned requires CUDA",
)
def test_cb_lookup_v2_multiple_chunks_unaligned(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store two independent chunks then look up with a query where each chunk
    appears at a non-zero, non-consecutive position (separated by dummy chunks).

    Stored:  chunk_A (32000..32256), chunk_B (33000..33256) – separate docs.
    Query:   [prefix_X || chunk_A || prefix_Y || chunk_B]  (4 × CHUNK_SIZE)
    Expected: cur_st=CHUNK_SIZE for chunk_A, cur_st=3×CHUNK_SIZE for chunk_B.
    """
    chunk_a = tuple(range(32000, 32000 + CHUNK_SIZE))
    chunk_b = tuple(range(33000, 33000 + CHUNK_SIZE))
    prefix_x = tuple(range(34000, 34000 + CHUNK_SIZE))
    prefix_y = tuple(range(35000, 35000 + CHUNK_SIZE))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_a, request_id="multi-unaligned-store-a"),
            0,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_b, request_id="multi-unaligned-store-b"),
            CHUNK_SIZE,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Query: prefix_X || chunk_A || prefix_Y || chunk_B  (non-consecutive positions)
    query_tokens = prefix_x + chunk_a + prefix_y + chunk_b
    lookup_key = create_cb_cache_key(
        query_tokens, request_id="multi-unaligned-lookup-v2"
    )

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 2, "Both chunks must be found at non-aligned positions"
    sorted_results = sorted(cb_results, key=lambda r: r.cur_st)
    # chunk_A appears after one prefix chunk
    assert sorted_results[0].cur_st == CHUNK_SIZE
    assert sorted_results[0].cur_ed == 2 * CHUNK_SIZE
    # chunk_B appears after prefix_X + chunk_A + prefix_Y
    assert sorted_results[1].cur_st == 3 * CHUNK_SIZE
    assert sorted_results[1].cur_ed == 4 * CHUNK_SIZE


# ---------------------------------------------------------------------------
# 5. CB Retrieve Pre-Computed V2
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Retrieve V2 basic requires CUDA"
)
def test_cb_retrieve_v2_after_store_and_lookup(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """Store → lookup (V2) → retrieve (V2): retrieve should return True."""
    token_ids = tuple(range(10000, 10000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="retrieve-basic-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert len(cb_results) > 0

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    result = (
        client.submit_request(
            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
            [key, cb_results, CHUNK_SIZE, cb_registered_instance, event2.ipc_handle()],
            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )

    assert result is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve V2 data correctness requires CUDA",
)
def test_cb_retrieve_v2_verify_data_correctness(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store known data at offset 0, retrieve to a different offset,
    then verify that the destination slice matches the source.
    """
    source_offset = 0
    dest_offset = CHUNK_SIZE
    source_value = 0.25

    # Fill the whole buffer with 0, then write a known value in the source region
    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 0.0)
    cb_client_context.set_tensor_slice(source_offset, CHUNK_SIZE, source_value)

    token_ids = tuple(range(11000, 11000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="retrieve-correctness-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, source_offset, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Reset buffer so destination area is zero before retrieve
    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 0.0)
    cb_client_context.set_tensor_slice(source_offset, CHUNK_SIZE, source_value)

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert len(cb_results) == 1

    # Sanity: destination slice differs before retrieve
    src_slice = cb_client_context.get_tensor_slice(source_offset, CHUNK_SIZE)
    dst_slice = cb_client_context.get_tensor_slice(dest_offset, CHUNK_SIZE)
    assert not torch.allclose(dst_slice, src_slice, atol=1e-4)

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    result = (
        client.submit_request(
            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
            [key, cb_results, dest_offset, cb_registered_instance, event2.ipc_handle()],
            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result is True

    torch.cuda.synchronize()
    src_slice = cb_client_context.get_tensor_slice(source_offset, CHUNK_SIZE)
    dst_slice = cb_client_context.get_tensor_slice(dest_offset, CHUNK_SIZE)
    assert torch.allclose(src_slice, dst_slice, atol=1e-4), (
        "Retrieved data at dest_offset should match the stored source data"
    )

    # Regions outside the retrieved range should remain zero
    beyond = cb_client_context.get_tensor_slice(
        dest_offset + CHUNK_SIZE,
        cb_client_context.num_tokens - (dest_offset + CHUNK_SIZE),
    )
    assert torch.allclose(beyond, torch.zeros_like(beyond), atol=1e-4), (
        "Regions beyond the retrieved window should be unchanged (zeros)"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Retrieve V2 sub-sequence requires CUDA"
)
def test_cb_retrieve_v2_sub_sequence(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    End-to-end V2 sub-sequence test:
    1. Store chunk A at offset 0 in the GPU buffer.
    2. Lookup with query = [prefix || chunk_A]; match should be at cur_st=CHUNK_SIZE.
    3. Retrieve with the CBMatchResult and offset=0.
       gpu_st = cur_st + offset = CHUNK_SIZE → data copied to slot CHUNK_SIZE.
    4. Verify the destination slice (starting at CHUNK_SIZE) matches the source.
    """
    source_value = 0.125
    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 0.0)
    cb_client_context.set_tensor_slice(0, CHUNK_SIZE, source_value)

    stored_tokens = tuple(range(12000, 12000 + CHUNK_SIZE))
    store_key = create_cb_cache_key(stored_tokens, request_id="sub-seq-e2e-store-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [store_key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Build query: dummy prefix + stored chunk
    prefix_tokens = tuple(range(99000, 99000 + CHUNK_SIZE))
    query_tokens = prefix_tokens + stored_tokens
    lookup_key = create_cb_cache_key(query_tokens, request_id="sub-seq-e2e-lookup-v2")

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert len(cb_results) == 1
    assert cb_results[0].cur_st == CHUNK_SIZE

    # Reset destination region before retrieve
    cb_client_context.set_tensor_slice(CHUNK_SIZE, CHUNK_SIZE, 0.0)

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    result = (
        client.submit_request(
            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
            [lookup_key, cb_results, 0, cb_registered_instance, event2.ipc_handle()],
            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result is True

    torch.cuda.synchronize()
    # Data should have been copied to gpu_st = cur_st + offset = CHUNK_SIZE + 0
    src_slice = cb_client_context.get_tensor_slice(0, CHUNK_SIZE)
    dst_slice = cb_client_context.get_tensor_slice(CHUNK_SIZE, CHUNK_SIZE)
    assert torch.allclose(src_slice, dst_slice, atol=1e-4), (
        "Sub-sequence retrieve must copy data to cur_st + offset in the GPU buffer"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve V2 multiple chunks data correctness requires CUDA",
)
def test_cb_retrieve_v2_multiple_chunks_data_correctness(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Store two chunks with distinct fill values, look up both, then retrieve
    both to separate destination slots and verify data at each slot.

    GPU layout (slots of CHUNK_SIZE tokens each):
      slot 0 (source A) → value 0.25  ← chunk_A stored here
      slot 1 (source B) → value 0.50  ← chunk_B stored here
      slot 2 (dest A)   ← retrieve with offset=2*CHUNK_SIZE, cur_st=0
      slot 3 (dest B)   ← retrieve with offset=2*CHUNK_SIZE, cur_st=CHUNK_SIZE

    gpu_st formula: cur_st + offset
      chunk_A: 0       + 2*CHUNK_SIZE = 2*CHUNK_SIZE  (slot 2)
      chunk_B: CHUNK_SIZE + 2*CHUNK_SIZE = 3*CHUNK_SIZE  (slot 3)
    """
    value_a, value_b = 0.25, 0.50

    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 0.0)
    cb_client_context.set_tensor_slice(0, CHUNK_SIZE, value_a)
    cb_client_context.set_tensor_slice(CHUNK_SIZE, CHUNK_SIZE, value_b)

    chunk_a = tuple(range(36000, 36000 + CHUNK_SIZE))
    chunk_b = tuple(range(37000, 37000 + CHUNK_SIZE))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_a, request_id="multi-ret-store-a"),
            0,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            create_cb_cache_key(chunk_b, request_id="multi-ret-store-b"),
            CHUNK_SIZE,
            cb_registered_instance,
            event.ipc_handle(),
        ],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Lookup with query = [chunk_A || chunk_B]
    query_tokens = chunk_a + chunk_b
    lookup_key = create_cb_cache_key(query_tokens, request_id="multi-ret-lookup-v2")

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert len(cb_results) == 2

    # Clear dest slots before retrieve
    dest_offset = 2 * CHUNK_SIZE
    cb_client_context.set_tensor_slice(dest_offset, 2 * CHUNK_SIZE, 0.0)

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    result = (
        client.submit_request(
            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
            [
                lookup_key,
                cb_results,
                dest_offset,
                cb_registered_instance,
                event2.ipc_handle(),
            ],
            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result is True

    torch.cuda.synchronize()

    # chunk_A → gpu_st = 0 + 2*CHUNK_SIZE = slot 2
    src_a = cb_client_context.get_tensor_slice(0, CHUNK_SIZE)
    dst_a = cb_client_context.get_tensor_slice(dest_offset, CHUNK_SIZE)
    assert torch.allclose(dst_a, src_a, atol=1e-4), (
        "Slot 2 must match slot 0 (chunk_A source)"
    )

    # chunk_B → gpu_st = CHUNK_SIZE + 2*CHUNK_SIZE = slot 3
    src_b = cb_client_context.get_tensor_slice(CHUNK_SIZE, CHUNK_SIZE)
    dst_b = cb_client_context.get_tensor_slice(dest_offset + CHUNK_SIZE, CHUNK_SIZE)
    assert torch.allclose(dst_b, src_b, atol=1e-4), (
        "Slot 3 must match slot 1 (chunk_B source)"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve V2 unaligned with non-zero offset requires CUDA",
)
def test_cb_retrieve_v2_unaligned_nonzero_offset(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Retrieve a chunk that is unaligned in the query (cur_st = CHUNK_SIZE) with
    a non-zero offset, so the effective GPU destination is cur_st + offset =
    2 × CHUNK_SIZE.

    GPU layout:
      slot 0 (source) → value 0.75  ← chunk stored here (offset=0)
      slot 1 (prefix) → irrelevant dummy prefix tokens in the query
      slot 2 (dest)   ← gpu_st = CHUNK_SIZE + CHUNK_SIZE = 2*CHUNK_SIZE

    Confirms that the gpu_st = cur_st + offset formula applies correctly when
    both terms are non-zero.
    """
    source_value = 0.75

    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 0.0)
    cb_client_context.set_tensor_slice(0, CHUNK_SIZE, source_value)

    stored_tokens = tuple(range(38000, 38000 + CHUNK_SIZE))
    store_key = create_cb_cache_key(
        stored_tokens, request_id="unaligned-offset-store-v2"
    )

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [store_key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Query: [prefix || stored_chunk] → cur_st = CHUNK_SIZE
    prefix_tokens = tuple(range(39000, 39000 + CHUNK_SIZE))
    query_tokens = prefix_tokens + stored_tokens
    lookup_key = create_cb_cache_key(
        query_tokens, request_id="unaligned-offset-lookup-v2"
    )

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)
    assert len(cb_results) == 1
    assert cb_results[0].cur_st == CHUNK_SIZE

    # Retrieve with offset=CHUNK_SIZE → gpu_st = CHUNK_SIZE + CHUNK_SIZE = 2*CHUNK_SIZE
    retrieve_offset = CHUNK_SIZE
    cb_client_context.set_tensor_slice(2 * CHUNK_SIZE, CHUNK_SIZE, 0.0)

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    result = (
        client.submit_request(
            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
            [
                lookup_key,
                cb_results,
                retrieve_offset,
                cb_registered_instance,
                event2.ipc_handle(),
            ],
            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result is True

    torch.cuda.synchronize()
    src_slice = cb_client_context.get_tensor_slice(0, CHUNK_SIZE)
    dst_slice = cb_client_context.get_tensor_slice(2 * CHUNK_SIZE, CHUNK_SIZE)
    assert torch.allclose(src_slice, dst_slice, atol=1e-4), (
        "gpu_st = cur_st + offset = 2*CHUNK_SIZE must hold the stored source data"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Retrieve V2 empty results requires CUDA"
)
def test_cb_retrieve_v2_empty_match_results(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """Retrieve with empty CBMatchResult list is a no-op → returns True."""
    token_ids = tuple(range(13000, 13000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="empty-results-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    result = (
        client.submit_request(
            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
            [key, [], 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )

    assert result is True


# ---------------------------------------------------------------------------
# 6. CB Store Final (bridge to normal operations)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CB Store Final basic requires CUDA"
)
def test_cb_store_final_basic_v2(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """CB_STORE_FINAL with a valid key and offset should return True."""
    token_ids = tuple(range(14000, 14000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="final-basic-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    result = (
        client.submit_request(
            RequestType.CB_STORE_FINAL,
            [key, 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_FINAL),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )

    assert result is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Final then normal lookup requires CUDA",
)
def test_cb_store_final_v2_then_normal_lookup(
    client: MessageQueueClient,
    client_context: ClientContext,
    cb_client_context: CBClientContext,
    registered_instance: int,
    cb_registered_instance: int,
):
    """
    Store via CB_STORE_FINAL, then verify that normal LOOKUP finds the chunks
    and RETRIEVE returns matching data.
    """
    source_value = 0.5
    cb_client_context.set_tensor_slice(0, CHUNK_SIZE, source_value)

    token_ids = tuple(range(15000, 15000 + CHUNK_SIZE))
    cb_key = create_cb_cache_key(token_ids, request_id="final-norm-lookup-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_result = (
        client.submit_request(
            RequestType.CB_STORE_FINAL,
            [cb_key, 0, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_FINAL),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert store_result is True

    # Normal LOOKUP with worker_id=None
    lookup_key = create_cb_cache_key(
        token_ids, request_id="final-norm-lookup-v2", worker_id=None
    )
    # Phase 1: LOOKUP returns a prefetch job ID, not the chunk count
    job_id = client.submit_request(
        RequestType.LOOKUP, [lookup_key], get_response_class(RequestType.LOOKUP)
    ).result(timeout=DEFAULT_TIMEOUT)

    # Phase 2: Poll QUERY_PREFETCH_STATUS until the result is ready
    lookup_result = None
    while True:
        lookup_result = client.submit_request(
            RequestType.QUERY_PREFETCH_STATUS,
            [job_id],
            get_response_class(RequestType.QUERY_PREFETCH_STATUS),
        ).result(timeout=DEFAULT_TIMEOUT)
        if lookup_result is not None:
            break

    expected_chunks = 1  # one full CHUNK_SIZE chunk
    assert isinstance(lookup_result, int)
    assert lookup_result == expected_chunks, (
        "Normal LOOKUP should find the chunk stored via CB_STORE_FINAL"
    )

    # Normal RETRIEVE
    retrieve_key = create_cb_cache_key(token_ids, request_id="final-norm-retrieve-v2")
    pages_per_chunk = 16
    gpu_block_ids = list(range(pages_per_chunk))
    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    retrieve_result = (
        client.submit_request(
            RequestType.RETRIEVE,
            [retrieve_key, registered_instance, gpu_block_ids, event2.ipc_handle(), 0],
            get_response_class(RequestType.RETRIEVE),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert retrieve_result is True

    torch.cuda.synchronize()
    for layer in range(client_context.num_layers):
        tensor_slice = client_context.get_tensor_slice(layer, 0, pages_per_chunk)
        assert tensor_slice.mean().item() == pytest.approx(source_value, abs=1e-4), (
            f"Layer {layer}: retrieved data should match the stored source value"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Final not visible to CB Lookup V2 requires CUDA",
)
def test_cb_store_final_v2_not_visible_to_cb_lookup_v2(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    ISOLATION: data stored via CB_STORE_FINAL must NOT be found by
    CB_LOOKUP_PRE_COMPUTED_V2 (the BlendTokenRangeMatcher is not updated
    by cb_store_final).
    """
    token_ids = tuple(range(16000, 16000 + CHUNK_SIZE))
    cb_key = create_cb_cache_key(token_ids, request_id="final-not-cb-v2")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    client.submit_request(
        RequestType.CB_STORE_FINAL,
        [cb_key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_FINAL),
    ).to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    cb_results = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        [cb_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(cb_results, list)
    assert len(cb_results) == 0, (
        "CB_LOOKUP_PRE_COMPUTED_V2 should NOT find data stored via CB_STORE_FINAL"
    )
