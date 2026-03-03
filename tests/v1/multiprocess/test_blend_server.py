# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the BlendEngine in blend_server.py.

This test file follows the same two-process architecture as test_cache_server.py:
- A server process running BlendEngine with ZMQ
- A client process using MessageQueueClient

Tests cover:
1. Server startup and basic connectivity
2. CB KV cache registration/unregistration
3. CB Store Pre-Computed
4. CB Lookup Pre-Computed (with isolation tests)
5. CB Retrieve Pre-Computed (with isolation tests)
6. CB Store Final (bridge to normal operations)
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
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_PROMETHEUS_CONFIG
from lmcache.v1.multiprocess.blend_server import get_sep_tokens
from lmcache.v1.multiprocess.custom_types import (
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
SERVER_PORT = 5556  # Different port from test_cache_server.py to avoid conflicts
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0
DEFAULT_TIMEOUT = 10.0
SUPPORT_PARTIAL_CHUNK = False

# =============================================================================
# Helper Functions and Classes
# =============================================================================


def initialize_plain_kv_cache(
    device: torch.device,
    num_layers: int = 32,
    num_tokens: int = 4096,
    hidden_dim: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Initialize a plain [2, L, T, D] KV cache tensor for PlainGPUCacheContext.

    Shape: [2, num_layers, num_tokens, hidden_dim]
    - 2: K and V
    - L: number of layers
    - T: number of tokens
    - D: hidden dimension
    """
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
    """
    Initialize paged KV cache tensors for standard GPUCacheContext.

    Shape per layer: [2, num_pages, page_size, num_heads, head_size]
    """
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
    """
    Client context for CB operations with plain [2, L, T, D] GPU buffer.
    """

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
        """Wrap single tensor in CudaIPCWrapper (list of length 1)."""
        return [CudaIPCWrapper(self.gpu_kv_cache)]

    def get_tensor_slice(self, start_token: int, num_tokens: int) -> torch.Tensor:
        """Get a slice of the KV cache tensor."""
        return self.gpu_kv_cache[:, :, start_token : start_token + num_tokens, :]

    def set_tensor_slice(self, start_token: int, num_tokens: int, value: float) -> None:
        """Set a slice of the KV cache tensor to a specific value."""
        self.gpu_kv_cache[:, :, start_token : start_token + num_tokens, :] = value


class ClientContext:
    """
    Client context for standard (non-CB) operations with paged GPU buffer.
    """

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
        """Wrap GPU tensors in CudaIPCWrapper for IPC communication."""
        return [CudaIPCWrapper(tensor) for tensor in self.gpu_kv_caches]

    def get_tensor_slice(
        self, layer: int, start_page: int, num_pages: int
    ) -> torch.Tensor:
        """Get a slice of the KV cache tensor for a specific layer."""
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


def lookup_keys(keys: list[IPCCacheEngineKey]) -> list[IPCCacheEngineKey]:
    """Create lookup keys: worker_id=None."""
    return [k.no_worker_id_version() for k in keys]


def create_token_ids_with_sep_tokens(token_ids: list[int]) -> tuple[int, ...]:
    """Insert separator tokens between token ids."""
    st_pattern, ed_pattern = get_sep_tokens()
    len_st = len(st_pattern)
    len_ed = len(ed_pattern)
    result = st_pattern + token_ids[len_st : len(token_ids) - len_ed] + ed_pattern
    return tuple(result)


def calculate_expected_hit_count(document_length: int, chunk_size: int = 256):
    if SUPPORT_PARTIAL_CHUNK:
        return document_length
    else:
        return (document_length // chunk_size) * chunk_size


# =============================================================================
# Server Process Runner
# =============================================================================


def server_process_runner(
    host: str, port: int, chunk_size: int, cpu_buffer_size: float
):
    """
    Entry point for the server process running BlendEngine.
    """
    # Import here to ensure environment variables are set before import
    # First Party
    from lmcache.v1.multiprocess.blend_server import run_cache_server

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
    """
    Fixture that starts the blend cache server in a separate process.
    The server runs for the entire test module.
    """
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=server_process_runner,
        args=(SERVER_HOST, SERVER_PORT, CHUNK_SIZE, CPU_BUFFER_SIZE),
        daemon=True,
    )
    process.start()

    # Wait for server to initialize
    time.sleep(3)

    yield process

    # Cleanup: terminate the server process
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="module")
def zmq_context() -> Generator[zmq.Context, None, None]:
    """
    Fixture that provides a ZMQ context for the test module.
    """
    context = zmq.Context.instance()
    yield context


@pytest.fixture(scope="function")
def client(
    server_process: mp.Process, zmq_context: zmq.Context
) -> Generator[MessageQueueClient, None, None]:
    """
    Fixture that provides a message queue client for each test function.
    """
    client = MessageQueueClient(server_url=SERVER_URL, context=zmq_context)
    yield client
    client.close()


@pytest.fixture(scope="function")
def cb_client_context() -> Generator[CBClientContext, None, None]:
    """
    Fixture that provides a CB client context with plain [2, L, T, D] GPU buffer.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda:0")
    ctx = CBClientContext(device=device)
    yield ctx

    del ctx.gpu_kv_cache
    torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def client_context() -> Generator[ClientContext, None, None]:
    """
    Fixture that provides a standard client context with paged GPU buffer.
    """
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
    """
    Fixture that registers a CB KV cache instance and returns the instance ID.
    Automatically unregisters after the test.
    """
    instance_id = os.getpid() + 1000  # Offset to avoid collision with standard instance

    # Register CB KV cache
    future = client.submit_request(
        RequestType.CB_REGISTER_KV_CACHE,
        [instance_id, cb_client_context.get_kv_cache(), "testmodel", 1],
        get_response_class(RequestType.CB_REGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None, "CB Register should return None"

    yield instance_id

    # Unregister CB KV cache
    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        future = client.submit_request(
            RequestType.CB_UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
        )
        future.result(timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        print(f"Error during CB unregister: {e}")


@pytest.fixture(scope="function")
def registered_instance(
    client: MessageQueueClient, client_context: ClientContext
) -> Generator[int, None, None]:
    """
    Fixture that registers a standard KV cache instance and returns the instance ID.
    Automatically unregisters after the test.
    """
    instance_id = os.getpid()

    # Register KV cache
    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [instance_id, client_context.get_kv_cache(), "testmodel", 1],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None, "Register should return None"

    yield instance_id

    # Unregister KV cache
    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        future = client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.UNREGISTER_KV_CACHE),
        )
        future.result(timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        print(f"Error during unregister: {e}")


# =============================================================================
# Test Functions - 1. Server Startup and Basic Connectivity
# =============================================================================


def test_server_running(server_process: mp.Process):
    """
    Test that the server process is running.
    """
    assert server_process.is_alive(), "Server process should be running"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NOOP request requires CUDA server"
)
def test_noop_request(client: MessageQueueClient):
    """
    Test that NOOP request returns 'OK'.
    """
    future = client.submit_request(
        RequestType.NOOP,
        [],
        get_response_class(RequestType.NOOP),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result == "OK", "NOOP should return 'OK'"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Get chunk size requires CUDA server"
)
def test_get_chunk_size(client: MessageQueueClient):
    """
    Test retrieving the chunk size from the server.
    """
    chunk_size = client.submit_request(
        RequestType.GET_CHUNK_SIZE,
        [],
        get_response_class(RequestType.GET_CHUNK_SIZE),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert chunk_size == CHUNK_SIZE, f"Chunk size should be {CHUNK_SIZE}"


# =============================================================================
# Test Functions - 2. CB KV Cache Registration/Unregistration
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Register/Unregister requires CUDA",
)
def test_cb_register_unregister_kv_cache(
    client: MessageQueueClient, cb_client_context: CBClientContext
):
    """
    Test registering and unregistering a CB KV cache.
    """
    instance_id = os.getpid() + 2000

    # Register
    future = client.submit_request(
        RequestType.CB_REGISTER_KV_CACHE,
        [instance_id, cb_client_context.get_kv_cache(), "testmodel", 1],
        get_response_class(RequestType.CB_REGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None, "CB Register should return None"

    # Unregister
    future = client.submit_request(
        RequestType.CB_UNREGISTER_KV_CACHE,
        [instance_id],
        get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None, "CB Unregister should return None"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Register multiple instances requires CUDA",
)
def test_cb_register_multiple_instances(
    client: MessageQueueClient, cb_client_context: CBClientContext
):
    """
    Test registering multiple CB instances with different IDs.
    """
    base_id = os.getpid() + 3000
    instance_ids = [base_id + i for i in range(3)]

    # Register all instances
    for instance_id in instance_ids:
        future = client.submit_request(
            RequestType.CB_REGISTER_KV_CACHE,
            [instance_id, cb_client_context.get_kv_cache(), "testmodel", 1],
            get_response_class(RequestType.CB_REGISTER_KV_CACHE),
        )
        result = future.result(timeout=DEFAULT_TIMEOUT)
        assert result is None, f"CB Register for {instance_id} should return None"

    # Unregister all instances
    for instance_id in instance_ids:
        future = client.submit_request(
            RequestType.CB_UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
        )
        result = future.result(timeout=DEFAULT_TIMEOUT)
        assert result is None, f"CB Unregister for {instance_id} should return None"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Unregister nonexistent requires CUDA",
)
def test_cb_unregister_nonexistent(client: MessageQueueClient):
    """
    Test unregistering a non-existent instance (should not raise, just warn).
    """
    nonexistent_id = 999999

    # This should not raise an exception
    future = client.submit_request(
        RequestType.CB_UNREGISTER_KV_CACHE,
        [nonexistent_id],
        get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None, "CB Unregister nonexistent should return None"


# =============================================================================
# Test Functions - 3. CB Store Pre-Computed
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Pre-Computed requires CUDA",
)
def test_cb_store_pre_computed_basic(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test storing pre-computed chunks with a key and offset.
    """
    # Create a key with some token ids (one "paragraph")
    token_ids = create_token_ids_with_sep_tokens(list(range(CHUNK_SIZE)))
    key = create_cb_cache_key(token_ids)

    # Create CUDA event
    event = torch.cuda.Event(interprocess=True)
    event.record()

    future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [
            key,
            0,
            cb_registered_instance,
            event.ipc_handle(),
        ],  # offset, instance_id, event
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Expected: returns (event_ipc_handle, success_bool)
    assert result is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Pre-Computed various offsets requires CUDA",
)
def test_cb_store_pre_computed_various_offsets(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test storing pre-computed chunks at different offsets.
    """
    token_ids = create_token_ids_with_sep_tokens(list(range(CHUNK_SIZE)))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Test at offset 0
    key1 = create_cb_cache_key(token_ids, request_id="req1")
    future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key1, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert result is True, "Store at offset 0 should succeed"

    # Test at middle offset
    token_ids = create_token_ids_with_sep_tokens(
        list(range(CHUNK_SIZE, CHUNK_SIZE * 2))
    )
    key2 = create_cb_cache_key(token_ids, request_id="req2")
    middle_offset = cb_client_context.num_tokens // 2
    future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key2, middle_offset, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert result is True, "Store at middle offset should succeed"

    # Test at near-end offset
    token_ids = create_token_ids_with_sep_tokens(
        list(range(CHUNK_SIZE * 2, CHUNK_SIZE * 3))
    )
    key3 = create_cb_cache_key(token_ids, request_id="req3")
    end_offset = cb_client_context.num_tokens - CHUNK_SIZE
    future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key3, end_offset, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert result is True, "Store at end offset should succeed"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Pre-Computed invalid offset requires CUDA",
)
def test_cb_store_pre_computed_long_doc(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test storing a long document that has multiple chunks
    """
    # Create token ids that exceed the client's total token capacity
    num_tokens = 1000
    token_ids = create_token_ids_with_sep_tokens(list(range(num_tokens)))
    key = create_cb_cache_key(token_ids, request_id="long-doc")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    assert result is True


# =============================================================================
# Test Functions - 4. CB Lookup Pre-Computed
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup after store requires CUDA",
)
def test_cb_lookup_after_store_single_paragraph(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Store one paragraph, then lookup with same tokens.
    Expected: Returns ranges matching the stored data.
    """
    paragraph_size = 1000
    expected_hit_count_per_paragraph = calculate_expected_hit_count(
        paragraph_size, chunk_size=CHUNK_SIZE
    )

    # Store one paragraph
    token_ids = create_token_ids_with_sep_tokens(list(range(100, 100 + paragraph_size)))
    key = create_cache_key(token_ids, request_id="store-lookup-single")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True, "Store should succeed"

    # Lookup with same tokens
    lookup_key = create_cb_cache_key(token_ids, request_id="lookup-test-1")
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)

    # Verify result type and expected content
    assert isinstance(ranges, list), "Lookup should return a list"
    # Expected: should return [(0, 1)] for 1 chunk match
    assert len(ranges) > 0, "Lookup should find the stored paragraph"
    expected_ranges = [(0, expected_hit_count_per_paragraph)]

    assert len(ranges) == len(expected_ranges), (
        f"Expected {len(expected_ranges)} ranges but got {len(ranges)}"
    )
    for expected, actual in zip(expected_ranges, ranges, strict=False):
        assert expected == actual, f"Expected range {expected} but got {actual}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup after store multiple paragraphs requires CUDA",
)
def test_cb_lookup_after_store_multiple_paragraphs(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Store multiple paragraphs, then lookup sequence containing all.
    Expected: Each paragraph's range is returned.
    """
    num_paragraphs = 3
    paragraph_size = 800
    expected_hit_count_per_paragraph = calculate_expected_hit_count(
        paragraph_size, chunk_size=CHUNK_SIZE
    )
    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Store multiple paragraphs
    for i in range(num_paragraphs):
        token_ids = create_token_ids_with_sep_tokens(
            list(range(i * 1000, i * 1000 + paragraph_size))
        )
        key = create_cb_cache_key(token_ids, request_id=f"multi-para-{i}")
        offset = i * CHUNK_SIZE

        store_future = client.submit_request(
            RequestType.CB_STORE_PRE_COMPUTED,
            [key, offset, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
        )
        store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
        assert store_result is True, f"Store for paragraph {i} should succeed"

    # Lookup first paragraph
    lookup_token_ids = create_token_ids_with_sep_tokens(list(range(0, paragraph_size)))
    lookup_key = create_cb_cache_key(lookup_token_ids, request_id="multi-lookup")
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(ranges, list), "Lookup should return a list"
    # Expected: should find the first paragraph
    assert len(ranges) > 0, "Lookup should find stored paragraph"
    assert ranges[0] == (0, expected_hit_count_per_paragraph), (
        f"Expected range (0, {expected_hit_count_per_paragraph}) but got {ranges[0]}"
    )

    # Construct a big paragraph with sep tokens
    lookup_token_ids_list: list[int] = []
    for i in range(num_paragraphs):
        lookup_token_ids_list.extend(
            create_token_ids_with_sep_tokens(
                list(range(i * 1000, i * 1000 + paragraph_size))
            )
        )

    lookup_key = create_cb_cache_key(
        tuple(lookup_token_ids_list), request_id="multi-lookup-all"
    )
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)
    assert isinstance(ranges, list), "Lookup should return a list"
    # Expected: should find all 3 paragraphs with correct ranges
    for idx, (start, end) in enumerate(ranges):
        assert (end - start) == expected_hit_count_per_paragraph, (
            "Each range should correspond to one chunk"
        )
        assert start == paragraph_size * idx, (
            "Ranges should be correctly spaced with sep tokens"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup partial match requires CUDA",
)
def test_cb_lookup_partial_match(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Store [A, B, C], then lookup [A, D, C].
    Expected: Get the ranges for A and C
    """
    # Store tokens [A, B, C]
    paragraph_size = 300
    expected_hit_count_per_paragraph = calculate_expected_hit_count(
        paragraph_size, chunk_size=CHUNK_SIZE
    )
    num_paragraphs = 3
    event = torch.cuda.Event(interprocess=True)
    event.record()

    for i in range(num_paragraphs):
        token_ids = create_token_ids_with_sep_tokens(
            list(range(i * 1000, i * 1000 + paragraph_size))
        )
        key = create_cb_cache_key(token_ids, request_id=f"partial-match-{i}")
        offset = i * paragraph_size

        store_future = client.submit_request(
            RequestType.CB_STORE_PRE_COMPUTED,
            [key, offset, cb_registered_instance, event.ipc_handle()],
            get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
        )
        store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
        assert store_result is True, f"Store for paragraph {i} should succeed"

    # Lookup with different ending: [A, D, C]
    lookup_token_ids: list[int] = []
    for i in range(num_paragraphs):
        lookup_token_ids.extend(
            create_token_ids_with_sep_tokens(
                list(range(i * 1000, i * 1000 + paragraph_size))
            )
        )
        if i == 1:  # Replace B with D
            lookup_token_ids[-paragraph_size:] = range(9999, 9999 + paragraph_size)

    lookup_key = create_cb_cache_key(
        tuple(lookup_token_ids), request_id="partial-lookup"
    )
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [lookup_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)
    assert isinstance(ranges, list), "Lookup should return a list"
    assert len(ranges) == 2, "Lookup should find 2 matching paragraphs"

    # Expected: should find the first and third paragraphs, but not the second
    expected_ranges = [
        (0, expected_hit_count_per_paragraph),
        (
            2 * paragraph_size,
            2 * paragraph_size + expected_hit_count_per_paragraph,
        ),
    ]
    for expected, actual in zip(expected_ranges, ranges, strict=False):
        assert expected == actual, f"Expected range {expected} but got {actual}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup no match requires CUDA",
)
def test_cb_lookup_no_match(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Lookup tokens that were never stored.
    Expected: Empty list [].
    """
    # Lookup tokens that were never stored
    token_ids = tuple(range(50000, 50000 + CHUNK_SIZE))
    key = create_cb_cache_key(token_ids, request_id="no-match")

    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(ranges, list), "Lookup should return a list"
    assert len(ranges) == 0, "Lookup for non-existent tokens should return empty list"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Lookup isolation test requires CUDA",
)
def test_cb_lookup_cannot_find_normal_store(
    client: MessageQueueClient,
    client_context: ClientContext,
    cb_client_context: CBClientContext,
    registered_instance: int,
    cb_registered_instance: int,
):
    """
    ISOLATION TEST: Store via normal STORE, then CB_LOOKUP_PRE_COMPUTED.
    Expected: CB lookup should NOT see normal-stored data (returns empty []).
    """
    # Store via normal STORE operation (one key at a time)
    num_keys = 5
    blocks_per_key = 16
    keys = [
        create_cache_key(
            tuple(range(CHUNK_SIZE * i, CHUNK_SIZE * (i + 1))),
            request_id=f"normal-store-{i}",
        )
        for i in range(num_keys)
    ]
    gpu_block_ids = list(range(0, blocks_per_key * num_keys))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    for i, key in enumerate(keys):
        start = i * blocks_per_key
        end = start + blocks_per_key
        block_ids = gpu_block_ids[start:end]
        store_future = client.submit_request(
            RequestType.STORE,
            [key, registered_instance, block_ids, event.ipc_handle()],
            get_response_class(RequestType.STORE),
        )
        store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
        assert store_result is True, f"Normal store should succeed for key {i}"

    # Now try CB lookup with same token pattern
    # Use same hash value converted to token_ids pattern
    token_ids = tuple(range(CHUNK_SIZE))
    cb_key = create_cb_cache_key(token_ids, request_id="isolation-test")

    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [cb_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(ranges, list), "CB Lookup should return a list"
    # CB lookup should NOT find data stored via normal STORE
    assert len(ranges) == 0, (
        "CB Lookup should NOT find data stored via normal STORE operation"
    )


# =============================================================================
# Test Functions - 5. CB Retrieve Pre-Computed
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve after store and lookup requires CUDA",
)
def test_cb_retrieve_after_store_and_lookup(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Store pre-computed, lookup to get ranges, then retrieve.
    Expected: Returns (event_handle, True) and data is copied to CB buffer.
    """
    paragraph_size = 600
    expected_hit_count_per_paragraph = calculate_expected_hit_count(
        paragraph_size, chunk_size=CHUNK_SIZE
    )
    # Store pre-computed chunks
    token_ids = create_token_ids_with_sep_tokens(list(range(200, 200 + paragraph_size)))
    key = create_cb_cache_key(token_ids, request_id="retrieve-test")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True, "Store should succeed"

    # Lookup to get ranges
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)
    assert len(ranges) > 0, "Lookup should find the stored data"
    assert ranges[0] == (0, expected_hit_count_per_paragraph), (
        f"Expected range (0, {expected_hit_count_per_paragraph}) but got {ranges[0]}"
    )

    # Retrieve
    retrieve_ranges = ranges
    retrieve_offset = paragraph_size  # Retrieve to a different offset

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    retrieve_future = client.submit_request(
        RequestType.CB_RETRIEVE_PRE_COMPUTED,
        [
            key,
            retrieve_ranges,
            retrieve_offset,
            cb_registered_instance,
            event2.ipc_handle(),
        ],
        get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED),
    )
    result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert result is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve verify data correctness requires CUDA",
)
def test_cb_retrieve_verify_data_correctness(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Store known data, retrieve to different offset, verify correctness.
    """
    paragraph_size = 600
    expected_hit_count_per_paragraph = calculate_expected_hit_count(
        paragraph_size, chunk_size=CHUNK_SIZE
    )

    # Set known values in the source region
    source_offset = 0
    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 1.0)
    cb_client_context.set_tensor_slice(source_offset, paragraph_size, 0.5)

    # Store pre-computed chunks
    token_ids = create_token_ids_with_sep_tokens(list(range(300, 300 + paragraph_size)))
    key = create_cb_cache_key(token_ids, request_id="verify-data")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_future = client.submit_request(
        RequestType.CB_STORE_PRE_COMPUTED,
        [key, source_offset, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True, "Store should succeed"

    # reset other region to 0.0 to catch the retrieve overflow error
    cb_client_context.set_tensor_slice(0, cb_client_context.num_tokens, 0.0)
    cb_client_context.set_tensor_slice(source_offset, paragraph_size, 0.5)

    # Lookup to get ranges
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)
    assert ranges[0] == (0, expected_hit_count_per_paragraph), (
        f"Expected range (0, {expected_hit_count_per_paragraph}) but got {ranges[0]}"
    )

    # Very data is not the same before retrieve
    dest_offset = paragraph_size  # Retrieve to a different offset
    source_slice = cb_client_context.get_tensor_slice(
        source_offset, expected_hit_count_per_paragraph
    )
    dest_slice = cb_client_context.get_tensor_slice(
        dest_offset, expected_hit_count_per_paragraph
    )
    assert not torch.allclose(dest_slice, source_slice, atol=1e-4), (
        "Before retrieve, destination slice should not match source values"
    )

    # Retrieve to dest_offset
    retrieve_ranges = ranges

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    retrieve_future = client.submit_request(
        RequestType.CB_RETRIEVE_PRE_COMPUTED,
        [
            key,
            retrieve_ranges,
            dest_offset,
            cb_registered_instance,
            event2.ipc_handle(),
        ],
        get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED),
    )
    result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    assert result is True, "Retrieve should succeed"

    # Verify tensor values match after GPU sync
    torch.cuda.synchronize()
    source_slice = cb_client_context.get_tensor_slice(
        source_offset, expected_hit_count_per_paragraph
    )
    dest_slice = cb_client_context.get_tensor_slice(
        dest_offset, expected_hit_count_per_paragraph
    )

    assert torch.allclose(source_slice, dest_slice, atol=1e-4), (
        "Retrieved data should match stored data"
    )

    # Verify that only the retrieved range is updated,
    # and other regions are unchanged
    other_slice = cb_client_context.get_tensor_slice(
        dest_offset + expected_hit_count_per_paragraph,
        cb_client_context.num_tokens - (dest_offset + expected_hit_count_per_paragraph),
    )
    assert torch.allclose(other_slice, torch.zeros_like(other_slice), atol=1e-4), (
        "Other regions of the buffer should be unchanged (zeros)"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve empty ranges requires CUDA",
)
def test_cb_retrieve_empty_ranges(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Retrieve with empty ranges [].
    Expected: Returns (event_handle, True) (no-op is success).
    """
    token_ids = create_token_ids_with_sep_tokens(list(range(CHUNK_SIZE)))
    key = create_cb_cache_key(token_ids, request_id="empty-ranges")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    retrieve_future = client.submit_request(
        RequestType.CB_RETRIEVE_PRE_COMPUTED,
        [key, [], 0, cb_registered_instance, event.ipc_handle()],  # Empty ranges
        get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED),
    )
    result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    assert result is True, "Empty ranges should be a no-op, returning True"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Retrieve invalid ranges requires CUDA",
)
def test_cb_retrieve_invalid_ranges(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Retrieve with ranges that don't exist in storage.
    Expected: Returns (event_handle, False).
    """
    # Don't store anything, just try to retrieve
    token_ids = create_token_ids_with_sep_tokens(list(range(60000, 60000 + CHUNK_SIZE)))
    key = create_cb_cache_key(token_ids, request_id="invalid-ranges")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    retrieve_future = client.submit_request(
        RequestType.CB_RETRIEVE_PRE_COMPUTED,
        [key, [(0, CHUNK_SIZE)], 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED),
    )
    result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    # Expected: should return False for non-existent ranges
    assert result is False, "Retrieve should fail for non-existent ranges"


# =============================================================================
# Test Functions - 6. CB Store Final (Bridge to Normal Operations)
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Final basic requires CUDA",
)
def test_cb_store_final_basic(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Call CB_STORE_FINAL with key and offset.
    Expected: Returns (event_handle, True).
    """
    paragraph_size = 600
    token_ids = create_token_ids_with_sep_tokens(list(range(400, 400 + paragraph_size)))
    key = create_cb_cache_key(token_ids, request_id="final-basic")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    future = client.submit_request(
        RequestType.CB_STORE_FINAL,
        [key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_FINAL),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    assert result is True, "CB Store Final should return True for success"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Final then normal lookup requires CUDA",
)
def test_cb_store_final_then_normal_lookup_retrieve(
    client: MessageQueueClient,
    client_context: ClientContext,
    cb_client_context: CBClientContext,
    registered_instance: int,
    cb_registered_instance: int,
):
    """
    Test: Store via CB_STORE_FINAL, then normal LOOKUP.
    Expected: Chunks ARE found, and can be retrieved
    """

    # Store via CB_STORE_FINAL
    paragraph_size = 600
    source_value = 0.125
    cb_client_context.set_tensor_slice(0, paragraph_size, source_value)

    expected_hit_count_per_paragraph = calculate_expected_hit_count(
        paragraph_size, chunk_size=CHUNK_SIZE
    )
    expected_hit_chunks = expected_hit_count_per_paragraph // CHUNK_SIZE
    token_ids = create_token_ids_with_sep_tokens(list(range(500, 500 + paragraph_size)))
    cb_key = create_cb_cache_key(token_ids, request_id="final-lookup-test")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_future = client.submit_request(
        RequestType.CB_STORE_FINAL,
        [cb_key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_FINAL),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True, "CB Store Final should succeed"

    # Create equivalent key for normal lookup
    # The normal lookup should find data stored by CB_STORE_FINAL
    lookup_key = create_cb_cache_key(
        token_ids, request_id="final-lookup-test", worker_id=None
    )

    lookup_future = client.submit_request(
        RequestType.LOOKUP,
        [lookup_key],
        get_response_class(RequestType.LOOKUP),
    )
    lookup_result = lookup_future.result(timeout=DEFAULT_TIMEOUT)

    # Expected: lookup_result should be > 0 (chunks found)
    assert isinstance(lookup_result, int), "Lookup should return an int"
    assert lookup_result == expected_hit_chunks, (
        "Normal LOOKUP should find data stored by CB_STORE_FINAL"
    )

    # Test retrieve
    retrieve_key = create_cb_cache_key(
        token_ids[:expected_hit_count_per_paragraph],
        request_id="final-retrieve-test",
    )
    gpu_block_ids = list(range(0, expected_hit_chunks * 16))  # Retrieve to first
    event2 = torch.cuda.Event(interprocess=True)
    event2.record()
    retrieve_future = client.submit_request(
        RequestType.RETRIEVE,
        [retrieve_key, registered_instance, gpu_block_ids, event2.ipc_handle()],
        get_response_class(RequestType.RETRIEVE),
    )
    retrieve_result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert retrieve_result is True, "Retrieve should succeed"

    # Verify the correctness
    torch.cuda.synchronize()
    for layer in range(client_context.num_layers):
        tensor_slice = client_context.get_tensor_slice(
            layer, 0, expected_hit_chunks * 16
        )
        assert tensor_slice.mean().item() == source_value, (
            "Retrieved tensor data should match stored values"
        )
        assert tensor_slice.std().item() == 0.0, (
            "Retrieved tensor data should have zero stddev"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CB Store Final not visible to CB lookup requires CUDA",
)
def test_cb_store_final_not_visible_to_cb_lookup(
    client: MessageQueueClient,
    cb_client_context: CBClientContext,
    cb_registered_instance: int,
):
    """
    Test: Store via CB_STORE_FINAL, then CB_LOOKUP_PRE_COMPUTED.
    Expected: Returns empty [] (CB lookup only sees pre-computed, not final).
    """
    # Store via CB_STORE_FINAL
    paragraph_size = 600
    token_ids = create_token_ids_with_sep_tokens(list(range(800, 800 + paragraph_size)))
    cb_key = create_cb_cache_key(token_ids, request_id="final-not-cb-visible")

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_future = client.submit_request(
        RequestType.CB_STORE_FINAL,
        [cb_key, 0, cb_registered_instance, event.ipc_handle()],
        get_response_class(RequestType.CB_STORE_FINAL),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True, "CB Store Final should succeed"

    # Try CB lookup - should NOT find it
    lookup_future = client.submit_request(
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        [cb_key],
        get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
    )
    ranges = lookup_future.result(timeout=DEFAULT_TIMEOUT)

    assert isinstance(ranges, list), "CB Lookup should return a list"
    # CB lookup should NOT find data stored via CB_STORE_FINAL
    assert len(ranges) == 0, "CB Lookup should NOT find data stored via CB_STORE_FINAL"
