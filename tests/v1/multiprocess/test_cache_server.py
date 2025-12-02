# SPDX-License-Identifier: Apache-2.0
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
from lmcache.v1.multiprocess.server import run_cache_server

# Configuration constants
SERVER_HOST = "localhost"
SERVER_PORT = 5555
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0
DEFAULT_TIMEOUT = 5.0


def initialize_kv_cache(
    device: torch.device,
    num_pages: int = 1024,
    num_layers: int = 32,
    page_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """
    Initialize KV cache tensors on GPU for testing.
    """
    torch.random.manual_seed(42)

    gpu_tensors = [
        torch.rand(
            (2, num_pages, page_size, num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    return gpu_tensors


class ClientContext:
    """
    Client context that manages GPU KV cache tensors.
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

        self.gpu_kv_caches = initialize_kv_cache(
            device, num_pages, num_layers, page_size, num_heads, head_size, dtype
        )

    def get_kv_cache(self) -> KVCache:
        """
        Wrap GPU tensors in CudaIPCWrapper for IPC communication.
        """
        return [CudaIPCWrapper(tensor) for tensor in self.gpu_kv_caches]

    def get_tensor_slice(
        self, layer: int, start_page: int, num_pages: int
    ) -> torch.Tensor:
        """
        Get a slice of the KV cache tensor for a specific layer.
        """
        return self.gpu_kv_caches[layer][:, start_page : start_page + num_pages]


def create_cache_key(index: int, model: str = "testmodel") -> IPCCacheEngineKey:
    """
    Create a cache key for testing.
    """
    return IPCCacheEngineKey.from_int_hash(model, 1, 0, index)


def server_process_runner(
    host: str, port: int, chunk_size: int, cpu_buffer_size: float
):
    """
    Entry point for the server process.
    """
    run_cache_server(
        host=host, port=port, chunk_size=chunk_size, cpu_buffer_size=cpu_buffer_size
    )


@pytest.fixture(scope="module")
def server_process() -> Generator[mp.Process, None, None]:
    """
    Fixture that starts the cache server in a separate process.
    The server runs for the entire test module.
    """
    # Start server process
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=server_process_runner,
        args=(SERVER_HOST, SERVER_PORT, CHUNK_SIZE, CPU_BUFFER_SIZE),
        daemon=True,
    )
    process.start()

    # Wait for server to initialize
    time.sleep(2)

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
    # Context cleanup is handled by ZMQ


@pytest.fixture(scope="function")
def client(
    server_process: mp.Process, zmq_context: zmq.Context
) -> Generator[MessageQueueClient, None, None]:
    """
    Fixture that provides a message queue client for each test function.
    """
    client = MessageQueueClient(server_url=SERVER_URL, context=zmq_context)
    yield client
    # Client cleanup
    client.close()


@pytest.fixture(scope="function")
def client_context() -> Generator[ClientContext, None, None]:
    """
    Fixture that provides a client context with initialized KV cache.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda:0")
    ctx = ClientContext(device=device)
    yield ctx

    # Cleanup GPU memory
    del ctx.gpu_kv_caches
    torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def registered_instance(
    client: MessageQueueClient, client_context: ClientContext
) -> Generator[int, None, None]:
    """
    Fixture that registers a KV cache instance and returns the instance ID.
    Automatically unregisters after the test.
    """
    instance_id = os.getpid()

    # Register KV cache
    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [instance_id, client_context.get_kv_cache()],
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


# ============================================================================
# Test Functions
# ============================================================================


def test_server_running(server_process: mp.Process):
    """
    Test that the server process is running.
    """
    assert server_process.is_alive(), "Server process should be running"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Register/Unregister KV cache requires CUDA",
)
def test_register_unregister_kv_cache(
    client: MessageQueueClient, client_context: ClientContext
):
    """
    Test registering and unregistering a KV cache.
    """
    instance_id = os.getpid()

    # Register
    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [instance_id, client_context.get_kv_cache()],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None

    # Unregister
    future = client.submit_request(
        RequestType.UNREGISTER_KV_CACHE,
        [instance_id],
        get_response_class(RequestType.UNREGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Store and Lookup require CUDA",
)
def test_store_and_lookup(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test storing KV cache entries and looking them up.
    """
    num_keys = 10
    keys = [create_cache_key(i) for i in range(num_keys)]
    gpu_block_ids = list(range(0, 16 * num_keys))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Store
    store_future = client.submit_request(
        RequestType.STORE,
        [keys, registered_instance, gpu_block_ids, event.ipc_handle()],
        get_response_class(RequestType.STORE),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True, "Store should succeed"

    # Lookup - keys that exist
    lookup_future = client.submit_request(
        RequestType.LOOKUP,
        [keys, False],
        get_response_class(RequestType.LOOKUP),
    )
    lookup_result = lookup_future.result(timeout=DEFAULT_TIMEOUT)
    assert len(lookup_result) == num_keys
    assert all(lookup_result), "All stored keys should exist"

    # Lookup - keys that don't exist
    non_existent_keys = [create_cache_key(i + 1000) for i in range(5)]
    lookup_future2 = client.submit_request(
        RequestType.LOOKUP,
        [non_existent_keys, False],
        get_response_class(RequestType.LOOKUP),
    )
    lookup_result2 = lookup_future2.result(timeout=DEFAULT_TIMEOUT)
    assert len(lookup_result2) == 5
    assert not any(lookup_result2), "Non-existent keys should not be found"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Store, Retrieve, and Verify require CUDA",
)
def test_store_retrieve_verify(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test storing and retrieving KV cache entries, verifying correctness.
    """
    num_keys = 20
    keys = [create_cache_key(i) for i in range(num_keys)]
    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Store at the beginning of the cache
    store_block_ids = list(range(0, 16 * num_keys))
    store_future = client.submit_request(
        RequestType.STORE,
        [keys, registered_instance, store_block_ids, event.ipc_handle()],
        get_response_class(RequestType.STORE),
    )
    store_result = store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert store_result is True

    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Retrieve to a different location in the cache
    # Use offset of 40 blocks (640 pages total needed: 320 + 320)
    retrieve_offset = 40 * 16
    retrieve_block_ids = list(range(retrieve_offset, retrieve_offset + 16 * num_keys))
    retrieve_future = client.submit_request(
        RequestType.RETRIEVE,
        [keys, registered_instance, retrieve_block_ids, event.ipc_handle()],
        get_response_class(RequestType.RETRIEVE),
    )
    retrieve_result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    assert len(retrieve_result) == num_keys
    assert all(retrieve_result), "All keys should be retrieved successfully"

    # Verify correctness by comparing tensors
    for i in range(num_keys):
        for layer in range(client_context.num_layers):
            original_block = i * 16
            retrieved_block = retrieve_offset + i * 16

            original_tensor = client_context.gpu_kv_caches[layer][
                :, original_block : original_block + 16
            ]
            retrieved_tensor = client_context.gpu_kv_caches[layer][
                :, retrieved_block : retrieved_block + 16
            ]

            assert torch.allclose(original_tensor, retrieved_tensor, atol=1e-4), (
                f"Mismatch for key {i}, layer {layer}"
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Partial miss retrieval requires CUDA",
)
def test_retrieve_partial_miss(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test retrieving when some keys exist and some don't.
    The retrieve should return ALL FALSE if any key is missing.
    """
    # Store first 30 keys (480 pages)
    num_stored = 30
    stored_keys = [create_cache_key(i) for i in range(num_stored)]
    store_block_ids = list(range(0, 16 * num_stored))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_future = client.submit_request(
        RequestType.STORE,
        [stored_keys, registered_instance, store_block_ids, event.ipc_handle()],
        get_response_class(RequestType.STORE),
    )
    assert store_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT) is True

    # Try to retrieve 60 keys (only first 30 exist)
    # Total pages needed: 60 * 16 = 960 (< 1024)
    num_requested = 60
    all_keys = [create_cache_key(i) for i in range(num_requested)]
    # Start retrieve at offset 2 keys (32 pages)
    retrieve_offset_keys = 2
    retrieve_block_ids = list(
        range(retrieve_offset_keys * 16, (retrieve_offset_keys + num_requested) * 16)
    )

    event = torch.cuda.Event(interprocess=True)
    event.record()

    retrieve_future = client.submit_request(
        RequestType.RETRIEVE,
        [all_keys, registered_instance, retrieve_block_ids, event.ipc_handle()],
        get_response_class(RequestType.RETRIEVE),
    )
    retrieve_result = retrieve_future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)

    assert len(retrieve_result) == num_requested
    # assert all(retrieve_result[:num_stored]), "Stored keys should be retrieved"
    # Remaining should fail
    assert not any(retrieve_result), (
        "Retrieve is expected to return all FALSE if any key is missing"
    )

    # Try to retrieve the first 30 keys only (all exist)
    retrieve_block_ids_2 = list(range(0, 16 * num_stored))
    event = torch.cuda.Event(interprocess=True)
    event.record()
    retrieve_future_2 = client.submit_request(
        RequestType.RETRIEVE,
        [stored_keys, registered_instance, retrieve_block_ids_2, event.ipc_handle()],
        get_response_class(RequestType.RETRIEVE),
    )
    retrieve_result_2 = retrieve_future_2.to_cuda_future().result(
        timeout=DEFAULT_TIMEOUT
    )
    assert len(retrieve_result_2) == num_stored
    assert all(retrieve_result_2), "All stored keys should be retrieved successfully"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Multiple retrieve operations require CUDA",
)
def test_multiple_retrieve_operations(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test multiple retrieve operations:
    Store 8,8,8,8 keys and then retrieve 8,8,8,8 keys in sequence.
    """
    num_batches = 4
    keys_per_batch = 8
    pages_per_key = 16

    # Initialize the values in GPU KV cache
    for layer in range(client_context.num_layers):
        layer_cache = client_context.gpu_kv_caches[layer]
        for i in range(num_batches):
            start_page = (i * keys_per_batch) * pages_per_key
            end_page = start_page + (keys_per_batch * pages_per_key)
            layer_cache[:, start_page:end_page] = (i + 1) / num_batches

    # Store in batches
    for batch_idx in range(num_batches):
        keys = [
            create_cache_key(batch_idx * keys_per_batch + i)
            for i in range(keys_per_batch)
        ]
        blocks = list(
            range(
                (batch_idx * keys_per_batch) * 16,
                (batch_idx * keys_per_batch + keys_per_batch) * 16,
            )
        )
        event = torch.cuda.Event(interprocess=True)
        event.record()

        store_result = (
            client.submit_request(
                RequestType.STORE,
                [keys, registered_instance, blocks, event.ipc_handle()],
                get_response_class(RequestType.STORE),
            )
            .to_cuda_future()
            .result(timeout=DEFAULT_TIMEOUT)
        )
        assert store_result is True

    # Retrieve in batches
    retrieve_offset = 32  # Start retrieving at offset of 32 chunks
    retrieve_futures = []
    event = torch.cuda.Event(interprocess=True)
    event.record()
    for batch_idx in range(num_batches):
        keys = [
            create_cache_key(batch_idx * keys_per_batch + i)
            for i in range(keys_per_batch)
        ]
        blocks = list(
            range(
                (batch_idx * keys_per_batch + retrieve_offset) * pages_per_key,
                (batch_idx * keys_per_batch + retrieve_offset + keys_per_batch)
                * pages_per_key,
            )
        )

        retrieve_future = client.submit_request(
            RequestType.RETRIEVE,
            [keys, registered_instance, blocks, event.ipc_handle()],
            get_response_class(RequestType.RETRIEVE),
        )
        retrieve_futures.append(retrieve_future.to_cuda_future())

    for retrieve_future in retrieve_futures:
        retrieve_result = retrieve_future.result(timeout=DEFAULT_TIMEOUT)
        assert len(retrieve_result) == keys_per_batch
        assert all(retrieve_result), "All keys should be retrieved successfully"

    # Verify correctness
    for layer in range(client_context.num_layers):
        layer_cache = client_context.gpu_kv_caches[layer]
        for batch_idx in range(num_batches):
            start_page = (retrieve_offset + batch_idx * keys_per_batch) * pages_per_key
            end_page = start_page + (keys_per_batch * pages_per_key)
            retrieved_tensor = layer_cache[:, start_page:end_page]
            expected_value = (batch_idx + 1) / num_batches
            assert torch.allclose(
                retrieved_tensor,
                torch.full_like(retrieved_tensor, expected_value),
            ), f"Mismatch in batch {batch_idx}, layer {layer}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Multiple store operations require CUDA",
)
def test_multiple_store_operations(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test multiple store operations in sequence.
    """
    # Store batch 1
    keys1 = [create_cache_key(i) for i in range(30)]
    blocks1 = list(range(0, 16 * 30))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    result1 = (
        client.submit_request(
            RequestType.STORE,
            [keys1, registered_instance, blocks1, event.ipc_handle()],
            get_response_class(RequestType.STORE),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result1 is True

    # Store batch 2
    keys2 = [create_cache_key(i + 30) for i in range(20)]
    blocks2 = list(range(30 * 16, 50 * 16))

    # Test with the same event for 2 store requests
    result2 = (
        client.submit_request(
            RequestType.STORE,
            [keys2, registered_instance, blocks2, event.ipc_handle()],
            get_response_class(RequestType.STORE),
        )
        .to_cuda_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result2 is True

    # Verify all keys exist
    all_keys = keys1 + keys2
    lookup_result = client.submit_request(
        RequestType.LOOKUP,
        [all_keys, False],
        get_response_class(RequestType.LOOKUP),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert len(lookup_result) == 50
    assert all(lookup_result), "All stored keys from both batches should exist"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Get chunk size requires CUDA"
)
def test_get_chunk_size(
    client: MessageQueueClient,
):
    """
    Test retrieving the chunk size from the server.
    """
    chunk_size = client.submit_request(
        RequestType.GET_CHUNK_SIZE,
        [],
        get_response_class(RequestType.GET_CHUNK_SIZE),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert chunk_size == CHUNK_SIZE, f"Chunk size should be {CHUNK_SIZE}"
