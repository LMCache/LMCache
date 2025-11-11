# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing.synchronize import Event as EventClass
from typing import Any, Callable
import multiprocessing as mp
import sys
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper, IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import (
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)

# Test helpers
from tests.v1.multiprocess import test_mq_handler_helpers

# ==============================================================================
# MessageQueueServer and MessageQueueClient Tests Infrastructure
# ==============================================================================


def _server_process(
    server_url: str,
    ready_event: EventClass,
    shutdown_event: EventClass,
    request_handlers: dict[RequestType, Callable],
):
    """
    Server process that runs the MessageQueueServer.

    Args:
        server_url: URL to bind the server to
        ready_event: Event to signal when server is ready
        shutdown_event: Event to signal server shutdown
        request_handlers: Dict mapping RequestType to handler functions
    """
    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)

    # Register all handlers
    for request_type, handler in request_handlers.items():
        payload_classes = get_payload_classes(request_type)
        handler_type = get_handler_type(request_type)
        server.add_handler(request_type, payload_classes, handler_type, handler)

    server.start()

    # Signal that server is ready
    ready_event.set()

    # Wait for shutdown signal
    shutdown_event.wait()

    # Cleanup
    server.close()


def _run_client_test(
    server_url: str,
    ready_event: EventClass,
    request_type: RequestType,
    payloads: list[Any],
    expected_response: Any,
    num_requests: int = 1,
    client_id: int = 0,
) -> None:
    """
    Client process that sends requests and validates responses.

    Args:
        server_url: URL to connect to
        ready_event: Event to wait for server to be ready
        request_type: Type of request to send
        payloads: List of payloads for the request
        expected_response: Expected response from server
        num_requests: Number of requests to send
        client_id: ID of this client (for debugging)

    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Wait for server to be ready
    if not ready_event.wait(timeout=5):
        print(f"Client {client_id}: Server failed to start within timeout")
        sys.exit(1)

    # Small delay to ensure server is fully initialized
    time.sleep(0.1)

    context = zmq.Context.instance()
    client = MessageQueueClient(server_url, context)
    successful = True

    try:
        futures = []
        # Submit requests
        for _ in range(num_requests):
            future = client.submit_request(request_type, payloads)  # type: ignore
            futures.append(future)

        # Validate responses
        for i, future in enumerate(futures):
            response = future.result(timeout=5)
            if response != expected_response:
                print(
                    f"Client {client_id}, Request {i}: Expected "
                    f"{expected_response}, got {response}"
                )

                # Exit with error code
                client.close()
                sys.exit(1)

    except Exception as e:
        print(f"Client {client_id} test failed with exception: {e}")
        successful = False
    finally:
        client.close()
        if not successful:
            sys.exit(1)


class MessageQueueTestHelper:
    """
    Helper class to facilitate testing MessageQueueServer and MessageQueueClient.

    Supports testing with single or multiple concurrent clients, where each client
    can send multiple requests to the server.

    Usage:
        1. Create an instance with server URL
        2. Register handlers for different RequestTypes
        3. Call run_test() to execute the test with client requests

    Example:
        helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5556")
        helper.register_handler(RequestType.NOOP, noop_handler)
        helper.run_test(
            request_type=RequestType.NOOP,
            payloads=[],
            expected_response="NOOP_OK",
            num_requests=10,  # Each client sends 10 requests
            num_clients=3,    # Start 3 concurrent clients
        )
    """

    def __init__(self, server_url: str = "tcp://127.0.0.1:5556"):
        self.server_url = server_url
        self.handlers: dict[RequestType, Callable] = {}
        self.ctx = mp.get_context("spawn")

    def register_handler(
        self,
        request_type: RequestType,
        handler: Callable,
    ) -> "MessageQueueTestHelper":
        """
        Register a handler for a specific RequestType.

        Args:
            request_type: The type of request to handle
            handler: Handler function that matches the protocol signature

        Returns:
            self for method chaining
        """
        self.handlers[request_type] = handler
        return self

    def run_test(
        self,
        request_type: RequestType,
        payloads: list[Any],
        expected_response: Any,
        num_requests: int = 1,
        num_clients: int = 1,
        timeout: float = 10.0,
    ) -> None:
        """
        Run a test by starting server and client processes.

        Args:
            request_type: Type of request to send
            payloads: List of payloads for the request
            expected_response: Expected response from server
            num_requests: Number of requests each client should send
            num_clients: Number of client processes to start
            timeout: Maximum time to wait for test completion

        Raises:
            AssertionError: If test fails
        """
        ready_event = self.ctx.Event()
        shutdown_event = self.ctx.Event()

        # Start server process
        server_process = self.ctx.Process(
            target=_server_process,
            args=(self.server_url, ready_event, shutdown_event, self.handlers),
        )
        server_process.start()

        # Start multiple client processes
        client_processes = []
        for client_id in range(num_clients):
            client_process = self.ctx.Process(
                target=_run_client_test,
                args=(
                    self.server_url,
                    ready_event,
                    request_type,
                    payloads,
                    expected_response,
                    num_requests,
                    client_id,
                ),
            )
            client_process.start()
            client_processes.append(client_process)

        # Wait for all clients to complete
        failed_clients = []
        for client_id, client_process in enumerate(client_processes):
            client_process.join(timeout=timeout)

            # Check if client completed successfully
            if client_process.is_alive():
                client_process.terminate()
                client_process.join()
                failed_clients.append((client_id, "timeout"))
            elif client_process.exitcode != 0:
                failed_clients.append(
                    (client_id, f"exit code {client_process.exitcode}")
                )

        # Shutdown server
        shutdown_event.set()
        server_process.join(timeout=2)

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()

        # Report any failures
        if failed_clients:
            failure_details = ", ".join(
                [f"Client {cid}: {reason}" for cid, reason in failed_clients]
            )
            pytest.fail(f"Some clients failed: {failure_details}")

        if server_process.exitcode != 0:
            pytest.fail(
                f"Server process failed with exit code {server_process.exitcode}"
            )


# ==============================================================================
# Tests for Different RequestTypes
# ==============================================================================


def test_mq_noop_request():
    """
    Test MessageQueue with NOOP request type.
    NOOP takes no payloads and returns a string response.
    """
    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5556")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with single request
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=1,
    )


def test_mq_noop_multiple_requests():
    """
    Test MessageQueue with multiple NOOP requests.
    Verifies that server can handle multiple sequential requests.
    """
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5557")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with multiple requests
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=10,
    )


def test_mq_noop_multiple_clients():
    """
    Test MessageQueue with multiple concurrent clients.
    Verifies that server can handle requests from multiple clients simultaneously.
    """
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5558")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with multiple clients, each sending multiple requests
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=5,
        num_clients=3,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for REGISTER_KV_CACHE tests",
)
def test_mq_register_kv_cache():
    """
    Test MessageQueue with REGISTER_KV_CACHE request type.
    REGISTER_KV_CACHE takes (gpu_id: int, kv_cache: KVCache) and returns None.
    """
    # Create test KV cache (list of CudaIPCWrapper objects)
    kv_cache = []
    for _ in range(3):
        tensor = torch.randn(2, 4, device="cuda")
        wrapper = CudaIPCWrapper(tensor)
        kv_cache.append(wrapper)

    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5559")
    helper.register_handler(
        RequestType.REGISTER_KV_CACHE, test_mq_handler_helpers.register_kv_cache_handler
    )

    # Run test with REGISTER_KV_CACHE request
    helper.run_test(
        request_type=RequestType.REGISTER_KV_CACHE,
        payloads=[gpu_id, kv_cache],
        expected_response=None,
        num_requests=1,
    )


def test_mq_unregister_kv_cache():
    """
    Test MessageQueue with UNREGISTER_KV_CACHE request type.
    UNREGISTER_KV_CACHE takes (gpu_id: int) and returns None.
    """
    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5560")
    helper.register_handler(
        RequestType.UNREGISTER_KV_CACHE,
        test_mq_handler_helpers.unregister_kv_cache_handler,
    )

    # Run test with UNREGISTER_KV_CACHE request
    helper.run_test(
        request_type=RequestType.UNREGISTER_KV_CACHE,
        payloads=[gpu_id],
        expected_response=None,
        num_requests=1,
    )


def test_mq_unregister_kv_cache_multiple_clients():
    """
    Test MessageQueue with UNREGISTER_KV_CACHE from multiple clients.
    Verifies that multiple clients can unregister KV caches concurrently.
    """
    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5561")
    helper.register_handler(
        RequestType.UNREGISTER_KV_CACHE,
        test_mq_handler_helpers.unregister_kv_cache_handler,
    )

    # Run test with multiple clients
    helper.run_test(
        request_type=RequestType.UNREGISTER_KV_CACHE,
        payloads=[gpu_id],
        expected_response=None,
        num_requests=3,
        num_clients=2,
    )


def test_mq_store():
    """
    Test MessageQueue with STORE request type.
    STORE takes (keys: list[KeyType], gpu_id: int, gpu_block_ids: list[int])
    and returns bool.
    """
    # Create test keys
    keys = [
        IPCCacheEngineKey.from_int_hash(
            model_name="test_model", world_size=1, worker_id=0, chunk_hash=i
        )
        for i in range(3)
    ]
    gpu_id = 0
    gpu_block_ids = [0, 1, 2]
    test_handle = b"\x00" * 64

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5562")
    helper.register_handler(RequestType.STORE, test_mq_handler_helpers.store_handler)

    # Run test with STORE request
    helper.run_test(
        request_type=RequestType.STORE,
        payloads=[keys, gpu_id, gpu_block_ids, test_handle],
        expected_response=(b"\x01" * 64, True),
        num_requests=1,
    )


def test_mq_retrieve():
    """
    Test MessageQueue with RETRIEVE request type.
    RETRIEVE takes (keys: list[KeyType], gpu_id: int, gpu_block_ids: list[int])
    and returns bool.
    """
    # Create test keys
    keys = [
        IPCCacheEngineKey.from_int_hash(
            model_name="test_model", world_size=1, worker_id=0, chunk_hash=i
        )
        for i in range(3)
    ]
    gpu_id = 0
    gpu_block_ids = [0, 1, 2]
    test_handle = b"\x00" * 64

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5563")
    helper.register_handler(
        RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )

    # Run test with RETRIEVE request
    helper.run_test(
        request_type=RequestType.RETRIEVE,
        payloads=[keys, gpu_id, gpu_block_ids, test_handle],
        expected_response=(b"\x01" * 64, [True, True, True]),
        num_requests=1,
    )


def test_mq_lookup():
    """
    Test MessageQueue with LOOKUP request type.
    LOOKUP takes (keys: list[KeyType], lock: Optional[bool])
    and returns list[bool].
    """
    # Create test keys
    keys = [
        IPCCacheEngineKey.from_int_hash(
            model_name="test_model", world_size=1, worker_id=0, chunk_hash=i
        )
        for i in range(4)
    ]
    lock = True

    # Expected response: alternating True/False for each key
    expected_response = [True, False, True, False]

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5564")
    helper.register_handler(RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler)

    # Run test with LOOKUP request
    helper.run_test(
        request_type=RequestType.LOOKUP,
        payloads=[keys, lock],
        expected_response=expected_response,
        num_requests=1,
    )


def test_mq_lookup_with_none_lock():
    """
    Test MessageQueue with LOOKUP request type with None lock parameter.
    Tests that Optional[bool] parameter works correctly with None value.
    """
    # Create test keys
    keys = [
        IPCCacheEngineKey.from_int_hash(
            model_name="test_model", world_size=1, worker_id=0, chunk_hash=i
        )
        for i in range(3)
    ]
    lock = None

    # Expected response: alternating True/False for each key
    expected_response = [True, False, True]

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5565")
    helper.register_handler(RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler)

    # Run test with LOOKUP request with None lock
    helper.run_test(
        request_type=RequestType.LOOKUP,
        payloads=[keys, lock],
        expected_response=expected_response,
        num_requests=1,
    )
