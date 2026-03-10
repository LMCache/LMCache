# SPDX-License-Identifier: Apache-2.0
"""
Tests for server restart resilience.

When the LMCache server restarts, previously registered KV caches are
lost.  The worker adapter should detect the failure and automatically
re-register so that subsequent operations succeed.
"""

# Standard
from multiprocessing.synchronize import Event as EventClass
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
    OperationStatus,
)
from lmcache.v1.multiprocess.mq import (
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)

# Test helpers (reuse handler helpers from existing tests)
from tests.v1.multiprocess import test_mq_handler_helpers

# ================================================================
# Constants
# ================================================================

SERVER_HOST = "localhost"
SERVER_PORT = 5610
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0
DEFAULT_TIMEOUT = 5.0
BLOCKS_PER_KEY = 16


# ================================================================
# Helpers
# ================================================================


def _has_working_new_shared_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        buf = torch.empty(1024, device="cuda")
        shared = buf.untyped_storage()._share_cuda_()
        return shared is not None
    except Exception:
        return False


def create_cache_key(index: int, model: str = "testmodel") -> IPCCacheEngineKey:
    token_ids = [index] * CHUNK_SIZE
    return IPCCacheEngineKey.from_token_ids(
        model,
        1,
        0,
        token_ids,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"test_request_{index}",
    )


def server_process_runner(
    host: str,
    port: int,
    chunk_size: int,
    cpu_buffer_size: float,
):
    """Entry point for server subprocess.

    Heavy imports are deferred here so that the test module
    can be collected even without the native C extensions.
    """
    # First Party (deferred to avoid native import at module level)
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.mp_observability.config import (
        DEFAULT_PROMETHEUS_CONFIG,
    )
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.server import run_cache_server

    mp_config = MPServerConfig(host=host, port=port, chunk_size=chunk_size)
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
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        prometheus_config=DEFAULT_PROMETHEUS_CONFIG,
    )


def _start_server() -> mp.Process:
    """Start a cache server in a subprocess."""
    proc = mp.Process(
        target=server_process_runner,
        args=(
            SERVER_HOST,
            SERVER_PORT,
            CHUNK_SIZE,
            CPU_BUFFER_SIZE,
        ),
        daemon=True,
    )
    proc.start()
    time.sleep(2)  # wait for server to be ready
    return proc


def _stop_server(proc: mp.Process) -> None:
    """Terminate the server process."""
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
    # Give OS time to release the port
    time.sleep(1)


# ================================================================
# Protocol-level test: store on unregistered instance (no CUDA)
# ================================================================


def _server_with_store_handler(
    server_url: str,
    ready_event: EventClass,
    shutdown_event: EventClass,
):
    """Run a minimal MQ server with a STORE handler."""
    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)

    payload_classes = get_payload_classes(RequestType.STORE)
    handler_type = get_handler_type(RequestType.STORE)
    server.add_handler(
        RequestType.STORE,
        payload_classes,
        handler_type,
        test_mq_handler_helpers.store_handler,
    )

    server.start()
    ready_event.set()
    shutdown_event.wait()
    server.close()


def test_mq_store_path_with_unregistered_instance():
    """Verify that the MQ STORE path does not crash when the
    instance_id is unknown; the mock handler always succeeds,
    confirming the plumbing is intact."""
    key = create_cache_key(0)
    gpu_id = 9999  # deliberately unregistered
    gpu_block_ids = [0, 1, 2]
    test_handle = b"\x00" * 64

    ctx = mp.get_context("spawn")
    ready_event = ctx.Event()
    shutdown_event = ctx.Event()
    url = "tcp://127.0.0.1:5611"

    proc = ctx.Process(
        target=_server_with_store_handler,
        args=(url, ready_event, shutdown_event),
    )
    proc.start()
    assert ready_event.wait(timeout=5), "Server did not start"
    time.sleep(0.2)

    zmq_ctx = zmq.Context.instance()
    client = MessageQueueClient(url, zmq_ctx)

    try:
        future = client.submit_request(
            RequestType.STORE,
            [key, gpu_id, gpu_block_ids, test_handle],
            get_response_class(RequestType.STORE),
        )
        result = future.result(timeout=5)
        # Mock handler returns (b"\x01"*64, OperationStatus.SUCCESS).
        assert result == (
            b"\x01" * 64,
            OperationStatus.SUCCESS,
        )
    finally:
        client.close()
        shutdown_event.set()
        proc.join(timeout=3)


# ================================================================
# Integration test: server restart with auto re-registration
# ================================================================

if not _has_working_new_shared_cuda():
    SKIP_CUDA = True
else:
    SKIP_CUDA = False


@pytest.mark.skipif(SKIP_CUDA, reason="CUDA required")
class TestServerRestart:
    """Integration tests that verify the full server restart
    scenario using real GPU tensors."""

    @staticmethod
    def _initialize_kv_cache(
        device: torch.device,
        num_pages: int = 1024,
        num_layers: int = 32,
        page_size: int = 16,
        num_heads: int = 8,
        head_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[torch.Tensor]:
        torch.random.manual_seed(42)
        return [
            torch.rand(
                (2, num_pages, page_size, num_heads, head_size),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def test_store_fails_then_succeeds_after_reregister(
        self,
    ):
        """After a server restart the first store fails because
        the instance is no longer registered.  After re-registering
        the same instance, subsequent stores succeed."""
        mp.set_start_method("spawn", force=True)
        device = torch.device("cuda:0")
        kv_tensors = self._initialize_kv_cache(device)
        kv_cache: KVCache = [CudaIPCWrapper(t) for t in kv_tensors]
        instance_id = os.getpid()

        # -- Phase 1: start server, register, verify store works
        server_proc = _start_server()
        zmq_ctx = zmq.Context.instance()
        client = MessageQueueClient(SERVER_URL, zmq_ctx)

        try:
            # Register
            client.submit_request(
                RequestType.REGISTER_KV_CACHE,
                [instance_id, kv_cache, "testmodel", 1],
                get_response_class(RequestType.REGISTER_KV_CACHE),
            ).result(timeout=DEFAULT_TIMEOUT)

            # Store should succeed
            key = create_cache_key(0)
            event = torch.cuda.Event(interprocess=True)
            event.record()
            result = (
                client.submit_request(
                    RequestType.STORE,
                    [
                        key,
                        instance_id,
                        list(range(BLOCKS_PER_KEY)),
                        event.ipc_handle(),
                    ],
                    get_response_class(RequestType.STORE),
                )
                .to_cuda_future()
                .result(timeout=DEFAULT_TIMEOUT)
            )
            assert result == OperationStatus.SUCCESS, (
                "Store should succeed when registered"
            )

            client.close()
        finally:
            _stop_server(server_proc)

        # -- Phase 2: restart server (no registration)
        server_proc = _start_server()
        client = MessageQueueClient(SERVER_URL, zmq_ctx)

        try:
            # Store should FAIL (instance not registered)
            key2 = create_cache_key(1)
            event2 = torch.cuda.Event(interprocess=True)
            event2.record()
            result2 = (
                client.submit_request(
                    RequestType.STORE,
                    [
                        key2,
                        instance_id,
                        list(range(BLOCKS_PER_KEY)),
                        event2.ipc_handle(),
                    ],
                    get_response_class(RequestType.STORE),
                )
                .to_cuda_future()
                .result(timeout=DEFAULT_TIMEOUT)
            )
            assert result2 is not OperationStatus.SUCCESS, (
                "Store should fail on restarted server without re-registration"
            )

            # Re-register
            client.submit_request(
                RequestType.REGISTER_KV_CACHE,
                [instance_id, kv_cache, "testmodel", 1],
                get_response_class(RequestType.REGISTER_KV_CACHE),
            ).result(timeout=DEFAULT_TIMEOUT)

            # Store should succeed now
            key3 = create_cache_key(2)
            event3 = torch.cuda.Event(interprocess=True)
            event3.record()
            result3 = (
                client.submit_request(
                    RequestType.STORE,
                    [
                        key3,
                        instance_id,
                        list(range(BLOCKS_PER_KEY)),
                        event3.ipc_handle(),
                    ],
                    get_response_class(RequestType.STORE),
                )
                .to_cuda_future()
                .result(timeout=DEFAULT_TIMEOUT)
            )
            assert result3 == OperationStatus.SUCCESS, (
                "Store should succeed after re-registration"
            )

            client.close()
        finally:
            _stop_server(server_proc)

        # Cleanup GPU memory
        del kv_tensors
        torch.cuda.empty_cache()
