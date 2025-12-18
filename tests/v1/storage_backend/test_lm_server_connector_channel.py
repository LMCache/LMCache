# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import socket
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.server.__main__ import LMCacheServer
from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.transfer_channel.py_socket_channel import PySocketChannel
from tests.v1.utils import (
    close_asyncio_loop,
    dumb_cache_engine_key,
    get_available_port,
    init_asyncio_loop,
)

logger = init_logger(__name__)


class TestLMCacheServerConnectorChannel:
    """
    Test suite for LMCacheServer and LMCServerConnector communication
    using the refactored transfer channel implementation.

    Tests verify that the refactored code using PySocketChannel
    maintains the same functionality as the original implementation.
    """

    @pytest.fixture
    def server_port(self):
        """Get an available port for the server."""
        return get_available_port()

    @pytest.fixture
    def server(self, server_port):
        """Create and start a LMCacheServer instance."""
        host = "127.0.0.1"
        device = "cpu"
        server = LMCacheServer(host, server_port, device)

        # Start server in a separate thread
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # Give server time to start
        time.sleep(0.2)

        yield server

        # Cleanup
        try:
            server.server_socket.close()
        except Exception:
            pass

    @pytest.fixture
    def async_loop(self):
        """Create an async event loop for testing."""
        async_loop, async_thread = init_asyncio_loop()
        yield async_loop
        close_asyncio_loop(async_loop, async_thread)

    @pytest.fixture
    def local_cpu_backend(self):
        """Create a LocalCPUBackend instance for the connector."""
        # First Party
        from lmcache.config import LMCacheEngineMetadata
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults()
        metadata = LMCacheEngineMetadata(
            "test_model", 1, 0, "vllm", torch.bfloat16, (4, 2, 256, 8, 128)
        )
        backend = LocalCPUBackend(config, metadata, "cpu")
        yield backend
        backend.close()

    @pytest.fixture
    def connector(self, server, server_port, async_loop, local_cpu_backend):
        """Create a LMCServerConnector instance."""
        host = "127.0.0.1"
        # Wait for server to be ready by attempting to connect
        max_retries = 10
        retry_delay = 0.1
        for _ in range(max_retries):
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(0.1)
                test_socket.connect((host, server_port))
                test_socket.close()
                break
            except (socket.error, OSError):
                time.sleep(retry_delay)
        else:
            raise RuntimeError(
                f"Server not ready after {max_retries * retry_delay} seconds"
            )

        connector = LMCServerConnector(host, server_port, async_loop, local_cpu_backend)
        yield connector
        # Cleanup
        try:
            future = asyncio.run_coroutine_threadsafe(connector.close(), async_loop)
            future.result(timeout=2)
        except Exception:
            pass

    def test_channel_initialization(self, connector):
        """Test that the connector uses PySocketChannel."""
        assert hasattr(connector, "channel"), (
            "Connector should have a channel attribute"
        )
        assert isinstance(connector.channel, PySocketChannel), (
            "Connector should use PySocketChannel"
        )
        assert connector.channel.data_socket is not None, (
            "Channel should have a data socket initialized"
        )

    def test_put_command(self, server, connector, async_loop):
        """Test PUT command - server stores data correctly using channel."""
        key = dumb_cache_engine_key()
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 4, num_tokens, 1024])
        dtype = torch.bfloat16

        # Allocate memory object
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        # Fill with test data
        torch.manual_seed(42)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Test PUT command
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=5)

        # Verify data was stored by checking if key exists
        future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
        assert future.result(timeout=5), "Key should exist after PUT command"

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_get_command_success(self, server, connector, async_loop):
        """Test GET command - server retrieves data correctly using channel."""
        key = dumb_cache_engine_key()
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 4, num_tokens, 1024])
        dtype = torch.bfloat16

        # Allocate memory object
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        # Fill with test data
        torch.manual_seed(42)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Store data first
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=5)

        # Test GET command
        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_memory_obj = future.result(timeout=5)

        # Verify data was retrieved correctly
        assert retrieved_memory_obj is not None, (
            "GET should return data for existing key"
        )
        assert retrieved_memory_obj.get_shape() == memory_obj.get_shape()
        assert retrieved_memory_obj.get_dtype() == memory_obj.get_dtype()
        assert (
            retrieved_memory_obj.get_memory_format() == memory_obj.get_memory_format()
        )

        # Verify data content matches
        retrieved_tensor = retrieved_memory_obj.tensor
        original_tensor = memory_obj.tensor
        assert torch.allclose(retrieved_tensor, original_tensor)

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_get_command_fail(self, server, connector, async_loop):
        """Test GET command - server returns None for non-existent key."""
        key = dumb_cache_engine_key()

        # Test GET on non-existent key
        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_memory_obj = future.result(timeout=5)

        # Server should return None (FAIL response) for non-existent key
        assert retrieved_memory_obj is None, (
            "GET should return None for non-existent key"
        )

    def test_exist_command_success(self, server, connector, async_loop):
        """Test EXIST command - server correctly identifies existing key."""
        key = dumb_cache_engine_key()

        # Key should not exist initially
        future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
        assert not future.result(timeout=5), "Key should not exist initially"

        # Put data
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=5)

        # Test EXIST command - key should exist now
        future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
        assert future.result(timeout=5), "Key should exist after PUT"

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_exist_command_fail(self, server, connector, async_loop):
        """Test EXIST command - server correctly identifies non-existent key."""
        key = dumb_cache_engine_key()

        # Test EXIST on non-existent key
        future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
        assert not future.result(timeout=5), (
            "EXIST should return False for non-existent key"
        )

    def test_health_command(self, server, server_port):
        """Test HEALTH command - server responds with success using channel."""
        host = "127.0.0.1"
        key = CacheEngineKey(
            fmt="",
            model_name="",
            world_size=0,
            worker_id=0,
            chunk_hash=0,
            dtype=torch.float16,
        )

        # Create connection and send HEALTH command
        with socket.create_connection((host, server_port), timeout=5) as s:
            msg = ClientMetaMessage(
                ClientCommand.HEALTH,
                key=key,
                length=0,
                fmt=MemoryFormat(1),
                dtype=torch.float16,
                shape=torch.Size((0, 0, 0, 0)),
            )
            s.sendall(msg.serialize())

            # Receive and parse response
            resp = s.recv(ServerMetaMessage.packlength())
            assert resp is not None and len(resp) > 0, (
                "Server should respond to HEALTH command"
            )

            # Parse the response message
            meta = ServerMetaMessage.deserialize(resp)

            # Verify server responded with SUCCESS
            assert meta.code == ServerReturnCode.SUCCESS, (
                "HEALTH command should return SUCCESS"
            )

    def test_channel_data_transfer(self, server, connector, async_loop):
        """Test that channel correctly transfers data between server and connector."""
        key = dumb_cache_engine_key()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Create test data
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()
        torch.manual_seed(123)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Put using channel
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=5)

        # Get using channel
        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_memory_obj = future.result(timeout=5)

        # Verify channel transferred data correctly
        assert retrieved_memory_obj is not None
        assert torch.allclose(retrieved_memory_obj.tensor, memory_obj.tensor)

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_channel_handles_bytearray(self, server, connector, async_loop):
        """Test that channel correctly handles bytearray data."""
        key = dumb_cache_engine_key()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Create memory object with bytearray data
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        # Put and get to verify bytearray handling
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=5)

        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_memory_obj = future.result(timeout=5)

        assert retrieved_memory_obj is not None
        assert retrieved_memory_obj.get_shape() == memory_obj.get_shape()

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_multiple_operations(self, server, connector, async_loop):
        """Test multiple PUT, GET, and EXIST operations using channel."""
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        keys = []
        memory_objs = []

        # Put multiple keys
        for i in range(5):
            key = dumb_cache_engine_key(id=i)
            keys.append(key)
            mem_obj_shape = torch.Size([2, 4, 256, 1024])
            dtype = torch.bfloat16
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            # Fill with unique data
            torch.manual_seed(i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result(timeout=5)
            memory_objs.append(memory_obj)

        # Test EXIST for all keys
        for key in keys:
            future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
            assert future.result(timeout=5), f"Key {key} should exist"

        # Test GET for all keys
        for i, key in enumerate(keys):
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            retrieved_memory_obj = future.result(timeout=5)

            assert retrieved_memory_obj is not None
            assert torch.allclose(retrieved_memory_obj.tensor, memory_objs[i].tensor)

        # Cleanup
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
        memory_allocator.close()

    def test_concurrent_operations(self, server, connector, async_loop):
        """Test concurrent PUT and GET operations using channel."""
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        keys = []
        memory_objs = []

        # Create multiple keys
        for i in range(10):
            key = dumb_cache_engine_key(id=i)
            keys.append(key)
            mem_obj_shape = torch.Size([2, 4, 256, 1024])
            dtype = torch.bfloat16
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            torch.manual_seed(i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
            memory_objs.append(memory_obj)

        # Concurrent PUT operations
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            futures.append(future)

        # Wait for all PUTs to complete
        for future in futures:
            future.result(timeout=10)

        # Concurrent GET operations
        get_futures = []
        for key in keys:
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            get_futures.append(future)

        # Verify all GETs
        for i, future in enumerate(get_futures):
            retrieved_memory_obj = future.result(timeout=10)
            assert retrieved_memory_obj is not None
            assert torch.allclose(retrieved_memory_obj.tensor, memory_objs[i].tensor)

        # Cleanup
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
        memory_allocator.close()

    def test_large_data(self, server, connector, async_loop):
        """Test PUT and GET with large data using channel."""
        key = dumb_cache_engine_key()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)

        # Create larger memory object
        mem_obj_shape = torch.Size([2, 4, 2048, 1024])
        dtype = torch.bfloat16
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(42)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Put large data
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=10)

        # Get large data
        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_memory_obj = future.result(timeout=10)

        assert retrieved_memory_obj is not None
        assert retrieved_memory_obj.get_shape() == memory_obj.get_shape()
        assert torch.allclose(retrieved_memory_obj.tensor, memory_obj.tensor)

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_channel_close(self, server, connector, async_loop):
        """Test that channel closes correctly."""
        # Verify channel is open
        assert connector.channel.data_socket is not None
        assert connector.channel.remote_xfer_handler_exists("test")

        # Close connector (which closes channel)
        future = asyncio.run_coroutine_threadsafe(connector.close(), async_loop)
        future.result(timeout=2)

        # Channel should be closed
        assert (
            connector.channel.data_socket is None
            or connector.channel.data_socket.fileno() == -1
        )

    def test_server_uses_channel_per_connection(self, server, server_port):
        """Test that server creates a new channel for each client connection."""
        host = "127.0.0.1"

        # Create multiple connections
        connections = []
        for _ in range(3):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, server_port))
            connections.append(sock)

        # Each connection should be handled by server
        # (We can't directly verify channel creation, but we can verify
        # connections work)
        key = CacheEngineKey(
            fmt="",
            model_name="",
            world_size=0,
            worker_id=0,
            chunk_hash=0,
            dtype=torch.float16,
        )

        for sock in connections:
            msg = ClientMetaMessage(
                ClientCommand.HEALTH,
                key=key,
                length=0,
                fmt=MemoryFormat(1),
                dtype=torch.float16,
                shape=torch.Size((0, 0, 0, 0)),
            )
            sock.sendall(msg.serialize())
            resp = sock.recv(ServerMetaMessage.packlength())
            assert resp is not None and len(resp) > 0

        # Cleanup
        for sock in connections:
            sock.close()

    def test_channel_async_operations(self, server, connector, async_loop):
        """Test that channel async operations work correctly."""
        key = dumb_cache_engine_key()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Create memory object
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        # Test async send through put (which uses async_batched_send internally)
        # We don't test async_batched_send directly because it requires following
        # the protocol (sending proper ClientMetaMessage first)
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result(timeout=5)

        # Test async recv through get (which uses async_batched_recv internally)
        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved_memory_obj = future.result(timeout=5)

        assert retrieved_memory_obj is not None
        assert retrieved_memory_obj.get_shape() == memory_obj.get_shape()
        assert torch.allclose(retrieved_memory_obj.tensor, memory_obj.tensor)

        memory_obj.ref_count_down()
        memory_allocator.close()

    def test_channel_batched_send_recv_bytearray(self):
        """Test PySocketChannel batched_send/batched_recv with byte payloads.

        This is a direct unit-level test for the data-plane batch primitives,
        independent of the LMCache server protocol.
        """
        if not hasattr(socket, "socketpair"):
            pytest.skip("socketpair not supported on this platform")

        a, b = socket.socketpair()
        try:
            sender = PySocketChannel.__new__(PySocketChannel)
            receiver = PySocketChannel.__new__(PySocketChannel)
            sender.data_socket = a
            receiver.data_socket = b

            sizes = [1, 7, 4096, 12345]
            payloads = [bytes([i]) * sz for i, sz in enumerate(sizes, start=1)]
            buffers = [bytearray(sz) for sz in sizes]
            total_bytes = sum(sizes)

            t0 = time.perf_counter()
            sent = sender.batched_send(payloads)
            t1 = time.perf_counter()
            recvd = receiver.batched_recv(buffers, transfer_spec={})
            t2 = time.perf_counter()

            assert sent == len(payloads)
            assert recvd == len(payloads)
            assert [bytes(buf) for buf in buffers] == payloads

            send_s = max(t1 - t0, 1e-12)
            recv_s = max(t2 - t1, 1e-12)
            total_s = max(t2 - t0, 1e-12)
            throughput_mib_s = (total_bytes / (1024 * 1024)) / total_s

            print(f"\n{'=' * 100}")
            print("Batch Transfer Efficiency (PySocketChannel)")
            print(f"{'=' * 100}")
            print(
                f"{'Mode':10s} | {'Msgs':4s} | {'Bytes':10s} | {'Send (ms)':10s} | "
                f"{'Recv (ms)':10s} | {'Total (ms)':10s} | {'Throughput (MiB/s)':18s}"
            )
            print(f"{'-' * 100}")
            print(
                f"{'sync':10s} | {len(payloads):4d} | {total_bytes:10d} | "
                f"{send_s * 1e3:10.3f} | "
                f"{recv_s * 1e3:10.3f} | {total_s * 1e3:10.3f} | "
                f"{throughput_mib_s:18.2f}"
            )
            print(f"{'=' * 100}\n")

            logger.info(
                "PySocketChannel batched_send/recv metrics: msgs=%d bytes=%d "
                "send_ms=%.3f recv_ms=%.3f total_ms=%.3f throughput_MiBps=%.2f",
                len(payloads),
                total_bytes,
                send_s * 1e3,
                recv_s * 1e3,
                total_s * 1e3,
                throughput_mib_s,
            )
        finally:
            a.close()
            b.close()

    def test_channel_async_batched_send_recv_bytearray(self):
        """Test PySocketChannel async_batched_send/async_batched_recv with byte
        payloads.
        """
        if not hasattr(socket, "socketpair"):
            pytest.skip("socketpair not supported on this platform")

        async def _roundtrip():
            a, b = socket.socketpair()
            a.setblocking(False)
            b.setblocking(False)
            try:
                sender = PySocketChannel.__new__(PySocketChannel)
                receiver = PySocketChannel.__new__(PySocketChannel)
                sender.data_socket = a
                receiver.data_socket = b

                sizes = [1, 7, 4096, 100000]
                payloads = [bytes([i]) * sz for i, sz in enumerate(sizes, start=1)]
                buffers = [bytearray(sz) for sz in sizes]
                total_bytes = sum(sizes)

                async def _timed_send():
                    t0 = time.perf_counter()
                    sent = await sender.async_batched_send(payloads)
                    t1 = time.perf_counter()
                    return sent, max(t1 - t0, 1e-12)

                async def _timed_recv():
                    t0 = time.perf_counter()
                    recvd = await receiver.async_batched_recv(buffers, transfer_spec={})
                    t1 = time.perf_counter()
                    return recvd, max(t1 - t0, 1e-12)

                wall_t0 = time.perf_counter()
                (sent, send_s), (recvd, recv_s) = await asyncio.gather(
                    asyncio.create_task(_timed_send()),
                    asyncio.create_task(_timed_recv()),
                )
                wall_t1 = time.perf_counter()

                assert sent == len(payloads)
                assert recvd == len(payloads)
                assert [bytes(buf) for buf in buffers] == payloads

                wall_s = max(wall_t1 - wall_t0, 1e-12)
                throughput_mib_s = (total_bytes / (1024 * 1024)) / wall_s

                print(f"\n{'=' * 100}")
                print("Batch Transfer Efficiency (PySocketChannel)")
                print(f"{'=' * 100}")
                print(
                    f"{'Mode':10s} | {'Msgs':4s} | {'Bytes':10s} | {'Send (ms)':10s} | "
                    f"{'Recv (ms)':10s} | {'Wall (ms)':10s} | "
                    f"{'Throughput (MiB/s)':18s}"
                )
                print(f"{'-' * 100}")
                print(
                    f"{'async':10s} | {len(payloads):4d} | {total_bytes:10d} | "
                    f"{send_s * 1e3:10.3f} | "
                    f"{recv_s * 1e3:10.3f} | {wall_s * 1e3:10.3f} | "
                    f"{throughput_mib_s:18.2f}"
                )
                print(f"{'=' * 100}\n")

                logger.info(
                    "PySocketChannel async_batched_send/recv metrics: msgs=%d bytes=%d "
                    "send_ms=%.3f recv_ms=%.3f wall_ms=%.3f throughput_MiBps=%.2f",
                    len(payloads),
                    total_bytes,
                    send_s * 1e3,
                    recv_s * 1e3,
                    wall_s * 1e3,
                    throughput_mib_s,
                )
            finally:
                a.close()
                b.close()

        asyncio.run(_roundtrip())

    # ========== Performance Measurement Tests ==========

    def test_measure_put_efficiency(self, server, connector, async_loop):
        """Measure PUT operation efficiency (latency and throughput) using channel."""
        # Standard
        import statistics

        num_iterations = 100
        latencies = []
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Warm-up
        warmup_key = dumb_cache_engine_key(id=-1)
        warmup_memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        warmup_memory_obj.ref_count_up()
        future = asyncio.run_coroutine_threadsafe(
            connector.put(warmup_key, warmup_memory_obj), async_loop
        )
        future.result(timeout=5)
        warmup_memory_obj.ref_count_down()

        # Measure PUT operations
        for i in range(num_iterations):
            key = dumb_cache_engine_key(id=i)
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result(timeout=5)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            memory_obj.ref_count_down()

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        throughput = 1000 / avg_latency if avg_latency > 0 else 0

        # Print results
        print(f"\n{'=' * 60}")
        print(f"PUT Operation Performance with Channel (n={num_iterations})")
        print(f"{'=' * 60}")
        print(f"Average Latency: {avg_latency:.3f} ms")
        print(f"Median Latency:  {median_latency:.3f} ms")
        print(f"Min Latency:     {min_latency:.3f} ms")
        print(f"Max Latency:     {max_latency:.3f} ms")
        print(f"P95 Latency:     {p95_latency:.3f} ms")
        print(f"Throughput:      {throughput:.2f} ops/sec")
        print(f"{'=' * 60}\n")

        memory_allocator.close()

    def test_measure_get_efficiency(self, server, connector, async_loop):
        """Measure GET operation efficiency (latency and throughput) using channel."""
        # Standard
        import statistics

        num_iterations = 100
        latencies = []
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Prepare data first
        keys = []
        for i in range(num_iterations):
            key = dumb_cache_engine_key(id=i)
            keys.append(key)
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result(timeout=5)
            memory_obj.ref_count_down()

        time.sleep(0.1)  # Small delay to ensure data is stored

        # Warm-up
        future = asyncio.run_coroutine_threadsafe(connector.get(keys[0]), async_loop)
        future.result(timeout=5)

        # Measure GET operations
        for key in keys:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            future.result(timeout=5)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        throughput = 1000 / avg_latency if avg_latency > 0 else 0

        # Print results
        print(f"\n{'=' * 60}")
        print(f"GET Operation Performance with Channel (n={num_iterations})")
        print(f"{'=' * 60}")
        print(f"Average Latency: {avg_latency:.3f} ms")
        print(f"Median Latency:  {median_latency:.3f} ms")
        print(f"Min Latency:     {min_latency:.3f} ms")
        print(f"Max Latency:     {max_latency:.3f} ms")
        print(f"P95 Latency:     {p95_latency:.3f} ms")
        print(f"Throughput:      {throughput:.2f} ops/sec")
        print(f"{'=' * 60}\n")

        memory_allocator.close()

    def test_measure_exist_efficiency(self, server, connector, async_loop):
        """Measure EXIST operation efficiency (latency and throughput) using channel."""
        # Standard
        import statistics

        num_iterations = 100
        latencies = []
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Prepare data first
        keys = []
        for i in range(num_iterations):
            key = dumb_cache_engine_key(id=i)
            keys.append(key)
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result(timeout=5)
            memory_obj.ref_count_down()

        time.sleep(0.1)  # Small delay to ensure data is stored

        # Warm-up
        future = asyncio.run_coroutine_threadsafe(connector.exists(keys[0]), async_loop)
        future.result(timeout=5)

        # Measure EXIST operations
        for key in keys:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
            future.result(timeout=5)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        throughput = 1000 / avg_latency if avg_latency > 0 else 0

        # Print results
        print(f"\n{'=' * 60}")
        print(f"EXIST Operation Performance with Channel (n={num_iterations})")
        print(f"{'=' * 60}")
        print(f"Average Latency: {avg_latency:.3f} ms")
        print(f"Median Latency:  {median_latency:.3f} ms")
        print(f"Min Latency:     {min_latency:.3f} ms")
        print(f"Max Latency:     {max_latency:.3f} ms")
        print(f"P95 Latency:     {p95_latency:.3f} ms")
        print(f"Throughput:      {throughput:.2f} ops/sec")
        print(f"{'=' * 60}\n")

        memory_allocator.close()

    def test_measure_health_efficiency(self, server, server_port):
        """Measure HEALTH command efficiency (latency and throughput) using channel."""
        # Standard
        import statistics

        num_iterations = 100
        latencies = []
        host = "127.0.0.1"
        key = CacheEngineKey(
            fmt="",
            model_name="",
            world_size=0,
            worker_id=0,
            chunk_hash=0,
            dtype=torch.float16,
        )

        # Warm-up
        with socket.create_connection((host, server_port), timeout=5) as s:
            msg = ClientMetaMessage(
                ClientCommand.HEALTH,
                key=key,
                length=0,
                fmt=MemoryFormat(1),
                dtype=torch.float16,
                shape=torch.Size((0, 0, 0, 0)),
            )
            s.sendall(msg.serialize())
            s.recv(ServerMetaMessage.packlength())

        # Measure HEALTH operations
        for _ in range(num_iterations):
            with socket.create_connection((host, server_port), timeout=5) as s:
                msg = ClientMetaMessage(
                    ClientCommand.HEALTH,
                    key=key,
                    length=0,
                    fmt=MemoryFormat(1),
                    dtype=torch.float16,
                    shape=torch.Size((0, 0, 0, 0)),
                )
                start_time = time.perf_counter()
                s.sendall(msg.serialize())
                s.recv(ServerMetaMessage.packlength())
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        throughput = 1000 / avg_latency if avg_latency > 0 else 0

        # Print results
        print(f"\n{'=' * 60}")
        print(f"HEALTH Operation Performance with Channel (n={num_iterations})")
        print(f"{'=' * 60}")
        print(f"Average Latency: {avg_latency:.3f} ms")
        print(f"Median Latency:  {median_latency:.3f} ms")
        print(f"Min Latency:     {min_latency:.3f} ms")
        print(f"Max Latency:     {max_latency:.3f} ms")
        print(f"P95 Latency:     {p95_latency:.3f} ms")
        print(f"Throughput:      {throughput:.2f} ops/sec")
        print(f"{'=' * 60}\n")

    def test_measure_all_operations_efficiency(self, server, connector, async_loop):
        """Measure efficiency of all operations in a single test using channel."""
        # Standard
        import statistics

        num_iterations = 50
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        mem_obj_shape = torch.Size([2, 4, 256, 1024])
        dtype = torch.bfloat16

        # Prepare data
        keys = []
        memory_objs = []
        for i in range(num_iterations):
            key = dumb_cache_engine_key(id=i)
            keys.append(key)
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            memory_objs.append(memory_obj)

        # Measure PUT
        put_latencies = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result(timeout=5)
            end_time = time.perf_counter()
            put_latencies.append((end_time - start_time) * 1000)

        time.sleep(0.1)  # Small delay

        # Measure GET
        get_latencies = []
        for key in keys:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            future.result(timeout=5)
            end_time = time.perf_counter()
            get_latencies.append((end_time - start_time) * 1000)

        # Measure EXIST
        exist_latencies = []
        for key in keys:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
            future.result(timeout=5)
            end_time = time.perf_counter()
            exist_latencies.append((end_time - start_time) * 1000)

        # Calculate and print statistics
        def print_stats(op_name, latencies):
            avg = statistics.mean(latencies)
            median = statistics.median(latencies)
            p95 = statistics.quantiles(latencies, n=20)[18]
            throughput = 1000 / avg if avg > 0 else 0
            print(
                f"{op_name:8s} | {avg:8.3f} | {median:8.3f} | {p95:8.3f} | "
                f"{throughput:10.2f}"
            )

        print(f"\n{'=' * 70}")
        print(f"All Operations Performance Summary with Channel (n={num_iterations})")
        print(f"{'=' * 70}")
        print(
            f"{'Operation':8s} | {'Avg (ms)':8s} | "
            f"{'Median (ms)':8s} | {'P95 (ms)':8s} | "
            f"{'Throughput (ops/s)':10s}"
        )
        print(f"{'-' * 70}")
        print_stats("PUT", put_latencies)
        print_stats("GET", get_latencies)
        print_stats("EXIST", exist_latencies)
        print(f"{'=' * 70}\n")

        # Cleanup
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
        memory_allocator.close()

    def test_measure_operation_efficiency_by_data_size(
        self, server, connector, async_loop
    ):
        """Measure operation efficiency for different data sizes using channel."""
        # Standard

        data_sizes = [
            (64, "Small"),
            (256, "Medium"),
            (1024, "Large"),
            (2048, "XLarge"),
        ]
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        dtype = torch.bfloat16

        print(f"\n{'=' * 100}")
        print("Operation Efficiency by Data Size (with Channel)")
        print(f"{'=' * 100}")
        print(
            f"{'Shape':30s} | {'PUT (ms)':10s} | {'GET (ms)':10s} | {'EXIST (ms)':10s}"
        )
        print(f"{'-' * 100}")

        for num_tokens, size_label in data_sizes:
            mem_obj_shape = torch.Size([2, 4, num_tokens, 1024])
            shape_str = str(list(mem_obj_shape))
            key = dumb_cache_engine_key(id=hash(size_label))

            # Measure PUT
            memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result(timeout=10)
            put_latency = (time.perf_counter() - start_time) * 1000

            time.sleep(0.1)

            # Measure GET
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            future.result(timeout=10)
            get_latency = (time.perf_counter() - start_time) * 1000

            # Measure EXIST
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop)
            future.result(timeout=5)
            exist_latency = (time.perf_counter() - start_time) * 1000

            print(
                f"{shape_str:30s} | {put_latency:10.3f} | {get_latency:10.3f} | "
                f"{exist_latency:10.3f}"
            )

            memory_obj.ref_count_down()

        print(f"{'=' * 100}\n")
        memory_allocator.close()
