# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from unittest.mock import patch
import asyncio
import random
import shlex
import socket
import subprocess
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngineBuilder


# Monkey patch RemoteBackend.close() to add timeout for tests only
@pytest.fixture(scope="session", autouse=True)
def patch_remote_backend_close():
    """Monkey patch RemoteBackend.close() to add timeout behavior for tests."""
    try:
        # First Party
        from lmcache.logging import init_logger
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        logger = init_logger(__name__)
        original_close = RemoteBackend.close

        def close_with_timeout(self):
            """Test-only version of RemoteBackend.close() with timeout."""
            try:
                assert self.connection is not None
                future = asyncio.run_coroutine_threadsafe(
                    self.connection.close(), self.loop
                )
                future.result(timeout=10.0)  # 10 second timeout for tests
                logger.info("Remote backend closed.")
            except TimeoutError:
                logger.warning(
                    "Remote connection close timed out after 10s, forcing cleanup"
                )
                # Cancel the future to prevent the coroutine warning
                future.cancel()
            except Exception as e:
                logger.warning(f"Error occurred when closing remote connection: {e}")

        # Apply the monkey patch
        RemoteBackend.close = close_with_timeout

        yield

        # Restore original method after tests
        RemoteBackend.close = original_close

    except ImportError:
        # RemoteBackend not available (tests that don't use it)
        yield


class MockRedis:
    def __init__(
        self, host=None, port=None, url=None, decode_responses=False, **kwargs
    ):
        self.store = {}
        self.host = host
        self.port = port
        self.url = url
        self.decode_responses = decode_responses

    def set(self, key, value):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key, None)

    def exists(self, key):
        return key in self.store

    def scan(self, cursor=0, match=None):
        keys = [s.encode("utf-8") for s in self.store.keys()]
        return (0, keys)

    def close(self):
        pass

    @classmethod
    def from_url(cls, url, decode_responses=False, **kwargs):
        """Mock implementation of Redis.from_url"""
        return cls(url=url, decode_responses=decode_responses, **kwargs)


class MockRedisSentinel:
    def __init__(self, hosts_and_ports, socket_timeout=None, **kwargs):
        self.redis = MockRedis()
        self.hosts_and_ports = hosts_and_ports
        self.socket_timeout = socket_timeout

    def master_for(
        self, service_name, socket_timeout=None, username=None, password=None, **kwargs
    ):
        return self.redis

    def slave_for(
        self, service_name, socket_timeout=None, username=None, password=None, **kwargs
    ):
        return self.redis


@dataclass
class LMCacheServerProcess:
    server_url: str
    server_process: object


@pytest.fixture(scope="function", autouse=True)
def mock_redis():
    with (
        patch("redis.Redis", MockRedis) as mock_redis_class,
        patch("redis.from_url", MockRedis.from_url),
    ):
        yield mock_redis_class


@pytest.fixture(scope="function", autouse=True)
def mock_redis_sentinel():
    with patch("redis.Sentinel", MockRedisSentinel) as mock:
        yield mock


@pytest.fixture(scope="module")
def lmserver_v1_process(request):
    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set socket timeout to prevent infinite hangs
        client_socket.settimeout(2.0)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                continue
            except Exception:
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        proc = subprocess.Popen(
            shlex.split(
                f"python3 -m lmcache.v1.server localhost {port_number} {device}"
            )
        )

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            # Process has terminated - this is bad, server failed to start
            successful = False
        else:
            # Process is still running - try to connect to it
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

    # Terminate the process
    proc.terminate()
    try:
        proc.wait(timeout=10)  # Add 10 second timeout
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))


@pytest.fixture(scope="module")
def lmserver_process(request):
    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set socket timeout to prevent infinite hangs
        client_socket.settimeout(2.0)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                print("Probing connection, remaining retries: ", retries)
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                print("Connection refused!")
                continue
            except Exception as e:
                print(f"other Exception: {e}")
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        print("Starting the lmcache server process on port")
        proc = subprocess.Popen(
            shlex.split(f"python3 -m lmcache.server localhost {port_number} {device}")
        )

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            # Process has terminated - this is bad, server failed to start
            successful = False
        else:
            # Process is still running - try to connect to it
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

    # Terminate the process
    proc.terminate()
    proc.wait()

    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))


@pytest.fixture(scope="session", autouse=True)
def global_cuda_cleanup():
    """Global CUDA cleanup at session start and end."""
    # Standard
    import gc

    # Third Party
    import torch

    # Aggressive cleanup at start of session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset all CUDA devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    gc.collect()

    yield

    # Cleanup at end of session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@pytest.fixture(scope="module", autouse=True)
def module_level_allocator_cleanup():
    """Cleanup allocators between test modules to prevent cross-module leaks."""
    yield  # Let the module run first

    # Aggressive cleanup after each test module
    # Standard
    import gc

    # Third Party
    import torch

    print("🧹 [DEBUG] Module-level allocator cleanup starting...")

    # Force cleanup of any remaining allocators
    allocator_types = [
        "AdHocMemoryAllocator",
        "PinMemoryAllocator",
        "MixedMemoryAllocator",
        "GPUMemoryAllocator",
        "HostMemoryAllocator",
        "PagedTensorMemoryAllocator",
        "TensorMemoryAllocator",
    ]

    cleanup_count = 0
    for obj in gc.get_objects():
        try:
            if hasattr(obj, "__class__") and obj.__class__.__name__ in allocator_types:
                if hasattr(obj, "close"):
                    print(f"🔧 [DEBUG] Force-closing leaked {obj.__class__.__name__}")
                    obj.close()
                    cleanup_count += 1
        except Exception:
            # Ignore errors during force cleanup
            pass

    # Additional cleanup for test allocators
    for obj in gc.get_objects():
        try:
            if hasattr(obj, "_test_allocator"):
                if hasattr(obj._test_allocator, "close"):
                    print("🔧 [DEBUG] Force-closing _test_allocator")
                    obj._test_allocator.close()
                    cleanup_count += 1
        except Exception:
            pass

    print(f"🧹 [DEBUG] Module cleanup: closed {cleanup_count} allocators")

    # Force CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🔧 [DEBUG] CUDA cache cleared between modules")

    # Force garbage collection
    gc.collect()
    print("🧹 [DEBUG] Module-level cleanup complete")


@pytest.fixture(scope="function")
def autorelease(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    # Cleanup all objects created by the factory
    for obj in objects:
        obj.close()

    # Add CUDA cleanup after each test using legacy engines
    # Third Party
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(scope="function", autouse=True)
def aggressive_allocator_cleanup():
    """Force cleanup ALL allocators after every single test."""
    yield  # Let the test run first

    # Standard
    import gc

    # Third Party
    import torch

    # Force cleanup of ANY allocator in memory
    allocator_types = [
        "AdHocMemoryAllocator",
        "PinMemoryAllocator",
        "MixedMemoryAllocator",
        "GPUMemoryAllocator",
        "HostMemoryAllocator",
        "PagedTensorMemoryAllocator",
        "TensorMemoryAllocator",
    ]

    cleanup_count = 0
    for obj in gc.get_objects():
        try:
            if hasattr(obj, "__class__") and obj.__class__.__name__ in allocator_types:
                if hasattr(obj, "close"):
                    obj.close()
                    cleanup_count += 1
        except Exception:
            pass

    # Cleanup test allocators too
    for obj in gc.get_objects():
        try:
            if hasattr(obj, "_test_allocator") and hasattr(
                obj._test_allocator, "close"
            ):
                obj._test_allocator.close()
                cleanup_count += 1
        except Exception:
            pass

    if cleanup_count > 0:
        print(f"🧹 [DEBUG] Cleaned up {cleanup_count} allocators after test")

    # Force CUDA cleanup after every test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@pytest.fixture(scope="function")
def autorelease_v1(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    print("🧹 [DEBUG] autorelease_v1 cleanup starting - destroying engine 'test'")

    # Destroy engines
    try:
        for engine_name in ["test", "test_engine"]:
            try:
                LMCacheEngineBuilder.destroy(engine_name)
                print(f"✅ [DEBUG] Engine '{engine_name}' destroyed")
            except Exception:
                pass
    except Exception:
        pass

    # Cleanup tracked objects
    for i, obj in enumerate(objects):
        if hasattr(obj, "close"):
            try:
                obj.close()
                print(f"✅ [DEBUG] Object {i + 1} closed")
            except Exception as e:
                print(f"⚠️ [DEBUG] Error closing object {i + 1}: {e}")

    print("🎉 [DEBUG] autorelease_v1 cleanup complete!")
