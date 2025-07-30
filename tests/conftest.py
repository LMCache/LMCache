# SPDX-License-Identifier: Apache-2.0

# Set smaller memory allocation for tests at import time to avoid CUDA limits
# Standard
import os

# Force small allocation for tests regardless of CI environment settings
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "0.5"  # 512MB instead of 2GB

# Standard
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from unittest.mock import patch
import asyncio
import random
import shlex
import socket
import subprocess

# Global allocator tracking to ensure only one allocator active at any time
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# Global allocator tracking to ensure only one allocator active at any time
_allocator_lock = threading.Lock()


def _log_cuda_memory_status(context=""):
    """Log CUDA memory status for debugging."""
    # Third Party
    import torch

    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            print(
                f"🔍 [CUDA-{context}] Allocated: {allocated:.1f}MB, \
                    Reserved: {reserved:.1f}MB"
            )
        except Exception as e:
            print(f"🔍 [CUDA-{context}] Error getting memory status: {e}")


# Monkey patch from_legacy to respect LMCACHE_MAX_LOCAL_CPU_SIZE for tests
def _patch_from_legacy():
    """Patch LMCacheEngineConfig.from_legacy to respect environment variables."""
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

    original_from_legacy = LMCacheEngineConfig.from_legacy

    @staticmethod
    def from_legacy_with_env(*args, **kwargs):
        config = original_from_legacy(*args, **kwargs)
        # Override with environment variable if set
        env_cpu_size = os.environ.get("LMCACHE_MAX_LOCAL_CPU_SIZE")
        if env_cpu_size is not None:
            print(
                f"🔧 [PATCH] Overriding max_local_cpu_size \
                    from {config.max_local_cpu_size} to {env_cpu_size}"
            )
            config.max_local_cpu_size = float(env_cpu_size)
        return config

    LMCacheEngineConfig.from_legacy = from_legacy_with_env


# Apply the patch at import time
_patch_from_legacy()


# Also try a more aggressive approach - patch the hardcoded values in from_legacy method
def _patch_from_legacy_aggressive():
    """More aggressive patch that modifies the source code constants."""
    # First Party
    from lmcache.v1 import config as config_module

    # Override the hardcoded values if they exist
    if hasattr(config_module, "LMCacheEngineConfig"):
        original_method = config_module.LMCacheEngineConfig.from_legacy

        @staticmethod
        def from_legacy_override(*args, **kwargs):
            # Call original method
            result = original_method(*args, **kwargs)

            # Force override if environment variable is set
            env_cpu_size = os.environ.get("LMCACHE_MAX_LOCAL_CPU_SIZE")
            if env_cpu_size is not None:
                print(f"🔧 [AGGRESSIVE] Forcing max_local_cpu_size to {env_cpu_size}")
                result.max_local_cpu_size = float(env_cpu_size)

            return result

        config_module.LMCacheEngineConfig.from_legacy = from_legacy_override


# Apply aggressive patch as backup
_patch_from_legacy_aggressive()


def ensure_no_active_allocators():
    """Ensure no allocators are currently active before proceeding."""
    # Standard
    import gc

    # Third Party
    import torch

    _log_cuda_memory_status("PRE-CLEANUP")

    # Force CUDA cleanup BEFORE checking for allocators
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Give CUDA time to process deregistrations
        time.sleep(0.1)

    lock_acquired = _allocator_lock.acquire(timeout=30)
    if not lock_acquired:
        print("⚠️ [LOCK] Failed to acquire allocator lock within 30s, proceeding anyway")
        return

    try:
        # Comprehensive list of all allocator types
        allocator_types = [
            "AdHocMemoryAllocator",
            "PinMemoryAllocator",
            "MixedMemoryAllocator",
            "GPUMemoryAllocator",
            "HostMemoryAllocator",
            "PagedTensorMemoryAllocator",
            "TensorMemoryAllocator",
            "CuFileMemoryAllocator",
            "NixlBufferAllocator",
            "NixlCPUMemoryAllocator",
        ]

        # Collect all allocators to close
        # (separate collection from closing to minimize lock time)
        allocators_to_close = []
        test_allocators_to_close = []
        gpu_tensors_to_clear = []  # New list for GPU tensors
        cuda_contexts_to_clear = []  # New list for CUDA contexts

        for obj in gc.get_objects():
            try:
                if (
                    hasattr(obj, "__class__")
                    and obj.__class__.__name__ in allocator_types
                ):
                    if hasattr(obj, "close"):
                        allocators_to_close.append(obj)
                elif hasattr(obj, "_test_allocator") and hasattr(
                    obj._test_allocator, "close"
                ):
                    test_allocators_to_close.append(obj._test_allocator)
                # Enhanced GPU tensor detection
                elif hasattr(obj, "__class__") and obj.__class__.__name__ == "Tensor":
                    if hasattr(obj, "device") and str(obj.device).startswith("cuda"):
                        # Include ALL GPU tensors, not just pinned ones
                        gpu_tensors_to_clear.append(obj)
                # Look for CUDA context objects
                elif (
                    hasattr(obj, "__class__")
                    and "cuda" in obj.__class__.__name__.lower()
                ):
                    if hasattr(obj, "device") or hasattr(obj, "_device"):
                        cuda_contexts_to_clear.append(obj)
            except Exception:
                # Ignore objects that raise exceptions during attribute access
                pass
    finally:
        _allocator_lock.release()

    # Close allocators outside the lock
    cleanup_count = 0
    for allocator in allocators_to_close:
        try:
            allocator.close()
            cleanup_count += 1
        except Exception:
            pass

    for test_allocator in test_allocators_to_close:
        try:
            test_allocator.close()
            cleanup_count += 1
        except Exception:
            pass

    # Enhanced GPU tensor cleanup
    tensor_count = len(gpu_tensors_to_clear)
    context_count = len(cuda_contexts_to_clear)
    if tensor_count > 0:
        print(f"🧹 [GPU] Found {tensor_count} GPU tensors, clearing references")

        # AGGRESSIVE APPROACH: Try multiple cleanup strategies
        try:
            # Strategy 1: Set tensors to None to break references
            for i, tensor in enumerate(gpu_tensors_to_clear):
                try:
                    # Try to zero out the tensor first
                    if hasattr(tensor, "data") and tensor.data is not None:
                        tensor.data = None
                except Exception:
                    pass

            # Strategy 2: Clear the list and force immediate cleanup
            del gpu_tensors_to_clear

            # Strategy 3: Multiple aggressive CUDA cleanup cycles
            if torch.cuda.is_available():
                # Force CUDA context synchronization
                torch.cuda.synchronize()

                # Multiple empty_cache calls with delays
                for _ in range(5):
                    torch.cuda.empty_cache()
                    time.sleep(0.1)  # Longer delay

                # Force garbage collection between CUDA calls
                for _ in range(5):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(0.05)

        except Exception as e:
            print(f"⚠️ [GPU] Error during aggressive cleanup: {e}")
            # Fallback to original method
            del gpu_tensors_to_clear

    else:
        # Clean up the empty list
        del gpu_tensors_to_clear

    if context_count > 0:
        print(
            f"🧹 [CUDA] Found {context_count} CUDA context objects, clearing references"
        )
        del cuda_contexts_to_clear

    if cleanup_count > 0:
        print(
            f"🧹 [LOCK] Forcibly closed {cleanup_count} allocators for exclusive access"
        )

    _log_cuda_memory_status("MID-CLEANUP")

    # SUPER AGGRESSIVE CUDA cleanup after closing allocators
    if torch.cuda.is_available():
        print("🔥 [GPU] Starting super aggressive CUDA memory cleanup...")

        # Phase 1: Multiple sync + empty cycles
        for i in range(15):  # Increased from 10 to 15 cycles
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(0.1)  # Longer delay

        # Phase 2: Interleaved GC and CUDA cleanup
        for i in range(15):  # Increased from 10 to 15 cycles
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.05)

        # Phase 3: Try to reset all CUDA states
        try:
            torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                torch.cuda.reset_accumulated_memory_stats()
            # Try to reset memory stats multiple ways
            if hasattr(torch.cuda, "reset_max_memory_allocated"):
                torch.cuda.reset_max_memory_allocated()
            if hasattr(torch.cuda, "reset_max_memory_cached"):
                torch.cuda.reset_max_memory_cached()
        except Exception as e:
            print(f"Failed to reset CUDA stats: {e}")

        # Phase 4: Final ultra-aggressive cleanup
        for i in range(10):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.02)

        # Phase 5: Try manual memory freeing if available
        try:
            # Force release of cached memory
            if hasattr(torch.cuda, "memory"):
                if hasattr(torch.cuda.memory, "empty_cache"):
                    torch.cuda.memory.empty_cache()
        except Exception:
            pass

        print("🔥 [GPU] Super aggressive cleanup complete")

    _log_cuda_memory_status("POST-CLEANUP")


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
    print("🚀 [DEBUG] Session startup - ensuring exclusive allocator access...")
    ensure_no_active_allocators()
    print("✅ [DEBUG] Session startup complete")

    yield

    print("🏁 [DEBUG] Session teardown - final cleanup...")
    ensure_no_active_allocators()
    print("🎉 [DEBUG] Session teardown complete")


@pytest.fixture(scope="module", autouse=True)
def module_level_allocator_cleanup():
    """Cleanup allocators between test modules to prevent cross-module leaks."""
    yield  # Let the module run first

    print("🧹 [DEBUG] Module-level allocator cleanup starting...")
    ensure_no_active_allocators()
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
    """Ensure exclusive allocator access BEFORE every single test."""
    ensure_no_active_allocators()
    yield  # Now let the test run with a clean slate


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
