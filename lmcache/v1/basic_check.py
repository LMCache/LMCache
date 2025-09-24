# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import asyncio
import hashlib
import threading
import time

# Third Party
import torch
import tqdm  # Added for progress bar

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.utils import CacheEngineKey

# Import from lmcache with absolute paths
from lmcache.v1.memory_management import (
    MemoryObj,
    MixedMemoryAllocator,
)
from lmcache.v1.storage_backend import RemoteBackend
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


memory_allocator = MixedMemoryAllocator(5 * 1024 * 1024 * 1024)
model_name = "/lmcache_test_model/"


def parse_args():
    parser = argparse.ArgumentParser(description="LMCache basic check Tool")
    parser.add_argument(
        "--mode",
        choices=["test", "gen"],
        default="test",
        help="Operation mode: test (default) or gen",
    )
    parser.add_argument("--model", default=model_name, help="model name")
    parser.add_argument(
        "--num-keys",
        type=int,
        default=100,
        help="Number of keys to generate (gen mode only)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrency level for generation (gen mode only)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for key generation (gen mode only)",
    )
    return parser.parse_args()


def _get_default_metadata(model: str) -> LMCacheEngineMetadata:
    """Get default metadata for testing"""
    return LMCacheEngineMetadata(
        model_name=model,
        world_size=8,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 2, 16, 8, 16),
    )


def create_test_key(model: str, key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        "vllm", model, 8, 0, int(hashlib.sha256(key_id.encode()).hexdigest(), 16)
    )


def create_test_memory_obj(
    backend: RemoteBackend, local_cpu_backend: LocalCPUBackend
) -> MemoryObj:
    """Create a test MemoryObj using AdHocMemoryAllocator for testing."""
    # Check if connection exists and is not None
    if backend.connection is None:
        raise ValueError("Backend connection is None")

    if isinstance(backend.connection, InstrumentedRemoteConnector):
        connector = backend.connection.getWrappedConnector()
    else:
        connector = backend.connection

    memory_obj = local_cpu_backend.allocate(
        connector.meta_shape, connector.meta_dtype, connector.meta_fmt
    )
    return memory_obj


class EventLoopManager:
    """Manages a dedicated event loop in a separate thread"""

    def __init__(self):
        self.loop = None
        self.thread = None
        self._loop_started = threading.Event()

    def start(self):
        """Start the event loop in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            return

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        # Wait a bit to ensure the loop is running
        self._loop_started.wait(timeout=5.0)

    def _run_loop(self):
        """Run the event loop"""
        asyncio.set_event_loop(self.loop)
        self._loop_started.set()
        try:
            self.loop.run_forever()
        except Exception as e:
            print(f"Event loop error: {e}")
        finally:
            self.loop.close()

    def stop(self):
        """Stop the event loop and thread"""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def get_loop(self):
        """Get the event loop"""
        return self.loop


async def async_submit_put(backend, key, memory_obj):
    """Async wrapper for submit_put_task"""
    future = backend.submit_put_task(key, memory_obj)
    # Wait for the future to complete with timeout
    try:
        await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
        return True
    except asyncio.TimeoutError:
        print(f"Put task timed out for key: {key}")
        return False


async def test_connector(model: str):
    config = lmcache_get_or_create_config()
    metadata = _get_default_metadata(model)

    # Create and start event loop manager
    loop_manager = EventLoopManager()
    loop_manager.start()

    # Create local CPU backend as dependency
    local_cpu_backend = LocalCPUBackend(
        config=config, memory_allocator=memory_allocator
    )

    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop_manager.get_loop(),
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
    )

    try:
        # Test basic operations
        print("Testing basic operations...")

        # Performance test function with timeout
        async def run_perf_test_with_timeout(func, args_list, timeout=30.0):
            times = []
            for i, args in enumerate(args_list):
                try:
                    start = time.perf_counter()
                    # Add timeout to prevent hanging
                    await asyncio.wait_for(func(*args), timeout=timeout)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                    print(
                        f"  Test {i + 1}/{len(args_list)} completed in "
                        f"{(end - start) * 1000:.2f}ms"
                    )
                except asyncio.TimeoutError:
                    print(f"  Test {i + 1}/{len(args_list)} timed out after {timeout}s")
                    times.append(timeout * 1000)  # Record timeout as max time
                except Exception as e:
                    print(f"  Test {i + 1}/{len(args_list)} failed: {e}")
                    times.append(timeout * 1000)  # Record failure as max time

            if times:
                return {
                    "avg": sum(times) / len(times),
                    "max": max(times),
                    "min": min(times),
                }
            else:
                return {"avg": 0, "max": 0, "min": 0}

        # Create test data
        num_tests = 5  # Reduced for faster testing

        # Group 1: Non-existing keys
        non_exist_keys = [
            create_test_key(model, f"non_exist_{i}") for i in range(num_tests)
        ]

        # Group 2: Existing keys
        exist_keys = [create_test_key(model, f"exist_{i}") for i in range(num_tests)]
        exist_memories = [
            create_test_memory_obj(backend, local_cpu_backend) for _ in range(num_tests)
        ]

        # Phase 1: exists test (key does not exist)
        print("Phase 1: Testing exists for non-existing keys...")

        async def async_contains(key):
            """Async wrapper for contains method"""
            return backend.contains(key)

        exists_non_exist_stats = await run_perf_test_with_timeout(
            async_contains, [(key,) for key in non_exist_keys]
        )

        # Phase 2: put test (create new key)
        print("Phase 2: Testing put operations...")

        put_stats = await run_perf_test_with_timeout(
            async_submit_put,
            [(backend, exist_keys[i], exist_memories[i]) for i in range(num_tests)],
        )

        # Phase 3: exists test (key exists)
        print("Phase 3: Testing exists for existing keys...")
        exists_exist_stats = await run_perf_test_with_timeout(
            async_contains, [(key,) for key in exist_keys]
        )

        # Phase 4: get test (key exists)
        print("Phase 4: Testing get operations...")

        async def async_get_blocking(key):
            """Async wrapper for get_blocking method"""
            return backend.get_blocking(key)

        get_stats = await run_perf_test_with_timeout(
            async_get_blocking, [(key,) for key in exist_keys]
        )

        # Collect all stats for tabular output
        stats_data = [
            ("EXISTS (non-exist)", exists_non_exist_stats),
            ("PUT", put_stats),
            ("EXISTS (exist)", exists_exist_stats),
            ("GET", get_stats),
        ]

        # Print performance results in table format
        print("\nPerformance Results:")
        print("-" * 60)
        print(
            f"| {'Operation':<20} | {'Avg (ms)':>12} | {'Max (ms)':>12} "
            f"| {'Min (ms)':>12} |"
        )
        print("-" * 60)
        for op, stats in stats_data:
            print(
                f"| {op:<20} | {stats['avg']:>12.6f} | {stats['max']:>12.6f} "
                f"| {stats['min']:>12.6f} |"
            )
        print("-" * 60)

    except Exception as e:
        print(f"Test Failed - Error: {e}")
    finally:
        # Clean up
        try:
            if backend:
                backend.close()
        except Exception as e:
            print(f"Error closing backend: {e}")

        # Stop the event loop
        loop_manager.stop()


async def generate_keys(model: str, num_keys: int, concurrency: int, offset: int = 0):
    """Generate test keys with progress reporting using tqdm."""
    config = lmcache_get_or_create_config()
    metadata = _get_default_metadata(model)

    # Create and start event loop manager
    loop_manager = EventLoopManager()
    loop_manager.start()

    # Create local CPU backend as dependency
    local_cpu_backend = LocalCPUBackend(
        config=config, memory_allocator=memory_allocator
    )

    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop_manager.get_loop(),
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
    )
    try:
        print("Generate: Passed - Created connector with valid config")

        # Create test memory object
        memory_obj = create_test_memory_obj(backend, local_cpu_backend)

        # Create progress bar
        progress_bar = tqdm.tqdm(
            total=num_keys, desc="Generating keys", unit="key", unit_scale=True
        )

        # Generate keys with controlled concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_one_key(i):
            async with semaphore:
                # Use submit_put_task instead of put, and wait for the future
                future = backend.submit_put_task(
                    create_test_key(model, f"gen_{offset + i}"), memory_obj
                )
                # Wait for the future to complete with timeout
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"Put task timed out for key: gen_{offset + i}")
                progress_bar.update(1)

        # Create and run tasks dynamically,
        # ensuring pending tasks don't exceed batch size
        pending_tasks: set[asyncio.Task[None]] = set()
        for i in range(num_keys):
            # Wait if pending tasks exceed batch size
            while len(pending_tasks) >= 10 * concurrency:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
            task = asyncio.create_task(generate_one_key(i))
            pending_tasks.add(task)

        # Wait for all remaining tasks to complete
        await asyncio.wait(pending_tasks)

        progress_bar.close()
    except Exception as e:
        print(f"Generate: Failed - Error creating connector with valid config: {e}")
    finally:
        if backend:
            backend.close()


async def main():
    args = parse_args()

    if args.mode == "gen":
        await generate_keys(args.model, args.num_keys, args.concurrency, args.offset)
    else:
        await test_connector(args.model)


if __name__ == "__main__":
    asyncio.run(main())
