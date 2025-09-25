# SPDX-License-Identifier: Apache-2.0
"""Test mode implementation for basic checks"""

# Standard
import asyncio
import time

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.v1.check import check_mode

# Import from lmcache with absolute paths
from lmcache.v1.storage_backend import RemoteBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Local
# Import shared utilities
from .utils import (
    EventLoopManager,
    _get_default_metadata,
    create_test_key,
    create_test_memory_obj,
)


@check_mode("test")
async def run_test_mode(model: str, **kwargs):
    """Run connector test mode"""
    config = lmcache_get_or_create_config()
    metadata = _get_default_metadata(model)

    # Create and start event loop manager
    loop_manager = EventLoopManager()
    loop_manager.start()

    local_cpu_backend = LocalCPUBackend(
        config=config, metadata=metadata, dst_device="cpu"
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
            results = []  # Collect results for each operation
            for i, args in enumerate(args_list):
                try:
                    start = time.perf_counter()
                    result = await asyncio.wait_for(func(*args), timeout=timeout)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                    results.append(result)
                    print(
                        f"  Test {i + 1}/{len(args_list)} completed in "
                        f"{(end - start) * 1000:.2f}ms"
                    )
                except asyncio.TimeoutError:
                    print(f"  Test {i + 1}/{len(args_list)} timed out after {timeout}s")
                    times.append(timeout * 1000)
                    results.append(None)
                except Exception as e:
                    print(f"  Test {i + 1}/{len(args_list)} failed: {e}")
                    times.append(0)
                    results.append(None)

            if times:
                return {
                    "time_stats": {
                        "avg": sum(times) / len(times),
                        "max": max(times),
                        "min": min(times),
                    },
                    "results": results,
                }
            else:
                return {"time_stats": {"avg": 0, "max": 0, "min": 0}, "results": []}

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

        exists_non_exist_res = await run_perf_test_with_timeout(
            async_contains, [(key,) for key in non_exist_keys]
        )
        exists_non_exist_stats = exists_non_exist_res["time_stats"]
        # Validation: All non-existing keys should return False
        pass_count = sum(1 for r in exists_non_exist_res["results"] if r is False)
        pass_rate = pass_count / len(non_exist_keys) * 100
        print(
            f"  Validation: {pass_count}/{len(non_exist_keys)} "
            f"passed ({pass_rate:.1f}%)"
        )

        # Phase 2: put test (create new key)
        print("Phase 2: Testing put operations...")

        put_res = await run_perf_test_with_timeout(
            async_submit_put,
            [(backend, exist_keys[i], exist_memories[i]) for i in range(num_tests)],
        )
        put_stats = put_res["time_stats"]
        # Validation: All PUT operations should return True
        pass_count = sum(1 for r in put_res["results"] if r is True)
        pass_rate = pass_count / num_tests * 100
        print(f"  Validation: {pass_count}/{num_tests} passed ({pass_rate:.1f}%)")

        # Phase 3: exists test (key exists)
        print("Phase 3: Testing exists for existing keys...")

        exists_exist_res = await run_perf_test_with_timeout(
            async_contains, [(key,) for key in exist_keys]
        )
        exists_exist_stats = exists_exist_res["time_stats"]
        # Validation: All existing keys should return True
        pass_count = sum(1 for r in exists_exist_res["results"] if r is True)
        pass_rate = pass_count / num_tests * 100
        print(f"  Validation: {pass_count}/{num_tests} passed ({pass_rate:.1f}%)")

        # Phase 4: get test (key exists)
        print("Phase 4: Testing get operations...")

        async def async_get_blocking(key):
            """Async wrapper for get_blocking method"""
            return backend.get_blocking(key)

        get_res = await run_perf_test_with_timeout(
            async_get_blocking, [(key,) for key in exist_keys]
        )
        get_stats = get_res["time_stats"]
        # Validation: Check for non-None results and content correctness
        content_valid_count = 0
        for i, result in enumerate(get_res["results"]):
            if result is None:
                continue
            try:
                if result.tensor is None or exist_memories[i].tensor is None:
                    print(f"  GET for key {exist_keys[i]} returned None tensor")
                    continue

                data_match = torch.equal(result.tensor, exist_memories[i].tensor)
            except Exception as e:
                print(f"  Data comparison failed for key {exist_keys[i]}: {e}")
                data_match = False

            if data_match:
                content_valid_count += 1
            else:
                print(f"  GET for key {exist_keys[i]} returned incorrect memory object")
                if not data_match:
                    print("    Data content mismatch detected")
        # Calculate pass rates
        not_none_count = sum(1 for r in get_res["results"] if r is not None)
        content_pass_rate = content_valid_count / num_tests * 100
        print(f"  Validation (not None): {not_none_count}/{num_tests} passed")
        print(
            f"  Validation (content correct): {content_valid_count}/{num_tests}"
            f" passed ({content_pass_rate:.1f}%)"
        )

        stats_data = [
            (
                "EXISTS (non-exist)",
                exists_non_exist_stats,
                exists_non_exist_res["results"],
            ),
            ("PUT", put_stats, put_res["results"]),
            ("EXISTS (exist)", exists_exist_stats, exists_exist_res["results"]),
            ("GET", get_stats, get_res["results"]),
        ]

        print("\nPerformance Results:")
        print("-" * 100)
        print(
            f"| {'Operation':<20} | {'Avg (ms)':>12} | {'Max (ms)':>12} "
            f"| {'Min (ms)':>12} | {'Pass/All':>10} | {'Pass Rate':>10} |"
        )
        print("-" * 100)
        for op, stats, results in stats_data:
            # Calculate pass count
            if op == "EXISTS (non-exist)":
                pass_count = sum(1 for r in results if r is False)
            elif op == "PUT":
                pass_count = sum(1 for r in results if r is True)
            elif op == "EXISTS (exist)":
                pass_count = sum(1 for r in results if r is True)
            elif op == "GET":
                pass_count = sum(1 for r in results if r is not None)

            total = len(results)
            pass_all = f"{pass_count}/{total}"
            pass_rate = pass_count / total * 100 if total > 0 else 0

            print(
                f"| {op:<20} | {stats['avg']:>12.6f} | {stats['max']:>12.6f} "
                f"| {stats['min']:>12.6f} | {pass_all:>10} | {pass_rate:>9.1f}% |"
            )
        print("-" * 100)

    except Exception as e:
        print(f"Test Failed - Error: {e}")
        # Standard
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        try:
            if backend:
                backend.close()
        except Exception as e:
            print(f"Error closing backend: {e}")

        # Stop the event loop
        loop_manager.stop()


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
