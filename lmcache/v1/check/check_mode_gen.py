# SPDX-License-Identifier: Apache-2.0
"""Generate mode implementation for key generation"""

# Standard
import asyncio

# Third Party
import tqdm

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


@check_mode("gen")
async def run_gen_mode(
    model: str, num_keys: int, concurrency: int, offset: int = 0, **kwargs
):
    """Run key generation mode"""
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
