# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for check modes"""

# Standard
import asyncio
import hashlib
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey

# Import from lmcache with absolute paths
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend import RemoteBackend
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


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
    """Create a test MemoryObj for testing."""
    if backend.connection is None:
        raise ValueError("Backend connection is None")

    if isinstance(backend.connection, InstrumentedRemoteConnector):
        connector = backend.connection.getWrappedConnector()
    else:
        connector = backend.connection

    return local_cpu_backend.allocate(
        connector.meta_shape, connector.meta_dtype, connector.meta_fmt
    )


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
        self._loop_started.wait()

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
