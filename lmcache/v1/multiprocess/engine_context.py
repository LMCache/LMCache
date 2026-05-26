# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class LayoutDescRegistry:
    """Thread-safe registry mapping (model_name, world_size) to MemoryLayoutDesc.

    Modules write to this registry when KV caches are registered.
    Consumers (e.g. LookupModule) read from it to find layout descriptors
    for prefetch tasks.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, int], MemoryLayoutDesc] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_name: str,
        world_size: int,
        layout_desc: MemoryLayoutDesc,
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
        """
        with self._lock:
            self._registry[(model_name, world_size)] = layout_desc

    def unregister(self, model_name: str, world_size: int) -> None:
        """Remove a layout descriptor for a (model_name, world_size) pair.

        Args:
            model_name: The model name.
            world_size: The world size.
        """
        with self._lock:
            self._registry.pop((model_name, world_size), None)

    def find(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Look up a layout descriptor by (model_name, world_size).

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The layout descriptor if found, otherwise None.
        """
        with self._lock:
            return self._registry.get((model_name, world_size))


class MPCacheEngineContext:
    """Shared infrastructure for all engine modules.

    Holds the storage manager, token hasher, session manager, event bus,
    and layout descriptor registry. Modules receive this context at init
    and use it for shared operations.

    Args:
        storage_manager_config: Configuration for the storage manager.
        chunk_size: Chunk size for KV cache operations.
        hash_algorithm: Hash algorithm for token hashing.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ) -> None:
        self.chunk_size = chunk_size
        self.storage_manager = StorageManager(storage_manager_config)
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)
        self.event_bus = get_event_bus()
        self.layout_desc_registry = LayoutDescRegistry()

    def resolve_obj_keys(self, key: IPCCacheEngineKey) -> list[ObjectKey]:
        """Resolve object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range.

        Args:
            key: IPC cache key describing model/session/token range.

        Returns:
            Resolved object keys for the requested token range.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        return ipc_key_to_object_keys(key, chunk_hashes)
